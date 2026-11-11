import os
import pathlib
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torchvision.transforms.functional as F
from metrics import clip_alignment, image_similarity
from PIL import Image
from train import sac_steer_lib, transform_score

from .vae import CLIPVAE

LATENT_SCALE = 1.0


@dataclass
class Transition:
    state_img: torch.Tensor
    state_cap: torch.Tensor
    action: torch.Tensor
    task_reward: List[float]  # Sparse CLIP Reward (At end)
    safety_reward: List[float]  # Dense Negative L2 (Every step)
    final_reward_components: List[Optional[Tuple[float, float, float]]]
    next_state_img: torch.Tensor
    next_state_cap: torch.Tensor
    done: bool
    timestep: List[float]


class SteeringSimulator:
    def __init__(
        self,
        pipe_name: str,
        scheduler_name: str,
        num_timesteps: int,
        guidance_scale: float,
        sac_agent: sac_steer_lib.SAC,
        vae_path: str,
        tanh_l2_threshold: float,
        device: str = "cuda",
        train_mode: bool = False,
    ):
        self._pipe_name = pipe_name
        self._scheduler_name = scheduler_name
        self._num_timesteps = num_timesteps
        self._guidance_scale = guidance_scale
        self._sac_agent = sac_agent
        self._diffusion_model_type = diffusion_model_type
        self._tanh_l2_threshold = tanh_l2_threshold
        self._device = device
        self._train_mode = train_mode
        self._pipe = None
        self._vae = CLIPVAE().to(self._device)
        if os.path.exists(vae_path):
            self._vae.load_state_dict(torch.load(vae_path, map_location=self._device))
            print(f"Loaded VAE from {vae_path}")
        else:
            print(
                f"WARNING: VAE checkpoint not found at {vae_path}. Initializing random weights."
            )
        self._vae.eval()
        self._dataset_path = utils.paths.get_prompt_path(diffusion_model_type)
        self._parquet_dataset = prompt_utils.load_parquet_file(
            pathlib.Path(self._dataset_path)
        )

    def eval(self):
        self._sac_agent.eval()

    def reset_pipeline(self) -> None:
        if self._pipe is None:
            self._pipe = utils.pipe.load_pipe(self._pipe_name, self._scheduler_name)
        self._pipe.init(
            num_inference_steps=self._num_timesteps, guidance_scale=self._guidance_scale
        )
        self._pipe.to(self._device)

    def get_target_images(self, prompts: List[str]) -> torch.Tensor:
        images = []
        for prompt in prompts:
            caption_element_idx = self._parquet_dataset[
                self._parquet_dataset["caption"] == prompt
            ].index[0]
            image_path = os.path.join(
                utils.paths.get_dataset_dir(self._diffusion_model_type),
                "image",
                f"{caption_element_idx}.png",
            )
            pil_image = Image.open(image_path).convert("RGB")
            pil_image = pil_image.resize((512, 512), Image.Resampling.BICUBIC)
            images.append(F.to_tensor(pil_image))
        return torch.stack(images).to(dtype=torch.float32, device=self._device)

    def run_batch_episode(
        self, prompts: List[str], seeds: Optional[Union[int, List[int]]] = None
    ):
        self.reset_pipeline()
        batch_size = len(prompts)
        if not self._train_mode and seeds is None:
            raise ValueError("Seeds required for eval")

        target_images = self.get_target_images(prompts)
        aligner = clip_alignment.CLIPAlignment(
            memorized_images=target_images, device=self._device
        )

        with torch.no_grad():
            target_text_embeds = aligner.compute_text_embeddings(prompts)

        prompt_embeds = self._pipe.prepare_embeds(prompts)
        timesteps, _, _ = self._pipe.prepare_timesteps()

        diffusion_latents = self._pipe.prepare_latents(prompt_embeds, seed=seeds).to(
            self._device
        )

        with torch.no_grad():
            current_prompt_latents_flat, _ = self._vae.encode(prompt_embeds)
            neg_latents, pos_latents = current_prompt_latents_flat.chunk(2)
            current_prompt_latents = torch.stack([neg_latents, pos_latents], dim=1)

        for i, t in enumerate(timesteps):
            if isinstance(t, torch.Tensor):
                t = t.to(self._device)

            action_latent = self._sac_agent.select_action(
                diffusion_latents,
                current_prompt_latents,
                i / self._num_timesteps,
                evaluate=not self._train_mode,
            )

            with torch.no_grad():
                cond_prompt_embeds = prompt_embeds.chunk(2)[1]
                mu, _ = self._vae.encode(cond_prompt_embeds)
                z_steered = mu + (action_latent * LATENT_SCALE)
                steered_cond_embedding = self._vae.decode(z_steered)
                if steered_cond_embedding.dim() == 4:
                    steered_cond_embedding = steered_cond_embedding.squeeze(1)
                neg_embeds = prompt_embeds.chunk(2)[0]
                steered_embeds = torch.cat([neg_embeds, steered_cond_embedding], dim=0)

            steered_embeds = steered_embeds.to(dtype=diffusion_latents.dtype)

            noise_pred = self._pipe.predict_noise(diffusion_latents, steered_embeds, t)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self._guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            next_latents = self._pipe.step(noise_pred, t, diffusion_latents)

            # --- 1. Compute Dense Safety Reward (Negative L2) ---
            diff = noise_pred_uncond - noise_pred_cond
            l2_norms = torch.norm(diff, p=2, dim=(1, 2, 3))
            transformed_l2 = -1.0 * transform_score.tanh_transform(
                l2_norms, scale=0.1, threshold=self._tanh_l2_threshold
            )

            # Safety Reward: Negative L2 (Cost). Maximize towards 0.
            safety_rewards = transformed_l2.tolist()

            is_last_step = i == len(timesteps) - 1

            task_rewards = [0.0] * batch_size
            final_components_list = [None] * batch_size

            if is_last_step:
                # Decode batch
                images_pred = self._pipe.decode(next_latents)

                if isinstance(images_pred, torch.Tensor):
                    param_images = F.resize(
                        images_pred, target_images.shape[-2:], antialias=True
                    )
                else:
                    param_images = torch.stack(
                        [F.to_tensor(img) for img in images_pred]
                    ).to(self._device)
                    param_images = F.resize(
                        param_images, target_images.shape[-2:], antialias=True
                    )

                if not self._train_mode:
                    for idx, (img, prompt) in enumerate(zip(images_pred, prompts)):
                        os.makedirs(f"eval_images/{prompt}", exist_ok=True)
                        s_val = (
                            seeds[idx]
                            if isinstance(seeds, list)
                            else (seeds if isinstance(seeds, int) else idx)
                        )
                        if isinstance(img, torch.Tensor):
                            img = F.to_pil_image(img)
                        img.save(f"eval_images/{prompt}/seed_{s_val}_steered.png")

                with torch.no_grad():
                    gen_img_embeds = aligner._compute_embeddings(param_images)
                    target_image_embeds = aligner._compute_embeddings(target_images)

                # --- 2. Compute Sparse Task Reward (CLIP) ---
                text_score = (gen_img_embeds * target_text_embeds).sum(dim=-1).tolist()
                img_score = (gen_img_embeds * target_image_embeds).sum(dim=-1).tolist()

                task_rewards = text_score
                final_components_list = list(
                    zip(transformed_l2.tolist(), text_score, img_score)
                )

            yield Transition(
                state_img=diffusion_latents.cpu(),
                state_cap=current_prompt_latents.cpu(),
                action=action_latent.cpu(),
                task_reward=task_rewards,  # Sparse CLIP (for Q)
                safety_reward=safety_rewards,  # Dense Tanh-transformed L2 (for V/Q)
                final_reward_components=final_components_list,
                next_state_img=next_latents.cpu(),
                next_state_cap=current_prompt_latents.cpu(),
                done=is_last_step,
                timestep=[i / self._num_timesteps] * batch_size,
            )

            diffusion_latents = next_latents
