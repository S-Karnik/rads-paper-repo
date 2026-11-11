import os
from typing import Any, Dict, Tuple

import lightning as L
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import os

from train import sac_steer_lib
from clip_autoencoder import model as clip_vae_model

from ..dataset import MemorizedPrompts
from ..mitigate import MemorizationMitigator
from ..paths import LOG_DIR, root_dir
from ..pipe import StableDiffusion

from ..misc import load_runner
from ..paths import LOG_DIR, root_dir

__all__ = [
    "Runner",
]


class Wrapper(L.LightningModule):
    def __init__(
        self,
        pipe: StableDiffusion,
        config: Dict[str, Any],
    ) -> None:
        super().__init__()

        pipe_name: str = config.get("pipe", "SDV1")
        scheduler: str = config.get("scheduler", "DDIM")
        num_inference_steps: int = config.get("T", 50)
        guidance_scale: float = config.get("g", 7.5)
        seed: int = config.get("seed", 0)
        train_seed: int = config.get("train_seed", 0)
        batch_size = config.get("batch_size", 1)
        num_workers = config.get("num_workers", 0)
        devices = config.get("devices", "auto")
        mitigate_strategy: str = config.get("mitigate", "none")
        vae_path: str = config.get("vae_path", None)
        sac_actor_path: str = config.get("sac_actor_path", None)
        sac_critic_path: str = config.get("sac_critic_path", None)
        sac_encoder_path: str = config.get("sac_encoder_path", None)
        sac_safety_critic_path: str = config.get("sac_safety_critic_path", None)
        time_embed_dim: int = config.get("time_embed_dim", 16)
        learn_lambda: bool = config.get("learn_lambda", False)
        safety_target: float = config.get("safety_target", 0.0)
        do_not_use_safety_critic: bool = config.get("do_not_use_safety_critic", False)
        sum_ablation: bool = config.get("sum_ablation", False)

        if devices == "auto" and torch.cuda.device_count() > 1:
            self.sync_dist = True
        else:
            self.sync_dist = (
                isinstance(devices, str)
                and "gpu:" in devices
                and len(devices.split("gpu:")[-1].split(",")) > 1
            )

        self.pipe = pipe
        if "sac" in mitigate_strategy:
            self.log_dir = os.path.join(
                root_dir(
                    root=LOG_DIR,
                    config=config,
                ),
                "generated_samples",
                mitigate_strategy,
                f"train_seed={train_seed}",
                f"seed={seed}",
            )
        else:
            self.log_dir = os.path.join(
                root_dir(
                    root=LOG_DIR,
                    config=config,
                ),
                "generated_samples",
                mitigate_strategy,
                f"seed={seed}",
            )
        print("============")
        print("self.log_dir = ", self.log_dir)
        print("============")
        self._grad_flag = (
            torch.enable_grad()
            if mitigate_strategy == "wen_iclr2024"
            else torch.no_grad()
        )

        self.config = {
            "pipe": pipe_name,
            "scheduler": scheduler,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "devices": devices,
            "mitigate_strategy": mitigate_strategy,
            "vae_path": vae_path,
            "sac_actor_path": sac_actor_path,
            "sac_critic_path": sac_critic_path,
            "sac_encoder_path": sac_encoder_path,
            "sac_safety_critic_path": sac_safety_critic_path,
            "time_embed_dim": time_embed_dim,
            "learn_lambda": learn_lambda,
            "safety_target": safety_target,
            "do_not_use_safety_critic": do_not_use_safety_critic,
            "sum_ablation": sum_ablation,
        }

    def on_test_start(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)

        self.pipe.init(
            num_inference_steps=self.config["num_inference_steps"],
            guidance_scale=self.config["guidance_scale"],
        )
        if "sac" in self.config["mitigate_strategy"]:
            print("self.config = ", self.config)

            # Initialize SAC Agent
            args = argparse.Namespace(
                gamma=0.99,
                tau=0.005,
                alpha=0.2,
                action_reg=self.config.get("action_reg", 1.0),
                device=self.device.index if hasattr(self.device, "index") and self.device.index is not None else 0,
                caption_dim=self.config.get("caption_dim", 64),
                critic_lr=0.0001,
                actor_lr=0.0001,
                alpha_lr=0.0001,
                vae_path=self.config.get("vae_path"),
                saved_sac_actor_path=self.config.get("sac_actor_path"),
                saved_sac_critic_path=self.config.get("sac_critic_path"),
                saved_sac_encoder_path=self.config.get("sac_encoder_path"),
                saved_sac_safety_critic_path=self.config.get("sac_safety_critic_path"),
                time_embed_dim=self.config.get("time_embed_dim"),
                learn_lambda=self.config.get("learn_lambda"),
                safety_target=self.config.get("safety_target"),
                pipe=self.config.get("pipe"),
                scheduler=self.config.get("scheduler"),
                lambda_lr=0.0001,
                do_not_use_safety_critic=self.config.get("do_not_use_safety_critic"),
                sum_ablation=self.config.get("sum_ablation"),
            )
            self.sac_agent = sac_steer_lib.SAC(
                action_dim=args.caption_dim,
                caption_dim=args.caption_dim,
                args=args,
            )
            self.sac_agent.eval()
            self.generator = self.pipe
            
            # Load VAE for SAC
            self.vae = clip_vae_model.CLIPVAE(latent_dim=64, decoder_final_norm=False).to(self.device)
            print(f"self.config['vae_path']: {self.config['vae_path']}")
            if self.config.get("vae_path") and os.path.exists(self.config["vae_path"]):
                self.vae.load_state_dict(torch.load(self.config["vae_path"], map_location=self.device))
                print(f"Loaded VAE from {self.config['vae_path']}")
            else:
                print(f"WARNING: VAE checkpoint not found. Initializing random weights.")
            self.vae.eval()

        elif self.config["mitigate_strategy"] != "none":
            self.generator = MemorizationMitigator(
                pipe=self.pipe,
                strategy=self.config["mitigate_strategy"],
            )
        else:
            self.generator = self.pipe
        self.generator.to(self.device)

    def test_step(self, batch, batch_idx):
        # Unpack batch
        indices: Tuple[int] = batch[0]
        prompts: Tuple[str] = batch[2]

        # Generate
        if "sac" in self.config["mitigate_strategy"]:
            # Manual generation loop with SAC
            prompts_list = list(prompts)
            
            # Prepare
            prompt_embeds = self.generator.prepare_embeds(prompts_list)
            timesteps, num_timesteps, num_warmup_steps = self.generator.prepare_timesteps()
            latents = self.generator.prepare_latents(prompt_embeds, seed=self.config["seed"])
            
            # Encode prompts with VAE for SAC
            with torch.no_grad():
                current_prompt_latents_flat, _ = self.vae.encode(prompt_embeds)
                neg_latents, pos_latents = current_prompt_latents_flat.chunk(2)
                current_prompt_latents = torch.stack([neg_latents, pos_latents], dim=1)

            # Denoising loop
            for i, t in tqdm(enumerate(timesteps), total=num_timesteps, leave=False):
                # Select action
                action_latent = self.sac_agent.select_action(latents, current_prompt_latents, i/num_timesteps, evaluate=True)
                
                # Apply steering
                LATENT_SCALE = 1.0
                with torch.no_grad():
                    cond_prompt_embeds = prompt_embeds.chunk(2)[1]
                    mu, _ = self.vae.encode(cond_prompt_embeds)
                    z_steered = mu + (action_latent * LATENT_SCALE)
                    steered_cond_embedding = self.vae.decode(z_steered)
                    if steered_cond_embedding.dim() == 4: steered_cond_embedding = steered_cond_embedding.squeeze(1)
                    neg_embeds = prompt_embeds.chunk(2)[0]
                    steered_embeds = torch.cat([neg_embeds, steered_cond_embedding], dim=0)
                
                steered_embeds = steered_embeds.to(dtype=latents.dtype)
                
                # Predict noise
                noise_pred = self.generator.predict_noise(latents, steered_embeds, t)
                noise_pred = self.generator.perform_guidance(noise_pred)
                
                # Step
                latents = self.generator.step(noise_pred, t, latents)
                
            # Decode
            images = self.generator.decode(latents)

        else:
            with self._grad_flag:
                images = self.generator(
                    prompt=list(prompts),
                    seed=self.config["seed"],
                )
            if self.config["mitigate_strategy"] == "none":
                images = images.images

        # Save
        for index, image in zip(indices, images):
            image.save(os.path.join(self.log_dir, f"{index}.png"))

        del images
        torch.cuda.empty_cache()

        return {}

    def on_test_end(self) -> None:
        pass


class Runner:
    def __init__(
        self,
        config: Dict[str, Any],
    ) -> None:
        self.runner: L.Trainer = load_runner(config)
        self.config = config

    def run(
        self,
        dataset: MemorizedPrompts,
        pipe: StableDiffusion,
    ) -> None:
        dataloader: DataLoader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
        )
        model: L.LightningModule = Wrapper(
            pipe=pipe,
            config=self.config,
        )
        self.runner.test(
            model=model,
            dataloaders=dataloader,
        )
