import os
import json
import random
from typing import Any, Dict, Tuple, List
import argparse
from tqdm import tqdm
import numpy as np

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from train import sac_steer_lib
from clip_autoencoder import model as clip_vae_model
from wah.diffusion.models import StableDiffusion

from ..misc import load_runner
from ..paths import LOG_DIR, root_dir, get_prompt_path, get_dataset_dir
from ..dataset import utils as dataset_utils
from .. import model_types
from ..args import load_args
from PIL import Image
import utils

__all__ = [
    "Runner",
]

def load_and_filter_dataset(dataset_path, diffusion_model_type):
    # Logic adapted from train/compute_zero_threshold.py
    parquet_dataset = dataset_utils.load_parquet_file(dataset_path)
    
    valid_indices = []
    dataset_dir = get_dataset_dir(diffusion_model_type)
    
    print(f"Filtering dataset from {dataset_dir}...")
    for idx, row in tqdm(parquet_dataset.iterrows(), total=len(parquet_dataset), desc="Checking images"):
        image_path = os.path.join(
            dataset_dir,
            "image", f"{idx}.png",
        )
        
        if os.path.exists(image_path):
            try:
                with Image.open(image_path) as img:
                    img.verify()
                valid_indices.append(idx)
            except Exception:
                pass
                
    filtered_dataset = parquet_dataset.loc[valid_indices]
    print(f"Dataset filtered. Original: {len(parquet_dataset)}, Valid: {len(filtered_dataset)}")
    return filtered_dataset

class NormWrapper(L.LightningModule):
    def __init__(
        self,
        pipe: StableDiffusion,
        config: Dict[str, Any],
    ) -> None:
        super().__init__()

        self.config = config
        self.pipe = pipe
        self.results = [] # To store results for all batches

        # Setup basics (similar to generate.py)
        devices = config.get("devices", "auto")
        if devices == "auto" and torch.cuda.device_count() > 1:
            self.sync_dist = True
        else:
            self.sync_dist = (
                isinstance(devices, str)
                and "gpu:" in devices
                and len(devices.split("gpu:")[-1].split(",")) > 1
            )
        
        self.log_dir = self.config.get("log_dir", ".")

    def on_test_start(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)

        self.pipe.init(
            num_inference_steps=self.config.get("num_inference_steps", 50),
            guidance_scale=self.config.get("guidance_scale", 7.5),
        )
        
        mode = self.config.get("mode", "steered")
        
        if mode == "steered":
            # Initialize SAC Agent (only SAC is supported for this measurement)
            if self.config["mitigate"] == "sac_safety_critic":
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
                    time_embed_dim=self.config.get("time_embed_dim", 16),
                    learn_lambda=self.config.get("learn_lambda", True),
                    safety_target=self.config.get("safety_target", 9.0),
                    lambda_lr=self.config.get("lambda_lr", 0.0001),
                    pipe=self.config.get("pipe", "SDV1"),
                )
                 self.sac_agent = sac_steer_lib.SAC(
                    action_dim=args.caption_dim,
                    caption_dim=args.caption_dim,
                    args=args,
                )
                 self.sac_agent.eval()
                 self.generator = self.pipe
                
                 self.vae = clip_vae_model.CLIPVAE(latent_dim=64, decoder_final_norm=False).to(self.device)
                 
                 if self.config.get("vae_path") and os.path.exists(self.config["vae_path"]):
                    self.vae.load_state_dict(torch.load(self.config["vae_path"], map_location=self.device))
                    print(f"Loaded VAE from {self.config['vae_path']}")
                 else:
                    print(f"WARNING: VAE checkpoint not found. Initializing random weights.")
                 self.vae.eval()
            else:
                raise ValueError("Only 'sac_safety_critic' mitigation strategy is supported for 'steered' mode.")
        elif mode == "baseline":
            # For baseline, we just use the pipe directly as generator
             self.generator = self.pipe
        else:
            raise ValueError(f"Unknown mode: {mode}")

        self.generator.to(self.device)

    def run_diffusion_pass(self, latents, prompt_embeds, is_steered=False):
        timesteps, num_timesteps, num_warmup_steps = self.generator.prepare_timesteps()
        
        # Prepare for steering if needed
        if is_steered:
             # Encode prompts with VAE for SAC
            with torch.no_grad():
                current_prompt_latents_flat, _ = self.vae.encode(prompt_embeds)
                neg_latents, pos_latents = current_prompt_latents_flat.chunk(2)
                current_prompt_latents = torch.stack([neg_latents, pos_latents], dim=1)
        
        batch_size = latents.shape[0] if not self.generator.pipe.do_classifier_free_guidance else latents.shape[0] # latents already have batch dim
        # Actually latents are [B, C, H, W]. 
        
        norms = {i: [] for i in range(len(latents))} # Assuming latents correspond one-to-one with batch indices 0..B-1
        
        curr_latents = latents.clone()
        
        for i, t in enumerate(timesteps):
            # 1. Calculate Conditional Embeddings (potentially steered)
            current_prompt_embeds = prompt_embeds
            
            if is_steered:
                # Select action
                action_latent = self.sac_agent.select_action(curr_latents, current_prompt_latents, i/num_timesteps, evaluate=True)
                
                # Apply steering
                LATENT_SCALE = 1.0
                with torch.no_grad():
                    cond_prompt_embeds = prompt_embeds.chunk(2)[1]
                    mu, _ = self.vae.encode(cond_prompt_embeds)
                    z_steered = mu + (action_latent * LATENT_SCALE)
                    steered_cond_embedding = self.vae.decode(z_steered)
                    if steered_cond_embedding.dim() == 4: steered_cond_embedding = steered_cond_embedding.squeeze(1)
                    neg_embeds = prompt_embeds.chunk(2)[0]
                    current_prompt_embeds = torch.cat([neg_embeds, steered_cond_embedding], dim=0)
                    
                    # Ensure dtype match
                    current_prompt_embeds = current_prompt_embeds.to(dtype=curr_latents.dtype)

            # 2. Predict Noise
            noise_pred = self.generator.predict_noise(curr_latents, current_prompt_embeds, t)
            
            # 3. Compute Guidance Norm (Steering agnostic logic for measurement)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            # diff = noise_pred_uncond - noise_pred_text  <-- Order doesn't matter for norm
            guidance_diff = noise_pred_text - noise_pred_uncond
            guidance_norm = torch.linalg.norm(guidance_diff.flatten(1), dim=1) # [batch_size]
            
            for b_idx in range(len(guidance_norm)):
                norms[b_idx].append(guidance_norm[b_idx].item())
            
            # 4. Standard CFG Step
            # We use the standard guidance scale for the actual diffusion step to match 'compute_zero_threshold' logic
            # or just standard generation logic.
            # noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            # But we can use the pipeline's helper if we didn't need to inspect components, 
            # here we reconstruct it manually to be safe/explicit or just use the noise_pred we have.
            # Note: perform_guidance does exactly this.
            
            noise_pred_step = self.generator.perform_guidance(noise_pred)
            
            # Step
            curr_latents = self.generator.step(noise_pred_step, t, curr_latents)
            
        return norms

    def test_step(self, batch, batch_idx):
        # Unpack batch
        indices = batch[0]
        prompts: Tuple[str] = batch[2]
        is_memorized_batch = batch[3]

        prompts_list = list(prompts)
        
        # Prepare
        prompt_embeds = self.generator.prepare_embeds(prompts_list)
        # Latents need to be identical for both passes
        latents_initial = self.generator.prepare_latents(prompt_embeds, seed=self.config["seed"])
        
        mode = self.config.get("mode", "steered")
        
        norms_baseline = None
        norms_steered = None
        
        if mode == "baseline":
            # Pass 1: Baseline
            norms_baseline = self.run_diffusion_pass(latents_initial, prompt_embeds, is_steered=False)
        elif mode == "steered":
             # Pass 2: Steered
            norms_steered = self.run_diffusion_pass(latents_initial, prompt_embeds, is_steered=True)
            
        # Collect results
        for b_idx, prompt in enumerate(prompts_list):
            res = {
                "prompt": prompt,
                "is_memorized": bool(is_memorized_batch[b_idx]),
            }
            if norms_baseline:
                res["guidance_norm_baseline"] = norms_baseline[b_idx]
            if norms_steered:
                res["guidance_norm_steered"] = norms_steered[b_idx]
                
            self.results.append(res)

        return {}

    def on_test_end(self) -> None:
        # Save results
        mode = self.config.get("mode", "steered")
        if mode == "baseline":
            filename = f"{self.config['pipe']}_{self.config['scheduler']}_baseline_{self.config['seed']}_guidance_norms.json"
        else:
            filename = f"{self.config['pipe']}_{self.config['scheduler']}_steered_{self.config['train_seed']}_{self.config['seed']}_guidance_norms.json"
            
        output_path = os.path.join(self.log_dir, filename)
        
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        print(f"Saved Guidance Norms to {output_path}")

        # Compute and print statistics
        def get_means_stds(res_list, key):
             if not res_list: return None, None
             # Filter out if key doesn't exist (though it should for all if mode is consistent)
             valid = [r[key] for r in res_list if key in r]
             if not valid: return None, None
             arr = np.array(valid)
             return np.mean(arr, axis=0), np.std(arr, axis=0)

        mem_results = [r for r in self.results if r['is_memorized']]
        non_mem_results = [r for r in self.results if not r['is_memorized']]

        def print_stats(name, results, key):
            if not results:
                print(f"\nStats for {name}: No data.")
                return
            
            mean, std = get_means_stds(results, key)
            if mean is None: return

            print(f"\nStats for {name} ({key}):")
            print(f"{'Step':<5} | {'Mean':<10} ± {'Std':<10}")
            print("-" * 35)
            for t in range(len(mean)):
                print(f"{t:<5} | {mean[t]:.4f} ± {std[t]:.4f}")

        if mode == "baseline":
            print_stats("Memorized (Baseline)", mem_results, 'guidance_norm_baseline')
        else:
            print_stats("Memorized (Steered)", mem_results, 'guidance_norm_steered')

class CustomDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return format matching what Wrapper expects
        # Wrapper expects batch[0] = indices, batch[2] = prompts
        # We will pack: (index, index, prompt, is_memorized, ...)
        item = self.data[idx]
        return (idx, idx, item['prompt'], item['is_memorized'])


class Runner:
    def __init__(
        self,
        config: Dict[str, Any],
    ) -> None:
        self.runner: L.Trainer = load_runner(config)
        self.config = config

    def run(self) -> None:
        # Load Dataset
        pipe_name = self.config.get("pipe", "SDV1")
        if pipe_name == "SDV1":
            diffusion_model_type = model_types.ModelType.STABLE_DIFFUSION_V1
        elif pipe_name == "RealVis":
            diffusion_model_type = model_types.ModelType.REAL_VISION
        else:
            raise ValueError(f"Invalid pipe name: {pipe_name}")

        dataset_path = get_prompt_path(diffusion_model_type)
        print(f"Loading dataset from {dataset_path}")
        
        dataset_path = get_prompt_path(diffusion_model_type)
        print(f"Loading dataset from {dataset_path}")
        
        # Load and filter dataset
        parquet_dataset = load_and_filter_dataset(dataset_path, diffusion_model_type)
        
        # Filter Memorized vs Non-Memorized
        if 'overfit_type' not in parquet_dataset.columns:
             # Fallback or error? Assuming dataset has this as per user instructions
             print("Warning: 'overfit_type' not found, treating all as non-memorized for structure")
             parquet_dataset['overfit_type'] = 'N'
        
        mem_df = parquet_dataset[parquet_dataset['overfit_type'] != 'N']
        non_mem_df = parquet_dataset[parquet_dataset['overfit_type'] == 'N']
        
        # Sampling with fixed seed
        sampling_seed = 42
        rng = np.random.default_rng(sampling_seed)
        
        # Sample 50 from each
        n_sample = 50
        
        def sample_prompts(df, label_is_mem):
            if len(df) < n_sample:
                print(f"Warning: Not enough samples for is_memorized={label_is_mem} (Found {len(df)}). Taking all.")
                selected = df
            else:
                indices = rng.choice(len(df), size=n_sample, replace=False)
                selected = df.iloc[indices]
            
            data = []
            for _, row in selected.iterrows():
                data.append({
                    "prompt": row['caption'],
                    "is_memorized": label_is_mem
                })
            return data

        data_list = sample_prompts(mem_df, True)
        
        print(f"data_list: {data_list}")
        print(f"Selected {len(data_list)} prompts ({len([x for x in data_list if x['is_memorized']])} memorized)")

        dataset = CustomDataset(data_list)
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get("batch_size", 10), # Default larger batch size for speed
            shuffle=False,
            num_workers=self.config.get("num_workers", 0),
        )

        # Initialize Pipe (needed for Wrapper)
        pipe = utils.load_pipe(args.pipe, args.scheduler)

        model = NormWrapper(
            pipe=pipe,
            config=self.config,
        )
        
        self.runner.test(
            model=model,
            dataloaders=dataloader,
        )

if __name__ == "__main__":
    # Load args using the shared utility
    args = load_args()
    
    # Construct config
    config = vars(args)
    
    # Running
    runner = Runner(config)
    runner.run()
