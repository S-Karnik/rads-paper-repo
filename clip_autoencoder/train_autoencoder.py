import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from tqdm import tqdm
from typing import List
import wandb
from transformers import CLIPTextModel, CLIPTokenizer

from clip_autoencoder.model import CLIPVAE
from clip_autoencoder.eval_autoencoder import evaluate_vae
from utils.dataset import utils as dataset_utils
from PIL import Image
from utils import misc

def load_valid_prompts(prompts_root, diffusion_model_name, prompts, df) -> List[str]:
    remaining_prompts = []
    for prompt in prompts:
        try:
            caption_element_idx = df[df['caption'] == prompt].index[0]
            image_path = os.path.join(
                f"{prompts_root}/{diffusion_model_name}",
                "image", f"{caption_element_idx}.png",
            )
            pil_image = Image.open(image_path).convert("RGB")
            pil_image = pil_image.resize((512, 512), Image.Resampling.BICUBIC)
            remaining_prompts.append(prompt)
        except Exception as e:
            pass
    return remaining_prompts


def train_vae(args, validation_prompts=None):
    wandb.init(project="clip-autoencoder", config=vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    misc.set_seed(seed)
    
    # --- CONFIG ---
    BATCH_SIZE = 64
    TOTAL_STEPS = 50000
    NOISE_LEVEL = 0.05 

    embed_dim = None
    if args.diffusion_model_name == "SDV1":
        embed_dim = 768
    elif args.diffusion_model_name == "SDV2":
        embed_dim = 1024
    else:
        raise ValueError(f"Unknown diffusion model type: {args.diffusion_model_type}") 
    
    # 1. Load Models
    if args.diffusion_model_name == "SDV1":
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    elif args.diffusion_model_name == "SDV2":
        text_encoder = CLIPTextModel.from_pretrained("sd2-community/stable-diffusion-2-1").to(device).eval()
        tokenizer = CLIPTokenizer.from_pretrained("sd2-community/stable-diffusion-2-1", subfolder="tokenizer")
    else:
        raise ValueError(f"Unknown diffusion model type: {args.diffusion_model_type}") 
    
    # 2. DATASET MIXING STRATEGY
    print("Loading Datasets...")
    
    # Internet Data
    internet_dataset = load_dataset("conceptual_captions", split="train", streaming=True)
    internet_dataset = internet_dataset.shuffle(seed=seed, buffer_size=10_000)

    # If list is empty for testing, fill with dummy data
    if len(validation_prompts) < 5:
        print("WARNING: Using dummy prompts. Please load your real dataset.")
        validation_prompts = ["Test prompt"] * 500

    print(f"Mixing Internet Data with {len(validation_prompts)} Validation Prompts...")

    # Pre-tokenize special prompts for speed
    with torch.no_grad():
        # Process in chunks to avoid OOM
        special_embeds = []
        chunk_size = 100
        for i in range(0, len(validation_prompts), chunk_size):
            chunk = validation_prompts[i:i+chunk_size]
            tokens = tokenizer(chunk, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
            embeds = text_encoder(tokens.input_ids)[0].cpu()
            special_embeds.append(embeds)
        special_embeds = torch.cat(special_embeds, dim=0)

    # 3. Model Setup
    decoder_final_norm = False
    vae = CLIPVAE(latent_dim=64, dropout=0.1, decoder_final_norm=decoder_final_norm).to(device)
    optimizer = optim.AdamW(vae.parameters(), lr=2e-4, weight_decay=1e-3) 
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-4, total_steps=TOTAL_STEPS, pct_start=0.1
    )
    
    cosine_loss_fn = nn.CosineEmbeddingLoss() 
    mse_loss_fn = nn.MSELoss()
    target_flag = torch.ones(BATCH_SIZE * 77).to(device)
    MAX_KL_WEIGHT = 0.002
    
    # We will iterate manually to mix data
    internet_iter = iter(internet_dataset)
    
    progress_bar = tqdm(range(TOTAL_STEPS))

    save_dir = f"clip_vae_{args.diffusion_model_name}_decoder_final_norm_{decoder_final_norm}"
    os.makedirs(save_dir, exist_ok=True)

    # --- TRAINING ---    
    for step in progress_bar:
        optimizer.zero_grad()
        
        # --- BATCH CONSTRUCTION ---
        # Strategy: 80% Internet, 20% Special Data
        internet_batch_size = int(BATCH_SIZE * 0.8)
        special_batch_size = BATCH_SIZE - internet_batch_size
        
        # 1. Get Internet Data
        internet_captions = []
        try:
            for _ in range(internet_batch_size):
                internet_captions.append(next(internet_iter)['caption'])
        except StopIteration:
            internet_iter = iter(internet_dataset) # Restart
            internet_captions.append(next(internet_iter)['caption'])
            
        with torch.no_grad():
            i_tokens = tokenizer(internet_captions, padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
            i_embeds = text_encoder(i_tokens.input_ids)[0]
            
            # 2. Get Special Data
            indices = torch.randperm(len(special_embeds))[:special_batch_size]
            s_embeds = special_embeds[indices].to(device)
            
            # Combine
            target = torch.cat([i_embeds, s_embeds], dim=0)

        # --- DENOISING ---
        noise = torch.randn_like(target) * NOISE_LEVEL
        noisy_input = target + noise
        
        # Forward
        reconstructed, mu, logvar = vae(noisy_input)
        
        # Loss Calculation
        recon_flat = reconstructed.reshape(-1, embed_dim)
        target_flat = target.reshape(-1, embed_dim)
        
        # Note: We compare Reconstructed vs CLEAN Target
        recon_loss = cosine_loss_fn(recon_flat, target_flat, target_flag)

        mse_loss = mse_loss_fn(target_flat, recon_flat)
        
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / (BATCH_SIZE * 64)
        
        loss = recon_loss + (MAX_KL_WEIGHT * kl_loss) + 0.1 * mse_loss
        
        wandb.log({
            "train/loss": loss.item(),
            "train/recon_loss": recon_loss.item(),
            "train/mse_loss": mse_loss.item(),
            "train/kl_loss": kl_loss.item(),
            "train/beta": MAX_KL_WEIGHT,
            "train/step": step
        })

        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()
        
        if step % 100 == 0:
            progress_bar.set_description(f"Cos: {recon_loss.item():.4f} | KL: {kl_loss.item():.4f}")
            
            if validation_prompts:
                vae.eval()
                avg_sim, avg_mse = evaluate_vae(vae, text_encoder, tokenizer, validation_prompts, device, embed_dim=embed_dim)
                tqdm.write(f"Validation | Cosine: {avg_sim:.4f} | MSE: {avg_mse:.6f}")
                wandb.log({
                    "val/cosine_sim": avg_sim,
                    "val/mse": avg_mse,
                    "val/step": step
                })
                vae.train()

        if step % 5000 == 0 and step > 0:
            torch.save(vae.state_dict(), os.path.join(save_dir, f"clip_vae_general_{step}.pth"))

    torch.save(vae.state_dict(), os.path.join(save_dir, "clip_vae_general_final.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation_parquet_path", type=str, default=None, help="Path to validation parquet for evaluation during training")
    parser.add_argument("--diffusion_model_name", type=str, default="SDV1", help="Name of the diffusion model")
    parser.add_argument("--prompts_root", type=str, help="Root directory for prompts")
    args = parser.parse_args()

    validation_prompts = None
    if args.validation_parquet_path:
        print(f"Loading validation prompts from {args.validation_parquet_path}")
        df = dataset_utils.load_parquet_file(args.validation_parquet_path)
        all_captions = df['caption'].tolist()
        validation_prompts = load_valid_prompts(args.prompts_root, args.diffusion_model_name, all_captions, df)
        print(f"Loaded {len(validation_prompts)} validation prompts")

    train_vae(args, validation_prompts)