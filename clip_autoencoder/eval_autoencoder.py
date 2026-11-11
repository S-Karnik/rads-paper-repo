import argparse
import os
import torch
import torch.nn as nn
from tqdm import tqdm

def evaluate_vae(vae, text_encoder, tokenizer, prompts, device, embed_dim, use_penultimate=False):
    """
    Evaluates the VAE on a list of prompts.
    Returns the average cosine similarity and MSE.
    """
    # Metrics storage
    text_recon_sims = []
    text_recon_mses = []

    # Evaluation Loop
    cosine_loss_fn = nn.CosineEmbeddingLoss() 
    mse_loss = nn.MSELoss()

    print(f"Starting evaluation on {len(prompts)} prompts...")
    with torch.no_grad():
        for idx, prompt in tqdm(enumerate(prompts), total=len(prompts), desc="Evaluating"):
            # A. Text Embedding (Target)
            tokens = tokenizer([prompt], padding="max_length", max_length=77, truncation=True, return_tensors="pt").to(device)
            target_text_embed = text_encoder(tokens.input_ids)[0] # (1, 77, 768)
            target_flat = target_text_embed.reshape(-1, embed_dim)

            # B. VAE Reconstruction
            # Use deterministic reconstruction (mean) to match simulator/inference time behavior
            mu, _ = vae.encode(target_text_embed)
            reconstructed = vae.decode(mu)
            recon_flat = reconstructed.reshape(-1, embed_dim)

            # C. Text Reconstruction Metrics
            # Cosine similarity on the flattened full sequence
            sim = cosine_loss_fn(target_flat, recon_flat, torch.ones(1 * 77).to(device)).item()
            mse = mse_loss(target_flat, recon_flat).item()
            
            text_recon_sims.append(sim)
            text_recon_mses.append(mse)

    avg_sim = sum(text_recon_sims)/len(text_recon_sims)
    avg_mse = sum(text_recon_mses)/len(text_recon_mses)

    print("\nEvaluation Results")
    print(f"Processed: {len(text_recon_sims)} samples")
    print(f"Text Recon Cosine Sim: {avg_sim:.4f}")
    print(f"Text Recon MSE: {avg_mse:.6f}")
    
    return avg_sim, avg_mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", type=str, required=True, help="Path to groundtruth parquet")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained VAE checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    eval_autoencoder(args)
