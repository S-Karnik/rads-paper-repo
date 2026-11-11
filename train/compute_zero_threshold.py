
import torch
import os
import numpy as np
import argparse
from sklearn.metrics import accuracy_score
import pathlib
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as F

from utils.dataset import utils as dataset_utils
from utils import paths, model_types, pipe as pipe_module

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--pipe_name", type=str, default="SDV1")
    parser.add_argument("--scheduler_name", type=str, default="DDIM")
    return parser.parse_args()

def load_and_filter_dataset(dataset_path, diffusion_model_type):
    parquet_dataset = dataset_utils.load_parquet_file(pathlib.Path(dataset_path))
    
    valid_indices = []
    dataset_dir = paths.get_dataset_dir(diffusion_model_type)
    
    print(f"Filtering dataset from {dataset_dir}...")
    for idx, row in tqdm(parquet_dataset.iterrows(), total=len(parquet_dataset)):
        print(f"Processing idx: {idx}")
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

def compute_l2_norms(dataset, args):
    print("Initializing pipeline...")
    pipeline = pipe_module.load_pipe(args.pipe_name, args.scheduler_name)
    pipeline.init(num_inference_steps=args.num_timesteps, guidance_scale=args.guidance_scale)
    pipeline.to(args.device)
    
    l2_norms_list = []
    prompts = dataset['caption'].tolist()
    
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Computing L2 Norms"):
        batch_prompts = prompts[i : i + args.batch_size]
        
        prompt_embeds = pipeline.prepare_embeds(batch_prompts)
        timesteps, _, _ = pipeline.prepare_timesteps()
        diffusion_latents = pipeline.prepare_latents(prompt_embeds, seed=None).to(args.device)
        
        dataset_l2_norms = []
        
        for t in timesteps:
            if isinstance(t, torch.Tensor): t = t.to(args.device)
            
            noise_pred = pipeline.predict_noise(diffusion_latents, prompt_embeds, t)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            
            diff = noise_pred_uncond - noise_pred_cond
            current_l2_norms = torch.norm(diff, p=2, dim=(1,2,3)).cpu().numpy()
            dataset_l2_norms.append(current_l2_norms)
            
            noise_pred_cfg = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            diffusion_latents = pipeline.step(noise_pred_cfg, t, diffusion_latents)
            
        batch_l2_norms = np.stack(dataset_l2_norms, axis=1) # Shape: (B, T)
        l2_norms_list.append(batch_l2_norms)
        
    return np.concatenate(l2_norms_list, axis=0)

def compute_binary_labels(raw_labels):
    return [0 if label == "N" else 1 for label in raw_labels]

def find_best_threshold(l2_norms, binary_labels):
    best_acc = 0
    best_threshold = 0
    
    print("Finding best threshold...")
    final_step_norms = l2_norms[:, -1]
    
    thresholds = np.unique(final_step_norms)
    if len(thresholds) > 1000:
        thresholds = np.percentile(final_step_norms, np.linspace(0, 100, 1000))
    
    for threshold in thresholds[::-1]:
        predictions = [1 if norm > threshold else 0 for norm in final_step_norms]
        acc = accuracy_score(binary_labels, predictions)
        print(f"Threshold: {threshold:.4f}, Accuracy: {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
            
    return best_threshold, best_acc

def main():
    args = parse_args()
    if args.pipe_name == "SDV1":
        diffusion_model_type = model_types.ModelType.STABLE_DIFFUSION_V1
    elif args.pipe_name == "RealVis":
        diffusion_model_type = model_types.ModelType.REAL_VISION
    elif args.pipe_name == "SDV2":
        diffusion_model_type = model_types.ModelType.STABLE_DIFFUSION_V2
    else:
        raise ValueError(f"Invalid pipe name: {args.pipe_name}")
    
    dataset_path = paths.get_prompt_path(diffusion_model_type)
    print(f"Loading dataset from {dataset_path}")
    dataset = load_and_filter_dataset(dataset_path, diffusion_model_type)
    
    l2_norms = compute_l2_norms(dataset, args)

    if 'overfit_type' in dataset.columns:
        binary_labels = compute_binary_labels(dataset['overfit_type'].tolist())
    else:
        print("Warning: 'overfit_type' column not found. Using dummy labels for testing.")
        binary_labels = [0] * len(dataset)
        
    threshold, acc = find_best_threshold(l2_norms, binary_labels)
    
    print(f"Best threshold: {threshold:.4f}")
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()