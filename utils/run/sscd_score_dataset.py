import os
from typing import Any, Dict, Tuple, List

import lightning as L
import torch
import tqdm
from PIL import Image
from torch.utils.data import DataLoader
from wah.diffusion.memorization import MemorizedPrompts
from wah.feature_extraction import SSCD

from ..misc import load_runner
from ..paths import LOG_DIR, root_dir

__all__ = [
    "Runner",
]


class Wrapper(L.LightningModule):
    def __init__(
        self,
        config: Dict[str, Any],
    ) -> None:
        super().__init__()

        pipe_name: str = config.get("pipe", "SDV1")
        scheduler: str = config.get("scheduler", "DDIM")
        num_inference_steps: int = config.get("T", 50)
        guidance_scale: float = config.get("g", 7.5)
        num_seeds: int = config.get("N", 50)
        batch_size = config.get("batch_size", 1)
        num_workers = config.get("num_workers", 0)
        devices = config.get("devices", "auto")
        mitigate_strategy: str = config.get("mitigate", "none")
        train_seed: int = config.get("train_seed", 0)
        reference_dir: str = config.get("reference_dir", "")

        if devices == "auto" and torch.cuda.device_count() > 1:
            self.sync_dist = True
        else:
            self.sync_dist = (
                isinstance(devices, str)
                and "gpu:" in devices
                and len(devices.split("gpu:")[-1].split(",")) > 1
            )

        self.log_dir = root_dir(
            root=LOG_DIR,
            config=config,
        )
        self.sscd = SSCD()
        
        # To hold reference dataset features
        self.reference_features = None

        self.config = {
            "pipe": pipe_name,
            "scheduler": scheduler,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_seeds": num_seeds,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "devices": devices,
            "mitigate_strategy": mitigate_strategy,
            "train_seed": train_seed,
            "reference_dir": reference_dir,
        }

    def on_test_start(self) -> None:
        # Create output directory
        base_dir = os.path.join(
            self.log_dir,
            "sscd_dataset_scores",
            self.config["mitigate_strategy"],
        )
        if self.config["mitigate_strategy"] == "sac_safety_critic":
             base_dir = os.path.join(base_dir, f"train_seed={self.config['train_seed']}")
        
        os.makedirs(base_dir, exist_ok=True)
        
        self.sscd.to(self.device)
        self.sscd.eval()

        # Load reference images
        reference_dir = self.config["reference_dir"]
        if not reference_dir:
            raise ValueError("reference_dir must be provided in config")
        if not os.path.isdir(reference_dir):
            raise FileNotFoundError(f"Reference directory not found: {reference_dir}")

        print(f"Scanning for images in {reference_dir}...")
        files = [
            f for f in os.listdir(reference_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        files.sort()
        
        if not files:
            raise ValueError(f"No image files found in {reference_dir}")

        print(f"Found {len(files)} reference images. Extracting SSCD features...")
        
        ref_feats = []
        batch_size = 32
        
        # Process in batches
        for i in tqdm.tqdm(range(0, len(files), batch_size), desc="Processing reference"):
            batch_files = files[i : i + batch_size]
            batch_imgs = []
            for fname in batch_files:
                try:
                    path = os.path.join(reference_dir, fname)
                    img = Image.open(path).convert("RGB")
                    batch_imgs.append(img)
                except Exception as e:
                    print(f"Skipping {fname}: {e}")
            
            if batch_imgs:
                # Move batch to device for processing, then back to CPU to save memory if dataset is large
                # But we need them for comparison in test_step.
                # If dataset is huge, we might need to keep on CPU and compute batch-wise on GPU during test.
                # Assuming dataset fits in GPU memory for now or we just keep it on device if small enough.
                # If it's large, we should probably keep `reference_features` on CPU and move small chunks or move query to CPU?
                # Best to keep reference on device if possible for speed.
                
                with torch.no_grad():
                    # SSCD expects list of PIL images
                    feats = self.sscd(batch_imgs) # (B, 512)
                    feats = torch.nn.functional.normalize(feats, p=2, dim=1)
                    ref_feats.append(feats)

        if not ref_feats:
            raise RuntimeError("No valid features extracted from reference dataset")
            
        self.reference_features = torch.cat(ref_feats, dim=0) # (M, 512)
        print(f"Stored {self.reference_features.shape[0]} reference features on {self.reference_features.device}.")


    def test_step(self, batch, batch_idx):
        # Unpack batch
        indices: Tuple[int] = batch[0]

        for index in indices:
            images = []

            # Load generated images
            if "sac" in self.config["mitigate_strategy"]:
                for seed in range(self.config["num_seeds"]):
                    try:
                        images.append(
                            Image.open(
                                    os.path.join(
                                        self.log_dir,
                                        "generated_samples",
                                    self.config["mitigate_strategy"],
                                    f"train_seed={self.config['train_seed']}",
                                    f"seed={seed}",
                                    f"{index}.png",
                                )
                            ).convert("RGB")
                        )
                    except Exception as e:
                        print(f"Error loading image {index} for seed {seed}: {e}")
                        # Ensure we don't crash if one image missing, but we should probably handle list length mismatch
                        pass
            else:
                for seed in range(self.config["num_seeds"]):
                    try:
                        images.append(
                            Image.open(
                                os.path.join(
                                    self.log_dir,
                                    "generated_samples",
                                    self.config["mitigate_strategy"],
                                    f"seed={seed}",
                                    f"{index}.png",
                                )
                            ).convert("RGB")
                        )
                    except Exception as e:
                        pass
            
            if not images:
                print(f"No images found for index {index}")
                continue

            # Extract features for generated images
            with torch.no_grad():
                features = self.sscd(images)  # (N, 512)
                
            # Normalize
            sscd_norm = torch.nn.functional.normalize(features, p=2, dim=1) # (N, 512)
            
            # Compute similarity: (N, 512) @ (M, 512)^T -> (N, M)
            # Ensure devices match
            ref_feats = self.reference_features
            if sscd_norm.device != ref_feats.device:
                ref_feats = ref_feats.to(sscd_norm.device)
                
            similarities = torch.mm(sscd_norm, ref_feats.t())  # (N, M)
            
            # Compute max similarity per generated image
            max_scores = similarities.max(dim=1).values # (N,)

            # Save dataset SSCD scores
            if "sac" in self.config["mitigate_strategy"]:
                dir_path = os.path.join(
                    self.log_dir,
                    "sscd_dataset_scores",
                    self.config["mitigate_strategy"],
                    f"train_seed={self.config['train_seed']}",
                )
            else:
                dir_path = os.path.join(
                    self.log_dir,
                    "sscd_dataset_scores",
                    self.config["mitigate_strategy"],
                )
                
            os.makedirs(dir_path, exist_ok=True)
            torch.save(
                max_scores,
                os.path.join(
                    dir_path,
                    f"{index}.pt",
                ),
            )

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
    ) -> None:
        dataloader: DataLoader = DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
        )
        model: L.LightningModule = Wrapper(
            config=self.config,
        )
        self.runner.test(
            model=model,
            dataloaders=dataloader,
        )
