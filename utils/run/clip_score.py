import os
from typing import Any, Dict, Tuple

import lightning as L
import torch
from PIL import Image
from torch.utils.data import DataLoader
from wah.diffusion.memorization import MemorizedPrompts
from wah.feature_extraction import CLIP

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
        self.clip = CLIP()

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
        }

    def on_test_start(self) -> None:
        os.makedirs(
            os.path.join(
                self.log_dir,
                "clip_scores",
                self.config["mitigate_strategy"],
            ),
            exist_ok=True,
        )
        self.clip.to(self.device)

    def test_step(self, batch, batch_idx):
        # Unpack batch
        indices: Tuple[int] = batch[0]
        prompts: Tuple[str] = batch[2]

        for index, prompt in zip(indices, prompts):
            images = []

            # Load generated images
            for seed in range(self.config["num_seeds"]):
                try:
                    if "sac" in self.config["mitigate_strategy"]:
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
                    else:
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
                    print(e)
                    pass

            # Extract features
            text_features = self.clip(prompt) # (1, 512)
            image_features = self.clip(images) # (N, 512)

            # Compute cosine similarity
            text_norm = torch.nn.functional.normalize(text_features, p=2, dim=1)
            image_norm = torch.nn.functional.normalize(image_features, p=2, dim=1)
            similarities = torch.mm(text_norm, image_norm.t())  # (1, N)

            # Save clip scores
            if "sac" in self.config["mitigate_strategy"]:
                dir_path = os.path.join(
                    self.log_dir,
                    "clip_scores",
                    self.config["mitigate_strategy"],
                    f"train_seed={self.config['train_seed']}",
                )
                os.makedirs(dir_path, exist_ok=True)
                torch.save(
                    similarities[0],
                    os.path.join(
                        dir_path,
                        f"{index}.pt",
                    ),
                )
            else:
                dir_path = os.path.join(
                    self.log_dir,
                    "clip_scores",
                    self.config["mitigate_strategy"],
                )
                os.makedirs(dir_path, exist_ok=True)
                torch.save(
                    similarities[0],
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
