import os
from typing import Any, Dict, Tuple

import lightning as L
import torch
from torch.utils.data import DataLoader

from ..misc import load_runner
from ..paths import LOG_DIR, root_dir
from ..pipe import StableDiffusion
from ..prompts import PromptDataset

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
        num_seeds: int = config.get("N", 50)
        batch_size = config.get("batch_size", 1)
        num_workers = config.get("num_workers", 0)
        devices = config.get("devices", "auto")

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

        self.config = {
            "pipe": pipe_name,
            "scheduler": scheduler,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_seeds": num_seeds,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "devices": devices,
        }

    def on_test_start(self) -> None:
        os.makedirs(os.path.join(self.log_dir, "wen_et_al"), exist_ok=True)

    def test_step(self, batch, batch_idx):
        # Unpack batch
        indices: Tuple[int] = batch[0]

        for index in indices:
            l2s = []

            for seed in range(self.config["num_seeds"]):
                # Load noise predictions
                eps_ts_uncond, eps_ts_cond = torch.load(
                    os.path.join(
                        self.log_dir,
                        "generated_samples",
                        f"seed={seed}",
                        "noise_pred",
                        f"{index}.pt",
                    ),
                    weights_only=True,
                    map_location="cpu",
                )

                eps_ts_uncond = eps_ts_uncond.reshape(
                    self.config["num_inference_steps"], -1
                ).to(self.device)
                eps_ts_cond = eps_ts_cond.reshape(
                    self.config["num_inference_steps"], -1
                ).to(self.device)

                # Compute metrics
                l2s.append((eps_ts_cond - eps_ts_uncond).norm(dim=-1))

            # (N, T)
            l2s = torch.stack(l2s).cpu()

            # Save
            torch.save(
                l2s,
                os.path.join(self.log_dir, "wen_et_al", f"{index}.pt"),
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
        dataset: PromptDataset,
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
