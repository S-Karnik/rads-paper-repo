import os
import random
import numpy as np
import lightning as L
import torch
from torch import Tensor
from PIL import Image
from typing import Dict, Any, List, Tuple

from .paths import FID_REF_ROOT, LOG_DIR, root_dir

__all__ = [
    "filter_for_invalid_indices",
    "filter_for_valid_indices",
    "load_clip_scores",
    "load_fid_scores",
    "load_sscd_scores_seed",
    "load_sscd_scores_prompt",
    "load_sscd_dataset_scores",
]

def set_seed(
    seed: int,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def filter_for_invalid_indices(
    prompts_dir: str,
    diffusion_model_name: str,
    indices: List[int],
) -> List[int]:
    remaining_indices = []
    for index in indices:
        try:
            image_path = os.path.join(
                f"{prompts_dir}/{diffusion_model_name}",
                "image", f"{index}.png",
            )
            pil_image = Image.open(image_path).convert("RGB")
            pil_image = pil_image.resize((512, 512), Image.Resampling.BICUBIC)
        except Exception as e:
            remaining_indices.append(index)
    return remaining_indices

def filter_for_valid_indices(
    prompts_dir: str,
    diffusion_model_name: str,
    indices: List[int],
) -> List[int]:
    remaining_indices = []
    for index in indices:
        try:
            image_path = os.path.join(
                f"{prompts_dir}/{diffusion_model_name}",
                "image", f"{index}.png",
            )
            pil_image = Image.open(image_path).convert("RGB")
            pil_image = pil_image.resize((512, 512), Image.Resampling.BICUBIC)
            remaining_indices.append(index)
        except Exception as e:
            pass
    return remaining_indices

def load_accelerator_and_devices(
    devices: str = "auto",
) -> Tuple[str, List[int]]:
    if isinstance(devices, str) and "cuda" in devices:
        devices = devices.replace("cuda", "gpu")
    devices_cfg: List[str] = devices.split(":")
    accelerator = "auto"
    devices = "auto"
    if len(devices_cfg) == 1:
        accelerator = devices_cfg[0]
    else:
        accelerator = devices_cfg[0]
        devices = [int(d) for d in devices_cfg[1].split(",")]
    return accelerator, devices


def load_runner(
    config: Dict[str, Any],
    use_half: bool = True,
) -> L.Trainer:
    # Load accelerator and devices
    accelerator, devices = load_accelerator_and_devices(config["devices"])
    # Set precision
    if use_half:
        torch.set_float32_matmul_precision("medium")
    # Load runner
    runner = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision="16-mixed" if use_half else "32-true",
        logger=False,
        callbacks=None,
        max_epochs=1,
        log_every_n_steps=None,
        enable_checkpointing=False,
        deterministic=False,
        inference_mode=False,
    )
    # Return runner
    return runner


def load_clip_scores(
    config,
    indices,
) -> Tensor:
    clip_scores = []
    if "sac" in config["mitigate_strategy"]:
        for index in indices:
            for train_seed in config["train_seeds"]:
                path = os.path.join(
                    root_dir(
                        root=LOG_DIR,
                        config=config,
                    ),
                    "clip_scores",
                    config["mitigate_strategy"],
                    f"train_seed={train_seed}",
                    f"{index}.pt",
                )
                if os.path.exists(path):
                    clip_scores.append(torch.load(path, weights_only=True, map_location="cpu"))
                else:
                    print(path)
    else:
        for index in indices:
            path = os.path.join(
                root_dir(
                    root=LOG_DIR,
                    config=config,
                ),
            "clip_scores",
            config["mitigate_strategy"],
            f"{index}.pt",
            )
            if os.path.exists(path):
                clip_scores.append(torch.load(path, weights_only=True, map_location="cpu"))
            else:
                print(path)
    # (num_data, N)
    clip_scores = torch.stack(clip_scores, dim=0)

    return clip_scores


def _compute_fid(
    mu1: torch.Tensor,
    sigma1: torch.Tensor,
    mu2: torch.Tensor,
    sigma2: torch.Tensor,
) -> float:
    # The Fréchet Inception Distance (FID)
    # between two multivariate Gaussians X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2) is
    # d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    # Compute covariance matrices (assume diagonal for efficiency)
    C1 = torch.diag(sigma1**2)
    C2 = torch.diag(sigma2**2)

    # Compute FID
    diff = mu1 - mu2
    sqrt_C1_C2 = torch.diag((C1 * C2).sqrt())
    fid = diff.dot(diff) + torch.trace(C1 + C2 - 2 * sqrt_C1_C2)

    return fid.item()


def load_fid_scores(
    config,
) -> Tensor:
    prefix = "filtered_" if config.get("filter_ood_indices", False) else ""
    if "sac" in config["mitigate_strategy"]:
        fid_scores = []
        for train_seed in config["train_seeds"]:
            scores = torch.load(
                os.path.join(
                    root_dir(
                        root=LOG_DIR,
                        config=config,
                    ),
                    "fid_scores",
                    f"train_seed={train_seed}",
                    f"{prefix}{config['mitigate_strategy']}.pt",
                ),
                weights_only=False,
                map_location="cpu",
            )
            fid_scores.append(torch.tensor(scores))  # (N,)
        fid_scores = torch.stack(fid_scores, dim=0)
    else:
        fid_scores = torch.load(
            os.path.join(
                root_dir(
                    root=LOG_DIR,
                    config=config,
                ),
                "fid_scores",
                f"{prefix}{config['mitigate_strategy']}.pt",
            ),
            weights_only=False,
            map_location="cpu",
        )  # (N,)

    return torch.tensor(fid_scores, dtype=torch.float32)


def load_sscd_scores_seed(
    config,
    indices,
    return_mean: bool = True,
) -> Tensor:
    sscd_scores = []
    if "sac" in config["mitigate_strategy"]:
        for index in indices:
            for train_seed in config["train_seeds"]:
                path = os.path.join(
                    root_dir(
                        root=LOG_DIR,
                        config=config,
                    ),
                    "sscd_scores_seed",
                    config["mitigate_strategy"],
                    f"train_seed={train_seed}",
                    f"{index}.pt",
                )
                if os.path.exists(path):
                    sscd_scores.append(torch.load(path, weights_only=True, map_location="cpu"))
    else:
        for index in indices:
            path = os.path.join(
                root_dir(
                    root=LOG_DIR,
                    config=config,
                ),
                "sscd_scores_seed",
                config["mitigate_strategy"],
                f"{index}.pt",
            )
            sscd_scores.append(
                torch.load(
                    path,
                    weights_only=True,
                    map_location="cpu",
                )
            )
    # (num_data, N, N)
    # If sac_safety_critic, (num_data, N, N, num_seeds)
    sscd_scores = torch.stack(sscd_scores, dim=0)

    if return_mean:
        # Scores between generated images: (num_data, N, N)
        # Average over different seeds excluding scores with itself: (num_data, N)
        mask = ~torch.eye(sscd_scores.shape[1], dtype=bool)  # Mask to ignore diagonal terms
        sscd_scores = (sscd_scores * mask).sum(dim=1) / mask.sum(dim=1)

    return sscd_scores


def load_sscd_scores_prompt(
    config,
) -> Tensor:
    sscd_scores = []
    prefix = "filtered_" if config["filter_ood_indices"] else ""
    if "sac" in config["mitigate_strategy"]:
        for seed in range(config["N"]):
            for train_seed in config["train_seeds"]:
                path = os.path.join(
                    root_dir(
                        root=LOG_DIR,
                        config=config,
                    ),
                    "sscd_scores_prompt",
                    config["mitigate_strategy"],
                    f"train_seed={train_seed}",
                    f"{prefix}seed={seed}.pt",
                )
                if os.path.exists(path):
                    sscd_scores.append(
                        torch.load(path, weights_only=True, map_location="cpu")
                    )
                else:
                    print(f"Path not found: {path}")
    else:
        for seed in range(config["N"]):
            sscd_scores.append(
                torch.load(
                    os.path.join(
                        root_dir(
                            root=LOG_DIR,
                            config=config,
                        ),
                        "sscd_scores_prompt",
                        config["mitigate_strategy"],
                        f"{prefix}seed={seed}.pt",
                    ),
                    weights_only=True,
                    map_location="cpu",
                )
            )
    # (N, num_data, num_data)
    sscd_scores = torch.stack(sscd_scores, dim=0)

    # Scores between generated images: (N, num_data, num_data)
    # Average over different seeds excluding scores with itself: (N, num_data)
    mask = ~torch.eye(sscd_scores.shape[1], dtype=bool)  # Mask to ignore diagonal terms
    sscd_scores = (sscd_scores * mask).sum(dim=1) / mask.sum(dim=1)

    return sscd_scores
    return sscd_scores


def load_sscd_dataset_scores(
    config,
    indices,
) -> Tensor:
    sscd_scores = []
    if "sac" in config["mitigate_strategy"]:
        for index in indices:
            for train_seed in config["train_seeds"]:
                path = os.path.join(
                    root_dir(
                        root=LOG_DIR,
                        config=config,
                    ),
                    "sscd_dataset_scores",
                    config["mitigate_strategy"],
                    f"train_seed={train_seed}",
                    f"{index}.pt",
                )
                if os.path.exists(path):
                    sscd_scores.append(torch.load(path, weights_only=True, map_location="cpu"))

    else:
        for index in indices:
            path = os.path.join(
                root_dir(
                    root=LOG_DIR,
                    config=config,
                ),
                "sscd_dataset_scores",
                config["mitigate_strategy"],
                f"{index}.pt",
            )
            if os.path.exists(path):
                sscd_scores.append(
                    torch.load(
                        path,
                        weights_only=True,
                        map_location="cpu",
                    )
                )
    
    if sscd_scores:
        sscd_scores = torch.stack(sscd_scores, dim=0) # (num_items, N)
    else:
        return torch.tensor([])

    return sscd_scores
