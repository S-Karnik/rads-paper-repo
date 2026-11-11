from .stable_diffusion import StableDiffusion
from wah.diffusion.models import RealisticVision

import torch

__all__ = [
    "StableDiffusion",
]


def load_pipe(
    pipe_name: str,
    scheduler: str,
) -> StableDiffusion:
    if pipe_name == "SDV1":
        return StableDiffusion(
            version="1.4",
            scheduler=scheduler,
            variant="fp16",
            verbose=False,
            safety_check=False,
            torch_dtype=torch.float16
        )
    elif pipe_name == "RealVis":
        return RealisticVision(
            version="1.4",
            scheduler=scheduler,
            verbose=False,
            safety_check=False,
        )
    elif pipe_name == "MemBench-SDV1":
        return StableDiffusion(
            version="1.4",
            scheduler=scheduler,
            variant="fp16",
            verbose=False,
            safety_check=False,
            torch_dtype=torch.float16
        )
    else:
        raise ValueError(f"Unsupported pipe: {pipe_name}")
