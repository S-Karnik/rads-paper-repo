import random

import numpy as np
import torch
import torch.distributed as dist

__all__ = [
    "setup_for_distributed",
    "cleanup",
    "seed_everything",
]


def setup_for_distributed(rank: int, world_size: int) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        device_id=rank,
    )


def cleanup() -> None:
    dist.destroy_process_group()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
