import os
from typing import Literal

from ..paths import DATASET_DIR
from . import realvis, sdv1
from .base import PromptDataset
from .coco import download_coco

__all__ = [
    "MemorizedPrompts",
    "load_prompts",
    "download_coco",
]

URLS = {
    "sdv1": sdv1.URL,
    "realvis": realvis.URL,
    "membench_sdv1": "membench_sdv1.parquet",
}
CHECKSUMS = {
    "sdv1": sdv1.CHECKSUM,
    "realvis": realvis.CHECKSUM,
    "membench_sdv1": "e20f46842dafbd530b5ddf909101be29",
}


class MemorizedPrompts(PromptDataset):
    def __init__(
        self,
        root: os.PathLike = "memorized_prompts",
        pipe: Literal["sdv1", "realvis", "membench_sdv1"] = "sdv1",
    ) -> None:
        super().__init__(
            root=root,
            url=URLS[pipe],
            checksum=CHECKSUMS[pipe],
        )


def load_prompts(
    dataset_name: str,
) -> MemorizedPrompts:
    if dataset_name == "SDV1":
        return MemorizedPrompts(root=DATASET_DIR, pipe="sdv1")
    elif dataset_name == "RealVis":
        return MemorizedPrompts(root=DATASET_DIR, pipe="realvis")
    elif dataset_name == "MemBench-SDV1":
        return MemorizedPrompts(root=DATASET_DIR, pipe="membench_sdv1")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
