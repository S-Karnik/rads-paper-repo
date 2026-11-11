import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict

from utils import model_types

__all__ = [
    "DATASET_DIR",
    "LOG_DIR",
    "FIG_DIR",
    "FID_REF_DIR",
    "root_dir",
]

# --- Configuration Enum ---
class PathConfig(Enum):
    """
    Defines environment-specific paths.
    Values are (env_name, base_dir).
    """

    def __init__(self, env_name, base_dir):
        self.env_name = env_name
        self.base_dir = base_dir

    @property
    def DATASET_DIR(self):
        return os.path.join(self.base_dir, "prompts")

    @property
    def LOG_DIR(self):
        return os.path.join(self.base_dir, "logs")

    @property
    def FIG_DIR(self):
        return os.path.join(self.base_dir, "figs")

    @property
    def FID_REF_DIR(self):
        return os.path.join(self.base_dir, "fid_ref", "coco", "images", "val2017")

    @property
    def FID_REF_ROOT(self):
        return os.path.join(self.base_dir, "fid_ref", "coco")

# --- Selection Logic ---

# 1. Create a lookup dictionary
CONFIG_MAP = {cfg.env_name: cfg for cfg in PathConfig}

# 2. Get user from env var
current_user_key = os.getenv("DIRECTORY_TYPE", "").lower()

# 3. Validate and Select
valid_keys = list(CONFIG_MAP.keys())
assert current_user_key in CONFIG_MAP, \
    f"DIRECTORY_TYPE '{current_user_key}' invalid. Must be one of: {valid_keys}"

_selected_config = CONFIG_MAP[current_user_key]

# 4. Export Module-Level Constants
DATASET_DIR = _selected_config.DATASET_DIR
LOG_DIR = _selected_config.LOG_DIR
FIG_DIR = _selected_config.FIG_DIR
FID_REF_DIR = _selected_config.FID_REF_DIR
FID_REF_ROOT = _selected_config.FID_REF_ROOT

def root_dir(
    root: Path,
    config: Dict[str, Any],
) -> Path:
    """
    Generates a unique directory path for a specific experiment run.
    """
    return os.path.join(
        root,
        config["pipe"],
        f"{config['scheduler']},T={config['T']},g={config['g']},N={config['N']}",
    )

def get_dataset_dir(diffusion_model_type: model_types.ModelType) -> Path:
    if diffusion_model_type == model_types.ModelType.STABLE_DIFFUSION_V1:
        return os.path.join(DATASET_DIR, "SDV1")
    elif diffusion_model_type == model_types.ModelType.REAL_VISION:
        return os.path.join(DATASET_DIR, "RealVis")
    else:
        raise ValueError(f"Invalid diffusion model type: {diffusion_model_type}")

def get_prompt_path(diffusion_model_type: model_types.ModelType) -> Path:
    if diffusion_model_type == model_types.ModelType.STABLE_DIFFUSION_V1:
        return os.path.join(DATASET_DIR, "SDV1/sdv1_bb_edge_groundtruth.parquet")
    elif diffusion_model_type == model_types.ModelType.REAL_VISION:
        return os.path.join(DATASET_DIR, "RealVis/realistic_vision_sdv1_edge_groundtruth.parquet")
    else:
        raise ValueError(f"Invalid diffusion model type: {diffusion_model_type}")