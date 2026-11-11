import os
import shutil
import tempfile
from typing import Any, Dict

import torch
from wah.metric import FID

from ..dataset import load_prompts
from ..misc import filter_for_invalid_indices
from ..paths import FID_REF_DIR, LOG_DIR, root_dir

__all__ = [
    "Runner",
]


class Runner:
    def __init__(
        self,
        config: Dict[str, Any],
    ) -> None:
        self.config = config

        self.metric = FID(
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            devices=config["devices"],
            use_half=True,
        )

    def run(
        self,
    ) -> None:
        # Compute FID scores
        fid_scores = []
        
        # Load indices and filter if requested
        indices = load_prompts(self.config["pipe"]).indices
        if self.config.get("filter_ood_indices", False):
            indices = filter_for_invalid_indices(self.config["pipe"], indices)
            
        for seed in range(self.config["N"]):
            if "sac" in self.config["mitigate"]:
                mitigated_dir = os.path.join(
                    root_dir(root=LOG_DIR, config=self.config),
                    "generated_samples",
                    self.config["mitigate"],
                    f"train_seed={self.config['train_seed']}",
                    f"seed={seed}",
                )
            else:
                mitigated_dir = os.path.join(
                    root_dir(root=LOG_DIR, config=self.config),
                    "generated_samples",
                    self.config["mitigate"],
                    f"seed={seed}",
                )
            
            if self.config.get("filter_ood_indices", False):
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Symlink valid images to temp directory
                    for idx in indices:
                        src = os.path.join(mitigated_dir, f"{idx}.png")
                        dst = os.path.join(temp_dir, f"{idx}.png")
                        # Only link if source exists (it should if generation worked)
                        if os.path.exists(src):
                            os.symlink(src, dst)
                    fid_score = self.metric(FID_REF_DIR, temp_dir)
            else:
                fid_score = self.metric(FID_REF_DIR, mitigated_dir)
                
            fid_scores.append(fid_score)
            print(f"Seed {seed}: {fid_score}", flush=True)

        # Save FID scores
        if "sac" in self.config["mitigate"]:
            save_dir = os.path.join(
                root_dir(root=LOG_DIR, config=self.config), "fid_scores", f"train_seed={self.config['train_seed']}"
            )
        else:
            save_dir = os.path.join(
                root_dir(root=LOG_DIR, config=self.config), "fid_scores"
            )
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{self.config['mitigate']}.pt"
        if self.config.get("filter_ood_indices", False):
            filename = f"filtered_{filename}"
        torch.save(fid_scores, os.path.join(save_dir, filename))
