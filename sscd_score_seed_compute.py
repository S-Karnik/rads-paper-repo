import os

import torch
import tqdm
from PIL import Image
from wah.feature_extraction import SSCD

from utils.paths import LOG_DIR, root_dir

import utils

if __name__ == "__main__":
    args = utils.load_args()

    # Load dataset
    dataset = utils.load_prompts(args.pipe)
    indices = dataset.indices
    filtered_index_prefix = ""
    if args.filter_ood_indices:
        indices = utils.filter_for_invalid_indices(args.pipe, indices)
        filtered_index_prefix = "filtered_"


    # Compute SSCD scores
    sscd = SSCD().to(args.device)

    train_seeds = args.train_seeds if "sac" in args.mitigate else [None]

    for train_seed in train_seeds:
        sscd_features = []
        for index in tqdm.tqdm(
            indices,
            desc=f"Computing SSCD features (train_seed={train_seed})..." if train_seed is not None else "Computing SSCD features...",
            total=len(indices),
        ):
            if train_seed is not None:
                image_path = os.path.join(
                    root_dir(root=LOG_DIR, config=vars(args)),
                    "generated_samples",
                    args.mitigate,
                    f"train_seed={train_seed}",
                    f"seed={args.seed}",
                    f"{index}.png",
                )
            else:
                image_path = os.path.join(
                    root_dir(root=LOG_DIR, config=vars(args)),
                    "generated_samples",
                    args.mitigate,
                    f"seed={args.seed}",
                    f"{index}.png",
                )
            
            try:
                image = Image.open(image_path).convert("RGBA").convert("RGB")
            except Exception as e:
                print(e)
                continue
            sscd_features.append(sscd(image))

        sscd_features = torch.cat(sscd_features, dim=0)  # (num_data, 512)
        sscd_features = torch.nn.functional.normalize(sscd_features, p=2, dim=1)
        sscd_scores = torch.mm(sscd_features, sscd_features.t())  # (num_data, num_data)

        if train_seed is not None:
            save_dir = os.path.join(
                utils.paths.root_dir(root=utils.paths.LOG_DIR, config=vars(args)),
                "sscd_scores_prompt",
                args.mitigate,
                f"train_seed={train_seed}",
                f"{filtered_index_prefix}seed={args.seed}.pt",
            )
        else:
            save_dir = os.path.join(
                utils.paths.root_dir(root=utils.paths.LOG_DIR, config=vars(args)),
                "sscd_scores_prompt",
                args.mitigate,
                f"{filtered_index_prefix}seed={args.seed}.pt",
            )
        
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        torch.save(sscd_scores, save_dir)
