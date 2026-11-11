import os

import matplotlib.pyplot as plt

import utils

if __name__ == "__main__":
    args = utils.load_args()

    # Load dataset
    dataset = utils.load_prompts(args.pipe)

    indices = dataset.indices
    if args.filter_ood_indices:
        indices = utils.filter_for_invalid_indices(args.pipe, indices)
    plt.rcParams["font.size"] = 15

    strategies = [
        "none",
        "wen_iclr2024",
        "ren_eccv2024",
        "hintersdorf_nips2024",
        "jain_cvpr2025",
        "sac_no_safety_critic",
    ]
    config = vars(args)

    for strategy in strategies:
        config["mitigate_strategy"] = strategy
        sscd_scores = utils.load_sscd_scores_seed(config, indices).flatten()

        print(
            f"[{strategy}] SSCD score (seed): {sscd_scores.mean().item():.4f} ± {sscd_scores.std().item():.4f}"
        )
