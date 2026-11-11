import os

import matplotlib.pyplot as plt

import utils

if __name__ == "__main__":
    args = utils.load_args()

    # Load dataset
    dataset = utils.load_prompts(args.pipe)

    if not args.aggregate_metrics:
        # Compute SSCD scores
        utils.sscd_score_prompt.Runner(vars(args)).run(dataset)

    else:
        fig_dir = os.path.join(
            utils.paths.root_dir(root=utils.paths.FIG_DIR, config=vars(args)),
            "sscd_scores_prompt",
        )
        os.makedirs(fig_dir, exist_ok=True)
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
            sscd_scores = utils.load_sscd_scores_prompt(config).flatten()

            print(
                f"[{strategy}] SSCD score (prompt): {sscd_scores.mean().item():.4f} ± {sscd_scores.std().item():.4f}"
            )
