import os

import matplotlib.pyplot as plt
import tqdm

import utils

if __name__ == "__main__":
    args = utils.load_args()

    if not args.aggregate_metrics:
        # Compute FID scores
        runner = utils.fid_score.Runner(vars(args))
        runner.run()

    else:
        fig_dir = os.path.join(utils.paths.FIG_DIR, "fid_score")
        os.makedirs(fig_dir, exist_ok=True)
        plt.rcParams["font.size"] = 15

        strategies = [
            "none",
            "wen_iclr2024",
            "ren_eccv2024",
            "hintersdorf_nips2024",
            "jain_cvpr2025",
            "sac_safety_critic",
            "sac_no_safety_critic",
        ]
        config = vars(args)

        for strategy in tqdm.tqdm(strategies, desc="Printing FID scores..."):
            config["mitigate_strategy"] = strategy
            fid_scores = utils.load_fid_scores(config).flatten()
            print(
                f"{strategy}: {fid_scores.mean().item():.4f} ± {fid_scores.std().item():.4f}"
            )
