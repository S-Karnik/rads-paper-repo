import os

import matplotlib.pyplot as plt
import tqdm

import utils

if __name__ == "__main__":
    args = utils.load_args()

    # Load dataset
    dataset = utils.load_prompts(args.pipe)

    if not args.aggregate_metrics:
        # Compute CLIP scores
        runner = utils.clip_score.Runner(vars(args))
        runner.run(dataset)

    else:
        fig_dir = os.path.join(
            utils.paths.root_dir(root=utils.paths.FIG_DIR, config=vars(args)),
            "clip_score",
        )
        os.makedirs(fig_dir, exist_ok=True)
        plt.rcParams["font.size"] = 15

        strategies = [
            "none",
            "wen_iclr2024",
            "ren_eccv2024",
            "hintersdorf_nips2024",
            "jain_cvpr2025",
            "sac_safety_critic",
        ]
        config = vars(args)
        indices = dataset.indices
        if args.filter_ood_indices:
            indices = utils.filter_for_invalid_indices(config["pipe"], dataset.indices)

        for strategy in tqdm.tqdm(strategies, desc="Plotting CLIP scores..."):
            config["mitigate_strategy"] = strategy
            clip_scores = utils.load_clip_scores(config, indices).flatten()
            avg_clip_scores = clip_scores.mean()
            std_clip_scores = clip_scores.std()
            print(f"{strategy}: {avg_clip_scores.item():.4f} ± {std_clip_scores.item():.4f}")
