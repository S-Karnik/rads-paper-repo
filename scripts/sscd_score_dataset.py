import utils
from utils.run import sscd_score_dataset
import matplotlib.pyplot as plt

if __name__ == "__main__":
    args = utils.load_args()

    # Load dataset
    dataset = utils.load_prompts(args.pipe)

    if not args.aggregate_metrics:
        # Compute SSCD scores relative to dataset
        runner = sscd_score_dataset.Runner(vars(args))
        runner.run(dataset)
    
    else:
        indices = dataset.indices
        if args.filter_ood_indices:
            indices = utils.filter_for_invalid_indices(args.pipe, indices)
        elif args.filter_valid_indices:
            if args.pipe == "RealVis":
                # RealVis uses the same images as SDV1.
                indices = utils.filter_for_valid_indices("SDV1", indices)
            else:
                indices = utils.filter_for_valid_indices(args.pipe, indices)
        
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
        
        for strategy in strategies:
            config["mitigate_strategy"] = strategy
            # Returns (num_indices, num_seeds)
            sscd_scores = utils.misc.load_sscd_dataset_scores(config, indices)
            
            if sscd_scores.numel() == 0:
                # print(f"[{strategy}] No scores found.")
                continue
                
            # Flatten to get all scores from all seeds/indices
            scores_flat = sscd_scores.flatten()
            
            print(
                f"[{strategy}] SSCD Dataset Score: {scores_flat.mean().item():.4f} ± {scores_flat.std().item():.4f}"
            )
