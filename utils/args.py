import argparse
import os

__all__ = [
    "load_args",
]


def load_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Device
    parser.add_argument("--device", type=str, required=False, default="cuda")
    parser.add_argument("--devices", type=str, required=False, default="auto")

    # Pipeline
    parser.add_argument("--pipe", type=str, required=False, default="SDV1")
    parser.add_argument("--scheduler", type=str, required=False, default="DDIM")
    parser.add_argument(
        "--T", type=int, required=False, default=50
    )  # num_inference_steps
    parser.add_argument(
        "--g", type=float, required=False, default=7.5
    )  # guidance_scale
    parser.add_argument("--seed", type=int, required=False, default=0)
    parser.add_argument("--train-seed", type=int, required=False, default=0)

    # Run
    parser.add_argument("--batch-size", type=int, required=False, default=8)
    parser.add_argument("--num-workers", type=int, required=False, default=8)
    parser.add_argument("--N", type=int, required=False, default=10)  # num_seeds

    # Mitigate
    parser.add_argument("--mitigate", type=str, required=False, default="none")
    parser.add_argument("--vae-path", type=str, required=False, default=None)
    parser.add_argument("--sac-actor-path", type=str, required=False, default=None)
    parser.add_argument("--sac-critic-path", type=str, required=False, default=None)
    parser.add_argument("--sac-encoder-path", type=str, required=False, default=None)
    parser.add_argument("--sac-safety-critic-path", type=str, required=False, default=None)
    parser.add_argument("--action-reg", type=float, required=False, default=1.0)
    parser.add_argument("--caption-dim", type=int, required=False, default=64)
    parser.add_argument("--learn-lambda", action="store_true", default=False)
    parser.add_argument("--do-not-use-safety-critic", action="store_true", default=False)

    # Train VAE
    parser.add_argument("--noise-level", type=float, required=False, default=0.05)
    parser.add_argument("--max-kl-weight", type=float, required=False, default=0.002)

    # Aggregate metrics
    parser.add_argument("--aggregate-metrics", action="store_true", default=False)

    # For evaluation
    parser.add_argument("--train-seeds", type=list, required=False, default=[41, 42, 43, 44, 45])
    parser.add_argument("--filter-ood-indices", action="store_true", default=False)
    parser.add_argument("--filter-valid-indices", action="store_true", default=False)
    parser.add_argument("--log-dir", type=str, required=False, default="")
    parser.add_argument("--mode", type=str, required=False, default="baseline")
    parser.add_argument("--reference-dir", type=str, required=False, default="")

    args = parser.parse_args()
    
    return args