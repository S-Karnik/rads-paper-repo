import os

import matplotlib.pyplot as plt
import tqdm

import utils

if __name__ == "__main__":
    args = utils.load_args()

    # Load dataset
    dataset = utils.load_prompts(args.pipe)

    # Compute SSCD scores
    runner = utils.sscd_score.Runner(vars(args))
    runner.run(dataset)
