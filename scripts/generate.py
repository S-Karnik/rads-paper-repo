import utils

if __name__ == "__main__":
    args = utils.load_args()

    # Load dataset
    dataset = utils.load_prompts(args.pipe)

    # Load pipe
    pipe = utils.load_pipe(args.pipe, args.scheduler)

    # Generate samples
    runner = utils.run.generate.Runner(vars(args))
    runner.run(dataset, pipe)
