from tap import Tap


class MainArguments(Tap):
    task: str = "mnist_op"


args = MainArguments().parse_args(known_only=True)

if args.task == "mnist_op":
    from expressive.experiments.mnist_op.mnistop import main

    main()
else:
    raise ValueError(f"Task {args.task} not recognized.")
