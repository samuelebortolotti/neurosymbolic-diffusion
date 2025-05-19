import multiprocessing

import pandas as pd

from expressive.args import PathPlanningEvalArguments
from expressive.experiments.path_planning.absorbing_path import create_nesy_diffusion
from expressive.methods.logger import (
    TestLog,
    TestLogger,
)
from expressive.experiments.path_planning.path_planning import eval
from expressive.util import get_device
from torch.utils.data import DataLoader
import ray
import torch

from expressive.experiments.path_planning.data.dataloader import get_datasets


args = PathPlanningEvalArguments(explicit_bool=True).parse_args()
print(args)

all_results = []

if __name__ == "__main__":
    device = get_device(args)

    if args.use_ray:
        ray.init(num_cpus=multiprocessing.cpu_count())

    model = create_nesy_diffusion(args).to(device)
    _, val, test = get_datasets(args.grid_size)

    for run_id, eval_at_epoch in zip(args.run_ids, args.eval_at_epoch):
        model.load_state_dict(torch.load(f"{args.model_dir}/{run_id}/model_{eval_at_epoch}.pth"))
        print(f"Loaded model from {args.model_dir}/{run_id}/model_{eval_at_epoch}.pth")

        loader = DataLoader(test, args.batch_size_test) if args.test else DataLoader(test, args.batch_size_test)
        prefix = "test" if args.test else "val"
        test_logger = TestLogger(TestLog, args, prefix, enable_wandb=False)
        result = eval(loader, test_logger, model, device, args)
        result['run_id'] = run_id
        print(result)
        all_results.append(result)

    split_name = "test" if args.test else "val"
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f"out/all_results_{args.eval_name}_{split_name}.csv", index=False)
    print(f"All results saved to {args.eval_name}_{split_name}.csv")

