from expressive.args import RSBenchEvalArguments
from expressive.experiments.rsbench.datasets import get_dataset
from expressive.experiments.rsbench.rsbenchmodel import create_rsbench_diffusion
from expressive.methods.logger import (
    BOIATestLog,
    TestLog,
    TestLogger,
)
from expressive.experiments.rsbench.nesydiffusion import eval
from expressive.util import get_device
import torch
import glob
import json
import pandas as pd


if __name__ == "__main__":
    args = RSBenchEvalArguments(explicit_bool=True).parse_args()
    device = get_device(args)
    dataset = get_dataset(args)
    model = create_rsbench_diffusion(args, dataset).to(device)
    args.use_wandb = False

    _, val_loader, test_loader = dataset.get_data_loaders()

    loader = test_loader if args.test else val_loader
    
    ood_loaders = dataset.get_ood_loaders()
    
    # Create an empty list to store all results
    all_results = []

    for run_id in args.run_ids:
        model_pattern = f"{args.model_dir}/{run_id}/model_*.pth"
        model_files = glob.glob(model_pattern)
        
        if not model_files:
            raise FileNotFoundError(f"No model files found matching pattern: {model_pattern}")
        
        # Assert there is only one file (RSBench code only saves one model per run)
        assert len(model_files) == 1, f"Expected 1 model file, found {len(model_files)}"
        model_path = model_files[0]
        
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print(f"Loaded model from {model_path}")

        # ID evaluation
        clazz = BOIATestLog if args.dataset == "boia" else TestLog
        logger = TestLogger(clazz, args, "val" if not args.test else "test")
        result = eval(loader, logger, model, device, args)
        
        # OOD evaluation
        ood_loggers = [TestLogger(clazz, args, f"ood_{i + 1}") for i in range(len(ood_loaders))]

        for ood_loader, ood_logger in zip(ood_loaders, ood_loggers):
            result.update(eval(ood_loader, ood_logger, model, device, args))

        print(result)
        
        # Add run_id to the result dictionary
        result['run_id'] = run_id
        
        # Append result to list
        all_results.append(result)
        
        # Still save individual JSON files if needed
        jzon = json.dumps(result)
        with open(f"out/{run_id}_result.json", "w") as f:
            f.write(jzon)
    
    # Convert all results to a pandas DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save the DataFrame to a CSV file
    split_name = "test" if args.test else "val"
    csv_path = f"out/all_results_{args.eval_name}_{split_name}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"All results saved to {csv_path}")

