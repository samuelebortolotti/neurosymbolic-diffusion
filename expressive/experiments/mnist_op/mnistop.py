from __future__ import annotations
import math
import os
import time

from expressive.util import get_device
from torch.utils.data import DataLoader
import torch
import wandb

from expressive.experiments.mnist_op.absorbing_mnist import (
    MNISTAddProblem,
    create_mnistadd,
    vector_to_base10,
)
from expressive.args import MNISTAbsorbingArguments
from expressive.experiments.mnist_op.data import (
    create_nary_multidigit_operation,
    get_mnist_op_dataloaders,
)

from expressive.methods.logger import (
    TestLog,
    TrainingLog,
    TrainLogger,
    TestLogger,
)

SWEEP = True


def test(
    val_loader: DataLoader,
    test_logger: TestLog,
    model: MNISTAddProblem,
    device: torch.device,
):
    for i, batch in enumerate(val_loader):
        mn_digits, label_digits, label = (
            batch[: 2 * args.N],
            batch[2 * args.N : -1],
            batch[-1],
        )
        x = torch.cat(mn_digits, dim=1)
        model.evaluate(
            x.to(device),
            vector_to_base10(label.to(device), args.N + 1),
            torch.stack(label_digits, dim=-1).to(device),
            test_logger.log,
        )
        if args.DEBUG:
            break
    test_logger.push(len(val_loader))


args = MNISTAbsorbingArguments(explicit_bool=True).parse_args()


def main():
    # name = "addition_" + str(args.N)
    run = wandb.init(
        project=f"nesy-diffusion",
        # name=name,
        tags=[],
        config=args.__dict__,
        mode="disabled" if not args.use_wandb else "online",
    )

    device = get_device(args)

    model = create_mnistadd(args).to(device)
    arity = 2
    digits_per_number = args.N
    n_operands = arity * digits_per_number

    bin_op = sum if args.op == "sum" else math.prod if args.op == "product" else None
    op = create_nary_multidigit_operation(arity, bin_op)

    if args.DEBUG:
        # Enable anomaly detection in PyTorch for debugging NaNs
        torch.autograd.set_detect_anomaly(True)

        # Add hooks to check for NaNs in gradients
        def hook(grad):
            if torch.isnan(grad).any():
                print("NaN gradient detected!")
                raise RuntimeError("NaN gradient detected")

        for p in model.parameters():
            if p.requires_grad:
                p.register_hook(hook)

    train_size = 60000 if args.test else 50000
    val_size = 0 if args.test else 10000
    train_loader, val_loader, test_loader = get_mnist_op_dataloaders(
        count_train=int(train_size / n_operands),
        count_val=int(val_size / n_operands),
        count_test=int(10000 / n_operands),
        batch_size=args.batch_size,
        n_operands=n_operands,
        op=op,
        # This shuffle is very weird...
        shuffle=True,
    )

    log_iterations = len(train_loader) // args.log_per_epoch

    train_logger = TrainLogger(log_iterations, TrainingLog, args)
    val_logger = TestLogger(TestLog, args, "val")
    print("Length of val loader:", len(val_loader))

    optim = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0
    )

    os.makedirs(f"models/{run.id}", exist_ok=True)
    for epoch in range(args.epochs):
        print("----------------------------------------")
        print("NEW EPOCH", epoch)

        start_epoch_time = time.time()

        for i, batch in enumerate(train_loader):
            optim.zero_grad()
            mn_digits, label, w_labels = batch[: 2 * args.N], batch[-1], batch[2 * args.N : -1]

            x = torch.cat(mn_digits, dim=1).to(device)
            w_labels = torch.stack(w_labels, dim=1).to(device)
            label = vector_to_base10(label.to(device), args.N + 1)
            loss = model.loss(x, label, train_logger.log, w_labels)

            loss.backward()
            optim.step()

            train_logger.step()

            if args.DEBUG:
                break

        end_epoch_time = time.time()

        epoch_time = end_epoch_time - start_epoch_time
        print(f"Epoch time: {epoch_time} seconds")

        # If val not available, don't test during training
        if epoch % args.test_every_epochs == 0:
            if not args.test:
                print("----- VALIDATING -----")
                test(val_loader, val_logger, model, device)
                test_time = time.time() - end_epoch_time
                print(f"Test time: {test_time} seconds")
            
            print(f"Saving model to {run.id}")
            wandb.save(f"model_{epoch}_{run.id}.pth")
            torch.save(model.state_dict(), f"models/{run.id}/model_{epoch}.pth") 
            

    print("----- TESTING -----")
    test_logger = TestLogger(TestLog, args, "test")
    test(test_loader, test_logger, model, device)
    print(f"Saving model to {run.id}")
    wandb.save(f"model_{epoch}_{run.id}.pth")
    torch.save(model.state_dict(), f"models/{run.id}/model_{epoch}.pth") 


if __name__ == "__main__":
    main()
