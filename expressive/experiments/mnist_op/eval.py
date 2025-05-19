from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import torch
from torch.utils.data import DataLoader


@dataclass
class _CorrectCounts:
    correct_digits: int
    correct_op: int
    tested_digits: int
    tested_op: int

    def __add__(self, other: _CorrectCounts) -> _CorrectCounts:
        return _CorrectCounts(
            self.correct_digits + other.correct_digits,
            self.correct_op + other.correct_op,
            self.tested_digits + other.tested_digits,
            self.tested_op + other.tested_op,
        )


@dataclass
class AccuracyMetrics:
    digit_accuracy: float
    op_accuracy: float


def test_step(
    model: torch.nn.Module,
    imgs: tuple[torch.Tensor, ...],
    individual_labels: tuple[torch.Tensor, ...],
    op: Callable[[list[torch.Tensor]], torch.Tensor],
    result_label: torch.Tensor,
    device: torch.device,
) -> _CorrectCounts:
    """Where each tensor is batched"""
    correct_digits = 0
    pred_tuple = []
    for img, individual_label in zip(imgs, individual_labels):
        img = img.to(device)
        individual_label = individual_label.to(device)
        logits = model(img)
        pred = torch.argmax(logits, dim=-1)
        correct_digits += torch.eq(pred, individual_label).sum().item()
        pred_tuple.append(pred)
    correct_op = 0
    for i in range(len(result_label)):
        correct_op += result_label[i].item() == op([x[i].cpu() for x in pred_tuple])
    n_operands = len(imgs)
    n_operations = len(result_label)
    return _CorrectCounts(
        correct_digits, correct_op, n_operands * n_operations, n_operations
    )


def eval_on_op_loader(
    model: torch.nn.Module,
    mnistop_loader: DataLoader,
    n_operands: int,
    op: Callable[[list[torch.Tensor]], torch.Tensor],
    device: torch.device,
) -> AccuracyMetrics:
    model.eval()
    counts = _CorrectCounts(0, 0, 0, 0)
    for batch in mnistop_loader:
        imgs, individual_labels, result_label = (
            batch[:n_operands],
            batch[n_operands : n_operands * 2],
            batch[-1],
        )
        counts += test_step(model, imgs, individual_labels, op, result_label, device)
    return AccuracyMetrics(
        counts.correct_digits / counts.tested_digits,
        counts.correct_op / counts.tested_op,
    )
