from typing_extensions import override

from expressive.args import PathPlanningArguments
from expressive.experiments.path_planning.perception import CombResnet18
import torch
from torch import Tensor

from expressive.experiments.path_planning.dijkstra import compute_shortest_path
from expressive.methods.base_model import BaseNeSyDiffusion, Problem
from expressive.methods.cond_model import CondNeSyDiffusion
from expressive.methods.simple_nesy_diff import SimpleNeSyDiffusion
from expressive.models.diffusion_model import WY_DATA, UnmaskingModel
from math import prod
import torch.nn as nn

class PathAbsorbModel(UnmaskingModel):
    def __init__(self, args: PathPlanningArguments, out_feats: int) -> None:
        super().__init__(
            vocab_dim=out_feats,
            w_dims=args.grid_size**2,
            seq_length=3 * args.grid_size + 1,
            args=args,
        )
        self.model = CombResnet18(args.grid_size, out_feats, 3, y_embed=args.y_embed)
        self.grid_size = args.grid_size

    def encode_x(self, x: Tensor) -> Tensor:
        return x

    def logits_t0(self, wy_t, x_encoding: Tensor, t: Tensor) -> Tensor:
        if isinstance(wy_t, tuple):
            return self.model(x_encoding, wy_t[0], wy_t[1], t)
        return self.model(x_encoding, wy_t, None, t)


class PathAbsorbing(Problem, nn.Module):
    def __init__(self, args: PathPlanningArguments):
        super().__init__()
        self.grid_size: int = args.grid_size
        self.costs = args.costs
        # reverse = PathAbsorbModel(args, len(self.costs))
        # super().__init__(reverse, self.y_from_w, args)
        self.register_buffer("costs_t", torch.tensor(self.costs))
        self.debug = args.DEBUG

    @override
    def shape_w(self) -> torch.Size:
        return (self.grid_size**2, len(self.costs))

    @override
    def shape_y(self) -> torch.Size:
        return (self.grid_size**2, 2)

    @override
    def y_from_w(self, w_SBW: torch.Tensor) -> torch.Tensor:
        # Reshape to prepare for shortest path computations.
        w_KGG = w_SBW.reshape(prod(w_SBW.shape[:-1]), self.grid_size, self.grid_size)
        costs_KGG = self.costs_t[w_KGG]
        y_KGG = compute_shortest_path(costs_KGG, debug=self.debug)
        return y_KGG.reshape(w_SBW.shape).long()

    def eval_y(self, y_0_SBY: Tensor, y_0_BY: Tensor, w_0_BW: Tensor) -> Tensor:
        cost_grid = self.costs_t[w_0_BW]
        cost_gt = (cost_grid * y_0_BY).sum(dim=-1)
        cost_hat = (cost_grid * y_0_SBY).sum(dim=-1)
        return torch.isclose(cost_gt, cost_hat)

def create_nesy_diffusion(args: PathPlanningArguments) -> BaseNeSyDiffusion:
    model = PathAbsorbModel(args, len(args.costs))
    problem = PathAbsorbing(args)
    if args.simple_model:
        assert not args.y_embed
        return SimpleNeSyDiffusion(model, problem, args)
    else:
        return CondNeSyDiffusion(model, problem, args)
