from typing_extensions import override

from expressive.args import RSBenchArguments
import torch
from torch import Tensor

from expressive.experiments.rsbench.datasets.utils.base_dataset import BaseDataset
from expressive.methods.base_model import BaseNeSyDiffusion, Problem
from expressive.methods.cond_model import CondNeSyDiffusion
from expressive.methods.simple_nesy_diff import SimpleNeSyDiffusion
from expressive.models.diffusion_model import WY_DATA, UnmaskingModel
import torch.nn as nn
from torch.nn.functional import one_hot

class RSBenchModel(UnmaskingModel):
    def __init__(self, args: RSBenchArguments, dataset: BaseDataset) -> None:
        super().__init__(
            vocab_dim=dataset.get_w_dim()[1],
            w_dims=dataset.get_w_dim()[0],
            seq_length=None,
            args=args,
        )
        # Make sure to look at the correct code for the backbone
        self.encoder, self.classifier = dataset.get_backbone_nesydiff()
        self.dataset_name = args.dataset
        self.dataset = dataset

    def encode_x(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def logits_t0(self, wy_t, x_encodings: Tensor, t: Tensor) -> Tensor:
        w_SBW = wy_t
        if not isinstance(wy_t, Tensor):
            w_SBW = wy_t[0]
        one_hot_w = one_hot(w_SBW, self.vocab_dim + 1)
        if self.args.backbone in ["partialentangled", "fullentangled"]:
            return self.classifier(x_encodings, one_hot_w)
        if self.args.backbone == "disentangled":
            concept_preds = []
            for i in range(self.w_dims):
                encoding_split = torch.split(x_encodings, x_encodings.size(-1) // self.w_dims, dim=-1)
                enc_flat = self.classifier(encoding_split[i], one_hot_w, i)
                if len(one_hot_w.shape) == 5:
                    enc_flat = enc_flat.reshape(one_hot_w.shape[0], one_hot_w.shape[1], enc_flat.shape[-1])
                concept_preds.append(enc_flat)
            return torch.stack(concept_preds, dim=-2)
        raise NotImplementedError(f"Backbone {self.args.backbone} not implemented")


class RSBenchAdapter(Problem, nn.Module):
    def __init__(self, args: RSBenchArguments, dataset: BaseDataset):
        super().__init__()
        self.debug = args.DEBUG
        self.dataset = dataset
    @override
    def shape_w(self) -> torch.Size:
        return self.dataset.get_w_dim()

    @override
    def shape_y(self) -> torch.Size:
        return self.dataset.get_y_dim()

    @override
    def y_from_w(self, w_SBW: torch.Tensor) -> torch.Tensor:
        return self.dataset.y_from_w(w_SBW)


def create_rsbench_diffusion(args: RSBenchArguments, dataset: BaseDataset) -> BaseNeSyDiffusion:
    model = RSBenchModel(args, dataset)
    problem = RSBenchAdapter(args, dataset)
    if args.simple_model:
        return SimpleNeSyDiffusion(model, problem, args)
    else:
        return CondNeSyDiffusion(model, problem, args)
