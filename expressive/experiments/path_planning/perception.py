from math import sqrt
from typing import Optional

import torchvision
from torch import nn
import torch


def get_resnet(amt_classes):
    model = torchvision.models.resnet18(weights=None, num_classes=amt_classes)

    return model


class CombResnet18(nn.Module):
    def __init__(self, grid_size: int, out_features: int, in_channels: int, y_embed: bool):
        super().__init__()
        self.resnet_model = torchvision.models.resnet18(
            weights=None, num_classes=out_features
        )
        del self.resnet_model.conv1
        self.embedding_size = 64
        self.resnet_model.conv1 = nn.Conv2d(
            in_channels,
            self.embedding_size,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.grid_size = grid_size
        output_shape = (grid_size, grid_size)
        self.pool = nn.AdaptiveMaxPool2d(output_shape)

        # TODO: Initial outputs aren't very uniformly distributed
        self.out = nn.Linear(self.embedding_size, out_features)
        # self.last_conv = nn.Conv2d(128, 1, kernel_size=1,  stride=1)

        self.cost_embeddings = nn.Embedding(out_features + 1, self.embedding_size)
        if y_embed:
            self.path_embeddings = nn.Embedding(3, self.embedding_size)
        self.y_embed = y_embed

    def forward(self, x_SB3HW, w_SBD, y_SBD: Optional[torch.Tensor], t: Optional[torch.Tensor]):
        # Architecture is mostly copied over from Adaptive IMLE:
        # https://github.com/nec-research/tf-imle/blob/main/WARCRAFT/maprop/models.py#L33
        if len(x_SB3HW.shape) == 5:
            x_B3HW = x_SB3HW.reshape(
                x_SB3HW.shape[0] * x_SB3HW.shape[1],
                x_SB3HW.shape[2],
                x_SB3HW.shape[3],
                x_SB3HW.shape[4],
            )
        else:
            x_B3HW = x_SB3HW
        x = self.resnet_model.conv1(x_B3HW)
        x = self.resnet_model.bn1(x)
        x = self.resnet_model.relu(x)
        x_BCHW = self.resnet_model.maxpool(x)

        # Add embeddings for cost and path to the image embedding
        e_SBDC = self.cost_embeddings(w_SBD) 
        if self.y_embed:
            e_SBDC += self.path_embeddings(y_SBD.long())
        if len(e_SBDC.shape) == 4:
            e_BDC = e_SBDC.reshape(
                e_SBDC.shape[0] * e_SBDC.shape[1], e_SBDC.shape[2], e_SBDC.shape[3]
            )
        else:
            e_BDC = e_SBDC
        e_BGGC = e_BDC.reshape(
            e_BDC.shape[0], self.grid_size, self.grid_size, self.embedding_size
        )
        e_BCGG = e_BGGC.permute(0, 3, 1, 2)
        repeats = x_BCHW.shape[2] // self.grid_size
        e_BCHG = torch.repeat_interleave(e_BCGG, repeats, dim=2)
        e_BCHW = torch.repeat_interleave(e_BCHG, repeats, dim=3)
        x_BCHW = x_BCHW + e_BCHW

        x = self.resnet_model.layer1(x_BCHW)
        x_BCGG = self.pool(x)

        # CHANGED FROM OTHER WORK:
        # Instead of mean aggregation (for continuous cost prediction) use a linear layer and softmax
        # x = x.mean(dim=1)
        x_BGGC = x_BCGG.permute(0, 2, 3, 1)

        x_BGGD = self.out(x_BGGC)
        if len(x_SB3HW.shape) == 5:
            return x_BGGD.reshape(
                x_SB3HW.shape[0],
                x_SB3HW.shape[1],
                self.grid_size * self.grid_size,
                x_BGGD.shape[-1],
            )
        return x_BGGD.reshape(
            x_BGGD.shape[0], self.grid_size * self.grid_size, x_BGGD.shape[-1]
        )
