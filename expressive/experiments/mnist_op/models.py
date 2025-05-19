import torch
import torch.nn as nn


class MNISTEncoder(nn.Module):
    def __init__(self, embedding_size: int, size=16 * 4 * 4):
        super(MNISTEncoder, self).__init__()
        self.size = size
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.mlp = nn.Sequential(
            nn.Linear(size, embedding_size),
        )

    def forward(self, x_B_D_28_28):
        x_BD_1_28_28 = x_B_D_28_28.reshape(-1, 1, 28, 28)
        x_BD_1_W_H = self.encoder(x_BD_1_28_28)
        x_BD_E = x_BD_1_W_H.view(-1, self.size)
        x_BD_E = self.mlp(x_BD_E)
        x_BDE = x_BD_E.view(x_B_D_28_28.shape[0], x_B_D_28_28.shape[1], -1)
        return x_BDE
