import torch.nn as nn


class MNIST_Net(nn.Module):
    def __init__(self, n_classes=10):
        # TODO: For some reason this code goes into infinite recursion when using super(MNIST_Net, self).__init__()
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n_classes),
        )

    def forward(self, x):
        b_size = x.size(0)
        x = x.view(-1, 1, 28, 28)
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        if x.shape[0] != b_size:
            x = x.view(b_size, -1, 10)
        self.logits = x
        return self.logits
