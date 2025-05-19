import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTAdditionCNN(nn.Module):
    def __init__(self):
        super(MNISTAdditionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 14, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 19)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 7 * 14)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

class EntangledDiffusionEncoder(nn.Module):
    def __init__(self):
        super(EntangledDiffusionEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 7 * 14)
        return x

class EntangledDiffusionClassifier(nn.Module):
    def __init__(self, n_images: int, n_classes: int):
        super(EntangledDiffusionClassifier, self).__init__()
        self.fc1 = nn.Linear(32 * 7 * 14 + n_images * (n_classes + 1), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_images * n_classes)
        self.softmax = nn.Softmax(dim=1)
        self.n_images = n_images
        self.n_classes = n_classes

    def forward(self, w_BWD: torch.Tensor, x_encoding: torch.Tensor):
        x = torch.cat((x_encoding, torch.reshape(w_BWD, w_BWD.shape[:-2] + (-1,))), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).reshape(w_BWD.shape[:-2] + (self.n_images, self.n_classes))
        x = self.softmax(x)
        return x


if __name__ == "__main__":
    model = MNISTAdditionCNN()
    dummy_input = torch.randn(64, 1, 28, 56)
    output = model(dummy_input)
    print(output.shape)
