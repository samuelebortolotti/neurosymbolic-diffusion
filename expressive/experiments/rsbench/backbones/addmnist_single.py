import torch.nn
from torch import nn
from backbones.base.ops import *


class MNISTSingleEncoder(nn.Module):
    def __init__(
        self, img_channels=1, hidden_channels=32, c_dim=10, latent_dim=16, dropout=0.5, n_images=2
    ):
        super(MNISTSingleEncoder, self).__init__()

        self.channels = 3
        self.img_channels = img_channels
        self.hidden_channels = hidden_channels
        self.c_dim = c_dim
        self.latent_dim = latent_dim
        self.n_images = n_images

        self.unflatten_dim = (3, 7)

        self.enc_block_1 = nn.Conv2d(
            in_channels=self.img_channels,
            out_channels=self.hidden_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.enc_block_2 = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.enc_block_3 = nn.Conv2d(
            in_channels=self.hidden_channels * 2,
            out_channels=self.hidden_channels * 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        self.flatten = Flatten()

        self.dense_logvar = nn.Linear(
            in_features=int(
                4
                * self.hidden_channels
                * self.unflatten_dim[0]
                * self.unflatten_dim[1]
                * (3 / 7)
            ),
            out_features=self.latent_dim * self.c_dim,
        )

        self.dense_mu = nn.Linear(
            in_features=int(
                4
                * self.hidden_channels
                * self.unflatten_dim[0]
                * self.unflatten_dim[1]
                * (3 / 7)
            ),
            out_features=self.latent_dim * self.c_dim,
        )

        self.dense_c = nn.Linear(
            in_features=int(
                4
                * self.hidden_channels
                * self.unflatten_dim[0]
                * self.unflatten_dim[1]
                * (3 / 7)
            ),
            out_features=self.c_dim,
        )

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # MNISTPairsEncoder block 1
        x = self.enc_block_1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # MNISTPairsEncoder block 2
        x = self.enc_block_2(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # MNISTPairsEncoder block 3
        x = self.enc_block_3(x)
        x = nn.ReLU()(x)

        # mu and logvar
        x = self.flatten(
            x
        )  # batch_size, dim1, dim2, dim3 -> batch_size, dim1*dim2*dim3
        c, mu, logvar = self.dense_c(x), self.dense_mu(x), self.dense_logvar(x)

        # return encodings for each object involved
        c = torch.stack(torch.split(c, self.c_dim, dim=-1), dim=1)
        mu = torch.stack(torch.split(mu, self.latent_dim, dim=-1), dim=1)
        logvar = torch.stack(torch.split(logvar, self.latent_dim, dim=-1), dim=1)

        return c, mu, logvar

class MNISTNeSyDiffEncoder(MNISTSingleEncoder):

    def encode_digit(self, x: torch.Tensor) -> torch.Tensor:
        # MNISTPairsEncoder block 1
        x = self.enc_block_1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # MNISTPairsEncoder block 2
        x = self.enc_block_2(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)

        # MNISTPairsEncoder block 3
        x = self.enc_block_3(x)
        x = nn.ReLU()(x)

        # mu and logvar
        return self.flatten(
            x
        )  # batch_size, dim1, dim2, dim3 -> batch_size, dim1*dim2*dim3

    def forward(self, x):
        x_encoding_flat = x
        if len(x.shape) == 5:
            x_encoding_flat = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        x_split = torch.split(x_encoding_flat, x.size(-1) // self.n_images, dim=-1)
        
        encodings = []
        for i in range(self.n_images):
            enc_flat = self.encode_digit(x_split[i])
            if len(x.shape) == 5:
                enc_flat = enc_flat.reshape(x.shape[0], x.shape[1], enc_flat.shape[-1])
            else:
                enc_flat = enc_flat.squeeze()
            encodings.append(enc_flat)
        return torch.cat(encodings, dim=-1)

class MNISTNeSyDiffClassifier(nn.Module):
    def __init__(
        self, n_images, embed_all_images=False, hidden_channels=32, c_dim=10, latent_dim=16, dropout=0.5
    ):
        super(MNISTNeSyDiffClassifier, self).__init__()
        assert n_images == 2, "Only 2 images are supported for now"

        self.hidden_channels = hidden_channels
        self.c_dim = c_dim
        self.latent_dim = latent_dim
        self.n_images = n_images

        self.unflatten_dim = (3, 7)
        self.embed_all_images = embed_all_images

        self.dense_c = nn.Linear(
            in_features=int(
                4
                * self.hidden_channels
                * self.unflatten_dim[0]
                * self.unflatten_dim[1]
                * (3 / 7)
                * (n_images if embed_all_images else 1)
                + n_images * (self.c_dim + 1)
            ),
            out_features=self.c_dim,
        )

    def forward(self, x_encodings, w_0_BWD, image_to_classify: int = -1):
        w_0_1_BD = w_0_BWD[..., 0, :]
        w_0_2_BD = w_0_BWD[..., 1, :]
        if self.embed_all_images:
            # Commutativity equivariance: Ensure the image to classify is always the first one, then share weights
            # Ie, this predicts p(y_1|x_1, x_2, w) = f(w, x_1, x_2), and p(y_2|x_1, x_2, w) = f(w, x_2, x_1)
            c1 = self.dense_c(torch.cat((w_0_1_BD, w_0_2_BD, x_encodings), dim=-1))
            encoding_split = torch.split(x_encodings, x_encodings.size(-1) // self.n_images, dim=-1)
            c2 = self.dense_c(torch.cat((w_0_2_BD, w_0_1_BD, encoding_split[1], encoding_split[0]), dim=-1))
            return torch.stack([c1, c2], dim=-2)

        if image_to_classify == 0:
            return self.dense_c(torch.cat((w_0_1_BD, w_0_2_BD, x_encodings), dim=-1))
        elif image_to_classify == 1:
            return self.dense_c(torch.cat((w_0_2_BD, w_0_1_BD, x_encodings), dim=-1))
        raise ValueError(f"Invalid image to classify: {image_to_classify}")