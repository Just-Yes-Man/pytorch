import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 2),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
