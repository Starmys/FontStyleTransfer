import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            torch.nn.Linear(32 * 32, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32 * 32)
        )

    def forward(self, x):
        latency = self.encoder(self.flatten(x))
        return self.decoder(latency).view(1, 1, 32, 32)
