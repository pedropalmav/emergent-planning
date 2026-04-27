import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock
from typing import Optional, Callable


class ResNetBC(nn.Module):
    def __init__(
        self,
        image_size: int,
        kernel_size: int,
        num_layers: int,
        num_classes: int,
        hidden_dim: int = 32,
        mlp_dim: int = 256,
        image_channels: int = 3,
        **kwargs
    ) -> None:
        super().__init__()

        self.encoder = nn.Conv2d(
            in_channels=image_channels,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.core = nn.Sequential(
            *[
                BasicBlock(hidden_dim, hidden_dim, stride=kernel_size // 2)
                for _ in range(num_layers)
            ]
        )
        self.linear = nn.Linear(hidden_dim * (image_size**2), mlp_dim)
        self.policy = nn.Linear(mlp_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.core(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear(x))
        x = self.policy(x)
        return x


if __name__ == "__main__":
    resnet = ResNetBC(
        image_size=8,
        kernel_size=3,
        num_layers=24,
        hidden_dim=32,
        mlp_dim=128,
        num_classes=5,
        image_channels=7,
    )

    print(resnet)
