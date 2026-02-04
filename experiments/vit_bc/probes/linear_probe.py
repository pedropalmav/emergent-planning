import torch
import torch.nn as nn
from typing import Optional


class LinearProbe(nn.Module):
    def __init__(self, in_channels: int, out_dim: int, nl: bool = False):
        super().__init__()
        self.ff = nn.Linear(
            in_features=in_channels * 64, out_features=out_dim * 64, bias=False
        )
        self.out_dim = out_dim
        self.loss_fnc = nn.CrossEntropyLoss()

    def forward(self, input: torch.tensor, targets: Optional[torch.tensor] = None):
        input = input.view(input.shape[0], -1)
        out = self.ff(input)
        if targets is not None:
            assert out.shape[0] == targets.shape[0]
            out = out.view(out.shape[0], self.out_dim, 64)
            targets = targets.view(out.shape[0], 64)
            loss = self.loss_fnc(out, targets)
        else:
            loss = None
        return out, loss
