from __future__ import annotations

import torch
from torch import nn


class TinyFeatOccNet(nn.Module):
    """Tiny MLP over log-PSD features.

    Input:  x [B, F]
    Output: y [B, F] logits
    """

    def __init__(self, freq_bins: int = 1024, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(freq_bins),
            nn.Linear(freq_bins, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, freq_bins),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
