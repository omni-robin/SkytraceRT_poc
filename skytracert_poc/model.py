from __future__ import annotations

import torch
from torch import nn


class TinyIQOccNet(nn.Module):
    """Tiny 1D CNN over raw IQ.

    Input:  x [B, 2, L] int16/float
    Output: y [B, F] logits (occupancy per frequency bin)

    Notes:
    - We keep it small so it can later be exported to ONNX/TensorRT.
    - We downsample aggressively, then pool to a fixed embedding.
    """

    def __init__(self, freq_bins: int = 1024, width: int = 32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(2, width, kernel_size=9, stride=4, padding=4, bias=False),
            nn.BatchNorm1d(width),
            nn.SiLU(),

            nn.Conv1d(width, width, kernel_size=9, stride=4, padding=4, bias=False),
            nn.BatchNorm1d(width),
            nn.SiLU(),

            nn.Conv1d(width, width * 2, kernel_size=9, stride=4, padding=4, bias=False),
            nn.BatchNorm1d(width * 2),
            nn.SiLU(),

            nn.Conv1d(width * 2, width * 2, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(width * 2),
            nn.SiLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(width * 2, freq_bins),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
