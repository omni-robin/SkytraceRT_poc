from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class WindowsMeta:
    sample_rate_hz: float
    freq_bins: int
    win_len: int
    win_hop: int
    freq_bin_edges_bb_hz: list[float]
    freq_bin_centers_bb_hz: list[float]
    ids: list[str]


class NPZWindowsDataset(Dataset):
    """Dataset for artifacts/windows_*.npz created by scripts/build_windows_npz.py."""

    def __init__(self, npz_path: str | Path):
        self.npz_path = Path(npz_path)
        z = np.load(self.npz_path, allow_pickle=True)
        self.X_i16 = z["X_i16"]  # [N,2,L] int16
        self.y_occ = z["y_occ"]  # [N,F] uint8
        self.meta = WindowsMeta(**json.loads(str(z["meta_json"])))

        if self.X_i16.ndim != 3 or self.X_i16.shape[1] != 2:
            raise ValueError(f"Bad X shape: {self.X_i16.shape}")

    def __len__(self) -> int:
        return int(self.X_i16.shape[0])

    def __getitem__(self, idx: int):
        # Convert to float in [-1,1] range. Keep it simple.
        x = self.X_i16[idx].astype(np.float32) / 32768.0
        y = self.y_occ[idx].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)
