#!/usr/bin/env python3
"""Run inference on a single SigMF capture and emit controller bands (absolute Hz).

This is a PoC inference script:
- slices the capture into windows
- runs occupancy model per window
- averages occupancy over windows
- post-processes into bands

Output: JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys

# Allow running from repo root without installing as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skytracert_poc.model import TinyIQOccNet
from skytracert_poc.postprocess import occ_to_bands


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_sigmf_meta(meta_path: Path) -> dict:
    return json.loads(meta_path.read_text())


def read_ci16(data_path: Path, sample_count: int) -> np.ndarray:
    raw = np.fromfile(data_path, dtype="<i2", count=2 * sample_count)
    i = raw[0::2]
    q = raw[1::2]
    x = np.stack([i, q], axis=0)
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--win-len", type=int, default=262_144)
    ap.add_argument("--win-hop", type=int, default=262_144)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min-bins", type=int, default=3)
    ap.add_argument("--merge-gap-bins", type=int, default=1)
    ap.add_argument("--max-bands", type=int, default=8)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_cfg = ckpt["model"]
    freq_bins = int(model_cfg["freq_bins"])
    width = int(model_cfg["width"])

    device = pick_device()

    model = TinyIQOccNet(freq_bins=freq_bins, width=width).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    meta = load_sigmf_meta(Path(args.meta))
    cap0 = (meta.get("captures") or [{}])[0] or {}
    g = meta.get("global") or {}

    sr_hz = float(g.get("core:sample_rate"))
    fc_hz = float(cap0.get("core:frequency"))
    sc = int(cap0.get("core:sample_count"))

    x_i16 = read_ci16(Path(args.data), sc)
    x_f32 = (x_i16.astype(np.float32) / 32768.0)

    # slice into windows
    L = x_f32.shape[1]
    win_len = min(args.win_len, L)
    hop = min(args.win_hop, win_len)

    outs = []
    for start in range(0, L - win_len + 1, hop):
        w = x_f32[:, start : start + win_len]
        xt = torch.from_numpy(w[None, ...])  # [1,2,L]
        xt = xt.to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type in ("mps", "cuda")):
            logits = model(xt)
            prob = torch.sigmoid(logits).float().cpu().numpy()[0]
        outs.append(prob)

    if not outs:
        raise SystemExit("No windows produced")

    occ = np.mean(np.stack(outs, axis=0), axis=0)

    # map bins to absolute Hz edges
    edges_bb = np.linspace(-sr_hz / 2.0, sr_hz / 2.0, freq_bins + 1)
    edges_abs = edges_bb + fc_hz

    bands = occ_to_bands(
        occ,
        edges_abs,
        thr=args.thr,
        min_bins=args.min_bins,
        merge_gap_bins=args.merge_gap_bins,
    )
    bands = bands[: args.max_bands]

    out = {
        "capture": {"center_frequency_hz": fc_hz, "sample_rate_hz": sr_hz, "sample_count": sc},
        "bands": [b.__dict__ for b in bands],
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
