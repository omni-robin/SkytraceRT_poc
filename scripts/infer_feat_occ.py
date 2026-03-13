#!/usr/bin/env python3
"""Inference using the feature (FFT/log-PSD) occupancy model.

This keeps the model tiny; the FFT frontend is computed explicitly.
Outputs controller bands in absolute Hz.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skytracert_poc.model_feat import TinyFeatOccNet
from skytracert_poc.postprocess import occ_to_bands


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_sigmf(meta_path: Path) -> tuple[dict, dict, float, float, int]:
    j = json.loads(meta_path.read_text())
    cap0 = (j.get("captures") or [{}])[0] or {}
    g = j.get("global") or {}
    sr_hz = float(g.get("core:sample_rate"))
    fc_hz = float(cap0.get("core:frequency"))
    sc = int(cap0.get("core:sample_count"))
    return j, cap0, sr_hz, fc_hz, sc


def read_ci16(data_path: Path, sample_count: int) -> np.ndarray:
    raw = np.fromfile(data_path, dtype="<i2", count=2 * sample_count)
    i = raw[0::2]
    q = raw[1::2]
    return np.stack([i, q], axis=0).astype(np.float32) / 32768.0


def window_logpsd(x_iq: np.ndarray, nfft: int, hop: int, eps: float = 1e-12) -> np.ndarray:
    """Compute log-PSD over one window, returning fftshifted bins length nfft."""
    iq = x_iq[0] + 1j * x_iq[1]
    win = np.hanning(nfft).astype(np.float32)
    acc = None
    nseg = 0
    L = iq.shape[0]
    for start in range(0, L - nfft + 1, hop):
        seg = iq[start : start + nfft] * win
        X = np.fft.fftshift(np.fft.fft(seg, nfft))
        p = (np.abs(X) ** 2).astype(np.float64)
        acc = p if acc is None else (acc + p)
        nseg += 1
    psd = (acc / max(nseg, 1)).astype(np.float32)
    return np.log(psd + eps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--win-len", type=int, default=262_144)
    ap.add_argument("--win-hop", type=int, default=262_144)
    ap.add_argument("--nfft", type=int, default=2048)
    ap.add_argument("--fft-hop", type=int, default=1024)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min-bins", type=int, default=3)
    ap.add_argument("--merge-gap-bins", type=int, default=1)
    ap.add_argument("--smooth-radius", type=int, default=2)
    ap.add_argument("--hysteresis", type=float, default=0.15)
    ap.add_argument("--max-bands", type=int, default=8)
    args = ap.parse_args()

    device = pick_device()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["model"]
    F = int(cfg["freq_bins"])
    hidden = int(cfg["hidden"])

    model = TinyFeatOccNet(freq_bins=F, hidden=hidden).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    _, _, sr_hz, fc_hz, sc = load_sigmf(Path(args.meta))
    x = read_ci16(Path(args.data), sc)

    L = x.shape[1]
    win_len = min(args.win_len, L)
    hop = min(args.win_hop, win_len)

    probs = []
    for start in range(0, L - win_len + 1, hop):
        w = x[:, start : start + win_len]
        logp = window_logpsd(w, nfft=args.nfft, hop=args.fft_hop)
        if logp.shape[0] != F:
            x_old = np.linspace(0.0, 1.0, logp.shape[0], dtype=np.float32)
            x_new = np.linspace(0.0, 1.0, F, dtype=np.float32)
            logp = np.interp(x_new, x_old, logp).astype(np.float32)
        xt = torch.from_numpy(logp[None, :]).to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type in ("mps", "cuda")):
            logits = model(xt)
            prob = torch.sigmoid(logits).float().cpu().numpy()[0]
        probs.append(prob)

    occ = np.mean(np.stack(probs, axis=0), axis=0)

    edges_bb = np.linspace(-sr_hz / 2.0, sr_hz / 2.0, F + 1)
    edges_abs = edges_bb + fc_hz

    bands = occ_to_bands(
        occ,
        edges_abs,
        thr=args.thr,
        min_bins=args.min_bins,
        merge_gap_bins=args.merge_gap_bins,
        smooth_radius=args.smooth_radius,
        hysteresis=args.hysteresis,
    )
    bands = bands[: args.max_bands]

    out = {"capture": {"center_frequency_hz": fc_hz, "sample_rate_hz": sr_hz, "sample_count": sc}, "bands": [b.__dict__ for b in bands]}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
