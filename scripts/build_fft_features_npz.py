#!/usr/bin/env python3
"""Convert raw IQ windows NPZ into FFT/PSD feature NPZ.

Why: learning frequency occupancy from raw time-domain IQ is hard with small models.
A deterministic FFT frontend produces a compact feature vector (log-PSD) that trains
well, stays small, and is still very fast to compute on edge devices.

Input:
- windows NPZ from scripts/build_windows_npz.py
  - X_i16 [N,2,L]
  - y_occ [N,F]

Output:
- features NPZ:
  - X_feat [N,F] float16 (log-PSD per frequency bin)
  - y_occ [N,F] uint8
  - meta_json passthrough
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-npz", required=True)
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--nfft", type=int, default=2048)
    ap.add_argument("--hop", type=int, default=1024)
    ap.add_argument("--eps", type=float, default=1e-12)
    args = ap.parse_args()

    z = np.load(args.in_npz, allow_pickle=True)
    X_i16 = z["X_i16"]  # [N,2,L]
    y = z["y_occ"]
    meta_json = str(z["meta_json"])
    meta = json.loads(meta_json)
    F = int(meta["freq_bins"])

    N, _, L = X_i16.shape

    nfft = args.nfft
    hop = args.hop
    if nfft % 2 != 0:
        raise ValueError("nfft must be even")

    # We'll compute an nfft FFT and then downsample/interpolate to F bins.
    # For PoC, pick nfft = F*2 so that FFT bins map cleanly (after fftshift).
    if nfft != F:
        # Allow nfft != F, but we will resample.
        pass

    win = np.hanning(nfft).astype(np.float32)

    X_feat = np.zeros((N, F), dtype=np.float32)

    for i in range(N):
        xi = X_i16[i].astype(np.float32) / 32768.0
        iq = xi[0] + 1j * xi[1]

        acc = None
        nseg = 0
        for start in range(0, L - nfft + 1, hop):
            seg = iq[start : start + nfft] * win
            X = np.fft.fftshift(np.fft.fft(seg, nfft))
            p = (np.abs(X) ** 2).astype(np.float64)
            acc = p if acc is None else (acc + p)
            nseg += 1
        psd = (acc / max(nseg, 1)).astype(np.float32)
        logp = np.log(psd + float(args.eps))

        # resample to F bins if needed
        if logp.shape[0] == F:
            feat = logp
        else:
            x_old = np.linspace(0.0, 1.0, logp.shape[0], dtype=np.float32)
            x_new = np.linspace(0.0, 1.0, F, dtype=np.float32)
            feat = np.interp(x_new, x_old, logp).astype(np.float32)

        X_feat[i] = feat

    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X_feat=X_feat.astype(np.float16), y_occ=y, meta_json=meta_json)
    print(f"Wrote {out} X_feat={X_feat.shape} y_occ={y.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
