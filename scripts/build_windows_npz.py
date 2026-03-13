#!/usr/bin/env python3
"""Build raw-IQ training windows + frequency-occupancy targets.

Assumptions:
- SigMF data is complex int16 little-endian (ci16_le), interleaved I,Q int16.
- Ground truth controller bands are absolute Hz in SigMF meta:
    global.annotations.custom.rc_configuration.rcs[].min_frequency_mhz/max_frequency_mhz

Output:
- NPZ with:
  - X_i16: int16 [N, 2, L]  (I and Q channels)
  - y_occ: uint8 [N, F]     (frequency occupancy mask, baseband bins)
  - meta: JSON string with bin edges and capture ids

Notes:
- This is a PoC dataset builder. It intentionally keeps things simple:
  targets are constant over time within a capture and are mapped into the capture span.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_sigmf_ci16(data_path: Path, sample_count: int | None = None) -> np.ndarray:
    """Return interleaved int16 samples as shape [2, L] (I,Q)."""
    if sample_count is None:
        # infer from file size
        n_i16 = data_path.stat().st_size // 2
        sample_count = n_i16 // 2
    raw = np.fromfile(data_path, dtype="<i2", count=2 * sample_count)
    if raw.size != 2 * sample_count:
        raise ValueError(f"Short read: {data_path} expected {2*sample_count} i16 got {raw.size}")
    i = raw[0::2]
    q = raw[1::2]
    return np.stack([i, q], axis=0)  # [2, L]


def make_freq_bins(sr_hz: float, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (edges_hz, centers_hz) in baseband [-sr/2, sr/2]."""
    edges = np.linspace(-sr_hz / 2.0, sr_hz / 2.0, n_bins + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def bands_to_mask(
    bands_abs_hz: list[tuple[float, float]],
    fc_hz: float,
    sr_hz: float,
    bin_centers_bb_hz: np.ndarray,
) -> np.ndarray:
    """Map absolute-Hz bands into baseband bins for this capture, producing {0,1} mask."""
    # capture span in absolute Hz
    cap_lo = fc_hz - sr_hz / 2.0
    cap_hi = fc_hz + sr_hz / 2.0

    # convert baseband bin centers to absolute Hz
    bin_abs = bin_centers_bb_hz + fc_hz

    y = np.zeros((bin_centers_bb_hz.size,), dtype=np.uint8)
    for lo, hi in bands_abs_hz:
        # intersect with capture span
        lo2 = max(lo, cap_lo)
        hi2 = min(hi, cap_hi)
        if hi2 <= lo2:
            continue
        y |= ((bin_abs >= lo2) & (bin_abs <= hi2)).astype(np.uint8)
    return y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-jsonl",
        required=True,
        help="JSONL index (one row per capture). See scripts/build_dataset_jsonl.py",
    )
    ap.add_argument("--out", required=True, help="Output NPZ path")
    ap.add_argument("--win-len", type=int, default=262_144, help="Window length in samples (default: 262144)")
    ap.add_argument("--win-hop", type=int, default=262_144, help="Hop length in samples (default: 262144)")
    ap.add_argument("--freq-bins", type=int, default=1024, help="Number of frequency bins for occupancy target")
    args = ap.parse_args()

    rows = []
    with open(args.dataset_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise SystemExit("No rows in dataset_jsonl")

    # Global assumptions for PoC (can be generalized later)
    sr_hz = float(rows[0]["sample_rate_hz"])
    edges_bb_hz, centers_bb_hz = make_freq_bins(sr_hz, args.freq_bins)

    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    id_list: list[str] = []

    for r in rows:
        if r.get("datatype") != "ci16_le":
            raise ValueError(f"Unsupported datatype {r.get('datatype')} in {r.get('id')}")
        sr = float(r["sample_rate_hz"])
        if abs(sr - sr_hz) > 1e-6:
            raise ValueError("Mixed sample rates not supported in this PoC")

        fc_hz = float(r.get("capture", {}).get("frequency_hz"))
        sc = r.get("capture", {}).get("sample_count")
        sc = int(sc) if sc is not None else None

        data_path = Path(r["data_path"])
        x = read_sigmf_ci16(data_path, sample_count=sc)  # [2, L]

        # collect GT bands (absolute Hz)
        bands = []
        for c in (r.get("gt", {}) or {}).get("controllers") or []:
            lo = c.get("min_frequency_hz")
            hi = c.get("max_frequency_hz")
            if lo is None or hi is None:
                continue
            bands.append((float(lo), float(hi)))

        y_occ = bands_to_mask(bands, fc_hz=fc_hz, sr_hz=sr_hz, bin_centers_bb_hz=centers_bb_hz)

        L = x.shape[1]
        win_len = min(args.win_len, L)
        hop = min(args.win_hop, win_len)

        start = 0
        while start + win_len <= L:
            X_list.append(x[:, start : start + win_len].copy())
            y_list.append(y_occ.copy())
            id_list.append(r["id"])
            start += hop

        # if capture shorter than win_len, include one padded window
        if L < args.win_len:
            pad = args.win_len - L
            xpad = np.pad(x, ((0, 0), (0, pad)), mode="constant")
            X_list.append(xpad.astype(np.int16, copy=False))
            y_list.append(y_occ.copy())
            id_list.append(r["id"])

    X = np.stack(X_list, axis=0)  # [N, 2, L]
    y = np.stack(y_list, axis=0)  # [N, F]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "freq_bin_edges_bb_hz": edges_bb_hz.tolist(),
        "freq_bin_centers_bb_hz": centers_bb_hz.tolist(),
        "sample_rate_hz": sr_hz,
        "win_len": int(args.win_len),
        "win_hop": int(args.win_hop),
        "freq_bins": int(args.freq_bins),
        "ids": id_list,
    }

    np.savez_compressed(out_path, X_i16=X, y_occ=y, meta_json=json.dumps(meta))
    print(f"Wrote {out_path} with X_i16={X.shape} y_occ={y.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
