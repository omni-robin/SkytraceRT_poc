#!/usr/bin/env python3
"""Evaluate feature (FFT/log-PSD) occupancy model vs SigMF GT bands.

This mirrors scripts/eval_occ.py but uses the TinyFeatOccNet + FFT frontend.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from skytracert_poc.metrics import Band2, band_iou_1d, band_recall_coverage
from skytracert_poc.model_feat import TinyFeatOccNet
from skytracert_poc.postprocess import occ_to_bands


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_sigmf(meta_path: Path) -> tuple[dict, dict, dict]:
    j = json.loads(meta_path.read_text())
    cap0 = (j.get("captures") or [{}])[0] or {}
    g = j.get("global") or {}
    rc = (g.get("annotations") or {}).get("custom", {}).get("rc_configuration") or {}
    return j, cap0, rc


def read_ci16(data_path: Path, sample_count: int) -> np.ndarray:
    raw = np.fromfile(data_path, dtype="<i2", count=2 * sample_count)
    i = raw[0::2]
    q = raw[1::2]
    return np.stack([i, q], axis=0).astype(np.float32) / 32768.0


def window_logpsd(x_iq: np.ndarray, nfft: int, hop: int, eps: float = 1e-12) -> np.ndarray:
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


def infer_capture_occ(
    model: TinyFeatOccNet,
    device: torch.device,
    x_f32: np.ndarray,
    win_len: int,
    win_hop: int,
    nfft: int,
    fft_hop: int,
    freq_bins: int,
) -> np.ndarray:
    model.eval()
    L = x_f32.shape[1]
    win_len = min(win_len, L)
    hop = min(win_hop, win_len)

    outs = []
    for start in range(0, L - win_len + 1, hop):
        w = x_f32[:, start : start + win_len]
        logp = window_logpsd(w, nfft=nfft, hop=fft_hop)
        if logp.shape[0] != freq_bins:
            x_old = np.linspace(0.0, 1.0, logp.shape[0], dtype=np.float32)
            x_new = np.linspace(0.0, 1.0, freq_bins, dtype=np.float32)
            logp = np.interp(x_new, x_old, logp).astype(np.float32)

        xt = torch.from_numpy(logp[None, :]).to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type in ("mps", "cuda")):
            logits = model(xt)
            prob = torch.sigmoid(logits).float().cpu().numpy()[0]
        outs.append(prob)

    return np.mean(np.stack(outs, axis=0), axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", default="artifacts/eval_feat.json")
    ap.add_argument("--win-len", type=int, default=262_144)
    ap.add_argument("--win-hop", type=int, default=262_144)
    ap.add_argument("--nfft", type=int, default=2048)
    ap.add_argument("--fft-hop", type=int, default=1024)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min-bins", type=int, default=3)
    ap.add_argument("--merge-gap-bins", type=int, default=1)
    ap.add_argument("--max-bands", type=int, default=8)
    args = ap.parse_args()

    device = pick_device()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["model"]
    F = int(cfg["freq_bins"])
    hidden = int(cfg["hidden"])

    model = TinyFeatOccNet(freq_bins=F, hidden=hidden).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    in_dir = Path(args.in_dir)

    rows = []
    for meta_path in sorted(in_dir.glob("*.sigmf-meta")):
        stem = meta_path.name.removesuffix(".sigmf-meta")
        data_path = in_dir / f"{stem}.sigmf-data"
        if not data_path.exists():
            continue

        j, cap0, rc = load_sigmf(meta_path)
        sr_hz = float((j.get("global") or {}).get("core:sample_rate"))
        fc_hz = float(cap0.get("core:frequency"))
        sc = int(cap0.get("core:sample_count"))

        x = read_ci16(data_path, sc)
        occ = infer_capture_occ(model, device, x, args.win_len, args.win_hop, args.nfft, args.fft_hop, F)

        edges_bb = np.linspace(-sr_hz / 2.0, sr_hz / 2.0, F + 1)
        edges_abs = edges_bb + fc_hz

        pred_bands = occ_to_bands(
            occ,
            edges_abs,
            thr=args.thr,
            min_bins=args.min_bins,
            merge_gap_bins=args.merge_gap_bins,
        )
        pred_bands = pred_bands[: args.max_bands]

        gt_bands = [
            Band2(lower_hz=float(r["min_frequency_mhz"]) * 1e6, upper_hz=float(r["max_frequency_mhz"]) * 1e6)
            for r in (rc.get("rcs") or [])
        ]
        pred_simple = [Band2(lower_hz=b.lower_hz, upper_hz=b.upper_hz) for b in pred_bands]

        best_ious = []
        for g in gt_bands:
            best_ious.append(max((band_iou_1d(g, p) for p in pred_simple), default=0.0))

        rows.append(
            {
                "id": stem,
                "gt_n": len(gt_bands),
                "pred_n": len(pred_simple),
                "best_match_mean_iou": float(np.mean(best_ious)) if best_ious else 1.0,
                "gt_coverage": float(band_recall_coverage(gt_bands, pred_simple)),
                "capture": {"fc_hz": fc_hz, "sr_hz": sr_hz, "sample_count": sc},
                "pred_bands": [asdict(b) for b in pred_bands],
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"rows": rows}, indent=2))

    if rows:
        print("captures:", len(rows))
        print("mean gt_coverage:", sum(r["gt_coverage"] for r in rows) / len(rows))
        print("mean best_match_mean_iou:", sum(r["best_match_mean_iou"] for r in rows) / len(rows))
        print("mean pred_n:", sum(r["pred_n"] for r in rows) / len(rows))
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
