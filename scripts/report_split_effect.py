#!/usr/bin/env python3
"""Report how much the split heuristic changes band count / edge metrics.

Goal: identify captures where GT has multiple bands but predictions are merged,
then see whether `occ_to_bands(split=True)` helps.

Outputs a JSON with per-capture stats + a short console summary.

Example:
  . .venv/bin/activate
  python scripts/report_split_effect.py \
    --ckpt artifacts/tiny_feat_occ_subset100_hz.pt \
    --in-dir /Users/omni/.openclaw/workspace/gcs_capture_data_config_info/subset100 \
    --out artifacts/report_split_subset100.json \
    --limit 30 \
    --split-min-peak-height 0.8 --split-min-peak-sep-bins 32 --split-min-valley-drop 0.12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# load_sigmf defined below (kept local to avoid packaging complexity)
from skytracert_poc.model_feat import TinyFeatOccNet
from skytracert_poc.postprocess import occ_to_bands
from skytracert_poc.metrics import Band2, band_recall_coverage, best_match_edge_error_hz


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


def read_ci16(path: Path, sample_count: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int16, count=2 * sample_count)
    if raw.size != 2 * sample_count:
        raise ValueError(f"Short read: expected {2*sample_count} int16, got {raw.size}")
    x = raw.reshape(-1, 2).T.astype(np.float32)
    return x


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


def load_gt_bands(meta_path: Path) -> list[Band2]:
    meta = json.loads(meta_path.read_text())
    rc = (
        meta.get("global", {})
        .get("annotations", {})
        .get("custom", {})
        .get("rc_configuration", {})
    )
    gt = []
    for r in (rc.get("rcs") or []):
        gt.append(
            Band2(
                lower_hz=float(r["min_frequency_mhz"]) * 1e6,
                upper_hz=float(r["max_frequency_mhz"]) * 1e6,
            )
        )
    return gt


def infer_occ_for_capture(
    *,
    model: TinyFeatOccNet,
    device: torch.device,
    meta_path: Path,
    data_path: Path,
    F: int,
    win_len: int,
    win_hop: int,
    nfft: int,
    fft_hop: int,
) -> tuple[np.ndarray, np.ndarray]:
    _, _, sr_hz, fc_hz, sc = load_sigmf(meta_path)
    x = read_ci16(data_path, sc)

    L = x.shape[1]
    win_len = min(win_len, L)
    hop = min(win_hop, win_len)

    probs = []
    for start in range(0, L - win_len + 1, hop):
        w = x[:, start : start + win_len]
        logp = window_logpsd(w, nfft=nfft, hop=fft_hop)
        if logp.shape[0] != F:
            x_old = np.linspace(0.0, 1.0, logp.shape[0], dtype=np.float32)
            x_new = np.linspace(0.0, 1.0, F, dtype=np.float32)
            logp = np.interp(x_new, x_old, logp).astype(np.float32)
        xt = torch.from_numpy(logp[None, :]).to(device)
        with (
            torch.no_grad(),
            torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type in ("mps", "cuda")),
        ):
            logits = model(xt)
            prob = torch.sigmoid(logits).float().cpu().numpy()[0]
        probs.append(prob)

    occ = np.mean(np.stack(probs, axis=0), axis=0)

    edges_bb = np.linspace(-sr_hz / 2.0, sr_hz / 2.0, F + 1)
    edges_abs = edges_bb + fc_hz
    return occ.astype(np.float32), edges_abs.astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--in-dir", required=True, help="folder with *.sigmf-meta/*.sigmf-data")
    ap.add_argument("--out", default="artifacts/report_split.json")
    ap.add_argument("--limit", type=int, default=30, help="max captures to process (multi-GT-band only)")

    ap.add_argument("--win-len", type=int, default=262_144)
    ap.add_argument("--win-hop", type=int, default=262_144)
    ap.add_argument("--nfft", type=int, default=2048)
    ap.add_argument("--fft-hop", type=int, default=1024)

    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min-bins", type=int, default=3)
    ap.add_argument("--merge-gap-bins", type=int, default=0)
    ap.add_argument("--smooth-radius", type=int, default=1)
    ap.add_argument("--hysteresis", type=float, default=0.05)
    ap.add_argument("--max-bands", type=int, default=8)

    ap.add_argument("--split-min-peak-height", type=float, default=0.8)
    ap.add_argument("--split-min-peak-sep-bins", type=int, default=32)
    ap.add_argument("--split-min-valley-drop", type=float, default=0.12)

    args = ap.parse_args()

    device = pick_device()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["model"]
    F = int(cfg["freq_bins"])
    hidden = int(cfg["hidden"])

    model = TinyFeatOccNet(freq_bins=F, hidden=hidden).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    in_dir = Path(args.in_dir)
    metas = sorted(in_dir.glob("*.sigmf-meta"))

    # Filter for multi-band GT
    cand = []
    for mp in metas:
        gt = load_gt_bands(mp)
        if len(gt) >= 2:
            cand.append((mp, gt))

    cand = cand[: max(0, int(args.limit))]

    rows = []
    for meta_path, gt in cand:
        data_path = meta_path.with_suffix(".sigmf-data")
        occ, edges_abs = infer_occ_for_capture(
            model=model,
            device=device,
            meta_path=meta_path,
            data_path=data_path,
            F=F,
            win_len=args.win_len,
            win_hop=args.win_hop,
            nfft=args.nfft,
            fft_hop=args.fft_hop,
        )

        bands0 = occ_to_bands(
            occ,
            edges_abs,
            thr=args.thr,
            min_bins=args.min_bins,
            merge_gap_bins=args.merge_gap_bins,
            smooth_radius=args.smooth_radius,
            hysteresis=args.hysteresis,
            split=False,
        )[: args.max_bands]

        bands1 = occ_to_bands(
            occ,
            edges_abs,
            thr=args.thr,
            min_bins=args.min_bins,
            merge_gap_bins=args.merge_gap_bins,
            smooth_radius=args.smooth_radius,
            hysteresis=args.hysteresis,
            split=True,
            split_min_peak_height=args.split_min_peak_height,
            split_min_peak_sep_bins=args.split_min_peak_sep_bins,
            split_min_valley_drop=args.split_min_valley_drop,
        )[: args.max_bands]

        pred0 = [Band2(b.lower_hz, b.upper_hz) for b in bands0]
        pred1 = [Band2(b.lower_hz, b.upper_hz) for b in bands1]

        cov0 = band_recall_coverage(gt, pred0)
        cov1 = band_recall_coverage(gt, pred1)

        # Edge error: mean over GT bands
        e0 = float(np.mean([best_match_edge_error_hz(g, pred0) for g in gt])) if gt else 0.0
        e1 = float(np.mean([best_match_edge_error_hz(g, pred1) for g in gt])) if gt else 0.0

        rows.append(
            {
                "stem": meta_path.name.removesuffix(".sigmf-meta"),
                "gt_n": len(gt),
                "pred_n_no_split": len(pred0),
                "pred_n_split": len(pred1),
                "cov_no_split": float(cov0),
                "cov_split": float(cov1),
                "edge_err_hz_no_split": float(e0),
                "edge_err_hz_split": float(e1),
            }
        )

    # Summary
    improved_n = sum(1 for r in rows if r["pred_n_split"] > r["pred_n_no_split"])
    improved_edge = sum(1 for r in rows if r["edge_err_hz_split"] < r["edge_err_hz_no_split"])

    summary = {
        "in_dir": str(in_dir),
        "ckpt": str(args.ckpt),
        "n_processed": len(rows),
        "n_multi_gt_total": len([1 for mp in metas if len(load_gt_bands(mp)) >= 2]),
        "improved_pred_n": int(improved_n),
        "improved_edge_err": int(improved_edge),
        "params": {
            "thr": args.thr,
            "hysteresis": args.hysteresis,
            "smooth_radius": args.smooth_radius,
            "merge_gap_bins": args.merge_gap_bins,
            "split_min_peak_height": args.split_min_peak_height,
            "split_min_peak_sep_bins": args.split_min_peak_sep_bins,
            "split_min_valley_drop": args.split_min_valley_drop,
        },
        "rows": rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print(f"processed={len(rows)} improved_pred_n={improved_n}/{len(rows)} improved_edge_err={improved_edge}/{len(rows)}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
