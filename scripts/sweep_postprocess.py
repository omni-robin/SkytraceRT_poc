#!/usr/bin/env python3
"""Sweep post-processing hyperparams to optimize band edges.

Goal (as requested): *tight* band edges.
Objective: configurable.

Modes:
- `iou` (default): maximize mean best-match IoU subject to mean GT coverage >= cov_min.
- `tight`: minimize mean edge error (Hz) subject to mean GT coverage >= cov_min.
  (Tie-breakers: higher IoU, higher coverage.)

This runs fully in-process (no subprocess spam) and reuses the loaded model.
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

from skytracert_poc.metrics import (
    Band2,
    band_iou_1d,
    band_recall_coverage,
    best_match_edge_error_hz,
    best_match_overshoot_ratio,
)
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


def score_params(
    metas: list[Path],
    model: TinyFeatOccNet,
    device: torch.device,
    thr: float,
    hysteresis: float,
    smooth_radius: int,
    nfft: int,
    fft_hop: int,
    win_len: int,
    win_hop: int,
    max_bands: int,
    cov_min: float,
    *,
    split: bool,
    split_min_peak_height: float | None,
    split_min_peak_sep_bins: int,
    split_min_valley_drop: float,
) -> dict:
    coverages: list[float] = []
    ious: list[float] = []
    overs: list[float] = []
    edge_errs: list[float] = []
    pred_ns: list[int] = []

    for meta_path in metas:
        stem = meta_path.name.removesuffix(".sigmf-meta")
        data_path = meta_path.with_suffix("").with_suffix(".sigmf-data")
        if not data_path.exists():
            continue

        j, cap0, rc = load_sigmf(meta_path)
        sr_hz = float((j.get("global") or {}).get("core:sample_rate"))
        fc_hz = float(cap0.get("core:frequency"))
        sc = int(cap0.get("core:sample_count"))

        x = read_ci16(data_path, sc)
        occ = infer_capture_occ(model, device, x, win_len, win_hop, nfft, fft_hop, freq_bins=model.net[0].normalized_shape[0])

        F = occ.shape[0]
        edges_bb = np.linspace(-sr_hz / 2.0, sr_hz / 2.0, F + 1)
        edges_abs = edges_bb + fc_hz

        pred_bands = occ_to_bands(
            occ,
            edges_abs,
            thr=thr,
            min_bins=3,
            merge_gap_bins=0,
            smooth_radius=smooth_radius,
            hysteresis=hysteresis,
            split=split,
            split_min_peak_height=split_min_peak_height,
            split_min_peak_sep_bins=split_min_peak_sep_bins,
            split_min_valley_drop=split_min_valley_drop,
        )
        pred_bands = pred_bands[:max_bands]

        gt_bands = [
            Band2(lower_hz=float(r["min_frequency_mhz"]) * 1e6, upper_hz=float(r["max_frequency_mhz"]) * 1e6)
            for r in (rc.get("rcs") or [])
        ]
        pred_simple = [Band2(lower_hz=b.lower_hz, upper_hz=b.upper_hz) for b in pred_bands]

        best_ious = []
        best_overs = []
        best_edge_errs = []
        for g in gt_bands:
            best_ious.append(max((band_iou_1d(g, p) for p in pred_simple), default=0.0))
            best_overs.append(best_match_overshoot_ratio(g, pred_simple))
            best_edge_errs.append(best_match_edge_error_hz(g, pred_simple))

        coverages.append(band_recall_coverage(gt_bands, pred_simple))
        ious.append(float(np.mean(best_ious)) if best_ious else 1.0)
        overs.append(float(np.mean(best_overs)) if best_overs else 0.0)
        edge_errs.append(float(np.mean(best_edge_errs)) if best_edge_errs else 0.0)
        pred_ns.append(len(pred_simple))

    cov = float(np.mean(coverages)) if coverages else 0.0
    iou = float(np.mean(ious)) if ious else 0.0
    over = float(np.mean(overs)) if overs else 0.0
    edge_err = float(np.mean(edge_errs)) if edge_errs else 0.0
    pred_n = float(np.mean(pred_ns)) if pred_ns else 0.0

    return {
        "thr": thr,
        "hysteresis": hysteresis,
        "smooth_radius": smooth_radius,
        "split": bool(split),
        "split_min_peak_height": split_min_peak_height,
        "split_min_peak_sep_bins": int(split_min_peak_sep_bins),
        "split_min_valley_drop": float(split_min_valley_drop),
        "mean_gt_coverage": cov,
        "mean_best_match_iou": iou,
        "mean_best_match_overshoot": over,
        "mean_best_match_edge_error_hz": edge_err,
        "mean_pred_n": pred_n,
        "ok": cov >= cov_min,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", default="artifacts/sweep_postprocess.json")
    ap.add_argument("--cov-min", type=float, default=0.98)
    ap.add_argument("--mode", choices=["iou", "tight"], default="iou")
    ap.add_argument("--win-len", type=int, default=262_144)
    ap.add_argument("--win-hop", type=int, default=262_144)
    ap.add_argument("--nfft", type=int, default=2048)
    ap.add_argument("--fft-hop", type=int, default=1024)
    ap.add_argument("--max-bands", type=int, default=8)
    ap.add_argument("--split", action="store_true", help="enable peak/valley splitting inside wide regions")
    ap.add_argument("--split-min-peak-height", type=float, default=None)
    ap.add_argument("--split-min-peak-sep-bins", type=int, default=24)
    ap.add_argument("--split-min-valley-drop", type=float, default=0.12)
    ap.add_argument(
        "--split-peak-heights",
        default=None,
        help="comma-separated grid for split_min_peak_height (e.g. 0.50,0.55,0.60). Defaults to just --split-min-peak-height.",
    )
    ap.add_argument(
        "--split-peak-seps",
        default=None,
        help="comma-separated grid for split_min_peak_sep_bins (e.g. 12,16,24,32). Defaults to just --split-min-peak-sep-bins.",
    )
    ap.add_argument(
        "--split-valley-drops",
        default=None,
        help="comma-separated grid for split_min_valley_drop (e.g. 0.01,0.02,0.03). Defaults to just --split-min-valley-drop.",
    )
    ap.add_argument("--limit", type=int, default=0, help="if >0, only score the first N captures")
    args = ap.parse_args()

    device = pick_device()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["model"]
    F = int(cfg["freq_bins"])
    hidden = int(cfg["hidden"])

    model = TinyFeatOccNet(freq_bins=F, hidden=hidden).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    metas = sorted(Path(args.in_dir).glob("*.sigmf-meta"))
    if args.limit and args.limit > 0:
        metas = metas[: int(args.limit)]

    thr_vals = [0.50, 0.55, 0.60, 0.65]
    hyst_vals = [0.0, 0.05, 0.10]
    smooth_vals = [0, 1, 2]

    def _parse_csv_floats(s: str | None) -> list[float] | None:
        if not s:
            return None
        return [float(x.strip()) for x in s.split(",") if x.strip()]

    def _parse_csv_ints(s: str | None) -> list[int] | None:
        if not s:
            return None
        return [int(x.strip()) for x in s.split(",") if x.strip()]

    peak_heights = _parse_csv_floats(args.split_peak_heights) if args.split else None
    peak_seps = _parse_csv_ints(args.split_peak_seps) if args.split else None
    valley_drops = _parse_csv_floats(args.split_valley_drops) if args.split else None

    if args.split:
        if peak_heights is None:
            peak_heights = [args.split_min_peak_height]
        if peak_seps is None:
            peak_seps = [int(args.split_min_peak_sep_bins)]
        if valley_drops is None:
            valley_drops = [float(args.split_min_valley_drop)]
    else:
        peak_heights = [None]
        peak_seps = [int(args.split_min_peak_sep_bins)]
        valley_drops = [float(args.split_min_valley_drop)]

    results = []
    best = None

    for thr in thr_vals:
        for hys in hyst_vals:
            for sm in smooth_vals:
                for ph in peak_heights:
                    for ps in peak_seps:
                        for vd in valley_drops:
                            r = score_params(
                                metas,
                                model,
                                device,
                                thr=thr,
                                hysteresis=hys,
                                smooth_radius=sm,
                                nfft=args.nfft,
                                fft_hop=args.fft_hop,
                                win_len=args.win_len,
                                win_hop=args.win_hop,
                                max_bands=args.max_bands,
                                cov_min=args.cov_min,
                                split=bool(args.split),
                                split_min_peak_height=ph,
                                split_min_peak_sep_bins=int(ps),
                                split_min_valley_drop=float(vd),
                            )
                            results.append(r)
                            if r["ok"]:
                                if args.mode == "iou":
                                    key = (r["mean_best_match_iou"], r["mean_gt_coverage"])
                                    best_key = (
                                        best["mean_best_match_iou"],
                                        best["mean_gt_coverage"],
                                    ) if best else None
                                    if best is None or key > best_key:
                                        best = r
                                else:  # tight
                                    # Minimize edge error, then maximize IoU/coverage as tie-break.
                                    key = (
                                        r["mean_best_match_edge_error_hz"],
                                        -r["mean_best_match_iou"],
                                        -r["mean_gt_coverage"],
                                    )
                                    best_key = (
                                        best["mean_best_match_edge_error_hz"],
                                        -best["mean_best_match_iou"],
                                        -best["mean_gt_coverage"],
                                    ) if best else None
                                    if best is None or key < best_key:
                                        best = r
                            print(
                                f"thr={thr:.2f} hys={hys:.2f} sm={sm} split={int(bool(args.split))} ph={ph} ps={ps} vd={vd} cov={r['mean_gt_coverage']:.4f} iou={r['mean_best_match_iou']:.4f} edge={r['mean_best_match_edge_error_hz']:.1f}Hz pred_n={r['mean_pred_n']:.2f}",
                                flush=True,
                            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"best": best, "results": results}, indent=2))
    print("best:", best)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
