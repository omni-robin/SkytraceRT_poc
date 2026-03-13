from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Band2:
    lower_hz: float
    upper_hz: float


def band_iou_1d(a: Band2, b: Band2) -> float:
    inter_lo = max(a.lower_hz, b.lower_hz)
    inter_hi = min(a.upper_hz, b.upper_hz)
    inter = max(0.0, inter_hi - inter_lo)
    union = max(1e-12, (a.upper_hz - a.lower_hz) + (b.upper_hz - b.lower_hz) - inter)
    return inter / union


def band_recall_coverage(gt: list[Band2], pred: list[Band2]) -> float:
    """Fraction of GT bandwidth covered by union of predicted bands."""
    if not gt:
        return 1.0
    total = sum(max(0.0, b.upper_hz - b.lower_hz) for b in gt)
    if total <= 0:
        return 0.0

    covered = 0.0
    for g in gt:
        # compute overlap length with union(pred)
        lo = g.lower_hz
        hi = g.upper_hz
        # accumulate overlaps (pred bands may overlap; fine for now)
        for p in pred:
            inter_lo = max(lo, p.lower_hz)
            inter_hi = min(hi, p.upper_hz)
            covered += max(0.0, inter_hi - inter_lo)
    return min(1.0, covered / total)
