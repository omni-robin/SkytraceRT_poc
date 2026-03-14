from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Band2:
    lower_hz: float
    upper_hz: float

    @property
    def bw_hz(self) -> float:
        return max(0.0, self.upper_hz - self.lower_hz)


def band_iou_1d(a: Band2, b: Band2) -> float:
    inter_lo = max(a.lower_hz, b.lower_hz)
    inter_hi = min(a.upper_hz, b.upper_hz)
    inter = max(0.0, inter_hi - inter_lo)
    union = max(1e-12, a.bw_hz + b.bw_hz - inter)
    return inter / union


def band_recall_coverage(gt: list[Band2], pred: list[Band2]) -> float:
    """Fraction of GT bandwidth covered by union of predicted bands."""
    if not gt:
        return 1.0
    total = sum(b.bw_hz for b in gt)
    if total <= 0:
        return 0.0

    covered = 0.0
    for g in gt:
        for p in pred:
            inter_lo = max(g.lower_hz, p.lower_hz)
            inter_hi = min(g.upper_hz, p.upper_hz)
            covered += max(0.0, inter_hi - inter_lo)
    return min(1.0, covered / total)


def best_match_overshoot_ratio(gt_band: Band2, pred_bands: list[Band2]) -> float:
    """How much wider than GT the best-matching predicted band is.

    We pick the prediction with maximum IoU to the GT band.

    Returns:
      max(0, pred_bw - gt_bw) / gt_bw

    If no predictions exist, returns 1.0.
    """
    if gt_band.bw_hz <= 0:
        return 0.0
    if not pred_bands:
        return 1.0

    best = max(pred_bands, key=lambda p: band_iou_1d(gt_band, p))
    return max(0.0, best.bw_hz - gt_band.bw_hz) / gt_band.bw_hz


def best_match_edge_error_hz(gt_band: Band2, pred_bands: list[Band2]) -> float:
    """Absolute edge error (Hz) for the best-matching predicted band.

    We pick the prediction with maximum IoU to the GT band, then compute:
      |pred.lower - gt.lower| + |pred.upper - gt.upper|

    If no predictions exist, returns the GT bandwidth (a rough, non-zero penalty).
    """
    if gt_band.bw_hz <= 0:
        return 0.0
    if not pred_bands:
        return gt_band.bw_hz

    best = max(pred_bands, key=lambda p: band_iou_1d(gt_band, p))
    best_iou = band_iou_1d(gt_band, best)
    if best_iou <= 0.0:
        # No overlap at all: treat as a miss, but don't let the metric explode.
        return gt_band.bw_hz
    return abs(best.lower_hz - gt_band.lower_hz) + abs(best.upper_hz - gt_band.upper_hz)
