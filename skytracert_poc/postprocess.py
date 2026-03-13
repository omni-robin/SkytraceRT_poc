from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Band:
    lower_hz: float
    upper_hz: float
    center_hz: float
    score: float


def _smooth1d(x: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return x
    k = 2 * radius + 1
    w = np.ones((k,), dtype=np.float32) / float(k)
    return np.convolve(x.astype(np.float32), w, mode="same")


def _refine_edge(
    occ: np.ndarray,
    edges_abs_hz: np.ndarray,
    idx: int,
    thr: float,
    side: str,
) -> float:
    """Refine a threshold crossing edge by linear interpolation.

    idx is a bin index; edge is between bins.
    side:
      - 'left': crossing between idx-1 and idx (idx is first >=thr)
      - 'right': crossing between idx and idx+1 (idx is last >=thr)

    Returns absolute Hz.
    """
    F = occ.shape[0]
    if side == "left":
        i0 = max(0, idx - 1)
        i1 = idx
        if i1 <= 0:
            return float(edges_abs_hz[0])
        y0 = float(occ[i0])
        y1 = float(occ[i1])
        x0 = float(edges_abs_hz[i1])  # left edge of bin i1
        x1 = float(edges_abs_hz[i1 + 1])  # right edge of bin i1
        # interpolate within bin i1 using y0->y1, but map to the bin edge span
        t = 0.0 if y1 == y0 else (thr - y0) / (y1 - y0)
        t = float(np.clip(t, 0.0, 1.0))
        return x0 + t * (x1 - x0)

    if side == "right":
        i0 = idx
        i1 = min(F - 1, idx + 1)
        if i0 >= F - 1:
            return float(edges_abs_hz[-1])
        y0 = float(occ[i0])
        y1 = float(occ[i1])
        x0 = float(edges_abs_hz[i0])
        x1 = float(edges_abs_hz[i0 + 1])
        t = 0.0 if y1 == y0 else (thr - y0) / (y1 - y0)
        t = float(np.clip(t, 0.0, 1.0))
        return x0 + t * (x1 - x0)

    raise ValueError("side must be left|right")


def occ_to_bands(
    occ_prob: np.ndarray,
    freq_edges_abs_hz: np.ndarray,
    thr: float = 0.5,
    min_bins: int = 3,
    merge_gap_bins: int = 1,
    smooth_radius: int = 2,
    hysteresis: float = 0.15,
) -> list[Band]:
    """Convert occupancy probabilities into frequency bands.

    Improvements vs the naive version:
    - optional smoothing (stabilizes edges)
    - hysteresis (reduces flicker / spurs): grow regions down to thr_low
    - linear interpolation at crossings for sub-bin edge refinement

    - occ_prob: shape [F]
    - freq_edges_abs_hz: shape [F+1] (absolute Hz)
    """
    assert occ_prob.ndim == 1
    F = occ_prob.shape[0]
    assert freq_edges_abs_hz.shape[0] == F + 1

    occ = _smooth1d(occ_prob, smooth_radius)

    thr_high = float(thr)
    thr_low = float(max(0.0, thr_high - hysteresis))

    seeds = occ >= thr_high
    grow = occ >= thr_low

    # find seed segments, then grow them within grow-mask
    segs: list[tuple[int, int]] = []  # [lo, hi) in bins
    i = 0
    while i < F:
        if not seeds[i]:
            i += 1
            continue
        j = i + 1
        while j < F and seeds[j]:
            j += 1

        # grow left
        lo = i
        while lo > 0 and grow[lo - 1]:
            lo -= 1
        # grow right
        hi = j
        while hi < F and grow[hi]:
            hi += 1

        if (hi - lo) >= min_bins:
            segs.append((lo, hi))
        i = j

    if not segs:
        return []

    # merge small gaps
    merged: list[tuple[int, int]] = [segs[0]]
    for lo, hi in segs[1:]:
        plo, phi = merged[-1]
        if lo - phi <= merge_gap_bins:
            merged[-1] = (plo, hi)
        else:
            merged.append((lo, hi))

    out: list[Band] = []
    for lo, hi in merged:
        # refine edges around thr_low crossing
        lower = _refine_edge(occ, freq_edges_abs_hz, lo, thr_low, side="left")
        upper = _refine_edge(occ, freq_edges_abs_hz, hi - 1, thr_low, side="right")
        if upper <= lower:
            continue

        center = 0.5 * (lower + upper)
        score = float(occ[lo:hi].mean())
        out.append(Band(lower_hz=float(lower), upper_hz=float(upper), center_hz=float(center), score=score))

    out.sort(key=lambda b: b.score, reverse=True)
    return out
