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


def _pick_peaks_1d(x: np.ndarray, *, min_height: float, min_sep_bins: int) -> list[int]:
    """Very small peak picker for 1D arrays.

    Returns peak indices i where x[i] is a local maximum and x[i] >= min_height.
    Applies a simple non-maximum suppression with min_sep_bins.
    """
    n = int(x.shape[0])
    if n < 3:
        return []

    cand: list[int] = []
    for i in range(1, n - 1):
        if x[i] >= min_height and x[i] >= x[i - 1] and x[i] >= x[i + 1]:
            cand.append(i)

    # NMS by height
    cand.sort(key=lambda i: float(x[i]), reverse=True)
    picked: list[int] = []
    for i in cand:
        if all(abs(i - j) >= min_sep_bins for j in picked):
            picked.append(i)
    picked.sort()
    return picked


def _split_segment_on_valleys(
    occ: np.ndarray,
    lo: int,
    hi: int,
    *,
    min_peak_height: float,
    min_peak_sep_bins: int,
    min_valley_drop: float,
) -> list[tuple[int, int]]:
    """Split a [lo,hi) segment if it contains multiple strong peaks separated by valleys."""
    seg = occ[lo:hi]
    peaks = _pick_peaks_1d(seg, min_height=min_peak_height, min_sep_bins=min_peak_sep_bins)
    if len(peaks) <= 1:
        return [(lo, hi)]

    # Convert to absolute indices
    peaks = [lo + p for p in peaks]

    cuts: list[int] = []
    for p0, p1 in zip(peaks[:-1], peaks[1:]):
        if p1 - p0 < 2:
            continue
        valley_rel = int(np.argmin(occ[p0:p1 + 1]))
        v = p0 + valley_rel
        drop = min(float(occ[p0]), float(occ[p1])) - float(occ[v])
        if drop >= min_valley_drop:
            cuts.append(v)

    if not cuts:
        return [(lo, hi)]

    # Build subsegments: cut at valley positions.
    segs: list[tuple[int, int]] = []
    start = lo
    for v in cuts:
        end = max(start, v)  # valley bin belongs to left side; right starts at v+1
        if end > start:
            segs.append((start, end))
        start = min(hi, v + 1)
    if start < hi:
        segs.append((start, hi))

    # Merge degenerate empties if any
    segs = [(a, b) for a, b in segs if b > a]
    return segs if segs else [(lo, hi)]


def occ_to_bands(
    occ_prob: np.ndarray,
    freq_edges_abs_hz: np.ndarray,
    thr: float = 0.5,
    min_bins: int = 3,
    merge_gap_bins: int = 1,
    smooth_radius: int = 2,
    hysteresis: float = 0.15,
    *,
    split: bool = False,
    split_min_peak_height: float | None = None,
    split_min_peak_sep_bins: int = 24,
    split_min_valley_drop: float = 0.12,
) -> list[Band]:
    """Convert occupancy probabilities into frequency bands.

    Improvements vs the naive version:
    - optional smoothing (stabilizes edges)
    - hysteresis (reduces flicker / spurs): grow regions down to thr_low
    - linear interpolation at crossings for sub-bin edge refinement
    - optional peak/valley based splitting inside a wide region (reduces merged bands)

    - occ_prob: shape [F]
    - freq_edges_abs_hz: shape [F+1] (absolute Hz)

    Splitting heuristic notes:
    - We look for multiple local maxima above `split_min_peak_height` (default: thr_high)
    - If adjacent peaks have a valley between them that drops by >= `split_min_valley_drop`,
      we split at that valley.
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
        sub_segs = [(lo, hi)]
        if split:
            mph = float(thr_high if split_min_peak_height is None else split_min_peak_height)
            sub_segs = _split_segment_on_valleys(
                occ,
                lo,
                hi,
                min_peak_height=mph,
                min_peak_sep_bins=int(split_min_peak_sep_bins),
                min_valley_drop=float(split_min_valley_drop),
            )

        for slo, shi in sub_segs:
            if (shi - slo) < min_bins:
                continue
            # refine edges around thr_low crossing
            lower = _refine_edge(occ, freq_edges_abs_hz, slo, thr_low, side="left")
            upper = _refine_edge(occ, freq_edges_abs_hz, shi - 1, thr_low, side="right")
            if upper <= lower:
                continue

            center = 0.5 * (lower + upper)
            score = float(occ[slo:shi].mean())
            out.append(
                Band(
                    lower_hz=float(lower),
                    upper_hz=float(upper),
                    center_hz=float(center),
                    score=score,
                )
            )

    out.sort(key=lambda b: b.score, reverse=True)
    return out
