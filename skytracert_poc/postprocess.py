from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Band:
    lower_hz: float
    upper_hz: float
    center_hz: float
    score: float


def occ_to_bands(
    occ_prob: np.ndarray,
    freq_edges_abs_hz: np.ndarray,
    thr: float = 0.5,
    min_bins: int = 3,
    merge_gap_bins: int = 1,
) -> list[Band]:
    """Convert occupancy probabilities into frequency bands.

    - occ_prob: shape [F]
    - freq_edges_abs_hz: shape [F+1] (absolute Hz)

    Returns a list of contiguous regions above threshold, optionally merged.
    """
    assert occ_prob.ndim == 1
    F = occ_prob.shape[0]
    assert freq_edges_abs_hz.shape[0] == F + 1

    mask = occ_prob >= thr

    # find segments
    segs: list[tuple[int, int]] = []  # [lo, hi) in bins
    i = 0
    while i < F:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < F and mask[j]:
            j += 1
        if (j - i) >= min_bins:
            segs.append((i, j))
        i = j

    # merge small gaps
    if not segs:
        return []

    merged: list[tuple[int, int]] = [segs[0]]
    for lo, hi in segs[1:]:
        plo, phi = merged[-1]
        if lo - phi <= merge_gap_bins:
            merged[-1] = (plo, hi)
        else:
            merged.append((lo, hi))

    out: list[Band] = []
    for lo, hi in merged:
        lower = float(freq_edges_abs_hz[lo])
        upper = float(freq_edges_abs_hz[hi])
        center = 0.5 * (lower + upper)
        score = float(occ_prob[lo:hi].mean())
        out.append(Band(lower_hz=lower, upper_hz=upper, center_hz=center, score=score))

    # sort by score desc
    out.sort(key=lambda b: b.score, reverse=True)
    return out
