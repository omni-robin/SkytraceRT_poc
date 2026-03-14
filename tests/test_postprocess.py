import unittest

import numpy as np

from skytracert_poc.postprocess import occ_to_bands


class TestPostprocess(unittest.TestCase):
    def test_occ_to_bands_splitting_two_peaks(self):
        # Synthetic occupancy with two peaks inside one wide above-thr_low region.
        F = 64
        x = np.zeros((F,), dtype=np.float32)

        # Make a wide plateau above thr_low
        x[10:54] = 0.55
        # Add two peaks
        x[18] = 0.95
        x[42] = 0.92
        # Create a valley between peaks (still above thr_low, but drops enough from peaks)
        x[30] = 0.60
        x[31] = 0.58
        x[32] = 0.57

        edges = np.linspace(0.0, 64.0, F + 1, dtype=np.float64)  # 1 Hz per bin for simplicity

        bands_nosplit = occ_to_bands(
            x,
            edges,
            thr=0.7,
            hysteresis=0.2,
            smooth_radius=0,
            min_bins=3,
            split=False,
        )
        self.assertEqual(len(bands_nosplit), 1)

        bands_split = occ_to_bands(
            x,
            edges,
            thr=0.7,
            hysteresis=0.2,
            smooth_radius=0,
            min_bins=3,
            split=True,
            split_min_peak_height=0.8,
            split_min_peak_sep_bins=8,
            split_min_valley_drop=0.25,
        )
        self.assertEqual(len(bands_split), 2)

        # Ensure they are separated in frequency.
        b0, b1 = sorted(bands_split, key=lambda b: b.center_hz)
        self.assertLess(b0.upper_hz, b1.lower_hz)


if __name__ == "__main__":
    unittest.main()
