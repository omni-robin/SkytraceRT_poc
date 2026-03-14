import unittest

from skytracert_poc.metrics import (
    Band2,
    band_iou_1d,
    band_recall_coverage,
    best_match_edge_error_hz,
    best_match_overshoot_ratio,
)


class TestMetrics(unittest.TestCase):
    def test_band_iou_1d(self):
        a = Band2(0.0, 10.0)
        b = Band2(5.0, 15.0)
        # intersection=5, union=15
        self.assertAlmostEqual(band_iou_1d(a, b), 5.0 / 15.0)

    def test_band_recall_coverage_perfect(self):
        gt = [Band2(0.0, 10.0)]
        pred = [Band2(0.0, 10.0)]
        self.assertAlmostEqual(band_recall_coverage(gt, pred), 1.0)

    def test_best_match_overshoot_ratio(self):
        gt = Band2(100.0, 200.0)
        pred = [Band2(90.0, 210.0)]
        # pred_bw=120, gt_bw=100 => overshoot 20%.
        self.assertAlmostEqual(best_match_overshoot_ratio(gt, pred), 0.2)

    def test_best_match_edge_error_hz(self):
        gt = Band2(100.0, 200.0)
        pred = [Band2(90.0, 210.0)]
        # |90-100| + |210-200| = 20
        self.assertAlmostEqual(best_match_edge_error_hz(gt, pred), 20.0)

    def test_best_match_edge_error_selects_best_iou(self):
        gt = Band2(100.0, 200.0)
        # First pred has worse IoU but smaller edge error; second has better IoU.
        p1 = Band2(80.0, 190.0)   # edge error 30
        p2 = Band2(95.0, 205.0)   # edge error 10
        self.assertAlmostEqual(best_match_edge_error_hz(gt, [p1, p2]), 10.0)

    def test_best_match_edge_error_no_overlap_is_capped(self):
        gt = Band2(100.0, 200.0)
        pred = [Band2(300.0, 400.0)]
        # No overlap => return gt_bw (cap)
        self.assertAlmostEqual(best_match_edge_error_hz(gt, pred), 100.0)


if __name__ == "__main__":
    unittest.main()
