from __future__ import annotations

import unittest

import numpy as np

from monthly_validation_selection import validation_score


class ValidationScoreTest(unittest.TestCase):
    def test_equal_four_season_rmse_not_pooled_pixel_rmse(self):
        # Each month has one observation; winter months receive three samples,
        # so the season score must aggregate pixel SSE/count before RMSE.
        counts = np.ones(12, dtype=np.float64)
        sse = np.array([4, 9, 1, 4, 9, 1, 4, 9, 1, 4, 9, 1], dtype=np.float64)
        score, seasons = validation_score(sse, counts)
        expected = np.mean([
            np.sqrt((sse[11] + sse[0] + sse[1]) / 3),
            np.sqrt((sse[2] + sse[3] + sse[4]) / 3),
            np.sqrt((sse[5] + sse[6] + sse[7]) / 3),
            np.sqrt((sse[8] + sse[9] + sse[10]) / 3),
        ])
        self.assertAlmostEqual(score, expected)
        self.assertEqual(tuple(seasons), ("DJF", "MAM", "JJA", "SON"))

    def test_sparse_month_is_allowed_when_all_seasons_are_covered(self):
        counts = np.ones(12, dtype=np.float64)
        counts[0] = 0.0
        sse = np.ones(12, dtype=np.float64)
        sse[0] = 0.0
        score, _ = validation_score(sse, counts)
        self.assertTrue(np.isfinite(score))


if __name__ == "__main__":
    unittest.main()
