import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from cpp_quality_mask import cot_reference_mask


class CPPMaskTests(unittest.TestCase):
    def test_declared_range_preserves_thick_cloud(self):
        cot = np.array([0, 85, 99, 100, 101, -1, np.nan])
        mask, _ = cot_reference_mask(cot, np.isfinite(cot), np.ones(7), np.full(7, 30))
        np.testing.assert_array_equal(mask, [1, 1, 1, 1, 0, 0, 0])

    def test_geometry_and_missing_not_clear(self):
        cot = np.array([5., 5., 5., 5., np.nan])
        mask, _ = cot_reference_mask(cot, np.isfinite(cot), np.array([1, 1, 1, 0, 1]),
                                      np.array([79.9, 80., np.nan, 20., 20.]))
        np.testing.assert_array_equal(mask, [1, 0, 0, 0, 0])
        self.assertTrue(np.isnan(cot[-1]))

    def test_reject_different_mask_semantics(self):
        with self.assertRaises(ValueError):
            cot_reference_mask(np.ones(2), np.array([True, False]), np.ones(2), np.ones(2))

    def test_reject_shape_broadcast(self):
        with self.assertRaises(ValueError):
            cot_reference_mask(np.ones((2, 2)), np.ones(2, dtype=bool), np.ones(2), np.ones(2))


if __name__ == '__main__':
    unittest.main()
