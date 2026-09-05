import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from cpp_grid_mapping import map_cpp_grid


class GridTests(unittest.TestCase):
    def test_one_pixel_shift(self):
        lat = np.array([26.04, 26., 25.96, 25.92])
        lon = np.array([112.28, 112.32, 112.36, 112.40])
        y, x = np.meshgrid(lat[2:4], lon[2:4], indexing='ij')
        r, c, e = map_cpp_grid(lat, lon, y, x)
        np.testing.assert_array_equal(r, [2, 3])
        np.testing.assert_array_equal(c, [2, 3])
        self.assertEqual(e['latitude_max_error_deg'], 0)

    def test_reject_unmatched_and_curved_target(self):
        lat, lon = np.arange(4.), np.arange(4.)
        y, x = np.meshgrid(lat[1:3], lon[1:3], indexing='ij')
        with self.assertRaises(ValueError):
            map_cpp_grid(lat, lon, y+0.5, x)
        y[0, 1] += 0.01
        with self.assertRaises(ValueError):
            map_cpp_grid(lat, lon, y, x)


if __name__ == '__main__':
    unittest.main()
