import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('splitter', Path(__file__).parents[1]/'scripts/repartition_cot_manifest.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BoundaryTests(unittest.TestCase):
    def test_bjt_boundary(self):
        self.assertEqual(module.paper_split('2025-06-30T15:59:00Z')[0], 'train')
        self.assertEqual(module.paper_split('2025-06-30T16:00:00Z')[0], 'validation')
        self.assertEqual(module.paper_split('2025-09-30T16:00:00Z')[0], 'test')

    def test_outside(self):
        self.assertIsNone(module.paper_split('2024-04-01T00:00:00Z')[0])
        self.assertIsNone(module.paper_split('2026-06-30T16:00:00Z')[0])

    def test_no_naive_timestamp(self):
        with self.assertRaises(ValueError):
            module.paper_split('2025-07-01T00:00:00')


if __name__ == '__main__':
    unittest.main()
