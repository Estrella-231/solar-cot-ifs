import unittest

import numpy as np

from train_hubei_cloud_classifier_oof import align_clp_sidecar_to_rows


class KeyAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"time_bjt": "202508220615", "split": "val"},
            {"time_bjt": "202508250615", "split": "train"},
            {"time_bjt": "202508220630", "split": "val"},
        ]
        self.keys = np.asarray([r["time_bjt"] for r in self.rows])[::-1]
        self.splits = np.asarray([r["split"] for r in self.rows])[::-1]
        self.clp = np.arange(3, dtype=np.uint8)[:, None, None] * np.ones((3, 2, 2), dtype=np.uint8)
        self.valid = np.ones((3, 2, 2), dtype=bool)

    def test_reorders_by_unique_bjt_key_and_checks_split(self):
        keys, splits, clp, valid, reordered = align_clp_sidecar_to_rows(
            self.rows, self.keys, self.splits, self.clp, self.valid)
        self.assertEqual(keys.tolist(), [r["time_bjt"] for r in self.rows])
        self.assertEqual(splits.tolist(), [r["split"] for r in self.rows])
        self.assertEqual(clp[:, 0, 0].tolist(), [2, 1, 0])
        self.assertTrue(valid.all())
        self.assertEqual(reordered, 2)

    def test_rejects_keyed_split_mismatch(self):
        wrong = self.splits.copy()
        wrong[0] = "train"
        with self.assertRaisesRegex(ValueError, "keyed split mismatch"):
            align_clp_sidecar_to_rows(self.rows, self.keys, wrong, self.clp, self.valid)

    def test_rejects_duplicate_key(self):
        duplicate = self.keys.copy()
        duplicate[1] = duplicate[0]
        with self.assertRaisesRegex(ValueError, "duplicate BJT keys in CPP/CLP sidecar"):
            align_clp_sidecar_to_rows(self.rows, duplicate, self.splits, self.clp, self.valid)


if __name__ == "__main__":
    unittest.main()
