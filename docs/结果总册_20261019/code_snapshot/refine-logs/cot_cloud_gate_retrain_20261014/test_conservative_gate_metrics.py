"""Small scientific-contract checks; no payloads, training, or network access."""
import unittest

import numpy as np

from conservative_gate_metrics import evaluate_threshold, select_threshold


def pair(p, raw, ref, clp):
    return (np.asarray([p, p], dtype=float), np.asarray([raw, raw], dtype=float),
            np.asarray([ref, ref], dtype=float), np.asarray([clp, clp], dtype=np.uint8),
            np.ones((2, len(p)), dtype=bool), np.asarray(["sili", "zhujia"]))


class ConservativeGateMetricsTest(unittest.TestCase):
    def test_actual_cot_cloud_recall_and_strict_equal_boundary(self):
        # Classifier says cloud for all four, but raw COT misses pixel 2.
        args = pair([0.7, 0.7, 0.9, 0.2], [40, 0, 40, 40], [40, 1, 40, 0], [1, 1, 2, 0])
        scores = evaluate_threshold(*args, threshold=0.7, split="validation")
        for st in ("sili", "zhujia"):
            arm = scores["by_station"][st]
            self.assertEqual(arm["gated"]["cloud"]["TP"], 2)
            self.assertAlmostEqual(arm["gated"]["cloud"]["POD"], 2 / 3)
            self.assertEqual(arm["gated"]["cloud"]["FP"], 0)
        chosen = select_threshold(*args)
        self.assertEqual(chosen["threshold"], 0.7)
        after = evaluate_threshold(*args, threshold=np.nextafter(0.7, 1.0), split="validation")
        self.assertFalse(after["guards_pass"])

    def test_maximum_feasible_event_matches_brute_force(self):
        rng = np.random.default_rng(42)
        for _ in range(16):
            p = rng.choice([0, 0.1, 0.2, 0.49, 0.5, 0.7, 0.9, 1], size=(2, 60))
            raw = rng.choice([0, 0.05, 0.1, 10, 30, 50], size=(2, 60))
            ref = rng.choice([0, 2, 30, 40], size=(2, 60))
            clp = np.where(ref > 0, 1, 0)
            args = (p, raw, ref, clp, np.ones_like(p, bool), ["sili", "zhujia"])
            selected = select_threshold(*args)
            candidates = np.unique(np.append(p.ravel(), 1.0))
            feasible = [t for t in candidates if evaluate_threshold(*args, threshold=t, split="train_oof")["guards_pass"]]
            self.assertEqual(selected["threshold"], max(feasible))
            self.assertEqual(selected["feasible_thresholds"], len(feasible))

    def test_probability_one_remains_and_is_largest_threshold(self):
        args = pair([0.2, 1.0, 1.0], [40, 40, 2], [0, 40, 2], [0, 1, 1])
        chosen = select_threshold(*args)
        self.assertEqual(chosen["threshold"], 1.0)
        self.assertTrue(chosen["selected_metrics"]["guards_pass"])

    def test_missing_station_thick_support_fails_closed(self):
        args = list(pair([0.2, 0.9], [40, 40], [0, 40], [0, 1]))
        args[2][1] = [0, 2]
        chosen = select_threshold(*args)
        self.assertEqual(chosen["state"], "FAIL_CLOSED")
        self.assertIsNone(chosen["threshold"])
        self.assertTrue(chosen["reasons"])

    def test_validation_cannot_select_and_invalid_mask_does_not_become_clear(self):
        args = list(pair([0.2, 0.9], [40, 40], [0, 40], [0, 1]))
        with self.assertRaisesRegex(ValueError, "train_oof"):
            select_threshold(*args, split="validation")
        args[0][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            select_threshold(*args)
        args[4][0, 0] = False
        scores = evaluate_threshold(*args, threshold=0.0, split="validation")
        self.assertEqual(scores["by_station"]["sili"]["valid_pixels"], 1)


if __name__ == "__main__":
    unittest.main()
