#!/usr/bin/env python3
"""Check R_U log1p(COT) teacher units at the P-model boundary."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from train_future_cot_u import DirectCotForecaster, UFrameSequenceDataset, predict_target_layout, safe_huber


def sha(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h


class TeacherUnitTest(unittest.TestCase):
    def test_log1p_teacher_is_used_once_and_targets_are_physical_cot(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            n = 24
            arrays = {
                "sequence_indices.npy": np.arange(n, dtype=np.int32)[None, :],
                "agri.npy": np.zeros((n, 2, 13, 16, 16), np.float32),
                "qa.npy": np.ones((n, 2, 13, 16, 16), np.uint8),
                "geometry.npy": np.zeros((n, 2, 3, 16, 16), np.float32),
                "cot_teacher.npy": np.zeros((n, 2, 1, 16, 16), np.float32),
                "cot_valid.npy": np.ones((n, 2, 1, 16, 16), np.uint8),
            }
            arrays["cot_teacher.npy"][:8] = np.log1p(1.0)
            arrays["cot_teacher.npy"][8:] = np.log1p(9.0)
            for name, arr in arrays.items():
                np.save(d / name, arr, allow_pickle=False)
            (d / "sequence_ids.txt").write_text("synthetic\n", encoding="utf-8")
            meta = {
                "state": "COMPLETE_U_ONLY_RU_TEACHER_CACHE",
                "teacher_value_space": "log1p_cot",
                "history_cot_log1p_mean": 999.0,  # legacy double-log statistic must be ignored
                "history_cot_log1p_std": 999.0,
                "agri_mean": [0.0] * 13,
                "agri_std": [1.0] * 13,
                "artifacts_sha256": {
                    name: sha(d / name)
                    for name in (*arrays.keys(), "sequence_ids.txt")
                },
            }
            (d / "cache_manifest.json").write_text(json.dumps(meta), encoding="utf-8")
            ds = UFrameSequenceDataset(d, "XC")
            x, _, y, mask = ds[0]
            self.assertTrue(np.isfinite(x).all())
            self.assertTrue(mask.all())
            self.assertAlmostEqual(float(y.mean()), 9.0, places=5)
            # History log1p(COT)=log(2), not log1p(log(2)).
            self.assertAlmostEqual(float(ds.cot_mean), float((8 * np.log1p(1.0) + 16 * np.log1p(9.0)) / 24), places=5)
            self.assertLess(float(x[0, 26].mean()), 0.0)

            # The training profile samples dataset rows before the first update.
            # Dataset items are NumPy arrays, so exercise the exact stack/convert
            # path that previously failed before P-C/P-XC training could start.
            # One synthetic sequence yields two station windows (indices 0,1).
            sample_ids = [0, 1, 0]
            profile_batch = [torch.from_numpy(np.stack([ds[i][j] for i in sample_ids]))
                             for j in range(4)]
            self.assertEqual(len(profile_batch), 4)
            self.assertTrue(all(torch.is_tensor(v) for v in profile_batch))
            self.assertEqual(tuple(profile_batch[0].shape), (3, 8, 31, 16, 16))
            self.assertEqual(tuple(profile_batch[1].shape), (3, 16, 3, 16, 16))
            self.assertEqual(tuple(profile_batch[2].shape), (3, 16, 1, 16, 16))
            self.assertEqual(tuple(profile_batch[3].shape), (3, 16, 1, 16, 16))

    def test_forecast_and_target_layout_match_without_broadcasting(self):
        class FakeChannelsFirstModel:
            def __call__(self, x, g):
                return torch.zeros((x.shape[0], 1, g.shape[1], 16, 16))

        x = torch.zeros((2, 8, 31, 16, 16))
        g = torch.zeros((2, 16, 3, 16, 16))
        target = torch.zeros((2, 16, 1, 16, 16))
        mask = torch.ones_like(target, dtype=torch.bool)
        pred = predict_target_layout(FakeChannelsFirstModel(), x, g)
        self.assertEqual(pred.shape, target.shape)
        self.assertTrue(torch.isfinite(safe_huber(pred, target, mask)))
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            safe_huber(torch.zeros((2, 1, 16, 16, 16)), target, mask)

    def test_real_forecaster_emits_single_cot_channel_in_target_layout(self):
        model = DirectCotForecaster().eval()
        x = torch.zeros((1, 8, 31, 16, 16))
        g = torch.zeros((1, 16, 3, 16, 16))
        with torch.inference_mode():
            pred = predict_target_layout(model, x, g)
        self.assertEqual(tuple(pred.shape), (1, 16, 1, 16, 16))


if __name__ == "__main__":
    unittest.main()
