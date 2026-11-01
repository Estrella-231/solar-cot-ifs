from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from merge_future_ghi_candidate_oof_validation_bank import merge, sha


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def save_index(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=("seq_id", "split"), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def save_array(path: Path, value: np.ndarray) -> str:
    np.save(path, value, allow_pickle=False)
    return sha(path)


class CandidateMergeTest(unittest.TestCase):
    def test_oof_train_and_candidate_validation_rows_both_replace_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            manifests = root / "manifests"
            val = root / "candidate_val"
            out = root / "merged"
            base.mkdir(); manifests.mkdir(); val.mkdir()
            for fold in range(5):
                (manifests / f"fold_{fold}").mkdir()
                (manifests / f"fold_{fold}" / "fold_oof_ids.txt").write_text(
                    f"train-{fold}\n", encoding="utf-8")
            rows = [{"seq_id": f"train-{i}", "split": "train"} for i in range(5)]
            rows.append({"seq_id": "val-0", "split": "val"})
            save_index(base / "index.csv", rows)
            layouts = {
                "history_agri_raw.npy": [6, 2, 2, 13, 3, 3],
                "history_geometry.npy": [6, 2, 2, 2, 3, 3],
                "history_cot_log1p.npy": [6, 2, 2, 1, 3, 3],
                "forecast_agri_raw.npy": [6, 2, 16, 13, 3, 3],
                "forecast_geometry.npy": [6, 2, 16, 2, 3, 3],
                "forecast_cot_log1p.npy": [6, 2, 16, 1, 3, 3],
            }
            for name, shape in layouts.items():
                np.save(base / name, np.zeros(shape, np.float32), allow_pickle=False)
            save_json(base / "bank_manifest.json", {
                "state": "COMPLETE_FORECAST_AGRI_R_TRAIN_VAL_BANK",
                "layouts": layouts,
                "test_payload_opened": False, "buffer_payload_opened": False,
                "future_observed_agri_read": False, "cpp_read": False,
            })

            fold_paths = []
            for fold in range(5):
                fb = root / f"fold-{fold}"; fb.mkdir(); fold_paths.append(fb)
                save_index(fb / "index.csv", [{"seq_id": f"train-{fold}", "split": "train"}])
                value = float(fold + 1)
                hashes = {
                    "forecast_agri_raw.npy": save_array(
                        fb / "forecast_agri_raw.npy",
                        np.full((1, 2, 16, 13, 3, 3), value, np.float32)),
                    "forecast_cot_log1p.npy": save_array(
                        fb / "forecast_cot_log1p.npy",
                        np.full((1, 2, 16, 1, 3, 3), value, np.float32)),
                }
                save_json(fb / "bank_manifest.json", {
                    "state": "COMPLETE_FUTURE_GHI_OOF_FOLD_FORECASTS", "fold": fold,
                    "fold_checkpoint_sha256": f"fold-sha-{fold}", "array_sha256": hashes,
                    "index_sha256": sha(fb / "index.csv"),
                    "future_observed_agri_read": False, "cpp_read": False,
                    "validation_test_payload_opened": False, "buffer_payload_opened": False,
                })

            save_index(val / "index.csv", [{"seq_id": "val-0", "split": "val"}])
            val_hashes = {
                "forecast_agri_raw.npy": save_array(
                    val / "forecast_agri_raw.npy",
                    np.full((1, 2, 16, 13, 3, 3), 9, np.float32)),
                "forecast_cot_log1p.npy": save_array(
                    val / "forecast_cot_log1p.npy",
                    np.full((1, 2, 16, 1, 3, 3), 9, np.float32)),
            }
            save_json(val / "bank_manifest.json", {
                "state": "COMPLETE_DROPPATH0_FULLTRAIN_VALIDATION_FORECASTS",
                "test_payload_opened": False, "buffer_payload_opened": False,
                "future_observed_agri_read": False, "cpp_read": False,
                "validation_payload_used_as_input": False,
                "simvp_checkpoint_sha256": "candidate-checkpoint",
                "selected_checkpoint_sha256": "candidate-checkpoint",
                "array_sha256": val_hashes,
                "index_sha256": sha(val / "index.csv"),
            })

            result = merge(base, manifests, fold_paths, val, out)
            self.assertEqual(result["state"], "PASS_CANDIDATE_OOF_VALIDATION_BANK_MERGE")
            merged_agri = np.load(out / "forecast_agri_raw.npy", allow_pickle=False)
            merged_cot = np.load(out / "forecast_cot_log1p.npy", allow_pickle=False)
            for fold in range(5):
                self.assertTrue(np.all(merged_agri[fold] == fold + 1))
                self.assertTrue(np.all(merged_cot[fold] == fold + 1))
            self.assertTrue(np.all(merged_agri[5] == 9))
            self.assertTrue(np.all(merged_cot[5] == 9))
            history = np.load(out / "history_agri_raw.npy", allow_pickle=False)
            self.assertTrue(np.all(history == 0))
            meta = json.loads((out / "bank_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["oof_fold_count"], 5)
            self.assertFalse(meta["future_observed_agri_read"])
            self.assertFalse(meta["test_payload_opened"])


if __name__ == "__main__":
    unittest.main()
