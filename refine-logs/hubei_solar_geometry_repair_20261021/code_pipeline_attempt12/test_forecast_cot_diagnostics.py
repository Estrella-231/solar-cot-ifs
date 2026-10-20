from __future__ import annotations

import csv
import hashlib
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate_forecast_cot as evaluator
import audit_oof_bank_lineage as oof_audit
from hubei_forecast_pack_gate import validate_input_pack_contract


class ForecastCOTDiagnosticsTests(unittest.TestCase):
    def test_paired_date_bootstrap_math(self):
        stats = {
            "20250801": {"forecast_simvp_to_R": (4.0, 1), "history_COT_persistence": (0.0, 1)},
            "20250802": {"forecast_simvp_to_R": (16.0, 1), "history_COT_persistence": (0.0, 1)},
        }
        out = evaluator.bootstrap_delta(stats, "forecast_simvp_to_R", "history_COT_persistence", 500, 7)
        self.assertAlmostEqual(out["delta_rmse_left_minus_right"], np.sqrt(10.0))
        self.assertEqual(out["n_target_dates"], 2)
        self.assertEqual(out["state"], "PAIRED_DATE_BLOCK_BOOTSTRAP")

    def test_exact_key_join_metrics_and_duplicate_key_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cot_path, keys_path = root / "cot.npy", root / "s_keys.csv"
            r_dir, out = root / "r_val", root / "out"
            r_dir.mkdir()
            first_init = datetime(2025, 8, 1, 21, 45)
            first_targets = [(first_init + timedelta(minutes=15 * (i + 1))).strftime("%Y%m%d%H%M") for i in range(16)]
            s_rows = []
            for i in range(407):
                dt = first_init if i == 0 else datetime(2025, 9, 1) + timedelta(days=i)
                targets = first_targets if i == 0 else [(dt + timedelta(minutes=15 * (j + 1))).strftime("%Y%m%d%H%M") for j in range(16)]
                s_rows.append({"split": "val", "init_time_bjt": dt.strftime("%Y%m%d%H%M"), "target_times_bjt": "|".join(targets)})
            with keys_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(s_rows[0]))
                writer.writeheader(); writer.writerows(s_rows)
            cot = np.zeros((407, 16, 1, 16, 16), dtype=np.float32)
            cot[0] = 1.0
            np.save(cot_path, cot)
            r_times = [first_init.strftime("%Y%m%d%H%M")] + first_targets
            with (r_dir / "val_keys.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["time_bjt", "split"])
                writer.writeheader(); writer.writerows({"time_bjt": t, "split": "val"} for t in r_times)
            zero = np.zeros((len(r_times), 1, 16, 16), dtype=np.float32)
            mask = np.ones_like(zero, dtype=np.bool_)
            np.save(r_dir / "val_prediction_cot.npy", zero)
            np.save(r_dir / "val_reference_cpp_cot.npy", zero)
            np.save(r_dir / "val_valid_mask.npy", mask)
            result = evaluator.evaluate(cot_path, keys_path, r_dir, out, draws=200, seed=19)
            self.assertEqual(result["paired_keys"], 16)
            import json
            report = json.loads((out / "forecast_cot_metrics.json").read_text(encoding="utf-8"))
            entry = report["metrics"]["15|all_patch|ALL_PATCH"]
            self.assertEqual(entry["forecast"]["n_pixels"], 256)
            self.assertAlmostEqual(entry["forecast"]["rmse_cot"], 1.0)
            self.assertEqual(entry["history_persistence"]["rmse_cot"], 0.0)

            duplicate_dir = root / "duplicate_r"
            duplicate_dir.mkdir()
            with (duplicate_dir / "val_keys.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["time_bjt", "split"])
                writer.writeheader(); writer.writerows([{"time_bjt": r_times[0], "split": "val"}] * 2)
            for name in ("val_prediction_cot.npy", "val_reference_cpp_cot.npy", "val_valid_mask.npy"):
                np.save(duplicate_dir / name, np.concatenate([np.load(r_dir / name)] * 2, axis=0))
            with self.assertRaisesRegex(ValueError, "Duplicate R validation time key"):
                evaluator.evaluate(cot_path, keys_path, duplicate_dir, root / "bad", draws=200)

    def test_pack_gate_checks_actual_oof_receipt_hash(self):
        with tempfile.TemporaryDirectory() as td:
            report_path = Path(td) / "oof.json"
            report_path.write_text('{"state":"PASS_OOF_TRAIN_BANK_LINEAGE_AND_GHI_ROW_BINDING","train_rows_checked":2,"validation_rows_checked":1,"test_payloads_read":0,"buffer_payloads_read":0}', encoding="utf-8")
            digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
            meta = {"state": "HUBEI_GHI_FORECAST_HEADS_TRAINVAL_PACK_TEST_CLOSED",
                    "test_and_buffer_values_read": 0, "train_forecast_inputs_are_oof": True,
                    "oof_fold_count": 5, "oof_train_bank_report_path": str(report_path),
                    "oof_train_bank_report_sha256": digest,
                    "validation_forecast_source": "frozen_full_train_S_R"}
            validate_input_pack_contract(meta, "forecast")
            meta["oof_train_bank_report_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                validate_input_pack_contract(meta, "forecast")

    def test_oof_time_key_uses_bjt_aux_and_utc_data(self):
        data, aux = [], []
        start = datetime(2025, 8, 1)
        for i in range(24):
            utc = start + timedelta(minutes=15 * i)
            data.append(f"/data/{utc:%Y%m%d%H%M}.npy")
            bjt = utc + timedelta(hours=8)
            aux.append(f"/aux/{bjt:%Y%m%d%H%M}_aux.npz")
        init, targets = oof_audit.bjt_sequence({"data_paths": "|".join(data), "aux_paths": "|".join(aux)})
        self.assertEqual(init, "202508010945")
        self.assertEqual(targets[0], "202508011000")
        self.assertEqual(targets[-1], "202508011345")


if __name__ == "__main__":
    unittest.main(verbosity=2)
