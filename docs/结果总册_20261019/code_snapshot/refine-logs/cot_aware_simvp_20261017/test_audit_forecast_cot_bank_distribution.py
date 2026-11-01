import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("audit_forecast_cot_bank_distribution.py")


def test_bank_distribution_and_provenance():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "bank"
        root.mkdir()
        index_rows = [
            {"bank_index": 0, "split": "train", "sequence_index_within_split": 0, "seq_id": "hunan_202607011200"},
            {"bank_index": 1, "split": "val", "sequence_index_within_split": 0, "seq_id": "hunan_202607021200"},
        ]
        with (root / "index.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(index_rows[0]))
            w.writeheader(); w.writerows(index_rows)
        cot = np.zeros((2, 2, 16, 1, 3, 3), dtype=np.float32)
        cot[0] = np.log1p(1.0)
        cot[1] = np.log1p(3.0)
        np.save(root / "forecast_cot_log1p.npy", cot)
        geom = np.zeros((2, 2, 16, 2, 3, 3), dtype=np.float32)
        geom[:, :, :, 0] = .5
        np.save(root / "forecast_geometry.npy", geom)
        hist = np.zeros((2, 2, 2, 1, 3, 3), dtype=np.float32)
        hist[0] = np.log1p(1.0)
        hist[1] = np.log1p(3.0)
        np.save(root / "history_cot_log1p.npy", hist)
        (root / "bank_manifest.json").write_text(json.dumps({
            "state": "COMPLETE_FORECAST_AGRI_R_TRAIN_VAL_BANK", "rows": 2,
            "test_payload_opened": False, "buffer_payload_opened": False,
            "future_observed_agri_read": False, "cpp_read": False,
            "in_sample_train_forecasts": True,
        }), encoding="utf-8")
        out = Path(td) / "shift.json"
        subprocess.run([sys.executable, str(SCRIPT), "--bank", str(root), "--output", str(out)], check=True, capture_output=True, text=True)
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["state"] == "COMPLETE_FORECAST_COT_BANK_INPUT_SHIFT_DIAGNOSTIC"
        assert report["train_forecasts_in_sample"] is True
        assert report["input_instances_by_split"] == {"train": 32, "val": 32}
        with out.with_suffix(".csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        train_row = next(r for r in rows if r["split"] == "train")
        val_row = next(r for r in rows if r["split"] == "val")
        assert float(train_row["mean"]) == 1.0
        assert float(val_row["mean"]) == 3.0
        assert float(train_row["mean_abs_15min_change"]) == 0.0
        assert train_row["solar_band"] == "sun"


if __name__ == "__main__":
    test_bank_distribution_and_provenance()
    print("FORECAST_COT_BANK_INPUT_SHIFT_WITNESS_PASS")
