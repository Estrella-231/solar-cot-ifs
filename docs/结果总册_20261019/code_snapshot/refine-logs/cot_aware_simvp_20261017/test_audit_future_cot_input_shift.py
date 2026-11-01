import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("audit_future_cot_input_shift.py")


def test_train_val_grouping_and_physical_cot_stats():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "pack"
        for split, vals in (("train", (1.0, 3.0)), ("val", (2.0, 4.0))):
            d = root / split
            d.mkdir(parents=True)
            # Three context frames, one COT channel, 3x3 patch.
            physical = np.zeros((2, 3, 1, 3, 3), dtype=np.float32)
            physical[:, 1, 0] = vals[0]
            physical[:, 2, 0] = np.asarray(vals)[:, None, None]
            np.save(d / "cot_log1p.npy", np.log1p(physical))
            geom = np.zeros((2, 3, 2, 3, 3), dtype=np.float32)
            geom[:, 2, 0] = .5
            np.save(d / "geometry.npy", geom)
            with (d / "keys.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=("station", "target_time_utc", "lead_minutes"))
                w.writeheader()
                for i in range(2):
                    w.writerow({"station": "sili", "target_time_utc": f"2026-07-0{i + 1} 12:00:00", "lead_minutes": 15})
        (root / "pack_manifest.json").write_text(json.dumps({
            "state": "COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK",
            "test_payload_opened": False, "buffer_payload_opened": False,
        }), encoding="utf-8")
        out = Path(td) / "shift.json"
        subprocess.run([sys.executable, str(SCRIPT), "--pack", str(root), "--output", str(out), "--chunk-rows", "1"], check=True, capture_output=True, text=True)
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["rows"] == {"train": 2, "val": 2}
        with out.with_suffix(".csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        train_row = next(r for r in rows if r["split"] == "train")
        val_row = next(r for r in rows if r["split"] == "val")
        assert train_row["solar_band"] == val_row["solar_band"] == "sun"
        assert float(train_row["mean"]) == 2.0
        assert float(val_row["mean"]) == 3.0
        assert float(train_row["mean_abs_15min_change"]) == 1.0


if __name__ == "__main__":
    test_train_val_grouping_and_physical_cot_stats()
    print("FUTURE_COT_INPUT_SHIFT_WITNESS_PASS")
