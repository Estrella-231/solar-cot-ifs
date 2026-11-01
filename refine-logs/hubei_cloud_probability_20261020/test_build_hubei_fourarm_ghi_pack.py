import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("build_hubei_fourarm_ghi_pack.py")


def write_rows(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


class CommonSupportPackTest(unittest.TestCase):
    def test_coverage_join_slices_all_base_arrays_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ghi, cot, pc, out = [root / x for x in ("ghi", "cot", "pc", "out")]
            for d in (ghi, cot, pc):
                d.mkdir()
            (ghi / "pack_report.json").write_text(json.dumps({
                "state": "HUBEI_GHI_HEADS_COMMON_TRAINVAL_PACK_TEST_CLOSED",
                "test_payloads_read": 0,
            }), encoding="utf-8")
            timekeys = ["202508010600", "202508010615", "202508010630"]
            rows = [{"station": "Niushou", "time_bjt": k, "split": "train"} for k in timekeys]
            write_rows(ghi / "rows.csv", rows, ["station", "time_bjt", "split"])
            base = {
                "agri_raw.npy": np.arange(3 * 13 * 3 * 3, dtype=np.float32).reshape(3, 13, 3, 3),
                "geometry.npy": np.arange(3 * 2 * 3 * 3, dtype=np.float32).reshape(3, 2, 3, 3),
                "ghi_wm2.npy": np.array([100, 200, 300], dtype=np.float32),
                "clear_sky_wm2.npy": np.array([500, 600, 700], dtype=np.float32),
            }
            for name, arr in base.items():
                np.save(ghi / name, arr, allow_pickle=False)

            for d, state, pred_name, pred in (
                (cot, "HUBEI_FROZEN_R_FULL_AGRI_TRAINVAL_BANK_COMPLETE_TEST_CLOSED",
                 "cot_pred_trainval_full.npy", np.ones((2, 1, 16, 16), dtype=np.float32)),
                (pc, "COMPLETE_HUBEI_FULL_AGRI_P_CLOUD_TRAINVAL_TEST_CLOSED",
                 "p_cloud_trainval.npy", np.full((2, 16, 16), 0.6, dtype=np.float32)),
            ):
                (d / "BANK_REPORT.json").write_text(json.dumps({
                    "state": state, "test_payloads_read": 0, "prediction_sha256": "fixture",
                }), encoding="utf-8")
                side_rows = [{"time_bjt": k, "split": "train", "prediction_source": "date_oof"}
                             for k in timekeys[:2]]
                write_rows(d / "rows.csv", side_rows, list(side_rows[0]))
                np.save(d / pred_name, pred, allow_pickle=False)

            subprocess.run([sys.executable, str(SCRIPT), "--ghi-pack", str(ghi),
                            "--cot-bank", str(cot), "--pcloud-bank", str(pc),
                            "--output", str(out)], check=True, capture_output=True, text=True)
            report = json.loads((out / "pack_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["source_ghi_rows"], 3)
            self.assertEqual(report["retained_common_support_rows"], 2)
            self.assertEqual(report["excluded_rows_due_aux_source_unavailable"], 1)
            with (out / "rows.csv").open(newline="", encoding="utf-8") as f:
                retained_keys = [r["time_bjt"] for r in csv.DictReader(f)]
            self.assertEqual(retained_keys, timekeys[:2])
            for name, arr in base.items():
                np.testing.assert_array_equal(np.load(out / name, allow_pickle=False), arr[:2])


if __name__ == "__main__":
    unittest.main()
