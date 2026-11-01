"""Small synthetic contract test for the H-only direct-COT GHI pack builder."""
from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BUILDER = HERE / "build_direct_cot_single_frame_ghi_pack.py"
ARMS = ("none", "simvp_r", "p_c", "p_xc")
STATIONS = ("sili", "zhujia")


class DirectCotPackTest(unittest.TestCase):
    def test_h_only_normalization_and_common_key_pack(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); bank = root / "bank"; out = root / "pack"
            bank.mkdir()
            manifest = {
                "state": "COMPLETE_DIRECT_COT_FORECAST_ROW_BANK",
                "arms": list(ARMS), "stations": list(STATIONS),
                "test_payload_opened": False, "buffer_payload_opened": False,
                "future_observed_agri_used_as_input": False, "cpp_used_as_input": False,
                "split_support_audit_sha256": "a" * 64,
                "normalization_fit_partition": "H",
                "agri_source": "fold1-SimVP-U-only-forecast",
                "artifact_sha256": {"fixture": "synthetic"},
            }
            for station in STATIONS:
                for split, n in (("train", 2), ("val", 1)):
                    d = bank / station / split; d.mkdir(parents=True)
                    agri = np.full((n, 13, 3, 3), 2.0 if split == "train" else 100.0, np.float32)
                    geo = np.zeros((n, 2, 3, 3), np.float32)
                    geo[:, 0] = 0.8
                    geo[:, 1] = 0.2
                    geo[:, 0, 0, 0] = 0.1  # Outside the future-COT support.
                    ghi = np.full(n, 300.0, np.float32)
                    clear = np.full(n, 600.0, np.float32)
                    for name, x in (("agri_raw.npy", agri), ("geometry_raw.npy", geo),
                                    ("ghi.npy", ghi), ("clear.npy", clear)):
                        np.save(d / name, x, allow_pickle=False)
                    keys = []
                    for i in range(n):
                        day = i + 1 if split == "train" else 11 + i
                        keys.append(json.dumps({"station": station, "init_time_utc": f"2024-04-{day:02d} 00:00:00",
                                                "target_time_utc": f"2024-04-{day:02d} 00:15:00",
                                                "lead_minutes": 15}, separators=(",", ":")))
                    (d / "keys.jsonl").write_text("\n".join(keys) + "\n", encoding="utf-8")
                    cd = d / "cot_raw"; cd.mkdir()
                    for arm in ARMS:
                        v = np.zeros((n, 1, 3, 3), np.float32)
                        if arm != "none":
                            v[:] = np.array([1.0, 3.0], np.float32)[:n, None, None, None]
                            if split == "val": v[:] = 90.0
                        np.save(cd / f"{arm}.npy", v, allow_pickle=False)
            manifest["artifact_sha256"] = {
                p.relative_to(bank).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in bank.rglob("*") if p.is_file()
            }
            (bank / "bank_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            proc = subprocess.run([sys.executable, str(BUILDER), "--bank", str(bank), "--output", str(out)],
                                  check=False, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            pm = json.loads((out / "pack_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(pm["state"], "COMPLETE_DIRECT_COT_SINGLE_FRAME_GHI_PACK")
            self.assertEqual(pm["normalization_and_common_key_contract"],
                             "PASS_TRAIN_ONLY_NORMALIZATION_AND_COMMON_KEYS")
            self.assertAlmostEqual(pm["normalization"]["sili"]["cot_by_arm"]["p_c"]["mean_log1p"],
                                   (np.log(2.0) + np.log(4.0)) / 2.0, places=6)
            outside = np.load(out / "sili" / "val" / "cot" / "p_c" / "val.npy", allow_pickle=False)
            self.assertEqual(float(outside[0, 0, 0, 0]), 0.0)
            none = np.load(out / "sili" / "train" / "cot" / "none" / "train.npy", allow_pickle=False)
            self.assertEqual(float(np.abs(none).max()), 0.0)
            train_agri = np.load(out / "sili" / "train" / "agri.npy", allow_pickle=False)
            self.assertTrue(np.all(train_agri == 0.0))
            self.assertFalse(pm["test_payload_opened"])
            self.assertFalse(pm["buffer_payload_opened"])


if __name__ == "__main__":
    unittest.main()
