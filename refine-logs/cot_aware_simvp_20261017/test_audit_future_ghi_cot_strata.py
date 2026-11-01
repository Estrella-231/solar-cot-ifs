"""Synthetic witness for key-joined forecast-COT stratified diagnostics."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def main():
    script = Path(__file__).with_name("audit_future_ghi_cot_strata.py")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); pack = root / "pack"; val = pack / "val"; val.mkdir(parents=True)
        (pack / "pack_manifest.json").write_text(json.dumps({
            "state": "COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK",
            "test_payload_opened": False, "buffer_payload_opened": False,
        }))
        keys = []
        cot = np.zeros((8, 3, 1, 3, 3), dtype=np.float32)
        cots = (0.5, 5., 20., 40.)
        leads = (15, 75, 135, 240)
        for si, station in enumerate(("sili", "zhujia")):
            for i, (c, lead) in enumerate(zip(cots, leads)):
                target = f"2024-05-0{1+i} 04:15:00"
                row = {"split": "val", "station": station,
                       "sequence_index_within_split": str(si*4+i),
                       "target_time_utc": target,
                       "init_time_utc": f"2024-05-0{1+i} 04:00:00",
                       "lead_minutes": str(lead), "ghi_row_index": str(si*4+i),
                       "crosswalk_row_index": str(si*4+i)}
                keys.append(row)
                cot[si*4+i, 2, 0, 1, 1] = np.log1p(c)
        with (val / "keys.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(keys[0])); w.writeheader(); w.writerows(keys)
        np.save(val / "cot_log1p.npy", cot)

        run = root / "runs"
        for seed in (42, 43, 44):
            for si, station in enumerate(("sili", "zhujia")):
                pred_keys = np.asarray([json.dumps(k, sort_keys=True) for k in keys[si*4:si*4+4]])
                truth = np.asarray([100., 200., 300., 400.])
                for arm, offset in (("A", 10.), ("AH", 7.), ("AC", 3.), ("AC50", 2.)):
                    d = run / station / f"seed{seed}" / arm; d.mkdir(parents=True)
                    np.savez_compressed(d / "validation_predictions.npz",
                        prediction_ghi_wm2=truth+offset, truth_ghi_wm2=truth, keys=pred_keys)
        out = root / "strata.json"
        subprocess.run([sys.executable, str(script), "--pack", str(pack),
                        "--run-root", str(run), "--output", str(out)],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["state"] == "COMPLETE_DESCRIPTIVE_FUTURE_GHI_COT_STRATA_AUDIT"
        assert report["test_payload_opened"] is False and report["buffer_payload_opened"] is False
        assert len(report["rows"]) == 24
        assert all(r["delta_AC_minus_AH_rmse_wm2"] < 0 for r in report["rows"])
        assert all(r["delta_AC50_minus_AH_rmse_wm2"] < 0 for r in report["rows"])
        assert (root / "strata.csv").is_file()
    print("FUTURE_GHI_COT_STRATA_WITNESS_PASS")


if __name__ == "__main__":
    main()
