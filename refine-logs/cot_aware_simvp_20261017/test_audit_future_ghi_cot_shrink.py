"""Synthetic witness for paired keys and target-date bootstrap aggregation."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def main():
    script = Path(__file__).with_name("audit_future_ghi_cot_shrink.py")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); run = root / "runs"; out = root / "audit.json"
        run.mkdir(parents=True)
        (run / "run_manifest.json").write_text(json.dumps({
            "state": "COMPLETE_FUTURE_GHI_COT_SHRINKAGE_SCREENING_PILOT",
            "train_forecast_oof": True,
            "forecast_bank_oof_fold_count": 5,
        }), encoding="utf-8")
        target_times = ["2024-05-01 04:15:00", "2024-05-02 04:15:00", "2024-05-03 04:15:00"]
        for seed in (42, 43, 44):
            for station in ("sili", "zhujia"):
                keys, truth = [], []
                for day_i, target in enumerate(target_times):
                    for lead in (15, 30, 60, 120, 240):
                        keys.append(json.dumps({"station": station,
                            "init_time_utc": f"2024-05-01 04:{day_i:02d}:00",
                            "target_time_utc": target, "lead_minutes": lead,
                            "ghi_row_index": day_i * 5 + lead // 15}, sort_keys=True))
                        truth.append(100. + day_i * 20. + lead)
                truth = np.asarray(truth, np.float64)
                keys = np.asarray(keys)
                for arm, offset in (("A", 8.), ("AH", 6.), ("AC", 4.), ("AC50", 3.)):
                    d = run / station / f"seed{seed}" / arm
                    d.mkdir(parents=True)
                    np.savez_compressed(d / "validation_predictions.npz",
                        prediction_ghi_wm2=truth + offset, truth_ghi_wm2=truth, keys=keys)
        proc = subprocess.run([sys.executable, str(script), "--run-root", str(run),
                               "--output", str(out), "--bootstrap", "100"],
                              check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode:
            raise RuntimeError(f"audit failed: stdout={proc.stdout}\nstderr={proc.stderr}")
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["state"] == "COMPLETE_PAIRED_FUTURE_GHI_COT_SHRINKAGE_VALIDATION_AUDIT"
        assert report["train_forecast_oof"] is True
        assert report["forecast_bank_oof_fold_count"] == 5
        assert "date-block OOF training forecasts" in report["scope"]
        assert report["pairwise_delta_rmse_wm2"]["AC_minus_A"] < 0
        assert report["pairwise_delta_rmse_wm2"]["AC_minus_AH"] < 0
        assert report["pairwise_delta_rmse_wm2"]["AC50_minus_AH"] < 0
        assert report["pairwise_delta_rmse_wm2"]["AC50_minus_AC"] < 0
        assert report["pairwise_delta_rmse_wm2"]["AH_minus_A"] < 0
        assert report["pairwise_relative_rmse_change_pct"]["AC_minus_AH"] < 0
        lo, hi = report["date_block_bootstrap_95pct_ci_wm2"]["AC_minus_AH"]
        assert lo < 0 and hi < 0
        assert report["target_bjt_date_blocks"] == 3
        assert report["test_payload_opened"] is False
        assert (root / "audit.csv").is_file()

        # Cross-seed labels must also be identical; a mismatched label pack
        # must fail closed before any paired score or bootstrap is reported.
        for arm in ("A", "AH", "AC", "AC50"):
            bad_path = run / "sili" / "seed43" / arm / "validation_predictions.npz"
            with np.load(bad_path, allow_pickle=False) as z:
                pred = z["prediction_ghi_wm2"]
                bad_truth = z["truth_ghi_wm2"].copy()
                keys = z["keys"]
            bad_truth[0] += 1.0
            np.savez_compressed(bad_path, prediction_ghi_wm2=pred,
                                truth_ghi_wm2=bad_truth, keys=keys)
        bad_out = root / "audit_bad_truth.json"
        bad_proc = subprocess.run([sys.executable, str(script), "--run-root", str(run),
                                   "--output", str(bad_out), "--bootstrap", "100"],
                                  check=False, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True)
        assert bad_proc.returncode != 0
        assert "paired truth arrays vary across seeds" in bad_proc.stderr
    print("FUTURE_GHI_COT_SHRINKAGE_PAIRED_BOOTSTRAP_WITNESS_PASS")


if __name__ == "__main__":
    main()
