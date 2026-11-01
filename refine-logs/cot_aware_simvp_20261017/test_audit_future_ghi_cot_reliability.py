"""Synthetic paired/date-block witness for the five-arm reliability audit."""
from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def main():
    script = Path(__file__).with_name("audit_future_ghi_cot_reliability.py")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); run = root / "runs"; out = root / "audit.json"
        run.mkdir(parents=True)
        (run / "run_manifest.json").write_text(json.dumps({
            "state": "COMPLETE_FUTURE_GHI_COT_RELIABILITY_SCREENING_PILOT",
            "train_forecast_oof": True,
            "forecast_bank_oof_fold_count": 5,
        }), encoding="utf-8")
        target_times = ["2024-05-01 04:15:00", "2024-05-02 04:15:00", "2024-05-03 04:15:00"]
        offsets = {"A": 8., "AH": 6., "AC": 4., "AC50": 3., "ACREL": 2.}
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
                truth = np.asarray(truth, np.float64); keys = np.asarray(keys)
                for arm, offset in offsets.items():
                    d = run / station / f"seed{seed}" / arm
                    d.mkdir(parents=True)
                    (d / "best.pt").write_bytes(f"{station}-{seed}-{arm}".encode())
                    np.savez_compressed(d / "validation_predictions.npz",
                        prediction_ghi_wm2=truth + offset, truth_ghi_wm2=truth, keys=keys)
                    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
                    (d / "metrics.json").write_text(json.dumps({
                        "checkpoint_sha256": sha(d / "best.pt"),
                        "predictions_sha256": sha(d / "validation_predictions.npz"),
                    }), encoding="utf-8")
                    run_manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
                    run_manifest.setdefault("runs", []).append({"station": station, "seed": seed, "arm": arm})
                    (run / "run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")
        proc = subprocess.run([sys.executable, str(script), "--run-root", str(run),
                               "--output", str(out), "--bootstrap", "100"],
                              check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode:
            raise RuntimeError(f"audit failed: stdout={proc.stdout}\nstderr={proc.stderr}")
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["state"] == "COMPLETE_PAIRED_FUTURE_GHI_COT_RELIABILITY_VALIDATION_AUDIT"
        assert report["train_forecast_oof"] is True
        assert report["forecast_bank_oof_fold_count"] == 5
        assert "ACREL" in report["arms"]
        assert report["pairwise_delta_rmse_wm2"]["ACREL_minus_AH"] < 0
        assert report["pairwise_delta_rmse_wm2"]["ACREL_minus_AC"] < 0
        assert report["pairwise_delta_rmse_wm2"]["ACREL_minus_AC50"] < 0
        assert report["pairwise_relative_rmse_change_pct"]["ACREL_minus_AH"] < 0
        lo, hi = report["date_block_bootstrap_95pct_ci_wm2"]["ACREL_minus_AH"]
        assert lo < 0 and hi < 0
        assert report["target_bjt_date_blocks"] == 3
        assert report["test_payload_opened"] is False
        assert (root / "audit.csv").is_file()

        bad_path = run / "sili" / "seed43" / "ACREL" / "validation_predictions.npz"
        with np.load(bad_path, allow_pickle=False) as z:
            pred, bad_truth, saved_keys = z["prediction_ghi_wm2"], z["truth_ghi_wm2"].copy(), z["keys"]
        bad_truth[0] += 1.0
        np.savez_compressed(bad_path, prediction_ghi_wm2=pred, truth_ghi_wm2=bad_truth, keys=saved_keys)
        bad_proc = subprocess.run([sys.executable, str(script), "--run-root", str(run),
                                   "--output", str(root / "bad.json"), "--bootstrap", "100"],
                                  check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert bad_proc.returncode != 0
        assert "hash mismatch" in bad_proc.stderr
    print("FUTURE_GHI_COT_RELIABILITY_PAIRED_BOOTSTRAP_WITNESS_PASS")


if __name__ == "__main__":
    main()
