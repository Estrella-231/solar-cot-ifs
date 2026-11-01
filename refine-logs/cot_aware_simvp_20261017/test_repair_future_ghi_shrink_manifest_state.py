"""Synthetic fail-closed witness for the AC50 aggregate state repair."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from repair_future_ghi_shrink_manifest_state import repair


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_root = root / "run"
        pack_path = root / "pack_manifest.json"
        pack_path.write_text(json.dumps({"state": "COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK"}))
        entries = []
        for station in ("sili", "zhujia"):
            for seed in (42, 43, 44):
                for arm in ("A", "AH", "AC", "AC50"):
                    output = run_root / station / ("seed%d" % seed) / arm
                    output.mkdir(parents=True)
                    for name in ("best.pt", "metrics.json", "training_curve.json",
                                 "validation_predictions.npz"):
                        (output / name).write_bytes((station + str(seed) + arm + name).encode())
                    entries.append({"station": station, "seed": seed, "arm": arm,
                        "checkpoint_sha256": sha(output / "best.pt"),
                        "predictions_sha256": sha(output / "validation_predictions.npz"),
                        "test_payload_opened": False})
        manifest = {"state": "COMPLETE_FUTURE_GHI_COT_SOURCE_SCREENING_PILOT",
            "seeds": [42, 43, 44], "stations": ["sili", "zhujia"],
            "test_payload_opened": False, "train_forecast_oof": True,
            "forecast_bank_oof_fold_count": 5,
            "future_cot_shrinkage": "AC50 = history COT + 0.5 * future forecast-COT anomaly in train-normalized space",
            "pack_manifest_sha256": sha(pack_path), "runs": entries}
        (run_root / "run_manifest.json").write_text(json.dumps(manifest))
        result = repair(run_root, pack_path, root / "receipt.json",
            "COMPLETE_FUTURE_GHI_COT_SOURCE_SCREENING_PILOT",
            "COMPLETE_FUTURE_GHI_COT_SHRINKAGE_SCREENING_PILOT")
        assert result["verified_run_count"] == 24
        repaired = json.loads((run_root / "run_manifest.json").read_text())
        assert repaired["state"] == "COMPLETE_FUTURE_GHI_COT_SHRINKAGE_SCREENING_PILOT"

        # Refuse a changed prediction artifact and a second repair attempt.
        out = run_root / "sili" / "seed42" / "A" / "validation_predictions.npz"
        out.write_bytes(b"changed")
        bad_receipt = root / "bad.json"
        try:
            repair(run_root, pack_path, bad_receipt,
                "COMPLETE_FUTURE_GHI_COT_SHRINKAGE_SCREENING_PILOT",
                "COMPLETE_FUTURE_GHI_COT_SHRINKAGE_SCREENING_PILOT")
        except ValueError:
            pass
        else:
            raise AssertionError("changed prediction artifact was not rejected")
    print("AC50_MANIFEST_REPAIR_SYNTHETIC_PASS")


if __name__ == "__main__":
    main()
