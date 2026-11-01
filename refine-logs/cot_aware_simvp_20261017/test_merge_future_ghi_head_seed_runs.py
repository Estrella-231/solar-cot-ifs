"""Synthetic witness for merging isolated seed head runs without collisions."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MERGER = HERE / "merge_future_ghi_head_seed_runs.py"


def make_run(root: Path, seed: int, base: dict, arms: tuple[str, ...], state: str) -> None:
    root.mkdir(parents=True)
    manifest = dict(base)
    stations = tuple(base["stations"])
    manifest.update({"state": state, "seeds": [seed], "runs": []})
    for station in stations:
        for arm in arms:
            d = root / station / f"seed{seed}" / arm
            d.mkdir(parents=True)
            for name in ("best.pt", "metrics.json", "training_curve.json", "validation_predictions.npz"):
                (d / name).write_bytes(f"{seed}:{station}:{arm}:{name}".encode())
            manifest["runs"].append({"station": station, "seed": seed, "arm": arm})
    (root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        base_contract = {"screening_pilot": True, "train_forecast_oof": True,
                         "scope": "synthetic", "split_id": "split-x", "epochs": 30,
                         "batch_size": 1792, "stations": ["sili", "zhujia"],
                         "architecture": "IntervalHead", "COT_source_arms": {"A": "none"},
                         "primary_future_cot_contrast": "AC-AH", "test_payload_opened": False,
                         "pack_manifest_sha256": "a" * 64,
                         "forecast_bank_manifest_sha256": "b" * 64,
                         "train_forecast_source": "OOF", "forecast_bank_oof_fold_count": 5}
        for tag, arms, state in (
                ("primary", ("A", "AH", "AC"), "COMPLETE_FUTURE_GHI_COT_SOURCE_SCREENING_PILOT"),
                ("shrink", ("A", "AH", "AC", "AC50"), "COMPLETE_FUTURE_GHI_COT_SHRINKAGE_SCREENING_PILOT")):
            b = root / f"{tag}_base"
            c43, c44 = root / f"{tag}_seed43", root / f"{tag}_seed44"
            make_run(b, 42, base_contract, arms, state)
            make_run(c43, 43, base_contract, arms, state)
            make_run(c44, 44, base_contract, arms, state)
            proc = subprocess.run([sys.executable, str(MERGER), "--base", str(b),
                                   "--child", str(c43), str(c44), "--complete-state", state,
                                   "--required-arms", ",".join(arms)],
                                  text=True, capture_output=True)
            assert proc.returncode == 0, proc.stderr
            merged = json.loads((b / "run_manifest.json").read_text())
            assert merged["state"] == state
            assert merged["seeds"] == [42, 43, 44]
            assert len(merged["runs"]) == 3 * 2 * len(arms)
            for seed in (42, 43, 44):
                for station in base_contract["stations"]:
                    for arm in arms:
                        assert (b / station / f"seed{seed}" / arm / "metrics.json").is_file()
    print("SYNTHETIC_HEAD_SEED_MERGE_PRIMARY_AND_SHRINK_PASS")


if __name__ == "__main__":
    main()
