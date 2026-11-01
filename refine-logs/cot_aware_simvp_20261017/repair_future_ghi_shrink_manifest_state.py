#!/usr/bin/env python3
"""Repair a known aggregate-state serialization bug after artifact verification.

This changes only the aggregate completion-state field. It refuses to run unless
the fixed three-seed, two-station, four-arm AC50 artifact set is complete and
the checkpoint/prediction hashes recorded in the manifest match disk.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path


EXPECTED_ARMS = ("A", "AH", "AC", "AC50")
EXPECTED_STATIONS = ("sili", "zhujia")
EXPECTED_SEEDS = (42, 43, 44)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repair(run_root: Path, pack_manifest: Path, receipt: Path,
           expected_prior_state: str, complete_state: str) -> dict:
    run_root = run_root.resolve()
    pack_manifest = pack_manifest.resolve()
    receipt = receipt.resolve()
    manifest_path = run_root / "run_manifest.json"
    if receipt.exists():
        raise FileExistsError(receipt)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("state") != expected_prior_state:
        raise ValueError("aggregate state differs from the known serialization bug")
    if manifest.get("seeds") != list(EXPECTED_SEEDS):
        raise ValueError("unexpected seed coverage")
    if manifest.get("stations") != list(EXPECTED_STATIONS):
        raise ValueError("unexpected station coverage")
    if manifest.get("test_payload_opened") is not False:
        raise ValueError("test payload must be explicitly closed")
    if manifest.get("train_forecast_oof") is not True:
        raise ValueError("training forecasts must be certified OOF")
    if manifest.get("forecast_bank_oof_fold_count") != 5:
        raise ValueError("expected five date-block OOF folds")
    if "0.5 * future forecast-COT anomaly" not in manifest.get("future_cot_shrinkage", ""):
        raise ValueError("manifest does not describe the registered AC50 intervention")
    if not pack_manifest.is_file():
        raise FileNotFoundError(pack_manifest)
    pack = json.loads(pack_manifest.read_text(encoding="utf-8"))
    if pack.get("state") != "COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK":
        raise ValueError("source pack is not complete")
    if sha256(pack_manifest) != manifest.get("pack_manifest_sha256"):
        raise ValueError("source pack manifest hash mismatch")

    expected = set(itertools.product(EXPECTED_STATIONS, EXPECTED_SEEDS, EXPECTED_ARMS))
    entries = manifest.get("runs", [])
    observed = [(x.get("station"), x.get("seed"), x.get("arm")) for x in entries]
    if len(entries) != len(expected) or set(observed) != expected or len(set(observed)) != len(expected):
        raise ValueError("aggregate run manifest does not have exact station/seed/arm coverage")

    checked = []
    for entry in entries:
        if entry.get("test_payload_opened") is not False:
            raise ValueError("a run entry does not explicitly certify test payload closure")
        out = run_root / entry["station"] / ("seed%d" % entry["seed"]) / entry["arm"]
        for name in ("best.pt", "metrics.json", "training_curve.json", "validation_predictions.npz"):
            if not (out / name).is_file():
                raise FileNotFoundError(out / name)
        ckpt_hash = sha256(out / "best.pt")
        pred_hash = sha256(out / "validation_predictions.npz")
        if ckpt_hash != entry.get("checkpoint_sha256"):
            raise ValueError("checkpoint hash mismatch: %s" % out)
        if pred_hash != entry.get("predictions_sha256"):
            raise ValueError("prediction hash mismatch: %s" % out)
        checked.append({"station": entry["station"], "seed": entry["seed"],
                        "arm": entry["arm"], "checkpoint_sha256": ckpt_hash,
                        "predictions_sha256": pred_hash})

    old_manifest_sha = sha256(manifest_path)
    manifest["state"] = complete_state
    temporary = manifest_path.with_suffix(".json.repair-tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, manifest_path)
    receipt_data = {
        "state": "PASS_AC50_AGGREGATE_MANIFEST_STATE_REPAIR",
        "run_root": str(run_root),
        "previous_state": expected_prior_state,
        "repaired_state": complete_state,
        "previous_manifest_sha256": old_manifest_sha,
        "repaired_manifest_sha256": sha256(manifest_path),
        "pack_manifest_sha256": sha256(pack_manifest),
        "verified_run_count": len(checked),
        "verified_runs": checked,
        "scientific_outputs_modified": False,
        "only_manifest_state_field_changed": True,
    }
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(receipt_data, indent=2) + "\n", encoding="utf-8")
    return receipt_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--pack-manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-prior-state", required=True)
    parser.add_argument("--complete-state", required=True)
    args = parser.parse_args()
    result = repair(args.run_root, args.pack_manifest, args.receipt,
                    args.expected_prior_state, args.complete_state)
    print(json.dumps({k: result[k] for k in ("state", "previous_state", "repaired_state",
        "verified_run_count", "previous_manifest_sha256", "repaired_manifest_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
