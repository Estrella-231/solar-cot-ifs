#!/usr/bin/env python3
"""Merge candidate OOF train forecasts and candidate full-train val forecasts.

Unlike merge_future_ghi_oof_bank.py, this deliberately replaces validation
forecast rows as well as OOF train rows. It never reads test/buffer payloads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


FORECAST_ARRAYS = ("forecast_agri_raw.npy", "forecast_cot_log1p.npy")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def verify_closed(meta: dict, fields: tuple[str, ...], label: str) -> None:
    for key in fields:
        if meta.get(key) is not False:
            raise ValueError(f"{label} provenance gate requires {key}=false")


def verify_array_file(root: Path, meta: dict, name: str) -> np.ndarray:
    path = root / name
    declared = meta.get("array_sha256", {}).get(name)
    if not path.is_file() or declared != sha(path):
        raise ValueError(f"{root}: missing or hash-mismatched {name}")
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if arr.dtype != np.float32 or not np.isfinite(arr).all():
        raise ValueError(f"{root}: invalid/nonfinite {name}")
    return arr


def merge(base: Path, manifests: Path, fold_banks: list[Path], validation_bank: Path,
          out: Path) -> dict:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    base_meta = json.loads((base / "bank_manifest.json").read_text(encoding="utf-8"))
    if base_meta.get("state") != "COMPLETE_FORECAST_AGRI_R_TRAIN_VAL_BANK":
        raise ValueError("base train/validation bank is not complete")
    verify_closed(base_meta, ("test_payload_opened", "buffer_payload_opened",
                              "future_observed_agri_read", "cpp_read"), "base bank")
    rows = read_rows(base / "index.csv")
    if not rows or not {"seq_id", "split"}.issubset(rows[0]):
        raise ValueError("base bank index lacks seq_id/split")
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    train_ids = [r["seq_id"] for r in train_rows]
    val_ids = [r["seq_id"] for r in val_rows]
    if len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids):
        raise ValueError("base bank contains duplicate split sequence keys")
    if set(train_ids) & set(val_ids):
        raise ValueError("train/validation sequence keys overlap")
    target_index = {r["seq_id"]: i for i, r in enumerate(rows)}

    # Validate the candidate full-train validation forecast bank before writing.
    vm = json.loads((validation_bank / "bank_manifest.json").read_text(encoding="utf-8"))
    if vm.get("state") != "COMPLETE_DROPPATH0_FULLTRAIN_VALIDATION_FORECASTS":
        raise ValueError("candidate validation forecast bank is not complete")
    verify_closed(vm, ("test_payload_opened", "buffer_payload_opened",
                       "future_observed_agri_read", "cpp_read"), "candidate validation bank")
    if vm.get("validation_payload_used_as_input") is not False:
        raise ValueError("validation payload may be used for scoring, never as model input")
    if vm.get("index_sha256") != sha(validation_bank / "index.csv"):
        raise ValueError("candidate validation index hash mismatch")
    vrows = read_rows(validation_bank / "index.csv")
    vids = [r["seq_id"] for r in vrows]
    if vids != val_ids:
        raise ValueError("candidate validation forecast keys/order differ from base validation keys")
    val_arrays = {name: verify_array_file(validation_bank, vm, name) for name in FORECAST_ARRAYS}
    expected_val_shapes = {
        "forecast_agri_raw.npy": (len(val_ids), 2, 16, 13, 3, 3),
        "forecast_cot_log1p.npy": (len(val_ids), 2, 16, 1, 3, 3),
    }
    for name, arr in val_arrays.items():
        if arr.shape != expected_val_shapes[name]:
            raise ValueError(f"candidate validation shape mismatch for {name}: {arr.shape}")
    if vm.get("simvp_checkpoint_sha256") != vm.get("selected_checkpoint_sha256"):
        raise ValueError("candidate validation bank is not bound to its selected full-train checkpoint")

    # Validate all five OOF fold banks against their frozen heldout IDs.
    fold_ids_seen: set[str] = set()
    fold_src: dict[int, tuple[Path, list[str], dict]] = {}
    for fb in fold_banks:
        fm = json.loads((fb / "bank_manifest.json").read_text(encoding="utf-8"))
        if fm.get("state") != "COMPLETE_FUTURE_GHI_OOF_FOLD_FORECASTS":
            raise ValueError(f"incomplete candidate OOF fold bank: {fb}")
        verify_closed(fm, ("future_observed_agri_read", "cpp_read",
                           "validation_test_payload_opened", "buffer_payload_opened"), str(fb))
        fold = int(fm["fold"])
        expected = (manifests / f"fold_{fold}/fold_oof_ids.txt").read_text().splitlines()
        frows = read_rows(fb / "index.csv")
        if fm.get("index_sha256") != sha(fb / "index.csv"):
            raise ValueError(f"fold {fold}: index hash mismatch")
        fids = [r["seq_id"] for r in frows]
        if fids != expected:
            raise ValueError(f"fold {fold}: candidate output keys differ from frozen OOF IDs")
        if fold_ids_seen.intersection(fids):
            raise ValueError("a train sequence has predictions from multiple folds")
        if not set(fids).issubset(target_index) or not set(fids).issubset(train_ids):
            raise ValueError(f"fold {fold}: OOF keys are absent from base train rows")
        for name in FORECAST_ARRAYS:
            arr = verify_array_file(fb, fm, name)
            expected_shape = ((len(fids), 2, 16, 13, 3, 3) if name == "forecast_agri_raw.npy"
                              else (len(fids), 2, 16, 1, 3, 3))
            if arr.shape != expected_shape:
                raise ValueError(f"fold {fold}: shape mismatch for {name}: {arr.shape}")
        if fm.get("fold_checkpoint_sha256") is None:
            raise ValueError(f"fold {fold}: missing checkpoint binding")
        fold_ids_seen.update(fids)
        fold_src[fold] = (fb, fids, fm)
    if len(fold_src) != 5:
        raise ValueError(f"expected exactly five OOF fold banks, received {len(fold_src)}")
    if fold_ids_seen != set(train_ids):
        raise ValueError(f"OOF train coverage mismatch: missing={len(set(train_ids)-fold_ids_seen)} "
                         f"extra={len(fold_ids_seen-set(train_ids))}")

    # All inputs are now validated. Create an independent candidate bank.
    out.mkdir(parents=True)
    arrays: dict[str, np.memmap] = {}
    for name, shape in base_meta["layouts"].items():
        src = np.load(base / name, mmap_mode="r", allow_pickle=False)
        if list(src.shape) != shape or src.dtype != np.float32:
            raise ValueError(f"base bank array schema mismatch: {name}")
        dst = np.lib.format.open_memmap(out / name, mode="w+", dtype=np.float32, shape=src.shape)
        for lo in range(0, len(src), 128):
            dst[lo:lo + 128] = src[lo:lo + 128]
        arrays[name] = dst

    for name in FORECAST_ARRAYS:
        for fold, (fb, fids, fm) in sorted(fold_src.items()):
            src = np.load(fb / name, mmap_mode="r", allow_pickle=False)
            for j, seq_id in enumerate(fids):
                arrays[name][target_index[seq_id]] = src[j]
        for j, seq_id in enumerate(val_ids):
            arrays[name][target_index[seq_id]] = val_arrays[name][j]
    for arr in arrays.values():
        arr.flush()
    with (out / "index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=tuple(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    meta = dict(base_meta)
    meta.update({
        "state": "COMPLETE_FORECAST_AGRI_R_TRAIN_VAL_BANK",
        "forecast_source": "candidate DropPath-0: train rows are five-fold date-heldout epoch-12 OOF; validation rows are the separately trained full-train candidate selected by equal-four-season validation RMSE; inference uses observed history only",
        "forecast_cot_source": "frozen R applied to candidate predicted AGRI and deterministic geometry; no CPP",
        "in_sample_train_forecasts": False,
        "oof_train_sequences": len(train_ids),
        "oof_fold_count": 5,
        "oof_fold_model_hashes": {str(k): v[2]["fold_checkpoint_sha256"] for k, v in sorted(fold_src.items())},
        "validation_forecast_model_hash": vm["selected_checkpoint_sha256"],
        "validation_forecast_bank_manifest_sha256": sha(validation_bank / "bank_manifest.json"),
        "source_bank_manifest_sha256": sha(base / "bank_manifest.json"),
        "test_payload_opened": False,
        "buffer_payload_opened": False,
        "future_observed_agri_read": False,
        "cpp_read": False,
    })
    meta["array_sha256"] = {p.name: sha(p) for p in out.glob("*.npy")}
    meta["index_sha256"] = sha(out / "index.csv")
    (out / "bank_manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return {"state": "PASS_CANDIDATE_OOF_VALIDATION_BANK_MERGE",
            "train_sequences": len(train_ids), "validation_sequences": len(val_ids),
            "fold_count": len(fold_src), "test_payload_opened": False,
            "buffer_payload_opened": False, "future_observed_agri_read": False,
            "cpp_read": False, "output_manifest_sha256": sha(out / "bank_manifest.json")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-bank", type=Path, required=True)
    p.add_argument("--fold-manifests", type=Path, required=True)
    p.add_argument("--fold-bank", type=Path, action="append", required=True)
    p.add_argument("--validation-bank", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(merge(a.base_bank, a.fold_manifests, a.fold_bank,
                           a.validation_bank, a.output), indent=2))


if __name__ == "__main__":
    main()
