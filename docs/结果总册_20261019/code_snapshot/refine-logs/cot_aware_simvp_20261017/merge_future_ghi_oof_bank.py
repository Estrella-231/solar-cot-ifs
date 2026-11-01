#!/usr/bin/env python3
"""Merge fold-heldout forecast AGRI/COT into a copy of the train/val bank."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def merge(base: Path, manifests: Path, fold_banks: list[Path], out: Path) -> dict:
    if out.exists():
        raise FileExistsError(out)
    source_meta = json.loads((base / "bank_manifest.json").read_text())
    if source_meta.get("state") != "COMPLETE_FORECAST_AGRI_R_TRAIN_VAL_BANK":
        raise ValueError("source forecast bank is not complete")
    if any(source_meta.get(k) is not False for k in ("test_payload_opened", "buffer_payload_opened",
                                                       "future_observed_agri_read", "cpp_read")):
        raise ValueError("source bank provenance gate failed")
    base_rows = read_rows(base / "index.csv")
    if len({r["seq_id"] for r in base_rows if r["split"] == "train"}) != sum(r["split"] == "train" for r in base_rows):
        raise ValueError("source bank has duplicate train sequence keys")
    target_index = {r["seq_id"]: i for i, r in enumerate(base_rows) if r["split"] == "train"}
    fold_ids_seen: set[str] = set()
    fold_src = {}
    for fb in fold_banks:
        meta = json.loads((fb / "bank_manifest.json").read_text())
        if meta.get("state") != "COMPLETE_FUTURE_GHI_OOF_FOLD_FORECASTS":
            raise ValueError(f"fold bank incomplete: {fb}")
        if any(meta.get(k) is not False for k in ("future_observed_agri_read", "cpp_read",
                                                   "validation_test_payload_opened", "buffer_payload_opened")):
            raise ValueError(f"fold bank causal-input gate failed: {fb}")
        fold = int(meta["fold"])
        expected_ids = (manifests / f"fold_{fold}/fold_oof_ids.txt").read_text().splitlines()
        rows = read_rows(fb / "index.csv")
        declared_arrays = meta.get("array_sha256", {})
        for name in ("forecast_agri_raw.npy", "forecast_cot_log1p.npy"):
            if declared_arrays.get(name) != sha(fb / name):
                raise ValueError(f"fold {fold} prediction array hash mismatch: {name}")
        if meta.get("index_sha256") != sha(fb / "index.csv"):
            raise ValueError(f"fold {fold} prediction index hash mismatch")
        ids = [r["seq_id"] for r in rows]
        if ids != expected_ids:
            raise ValueError(f"fold {fold} output keys differ from its precomputed heldout OOF IDs")
        if fold_ids_seen.intersection(ids):
            raise ValueError("a train sequence received OOF predictions from multiple folds")
        fold_ids_seen.update(ids)
        if not set(ids).issubset(target_index):
            raise ValueError(f"fold {fold} has keys absent from the GHI training bank")
        fold_src[fold] = (fb, ids, meta)
    if fold_ids_seen != set(target_index):
        missing = set(target_index) - fold_ids_seen
        extra = fold_ids_seen - set(target_index)
        raise ValueError(f"OOF bank coverage mismatch: missing={len(missing)} extra={len(extra)}")
    if not fold_src:
        raise ValueError("no fold forecast banks supplied")

    out.mkdir(parents=True)
    arrays = {}
    shapes = source_meta["layouts"]
    for name, shape in shapes.items():
        src = np.load(base / name, mmap_mode="r", allow_pickle=False)
        if list(src.shape) != shape or src.dtype != np.float32:
            raise ValueError(f"source bank array schema mismatch: {name}")
        dst = np.lib.format.open_memmap(out / name, mode="w+", dtype=np.float32, shape=src.shape)
        for lo in range(0, len(src), 128):
            dst[lo:lo+128] = src[lo:lo+128]
        arrays[name] = dst

    copied = 0
    for fold, (fb, ids, meta) in sorted(fold_src.items()):
        agri = np.load(fb / "forecast_agri_raw.npy", mmap_mode="r", allow_pickle=False)
        cot = np.load(fb / "forecast_cot_log1p.npy", mmap_mode="r", allow_pickle=False)
        if agri.shape != (len(ids), 2, 16, 13, 3, 3) or cot.shape != (len(ids), 2, 16, 1, 3, 3):
            raise ValueError(f"fold {fold} prediction array schema mismatch")
        if not np.isfinite(agri).all() or not np.isfinite(cot).all():
            raise ValueError(f"fold {fold} forecast contains nonfinite values")
        for j, seq_id in enumerate(ids):
            bi = target_index[seq_id]
            arrays["forecast_agri_raw.npy"][bi] = agri[j]
            arrays["forecast_cot_log1p.npy"][bi] = cot[j]
            copied += 1
    for arr in arrays.values():
        arr.flush()
    with (out / "index.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=tuple(base_rows[0].keys()), lineterminator="\n")
        w.writeheader(); w.writerows(base_rows)
    new_meta = dict(source_meta)
    new_meta.update({
        "state": "COMPLETE_FORECAST_AGRI_R_TRAIN_VAL_BANK",
        "forecast_source": "train rows use date-heldout five-fold SimVP epoch-12 OOF forecasts; validation rows use the fixed full-train SimVP; both use observed 8-frame input only",
        "forecast_cot_source": "frozen R applied to predicted AGRI and deterministic geometry; no CPP",
        "in_sample_train_forecasts": False,
        "oof_train_sequences": copied,
        "oof_fold_count": len(fold_src),
        "oof_fold_model_hashes": {str(k): v[2]["fold_checkpoint_sha256"] for k, v in sorted(fold_src.items())},
        "source_bank_manifest_sha256": sha(base / "bank_manifest.json"),
        "test_payload_opened": False,
        "buffer_payload_opened": False,
        "future_observed_agri_read": False,
        "cpp_read": False,
    })
    array_hashes = {p.name: sha(p) for p in out.glob("*.npy")}
    new_meta["array_sha256"] = array_hashes
    new_meta["index_sha256"] = sha(out / "index.csv")
    (out / "bank_manifest.json").write_text(json.dumps(new_meta, indent=2) + "\n", encoding="utf-8")
    return {"state": "PASS_OOF_BANK_MERGE", "train_keys": len(target_index),
            "merged_oof_keys": copied, "fold_count": len(fold_src),
            "validation_rows_preserved_from_source": sum(r["split"] == "val" for r in base_rows),
            "test_payload_opened": False, "future_observed_agri_read": False, "cpp_read": False,
            "output_manifest_sha256": sha(out / "bank_manifest.json")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-bank", type=Path, required=True)
    p.add_argument("--fold-manifests", type=Path, required=True)
    p.add_argument("--fold-bank", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(merge(a.base_bank, a.fold_manifests, a.fold_bank, a.output), indent=2))


if __name__ == "__main__":
    main()
