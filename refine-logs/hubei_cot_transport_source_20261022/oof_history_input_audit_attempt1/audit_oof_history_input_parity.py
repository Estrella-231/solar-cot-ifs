#!/usr/bin/env python3
"""CPU-only train/validation parity audit for existing Hubei R fold inputs.

This audit never reads CPP targets/masks, GHI labels, test, or buffer payloads.
It checks that staged train/validation AGRI NPZ inputs agree with the full-R
pack and every date-held-out R-fold pack after reversing each pack's
train-only normalization.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

AGRI_CH = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
CHANNELS = ["C01", "C02", "C03", "C04", "C05", "C06", "C09", "C10", "C11", "C12", "C13", "C14", "C15", "cosSOZ", "cosRAA", "day_mask"]
EXPECTED_STAGED_SHA = "8b1f95c2a2cf1d66cabd11539b975fe3dccd486146bf3bec17fb18e537f72183"
EXPECTED_FOLD_MAP_SHA = "c86bd6713d5afea1d6864cc4ef6048a93b596051ea4358e7954f39c32b054122"
TOL = 5e-4


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_norm(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    names = raw.get("input_channels")
    if names != CHANNELS:
        raise ValueError(f"Unexpected channel order in {path}: {names}")
    mean = np.asarray(raw["mean"], dtype=np.float64).reshape(16, 1, 1)
    std = np.asarray(raw["std"], dtype=np.float64).reshape(16, 1, 1)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError(f"Invalid normalization in {path}")
    return mean, std, names


def denormalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if x.shape != (16, 16, 16) or not np.isfinite(x).all():
        raise ValueError(f"Invalid normalized input shape/content: {x.shape}")
    return x.astype(np.float64) * std + mean


def load_staged(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        required = {"patch", "cosSOZ", "cosRAA", "day_mask"}
        if not required.issubset(z.files):
            raise ValueError(f"Missing required train/val AGRI keys in {path}")
        patch = np.asarray(z["patch"], dtype=np.float32)
        if patch.ndim != 3 or patch.shape[0] < 15 or patch.shape[1:] != (16, 16):
            raise ValueError(f"Unexpected patch shape in {path}: {patch.shape}")
        fields = [np.asarray(z[k], dtype=np.float32) for k in ("cosSOZ", "cosRAA", "day_mask")]
        if any(a.shape != (16, 16) for a in fields):
            raise ValueError(f"Unexpected geometry shape in {path}")
        out = np.concatenate((patch[AGRI_CH], *(a[None] for a in fields)), axis=0)
        if out.shape != (16, 16, 16):
            raise ValueError(f"Unexpected channel tensor in {path}: {out.shape}")
        return out


def max_error(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(a) & np.isfinite(b)
    if not np.all(np.isfinite(a[13:])) or not np.all(np.isfinite(b[13:])):
        raise ValueError("Non-finite solar geometry/day-mask field")
    err = np.zeros(16, dtype=np.float64)
    n = np.zeros(16, dtype=np.int64)
    for c in range(16):
        good = finite[c]
        n[c] = int(good.sum())
        if n[c]:
            err[c] = float(np.max(np.abs(a[c][good] - b[c][good])))
    return err, n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--staged-rows", type=Path, required=True)
    p.add_argument("--fold-map", type=Path, required=True)
    p.add_argument("--full-pack", type=Path, required=True)
    p.add_argument("--fold-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    if sha(a.staged_rows) != EXPECTED_STAGED_SHA or sha(a.fold_map) != EXPECTED_FOLD_MAP_SHA:
        raise ValueError("Frozen train/validation staged index or fold map hash mismatch")

    staged_rows = read_csv(a.staged_rows)
    fold_rows = read_csv(a.fold_map)
    if not staged_rows or any(r.get("split") not in {"train", "val"} for r in staged_rows):
        raise ValueError("Staged index must contain only frozen train/validation rows")
    staged_by_time = {r["time_bjt"]: r for r in staged_rows}
    if len(staged_by_time) != len(staged_rows):
        raise ValueError("Duplicate staged train/validation time key")
    if not all({"time_bjt", "split", "path_agri"}.issubset(r) for r in staged_rows):
        raise ValueError("Unexpected staged index schema")

    assignment_rows = [r for r in fold_rows if r.get("split") == "train"]
    if len(assignment_rows) != len(fold_rows) or not all({"date_BJT", "split", "heldout_fold"}.issubset(r) for r in fold_rows):
        raise ValueError("Fold map must contain only train-date assignments")
    assignments = {r["date_BJT"].replace("-", ""): int(r["heldout_fold"]) for r in fold_rows}
    if len(assignments) != len(fold_rows) or set(assignments.values()) != set(range(5)):
        raise ValueError("Duplicate date assignments or unexpected fold IDs")
    train_rows = [r for r in staged_rows if r["split"] == "train"]
    val_rows = [r for r in staged_rows if r["split"] == "val"]
    train_dates = {r["time_bjt"][:8] for r in train_rows}
    val_dates = {r["time_bjt"][:8] for r in val_rows}
    missing_dates = sorted(train_dates - set(assignments))
    val_overlap = sorted(val_dates & set(assignments))
    if missing_dates or val_overlap:
        raise ValueError(f"Date assignment mismatch: missing={missing_dates[:5]}, val_overlap={val_overlap[:5]}")

    full_status = json.loads((a.full_pack / "pack_status.json").read_text(encoding="utf-8"))
    if full_status.get("state") != "HUBEI_R_TRAIN_VAL_CORRECT_UTC_PACK_BUILT_TEST_CLOSED":
        raise ValueError("Full R pack is not the expected train/validation-only corrected-UTC pack")
    if any(full_status.get(k) != 0 for k in ("test_payloads_read", "buffer_payloads_read")):
        raise ValueError("Full R pack records test/buffer payload reads")
    full_rows = read_csv(a.full_pack / "rows.csv")
    full_norm_mean, full_norm_std, _ = read_norm(a.full_pack / "norm.json")
    full_x = np.load(a.full_pack / "x_norm.npy", mmap_mode="r", allow_pickle=False)
    if full_x.shape != (len(full_rows), 16, 16, 16):
        raise ValueError(f"Unexpected full R input shape: {full_x.shape}")
    full_lookup = {r["time_bjt"]: int(r["index"]) for r in full_rows}
    if len(full_lookup) != len(full_rows):
        raise ValueError("Duplicate full R pack time key")
    if not set(full_lookup).issubset(staged_by_time):
        raise ValueError("Full R pack contains a key absent from staged train/validation AGRI index")

    staged_arrays: dict[str, np.ndarray] = {}
    for time_bjt in full_lookup:
        row = staged_by_time[time_bjt]
        path = Path(row["path_agri"])
        if not path.is_file():
            raise FileNotFoundError(path)
        staged_arrays[time_bjt] = load_staged(path)

    full_channel_max = np.zeros(16, dtype=np.float64)
    full_counts = np.zeros(16, dtype=np.int64)
    for r in full_rows:
        idx = int(r["index"])
        if idx < 0 or idx >= len(full_rows):
            raise ValueError("Invalid full R pack row index")
        raw = denormalize(full_x[idx], full_norm_mean, full_norm_std)
        err, count = max_error(raw, staged_arrays[r["time_bjt"]])
        full_channel_max = np.maximum(full_channel_max, err)
        full_counts += count

    fold_results = []
    for fold in range(5):
        pack = a.fold_root / f"fold_{fold}"
        status = json.loads((pack / "pack_status.json").read_text(encoding="utf-8"))
        if status.get("state") != "HUBEI_R_TRAIN_VAL_CORRECT_UTC_PACK_BUILT_TEST_CLOSED":
            raise ValueError(f"Fold {fold} pack state mismatch")
        if any(status.get(k) != 0 for k in ("test_payloads_read", "buffer_payloads_read")):
            raise ValueError(f"Fold {fold} pack records test/buffer payload reads")
        if status.get("heldout_train_date_overlap") != 0:
            raise ValueError(f"Fold {fold} has held-out train-date overlap")
        rows = read_csv(pack / "rows.csv")
        mean, std, _ = read_norm(pack / "norm.json")
        x = np.load(pack / "x_norm.npy", mmap_mode="r", allow_pickle=False)
        if x.shape != (len(rows), 16, 16, 16):
            raise ValueError(f"Unexpected fold {fold} R input shape: {x.shape}")
        seen = set()
        channel_max = np.zeros(16, dtype=np.float64)
        counts = np.zeros(16, dtype=np.int64)
        by_split = Counter()
        for r in rows:
            t = r["time_bjt"]
            if t in seen:
                raise ValueError(f"Duplicate key in fold {fold}: {t}")
            seen.add(t)
            if t not in full_lookup or t not in staged_by_time:
                raise ValueError(f"Fold {fold} row missing from full/staged index: {t}")
            if r["split"] == "train":
                if assignments[t[:8]] == fold:
                    raise ValueError(f"Held-out train date leaked into fold {fold}: {t}")
            elif r["split"] != "val" or t[:8] not in val_dates:
                raise ValueError(f"Unexpected split/date in fold {fold}: {r}")
            by_split[r["split"]] += 1
            raw = denormalize(x[int(r["index"])], mean, std)
            err, count = max_error(raw, staged_arrays[t])
            channel_max = np.maximum(channel_max, err)
            counts += count
        fold_results.append({
            "fold": fold,
            "rows": len(rows),
            "rows_by_split": dict(by_split),
            "heldout_train_date_overlap": status.get("heldout_train_date_overlap"),
            "max_abs_raw_input_error_by_channel": channel_max.tolist(),
            "max_abs_geometry_day_mask_error": float(channel_max[13:].max()),
            "finite_values_compared_by_channel": counts.tolist(),
        })

    max_all = max(float(full_channel_max.max()), *(float(np.max(x["max_abs_raw_input_error_by_channel"])) for x in fold_results))
    max_geom = max(float(full_channel_max[13:].max()), *(x["max_abs_geometry_day_mask_error"] for x in fold_results))
    if max_all > TOL or max_geom > TOL:
        raise ValueError(f"Input parity failed: max_all={max_all}, max_geometry={max_geom}, tolerance={TOL}")

    train_rows_by_fold = Counter(assignments[r["time_bjt"][:8]] for r in train_rows)
    extra_fold_dates = sorted(set(assignments) - train_dates)
    result = {
        "state": "PASS_EXISTING_R_FOLD_INPUTS_MATCH_STAGED_TRAIN_VAL_AGRI_TEST_CLOSED",
        "purpose": "Read-only compatibility gate for possible OOF historical-COT inference; no model training or COT forecasting was run.",
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "cpp_target_or_quality_payloads_read": 0,
        "ghi_labels_read": 0,
        "staged_rows_sha256": sha(a.staged_rows),
        "fold_map_sha256": sha(a.fold_map),
        "full_pack_status_sha256": sha(a.full_pack / "pack_status.json"),
        "full_r_input_channel_max_abs_error": full_channel_max.tolist(),
        "full_r_input_max_abs_geometry_error": float(full_channel_max[13:].max()),
        "staged_rows": len(staged_rows),
        "staged_train_rows": len(train_rows),
        "staged_val_rows": len(val_rows),
        "staged_train_dates": len(train_dates),
        "staged_val_dates": len(val_dates),
        "date_fold_manifest_rows": len(assignments),
        "fold_date_counts": dict(sorted(Counter(assignments.values()).items())),
        "staged_train_rows_by_fold": dict(sorted(train_rows_by_fold.items())),
        "staged_train_dates_missing_from_fold_map": missing_dates,
        "staged_val_dates_in_fold_map": val_overlap,
        "fold_manifest_dates_without_staged_agri_rows": extra_fold_dates,
        "full_r_pack_rows_checked": len(full_rows),
        "full_r_pack_split_counts": dict(Counter(r["split"] for r in full_rows)),
        "folds": fold_results,
        "max_abs_raw_input_error_all_packs": max_all,
        "max_abs_geometry_day_mask_error_all_packs": max_geom,
        "absolute_tolerance": TOL,
        "r_model_source_sha256": "8af2d0e101b492bf2096223fe788b7f3a8a9b5ece2544db45903b95397620e05",
        "existing_train_bank_oof_state": "false; this audit does not regenerate it",
        "note": "Fold models are metadata-compatible and their pack inputs match the staged AGRI arrays. A future OOF bank still requires a separately approved run; current station-patch validation is exposed/frozen and the full-region CPP source gate remains blocked.",
    }
    a.out.parent.mkdir(parents=True, exist_ok=False)
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("state", "staged_rows", "staged_train_rows", "staged_val_rows", "full_r_pack_rows_checked", "max_abs_raw_input_error_all_packs", "max_abs_geometry_day_mask_error_all_packs", "test_payloads_read", "buffer_payloads_read")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
