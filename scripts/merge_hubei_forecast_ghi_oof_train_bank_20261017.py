#!/usr/bin/env python3
"""Merge five date-held-out SimVP -> fold-R forecast banks into canonical train order.

Only sequence keys are read from the full in-sample bank. Forecast tensors are
copied exclusively from the matching held-out S and R fold products. Test and
buffer payloads are never opened.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

EXPECTED_INDEX_SHA = "4eddd1649e90cd1a76c48264e972561338953d791cca021b3eeadc1101d22c62"
EXPECTED_SPLIT_SHA = "6ef8cbc7d2bb7d1b1d3defdfcba473cb171e06db0115c8206f7c289d8bad19c6"
EXPECTED_FOLD_SHA = "c86bd6713d5afea1d6864cc4ef6048a93b596051ea4358e7954f39c32b054122"
EXPECTED_R_PATHS_SHA = "3bae20a9607b941e544199b9f2900414b266ad11456e82d9bf51e7bd78c972f8"
PRED_TAIL = (16, 13, 16, 16)
GEOM_TAIL = (16, 3, 16, 16)
COT_TAIL = (16, 1, 16, 16)
KEY_FIELDS = ("sample_index", "split", "init_time_bjt", "target_times_bjt", "data_paths")
FULL_VAL_COUNT = 407
FULL_VAL_ELEMENTS = FULL_VAL_COUNT * 16 * 13 * 256 * 256
EXPECTED_STAGE_MANIFEST_SHA = "73e0f91955a7e8c07d69a1e6d79c900275ff000f350a826fb2fa04e79e47571a"
EXPECTED_CHANNEL_INDICES = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
EXPECTED_LEAD_MINUTES = list(range(15, 241, 15))
EXPECTED_R_CHANNELS = ["C01", "C02", "C03", "C04", "C05", "C06", "C09", "C10",
                       "C11", "C12", "C13", "C14", "C15", "cosSOZ", "cosRAA", "day_mask"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def valid_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def require_sha(value: object, label: str) -> str:
    if not valid_sha256(value):
        raise ValueError(f"{label} must be a non-empty lowercase SHA-256")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_single_fold_oof_receipt(audit: dict, fold: int, s_report: dict,
                                     s_report_sha: str, index_sha: str,
                                     sdir: Path) -> None:
    """Validate the explicit sidecar for the currently emitted one-fold S bank."""
    expected_shapes = {
        "pred_agri_patch": ["N", *PRED_TAIL],
        "true_agri_patch": ["N", *PRED_TAIL],
        "future_geometry_patch": ["N", *GEOM_TAIL],
    }
    s_count = s_report.get("split_counts", {}).get("val")
    checkpoint_sha = require_sha(s_report.get("checkpoint_sha256"), f"fold{fold} S checkpoint")
    norm_sha = require_sha(s_report.get("normalization_sha256"), f"fold{fold} S normalization")
    require_sha(index_sha, f"fold{fold} OOF index")
    require_sha(s_report_sha, f"fold{fold} S report")
    validate_closed_counters(s_report, f"Fold {fold} S OOF bank report")
    validate_closed_counters(audit, f"Fold {fold} S OOF audit")
    if (audit.get("state") != "HUBEI_SIMVP_SINGLE_FOLD_OOF_BANK_INTEGRITY_PASS_TEST_CLOSED"
            or audit.get("count") != s_count
            or audit.get("source_index_sha256") != index_sha
            or audit.get("source_bank_report_sha256") != s_report_sha
            or audit.get("checkpoint_sha256") != checkpoint_sha
            or audit.get("normalization_sha256") != norm_sha
            or s_report.get("local_stage", {}).get("stage_manifest_sha256") != EXPECTED_STAGE_MANIFEST_SHA
            or audit.get("test_payloads_read") != 0
            or audit.get("buffer_payloads_read") != 0
            or audit.get("test_or_buffer_rows") != 0
            or audit.get("ghi_labels_read") is not False
            or audit.get("channel_indices") != EXPECTED_CHANNEL_INDICES
            or audit.get("lead_minutes") != EXPECTED_LEAD_MINUTES
            or audit.get("patch_origin_crop_256") != [103, 118]
            or audit.get("patch_size") != 16
            or audit.get("finite") is not True
            or audit.get("keys_exactly_match_ordered_oof_index") is not True
            or audit.get("shapes") != {
                key: [s_count, *shape[1:]] for key, shape in expected_shapes.items()
            }):
        raise ValueError(f"Fold {fold} S OOF integrity receipt does not bind this bank/index/checkpoint")
    actual_arrays = {
        "pred_agri_patch": sdir / "val" / "pred_agri_patch.npy",
        "true_agri_patch": sdir / "val" / "true_agri_patch.npy",
        "future_geometry_patch": sdir / "val" / "future_geometry_patch.npy",
    }
    recorded_hashes = audit.get("arrays_sha256")
    if not isinstance(recorded_hashes, dict):
        raise ValueError(f"Fold {fold} S OOF receipt lacks actual-array hashes")
    for name, path in actual_arrays.items():
        if not path.is_file() or recorded_hashes.get(name) != sha(path):
            raise ValueError(f"Fold {fold} S OOF array hash mismatch: {name}")


def validate_full_validation_audit(audit: dict, fold: int, s_report: dict,
                                   validation_index: Path,
                                   canonical_rows: list[dict[str, str]],
                                   day_folds: dict[str, int],
                                   shard_dirs: dict[int, Path]) -> None:
    """Require selected S checkpoint to have a complete, test-closed val reload receipt."""
    expected_state = f"HUBEI_SIMVP_FOLD{fold}_CHECKPOINT_RELOAD_VALIDATION_AUDIT_PASSED_TEST_CLOSED"
    metrics = audit.get("full_field_validation_metrics", {})
    validate_closed_counters(audit, f"Fold {fold} full-validation audit")
    shards = audit.get("shards", [])
    shard_ids = [item.get("shard_index") for item in shards]
    shard_count = sum(item.get("sequence_count", 0) for item in shards)
    metrics_ok = all(
        isinstance(metrics.get(name), (int, float))
        and np.isfinite(metrics[name])
        and metrics[name] >= 0
        for name in ("normalized_mse", "normalized_mae")
    )
    index_sha = sha(validation_index)
    checkpoint_sha = require_sha(s_report.get("checkpoint_sha256"), f"fold{fold} S checkpoint")
    norm_sha = require_sha(s_report.get("normalization_sha256"), f"fold{fold} S normalization")
    require_sha(index_sha, f"fold{fold} fixed validation index")
    actual_rows = read_csv(validation_index)
    expected_rows: list[dict[str, str]] = []
    for row in canonical_rows:
        split = row.get("split")
        paths = row.get("data_paths", "").split("|")
        if split not in {"train", "val"} or len(paths) != 24:
            raise ValueError("Canonical frozen index has an unexpected split or sequence length")
        day = Path(paths[7]).stem[:8]
        if split == "val" or (split == "train" and day_folds.get(day) != fold):
            expected_rows.append({"split": split, "data_paths": row["data_paths"],
                                  "aux_paths": row.get("aux_paths", "")})
    if actual_rows != expected_rows:
        raise ValueError(f"Fold {fold} validation-reload index differs from the frozen canonical fold index")
    val_rows = [row for row in actual_rows if row.get("split") == "val"]
    if (audit.get("status") != expected_state
            or audit.get("fold_id") != fold
            or audit.get("validation_sequences_inferred") != FULL_VAL_COUNT
            or audit.get("unique_validation_key_coverage") != FULL_VAL_COUNT
            or audit.get("test_payloads_read") != 0
            or audit.get("buffer_payloads_read") != 0
            or audit.get("checkpoint_sha256") != checkpoint_sha
            or audit.get("normalization_sha256") != norm_sha
            or audit.get("stage_manifest_sha256") != EXPECTED_STAGE_MANIFEST_SHA
            or audit.get("index_sha256") != index_sha
            or audit.get("channel_indices") != EXPECTED_CHANNEL_INDICES
            or audit.get("lead_minutes") != EXPECTED_LEAD_MINUTES
            or audit.get("patch_origin_crop_256") != [103, 118]
            or audit.get("patch_size") != 16
            or len(val_rows) != FULL_VAL_COUNT
            or not metrics_ok
            or metrics.get("sequence_count") != FULL_VAL_COUNT
            or metrics.get("element_count") != FULL_VAL_ELEMENTS
            or not isinstance(audit.get("shard_count"), int)
            or audit.get("shard_count") != len(shards)
            or audit.get("shard_count") < 1
            or audit.get("shard_ids") != list(range(len(shards)))
            or shard_ids != list(range(len(shards)))
            or shard_count != FULL_VAL_COUNT):
        raise ValueError(f"Fold {fold} full-validation reload audit missing or does not bind the S checkpoint")
    required_shard_fields = {
        "prediction_sha256", "truth_sha256", "geometry_sha256", "keys_sha256",
        "bank_report_sha256",
    }
    for shard in shards:
        if (shard.get("channel_indices") != EXPECTED_CHANNEL_INDICES
                or shard.get("lead_minutes") != EXPECTED_LEAD_MINUTES
                or shard.get("patch_origin_crop_256") != [103, 118]
                or shard.get("patch_size") != 16
                or shard.get("test_payloads_read") != 0
                or shard.get("buffer_payloads_read") != 0
                or shard.get("test_or_buffer_rows") != 0
                or any(not valid_sha256(shard.get(name)) for name in required_shard_fields)):
            raise ValueError(f"Fold {fold} full-validation shard metadata/hash contract failed")
        shard_id = shard["shard_index"]
        shard_dir = shard_dirs.get(shard_id)
        if shard_dir is None:
            raise ValueError(f"Fold {fold} actual validation shard directory {shard_id} was not supplied")
        report_path = shard_dir / "BANK_REPORT.json"
        pred_path = shard_dir / "val" / "pred_agri_patch.npy"
        truth_path = shard_dir / "val" / "true_agri_patch.npy"
        geometry_path = shard_dir / "val" / "future_geometry_patch.npy"
        keys_path = shard_dir / "val" / "keys.csv"
        for path in (report_path, pred_path, truth_path, geometry_path, keys_path):
            if not path.is_file():
                raise FileNotFoundError(f"Fold {fold} validation shard artifact missing: {path}")
        bank_report = read_json(report_path)
        validate_closed_counters(bank_report, f"Fold {fold} validation shard {shard_id} report")
        if (bank_report.get("inferred_splits") != ["val"]
                or bank_report.get("shard_index") != shard_id
                or bank_report.get("shard_count") != len(shards)
                or bank_report.get("channel_indices") != EXPECTED_CHANNEL_INDICES
                or bank_report.get("lead_minutes") != EXPECTED_LEAD_MINUTES
                or bank_report.get("patch_origin_crop_256") != [103, 118]
                or bank_report.get("patch_size") != 16
                or bank_report.get("checkpoint_sha256") != checkpoint_sha
                or bank_report.get("normalization_sha256") != norm_sha
                or bank_report.get("index_sha256") != index_sha
                or bank_report.get("local_stage", {}).get("stage_manifest_sha256") != EXPECTED_STAGE_MANIFEST_SHA
                or bank_report.get("sequences_written_by_split", {}).get("val") != shard.get("sequence_count")
                or len(read_csv(keys_path)) != shard.get("sequence_count")
                or shard.get("keys_sha256") != sha(keys_path)):
            raise ValueError(f"Fold {fold} actual validation shard report/key contract failed for shard {shard_id}")
        actual_hashes = {
            "prediction_sha256": sha(pred_path),
            "truth_sha256": sha(truth_path),
            "geometry_sha256": sha(geometry_path),
            "bank_report_sha256": sha(report_path),
        }
        if any(shard.get(name) != value for name, value in actual_hashes.items()):
            raise ValueError(f"Fold {fold} validation shard {shard_id} receipt hashes do not match actual files")


def validate_closed_counters(report: dict, label: str) -> None:
    """Require explicit, non-defaulted test and buffer closure counters."""
    if (report.get("test_payloads_read") != 0
            or report.get("buffer_payloads_read") != 0
            or report.get("test_or_buffer_rows") != 0):
        raise ValueError(f"{label} must explicitly report test/buffer reads and rows as zero")


def validate_full_forecast_bank_closure(bank_dir: Path, report: dict,
                                        report_path: Path, stage_report_path: Path,
                                        canonical_index: Path) -> dict:
    """Independently close a legacy full-train/val bank whose report predates buffer counters.

    This is metadata-only: it reads the frozen sequence index, stage manifest,
    and train/val key CSVs, never any test or buffer payload. It does not edit or
    synthesize fields in the original bank report.
    """
    stage = read_json(stage_report_path)
    if (stage.get("status") != "LOCAL_STAGE_COMPLETE_TRAINVAL_ONLY_TEST_CLOSED"
            or stage.get("source_index_sha256") != EXPECTED_INDEX_SHA
            or stage.get("sequences") != {"train": 4086, "val": 407}
            or stage.get("test_sequences") != 0
            or stage.get("test_or_buffer_payloads_staged") != 0
            or stage.get("stage_root") != report.get("stage_root")
            or not valid_sha256(stage.get("stage_manifest_sha256"))
            or report.get("source_stage_report_sha256") != sha(stage_report_path)
            or report.get("source_stage_index_sha256") != stage.get("stage_index_sha256")
            or report.get("source_stage_manifest_sha256") != stage.get("stage_manifest_sha256")):
        raise ValueError("Full forecast bank does not bind a closed train/validation-only stage")
    stage_manifest = Path(stage.get("stage_manifest", ""))
    stage_index = Path(stage.get("stage_index", ""))
    if (not stage_manifest.is_file() or not stage_index.is_file()
            or sha(stage_manifest) != stage.get("stage_manifest_sha256")
            or sha(stage_index) != stage.get("stage_index_sha256")):
        raise ValueError("Full forecast bank stage manifest/index is missing or hash-mismatched")

    canonical_rows = read_csv(canonical_index)
    stage_rows = read_csv(stage_index)
    from collections import Counter
    expected_counts = Counter({"train": 4086, "val": 407})
    if (Counter(row.get("split") for row in canonical_rows) != expected_counts
            or Counter(row.get("split") for row in stage_rows) != expected_counts
            or len(canonical_rows) != 4493 or len(stage_rows) != 4493
            or any(row.get("split") not in {"train", "val"} for row in canonical_rows + stage_rows)):
        raise ValueError("Frozen canonical/stage index contains unexpected test/buffer rows or counts")

    # The stage manifest lists every copied train/validation payload. Bind its
    # destination set to the complete localized index without loading contents.
    stage_destinations = {row.get("destination", "") for row in read_csv(stage_manifest)}
    indexed_destinations = set()
    for row in stage_rows:
        data_paths = row.get("data_paths", "").split("|")
        aux_paths = row.get("aux_paths", "").split("|")
        if len(data_paths) != 24 or len(aux_paths) != 24:
            raise ValueError("Localized stage index has an unexpected sequence length")
        indexed_destinations.update(data_paths)
        indexed_destinations.update(aux_paths)
    if not stage_destinations or stage_destinations != indexed_destinations:
        raise ValueError("Stage payload manifest does not exactly match train/validation index paths")

    bank_children = {p.name for p in bank_dir.iterdir()}
    if bank_children != {"BANK_REPORT.json", "train", "val"}:
        raise ValueError("Full forecast bank contains an unexpected split directory")
    splits = report.get("splits", {})
    if (set(splits) != {"train", "val"}
            or splits.get("train", {}).get("sequences") != 4086
            or splits.get("val", {}).get("sequences") != 407
            or report.get("test_payloads_read") != 0):
        raise ValueError("Full forecast bank report does not explicitly close test and match train/val counts")
    key_counts = {}
    key_hashes = {}
    for split, expected_count in (("train", 4086), ("val", 407)):
        key_path = bank_dir / split / "keys.csv"
        rows = read_csv(key_path)
        if (len(rows) != expected_count
                or any(row.get("split") != split for row in rows)
                or [int(row.get("sample_index", -1)) for row in rows] != list(range(expected_count))):
            raise ValueError(f"Full forecast bank {split} keys do not match its declared split")
        key_counts[split] = len(rows)
        key_hashes[split] = sha(key_path)
    return {
        "state": "HUBEI_FORECAST_BANK_TRAIN_VAL_CLOSURE_AUDIT_TEST_BUFFER_CLOSED",
        "scope": "metadata-only source closure; no test/buffer payloads opened",
        "bank_report_sha256": sha(report_path),
        "stage_report_sha256": sha(stage_report_path),
        "stage_manifest_sha256": sha(stage_manifest),
        "stage_index_sha256": sha(stage_index),
        "canonical_index_sha256": sha(canonical_index),
        "splits": {"train": 4086, "val": 407, "test": 0, "buffer": 0},
        "bank_key_counts": key_counts,
        "bank_key_sha256": key_hashes,
        "bank_directories": sorted(bank_children),
        "stage_manifest_paths_exactly_match_trainval_index": True,
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "test_or_buffer_rows": 0,
    }


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        if not fields:
            raise ValueError(f"Missing CSV header: {path}")
        return fields, list(reader)


def validate_frozen_fold_r_manifest(r_pack: Path, fold: int) -> dict[str, str]:
    """Bind a fold R pack to the frozen global split, fold map and raw R path list."""
    fold_dir = r_pack.parent
    folds_dir = fold_dir.parent
    manifest_root = folds_dir.parent
    input_dir = manifest_root / "input"
    day_path = input_dir / "hubei_day_assignment_20261016.csv"
    fold_path = input_dir / "hubei_oof_train_date_fold_20261016.csv"
    raw_r_path = input_dir / "hubei_r_paths_trainval_20261016.csv"
    r_manifest = fold_dir / "r_trainval_manifest.csv"
    fold_report_path = fold_dir / "report.json"
    root_report_path = folds_dir / "root_report.json"
    for path in (day_path, fold_path, raw_r_path, r_manifest, fold_report_path, root_report_path):
        if not path.is_file():
            raise FileNotFoundError(f"Fold {fold} frozen R manifest provenance missing: {path}")
    day_sha, fold_sha, raw_r_sha = sha(day_path), sha(fold_path), sha(raw_r_path)
    if (day_sha != EXPECTED_SPLIT_SHA or fold_sha != EXPECTED_FOLD_SHA
            or raw_r_sha != EXPECTED_R_PATHS_SHA):
        raise ValueError(f"Fold {fold} R source manifests do not match frozen global hashes")
    root_report, fold_report = read_json(root_report_path), read_json(fold_report_path)
    if (root_report.get("state") != "HUBEI_FIVE_FOLD_S_R_MANIFESTS_BUILT_METADATA_ONLY"
            or root_report.get("source_index_sha256") != EXPECTED_INDEX_SHA
            or root_report.get("day_assignment_sha256") != day_sha
            or root_report.get("fold_assignment_sha256") != fold_sha
            or root_report.get("r_path_manifest_sha256") != raw_r_sha
            or root_report.get("fold_count") != 5
            or fold_report.get("state") != "METADATA_ONLY_FOLD_MANIFESTS_PASS_TEST_BUFFER_CLOSED"
            or fold_report.get("fold") != fold
            or fold_report.get("r_trainval_manifest_sha256") != sha(r_manifest)):
        raise ValueError(f"Fold {fold} manifest build report does not bind frozen source manifests")

    _, day_rows = read_csv_rows(day_path)
    _, fold_rows = read_csv_rows(fold_path)
    raw_fields, raw_rows = read_csv_rows(raw_r_path)
    manifest_fields, manifest_rows = read_csv_rows(r_manifest)
    if raw_fields != manifest_fields or not {"time_bjt", "split", "path_agri", "path_cpp_fixed_v2"}.issubset(raw_fields):
        raise ValueError(f"Fold {fold} R manifest schema differs from frozen source")
    split_by_date = {row["date_BJT"]: row["split"] for row in day_rows}
    fold_by_date = {row["date_BJT"]: int(row["heldout_fold"]) for row in fold_rows}
    if len(split_by_date) != len(day_rows) or len(fold_by_date) != len(fold_rows):
        raise ValueError("Duplicate frozen global date assignment")
    if set(fold_by_date) != {day for day, split in split_by_date.items() if split == "train"}:
        raise ValueError("Frozen fold assignment does not exactly cover global train dates")
    expected_rows: list[dict[str, str]] = []
    for row in raw_rows:
        stamp = row["time_bjt"]
        if len(stamp) != 12 or not stamp.isdigit():
            raise ValueError(f"Invalid frozen R timestamp: {stamp!r}")
        day = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
        if row["split"] not in {"train", "val"} or split_by_date.get(day) != row["split"]:
            raise ValueError(f"Frozen R source row split/date mismatch at {stamp}")
        if not (row["split"] == "train" and fold_by_date[day] == fold):
            expected_rows.append(row)
    if manifest_rows != expected_rows:
        raise ValueError(f"Fold {fold} R train/validation manifest differs from exact frozen derivation")
    pack_status = read_json(r_pack / "pack_status.json")
    rows_path = r_pack / "rows.csv"
    pack_fields, pack_rows = read_csv_rows(rows_path)
    expected_pack_fields = ["index", "split", "time_bjt", "path_agri", "path_cpp_fixed_v2"]
    if (pack_status.get("source_split_sha256") != EXPECTED_FOLD_SHA
            or pack_status.get("path_manifest_sha256") != sha(r_manifest)
            or pack_fields != expected_pack_fields
            or len(pack_rows) != len(manifest_rows)):
        raise ValueError(f"Fold {fold} R pack does not bind the frozen fold assignment/manifest")
    # The pack builder groups rows by split, whereas the frozen manifest is
    # chronological. Validate exact key-set equality and canonical row indices,
    # not incidental CSV ordering.
    from collections import Counter
    pack_key_fields = ("split", "time_bjt", "path_agri", "path_cpp_fixed_v2")
    expected_keys = Counter(tuple(row[field] for field in pack_key_fields)
                            for row in manifest_rows)
    actual_keys = Counter(tuple(row[field] for field in pack_key_fields)
                          for row in pack_rows)
    if (len(expected_keys) != len(manifest_rows)
            or len(actual_keys) != len(pack_rows)
            or any(row.get("index") != str(i) for i, row in enumerate(pack_rows))
            or actual_keys != expected_keys):
        raise ValueError(f"Fold {fold} R pack rows do not exactly cover the frozen manifest keys")
    return {
        "source_day_assignment_sha256": day_sha,
        "source_fold_assignment_sha256": fold_sha,
        "source_r_path_manifest_sha256": raw_r_sha,
        "fold_r_manifest_sha256": sha(r_manifest),
        "fold_r_manifest_report_sha256": sha(fold_report_path),
        "fold_manifest_root_report_sha256": sha(root_report_path),
    }


def validate_fold_r_provenance(c_report: dict, fold: int, r_run: Path,
                               r_pack: Path, model_source: Path,
                               oof_days: set[str], day_folds: dict[str, int]) -> dict[str, str]:
    """Recompute and verify the complete frozen-R source chain for one COT bank."""
    status_path = r_run / "status.json"
    metadata_path = r_run / "run_metadata.json"
    audit_path = r_run / "audit_report.json"
    checkpoint_path = r_run / "best.pt"
    norm_path = r_pack / "norm.json"
    rows_path = r_pack / "rows.csv"
    pack_status_path = r_pack / "pack_status.json"
    for path in (status_path, metadata_path, audit_path, checkpoint_path,
                 norm_path, rows_path, pack_status_path, model_source):
        if not path.is_file():
            raise FileNotFoundError(f"fold{fold} R provenance file missing: {path}")
    status, metadata, audit = read_json(status_path), read_json(metadata_path), read_json(audit_path)
    norm, pack_status = read_json(norm_path), read_json(pack_status_path)
    hashes = {
        "run_status_sha256": sha(status_path),
        "run_metadata_sha256": sha(metadata_path),
        "run_audit_sha256": sha(audit_path),
        "checkpoint_sha256": sha(checkpoint_path),
        "model_source_sha256": sha(model_source),
        "pack_norm_sha256": sha(norm_path),
        "pack_rows_sha256": sha(rows_path),
        "pack_status_sha256": sha(pack_status_path),
    }
    for key, value in hashes.items():
        require_sha(value, f"fold{fold} R {key}")
    expected = {
        "fold_R_run_audit_sha256": hashes["run_audit_sha256"],
        "fold_R_checkpoint_sha256": hashes["checkpoint_sha256"],
        "fold_R_model_source_sha256": hashes["model_source_sha256"],
        "fold_R_pack_norm_sha256": hashes["pack_norm_sha256"],
        "fold_R_pack_rows_sha256": hashes["pack_rows_sha256"],
    }
    if any(c_report.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Fold {fold} COT report does not bind the actual R lineage files")
    frozen_manifest_hashes = validate_frozen_fold_r_manifest(r_pack, fold)
    if (status.get("state") != "COMPLETE_VALIDATION_ONLY_TEST_CLOSED"
            or status.get("test_payloads_read") != 0
            or audit.get("state") != "HUBEI_OOF_R_FOLD_COMPLETION_AUDIT_PASSED_TEST_CLOSED"
            or audit.get("test_payloads_read") != 0
            or audit.get("fold_seed") != 42
            or metadata.get("seed") != 42
            or audit.get("best_checkpoint_sha256") != hashes["checkpoint_sha256"]
            or audit.get("pack_sha256") != metadata.get("source_pack_hashes")
            or metadata.get("model_source_sha256") != hashes["model_source_sha256"]
            or metadata.get("normalization") != norm
            or norm.get("input_channels") != EXPECTED_R_CHANNELS
            or norm.get("fit_split") != "train only"
            or norm.get("test_used") is not False
            or metadata.get("pack_status_sha256") != hashes["pack_status_sha256"]
            or pack_status.get("state") != "HUBEI_R_TRAIN_VAL_PACK_BUILT_TEST_CLOSED"
            or pack_status.get("test_cpp_payloads_read") != 0
            or pack_status.get("test_agri_payloads_read") != 0
            or pack_status.get("source_split_sha256") != EXPECTED_FOLD_SHA):
        raise ValueError(f"Fold {fold} R status/audit/metadata/norm contract failed")
    pack_source_hashes = metadata.get("source_pack_hashes")
    if not isinstance(pack_source_hashes, dict) or not pack_source_hashes:
        raise ValueError(f"Fold {fold} R metadata lacks source pack hashes")
    for filename, expected_sha in pack_source_hashes.items():
        actual_path = r_pack / filename
        if not actual_path.is_file() or not valid_sha256(expected_sha) or sha(actual_path) != expected_sha:
            raise ValueError(f"Fold {fold} R training-pack lineage mismatch: {filename}")
    train_dates = set()
    all_dates = set()
    for row in read_csv(rows_path):
        if row.get("split") not in {"train", "val"}:
            raise ValueError(f"Fold {fold} R pack has a non-train/val row")
        stamp = row.get("time_bjt", "").replace("-", "").replace(":", "").replace(" ", "")
        if len(stamp) < 8 or not stamp[:8].isdigit():
            raise ValueError(f"Fold {fold} R row has invalid date")
        day = stamp[:8]
        all_dates.add(day)
        if row.get("split") == "train":
            if day_folds.get(day) is None:
                raise ValueError(f"Fold {fold} R training date is absent from frozen fold map: {day}")
            train_dates.add(day)
    if (not train_dates or train_dates & oof_days or all_dates & oof_days
            or any(day_folds.get(day) == fold for day in train_dates)):
        raise ValueError(f"Fold {fold} R train/validation dates overlap S held-out OOF dates")
    hashes["run_root"] = str(r_run)
    hashes["pack_root"] = str(r_pack)
    hashes["model_source_path"] = str(model_source)
    hashes.update(frozen_manifest_hashes)
    return hashes


def sequence_identity(row: dict[str, str]) -> tuple[str, str, str]:
    """Stable sequence identity independent of fold-local sample_index/split."""
    if not all(row.get(k) for k in KEY_FIELDS):
        raise ValueError("Sequence key is missing a required canonical field")
    init = datetime.strptime(row["init_time_bjt"], "%Y%m%d%H%M")
    targets = row["target_times_bjt"].split("|")
    if len(targets) != 16:
        raise ValueError("Every forecast sequence must contain exactly 16 target times")
    for lead_i, value in enumerate(targets):
        target = datetime.strptime(value, "%Y%m%d%H%M")
        if target != init + timedelta(minutes=15 * (lead_i + 1)):
            raise ValueError("Forecast sequence target is not init +15..+240 minutes")
        if target.date() != init.date():
            raise ValueError("Forecast sequence crosses a BJT date boundary")
    return row["init_time_bjt"], row["target_times_bjt"], row["data_paths"]


def date_fold_map(path: Path) -> dict[str, int]:
    rows = read_csv(path)
    if not rows or not {"date_BJT", "split", "heldout_fold"}.issubset(rows[0]):
        raise ValueError("Unexpected frozen fold-assignment schema")
    result: dict[str, int] = {}
    for row in rows:
        if row["split"] != "train":
            continue
        day = row["date_BJT"].replace("-", "")
        fold = int(row["heldout_fold"])
        if not 0 <= fold < 5 or day in result:
            raise ValueError(f"Invalid or duplicate fold assignment for {day}")
        result[day] = fold
    return result


def canonical_key_from_index(row: dict[str, str], sample_index: int) -> dict[str, str]:
    if row.get("split") != "train":
        raise ValueError("Canonical sequence index must contain train rows only for this merge")
    paths = row.get("data_paths", "").split("|")
    if len(paths) != 24:
        raise ValueError(f"Canonical sequence {sample_index} must list exactly 24 AGRI source paths")
    from hubei_forecast_time_keys import canonical_bjt_stamps
    source_stamps = [Path(path).stem for path in paths]
    stamps = canonical_bjt_stamps(row)
    if any(len(stamp) != 12 or not stamp.isdigit() for stamp in stamps):
        raise ValueError(f"Canonical sequence {sample_index} has invalid BJT path timestamps")
    return {"sample_index": str(sample_index), "split": "train",
            "init_time_bjt": stamps[7], "target_times_bjt": "|".join(stamps[8:24]),
            "data_paths": "|".join(source_stamps)}


def canonicalize_keys(canonical: list[dict[str, str]], fold_rows: dict[int, list[dict[str, str]]],
                      day_folds: dict[str, int], expected_count: int = 4086) -> dict[int, dict[int, int]]:
    """Return canonical-index -> fold-local-index mappings; require exact coverage."""
    if len(canonical) != expected_count:
        raise ValueError(f"Expected {expected_count} canonical train sequences, got {len(canonical)}")
    identities: dict[tuple[str, str, str], int] = {}
    for i, row in enumerate(canonical):
        if int(row.get("sample_index", -1)) != i or row.get("split") != "train":
            raise ValueError(f"Noncanonical train key at row {i}")
        identity = sequence_identity(row)
        if identity in identities:
            raise ValueError(f"Duplicate canonical sequence key at row {i}")
        day = datetime.strptime(row["init_time_bjt"], "%Y%m%d%H%M").strftime("%Y%m%d")
        if day_folds.get(day) is None:
            raise ValueError(f"Canonical train date {day} has no frozen OOF fold")
        identities[identity] = i

    maps: dict[int, dict[int, int]] = {}
    covered: set[int] = set()
    for fold in range(5):
        rows = fold_rows[fold]
        local_seen: set[tuple[str, str, str]] = set()
        mapping: dict[int, int] = {}
        for local_i, row in enumerate(rows):
            if int(row.get("sample_index", -1)) != local_i or row.get("split") != "val":
                raise ValueError(f"Fold {fold} has noncanonical OOF row {local_i}")
            identity = sequence_identity(row)
            if identity in local_seen:
                raise ValueError(f"Fold {fold} repeats sequence {identity[0]}")
            local_seen.add(identity)
            canonical_i = identities.get(identity)
            if canonical_i is None:
                raise ValueError(f"Fold {fold} sequence absent from canonical train index: {identity[0]}")
            day = datetime.strptime(row["init_time_bjt"], "%Y%m%d%H%M").strftime("%Y%m%d")
            if day_folds.get(day) != fold:
                raise ValueError(f"OOF sequence {identity[0]} is not assigned to held-out fold {fold}")
            if canonical_i in covered:
                raise ValueError(f"Canonical sequence {canonical_i} appears in multiple OOF folds")
            covered.add(canonical_i)
            mapping[canonical_i] = local_i
        maps[fold] = mapping
    if len(covered) != len(canonical):
        raise ValueError(f"Five OOF folds cover {len(covered)}/{len(canonical)} canonical train sequences")
    return maps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--full-forecast-bank", type=Path, required=True,
                   help="Frozen canonical train/val forecast bank; only train keys are read")
    p.add_argument("--full-forecast-stage-report", type=Path, required=True,
                   help="Original stage report bound by the forecast bank; used for metadata-only test/buffer closure")
    p.add_argument("--canonical-index", type=Path, required=True,
                   help="Frozen canonical train/val sequence index; metadata only")
    p.add_argument("--fold-assignment", type=Path, required=True)
    p.add_argument("--fold", action="append", nargs=4, required=True, metavar=("ID", "S_BANK", "COT_BANK", "OOF_INDEX"),
                   help="Repeat exactly five times, one per frozen fold ID 0..4")
    p.add_argument("--fold-validation-audit", action="append", nargs=2, required=True,
                   metavar=("ID", "AUDIT_JSON"),
                   help="Repeat exactly five times; fixed 407-sequence full-validation reload receipt per fold")
    p.add_argument("--fold-validation-index", action="append", nargs=2, required=True,
                   metavar=("ID", "INDEX_CSV"),
                   help="Repeat exactly five times; the frozen train/validation index bound by each full-val receipt")
    p.add_argument("--fold-validation-shard", action="append", nargs=3, required=True,
                   metavar=("FOLD_ID", "SHARD_ID", "SHARD_DIR"),
                   help="Repeat once per actual validation shard; shard counts may differ by fold")
    p.add_argument("--fold-r-provenance", action="append", nargs=4, required=True,
                   metavar=("ID", "R_RUN", "R_PACK", "MODEL_SOURCE"),
                   help="Repeat exactly five times; actual fold-specific frozen-R files for lineage re-hashing")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.output.exists():
        raise FileExistsError(f"Refusing to overwrite {a.output}")
    if sha(a.fold_assignment) != EXPECTED_FOLD_SHA:
        raise ValueError("Frozen five-fold date assignment checksum mismatch")
    if sha(a.canonical_index) != EXPECTED_INDEX_SHA:
        raise ValueError("Frozen canonical sequence index checksum mismatch")
    full_report_path = a.full_forecast_bank / "BANK_REPORT.json"
    full_report = read_json(full_report_path)
    if (full_report.get("state") != "MERGED_SIMVP_R_PATCH_TRAINVAL_FORECAST_BANK"
            or full_report.get("canonical_index_sha256") != EXPECTED_INDEX_SHA
            or full_report.get("index_sha256") != EXPECTED_INDEX_SHA
            or full_report.get("splits", {}).get("train", {}).get("sequences") != 4086):
        raise ValueError("Canonical full forecast bank does not match frozen train/val contract")
    full_bank_closure = validate_full_forecast_bank_closure(
        a.full_forecast_bank, full_report, full_report_path,
        a.full_forecast_stage_report, a.canonical_index)
    canonical_keys = read_csv(a.full_forecast_bank / "train" / "keys.csv")
    index_rows = [row for row in read_csv(a.canonical_index) if row.get("split") == "train"]
    if len(index_rows) != len(canonical_keys):
        raise ValueError("Canonical full-bank train keys do not cover the frozen train index")
    for i, (actual, index_row) in enumerate(zip(canonical_keys, index_rows)):
        expected = canonical_key_from_index(index_row, i)
        actual_stems = "|".join(Path(p).stem for p in actual.get("data_paths", "").split("|"))
        if any(actual.get(k) != expected[k] for k in ("sample_index", "split", "init_time_bjt", "target_times_bjt")) or actual_stems != expected["data_paths"]:
            raise ValueError(f"Full forecast bank train keys differ from frozen canonical index at row {i}")

    fold_args: dict[int, tuple[Path, Path, Path]] = {}
    for fold_text, sdir_text, cdir_text, idx_text in a.fold:
        fold = int(fold_text)
        if fold in fold_args or not 0 <= fold < 5:
            raise ValueError(f"Duplicate or invalid fold ID {fold_text}")
        fold_args[fold] = (Path(sdir_text), Path(cdir_text), Path(idx_text))
    if set(fold_args) != set(range(5)):
        raise ValueError("Exactly folds 0,1,2,3,4 are required")
    val_audits: dict[int, Path] = {}
    for fold_text, audit_text in a.fold_validation_audit:
        fold = int(fold_text)
        if fold in val_audits or not 0 <= fold < 5:
            raise ValueError(f"Duplicate or invalid validation-audit fold ID {fold_text}")
        val_audits[fold] = Path(audit_text)
    if set(val_audits) != set(range(5)):
        raise ValueError("Exactly five fold validation audit receipts are required")
    val_indices: dict[int, Path] = {}
    for fold_text, index_text in a.fold_validation_index:
        fold = int(fold_text)
        if fold in val_indices or not 0 <= fold < 5:
            raise ValueError(f"Duplicate or invalid validation-index fold ID {fold_text}")
        val_indices[fold] = Path(index_text)
    if set(val_indices) != set(range(5)):
        raise ValueError("Exactly five fold validation index files are required")
    val_shards: dict[int, dict[int, Path]] = {fold: {} for fold in range(5)}
    for fold_text, shard_text, shard_dir_text in a.fold_validation_shard:
        fold, shard_id = int(fold_text), int(shard_text)
        if not 0 <= fold < 5 or shard_id < 0 or shard_id in val_shards[fold]:
            raise ValueError(f"Duplicate or invalid full-validation shard {fold_text}/{shard_text}")
        val_shards[fold][shard_id] = Path(shard_dir_text)
    if any(not val_shards[f] or set(val_shards[f]) != set(range(len(val_shards[f])))
           for f in range(5)):
        raise ValueError("Each fold needs a non-empty, consecutive set of actual validation shards")
    r_provenance: dict[int, tuple[Path, Path, Path]] = {}
    for fold_text, run_text, pack_text, source_text in a.fold_r_provenance:
        fold = int(fold_text)
        if fold in r_provenance or not 0 <= fold < 5:
            raise ValueError(f"Duplicate or invalid R-provenance fold ID {fold_text}")
        r_provenance[fold] = (Path(run_text), Path(pack_text), Path(source_text))
    if set(r_provenance) != set(range(5)):
        raise ValueError("Exactly five fold-specific R provenance tuples are required")
    day_folds = date_fold_map(a.fold_assignment)
    fold_rows: dict[int, list[dict[str, str]]] = {}
    sources: dict[int, tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray, dict[str, str]]] = {}

    for fold in range(5):
        sdir, cdir, index_path = fold_args[fold]
        s_report_path, c_report_path = sdir / "BANK_REPORT.json", cdir / "BANK_REPORT.json"
        s_report, c_report = read_json(s_report_path), read_json(c_report_path)
        s_index_sha = sha(index_path)
        s_report_sha = sha(s_report_path)
        s_sequences = s_report.get("sequences", s_report.get("split_counts", {}).get("val"))
        single_schema = s_report.get("state") == "SIMVP_R_PATCH_TRAINVAL_FORECAST_BANK"
        if single_schema:
            validate_closed_counters(s_report, f"Fold {fold} S OOF bank report")
            split_counts = s_report.get("split_counts", {})
            if (s_report.get("index_sha256") != s_index_sha
                    or s_report.get("channel_indices") != EXPECTED_CHANNEL_INDICES
                    or s_report.get("lead_minutes") != EXPECTED_LEAD_MINUTES
                    or s_report.get("patch_origin_crop_256") != [103, 118]
                    or s_report.get("patch_size") != 16
                    or s_report.get("sequences_by_split", {}).get("val") != s_sequences
                    or s_report.get("sequences_written_by_split", {}).get("train") != 0
                    or s_report.get("sequences_written_by_split", {}).get("val") != s_sequences
                    or any(split_counts.get(name) != 0 for name in ("train", "test", "buffer"))
                    or split_counts.get("val") != s_sequences):
                raise ValueError(f"Fold {fold} single-fold S OOF bank/report/index mismatch")
            oof_audit_path = sdir / "OOF_BANK_AUDIT_20261017_V5.json"
            validate_single_fold_oof_receipt(read_json(oof_audit_path), fold, s_report,
                                             s_report_sha, s_index_sha, sdir)
        else:
            raise ValueError(f"Fold {fold} S bank must use the fully audited single-fold report schema")
        validate_full_validation_audit(read_json(val_audits[fold]), fold, s_report,
                                       val_indices[fold], read_csv(a.canonical_index), day_folds,
                                       val_shards[fold])
        if not s_sequences:
            raise ValueError(f"Fold {fold} S OOF bank has no sequences")
        s_checkpoint_sha = require_sha(s_report.get("checkpoint_sha256"), f"fold{fold} S checkpoint")
        s_norm_sha = require_sha(s_report.get("normalization_sha256"), f"fold{fold} S norm")
        if (c_report.get("state") != f"HUBEI_SIMVP_OOF_FOLD{fold}_FORECAST_COT_BANK_TEST_CLOSED"
                or c_report.get("fold") != fold
                or c_report.get("test_payloads_read") != 0
                or c_report.get("test_or_buffer_rows") != 0
                or c_report.get("buffer_payloads_read") != 0
                or c_report.get("ghi_labels_read") is not False
                or c_report.get("r_reference_used_for_inference") is not False
                or c_report.get("train_forecast_predictions_are_oof") is not True
                or c_report.get("fold_assignment_sha256") != sha(a.fold_assignment)
                or c_report.get("canonical_oof_index_file_sha256") != sha(index_path)
                or c_report.get("forecast_agri_index_sha256") != (s_report.get("canonical_oof_index_sha256") or s_report.get("index_sha256"))
                or c_report.get("simvp_checkpoint_sha256") != s_checkpoint_sha
                or c_report.get("simvp_train_only_normalization_sha256") != s_norm_sha
                or c_report.get("forecast_agri_bank_report_sha256") != sha(s_report_path)
                or not valid_sha256(c_report.get("fold_R_checkpoint_sha256"))):
            raise ValueError(f"Fold {fold} R forecast-COT OOF report gate failed")
        validate_closed_counters(c_report, f"Fold {fold} forecast-COT OOF report")
        s_count_days = {datetime.strptime(row["init_time_bjt"], "%Y%m%d%H%M").strftime("%Y%m%d")
                        for row in read_csv(sdir / "val" / "keys.csv")}
        r_hashes = validate_fold_r_provenance(c_report, fold, *r_provenance[fold],
                                              s_count_days, day_folds)
        for name, source_path in (("keys.csv", sdir / "val" / "keys.csv"),
                                  ("pred_agri_patch.npy", sdir / "val" / "pred_agri_patch.npy"),
                                  ("future_geometry_patch.npy", sdir / "val" / "future_geometry_patch.npy")):
            actual_sha = sha(source_path)
            if c_report.get("forecast_agri_source_files_sha256", {}).get(name) != actual_sha:
                raise ValueError(f"Fold {fold} COT source lineage mismatch for {name}")
        if (c_report.get("keys_sha256") != sha(cdir / "keys.csv")
                or c_report.get("prediction_sha256") != sha(cdir / "forecast_cot.npy")):
            raise ValueError(f"Fold {fold} COT key/prediction checksum mismatch")
        s_keys = read_csv(sdir / "val" / "keys.csv")
        c_keys = read_csv(cdir / "keys.csv")
        if s_keys != c_keys or len(s_keys) != s_sequences:
            raise ValueError(f"Fold {fold} S/R OOF keys differ or counts mismatch")
        if len(s_keys) != c_report.get("oof_sequences_processed") or len(s_keys) != c_report.get("oof_sequences_available"):
            raise ValueError(f"Fold {fold} COT row count mismatch")
        index_rows = read_csv(index_path)
        if len(index_rows) != len(s_keys):
            raise ValueError(f"Fold {fold} frozen OOF index count mismatch")
        for i, (key, idx_row) in enumerate(zip(s_keys, index_rows)):
            if idx_row.get("split") != "val" or key.get("data_paths") != idx_row.get("data_paths"):
                raise ValueError(f"Fold {fold} OOF output is not aligned with its frozen index at row {i}")
        agri = np.load(sdir / "val" / "pred_agri_patch.npy", mmap_mode="r", allow_pickle=False)
        geom = np.load(sdir / "val" / "future_geometry_patch.npy", mmap_mode="r", allow_pickle=False)
        cot = np.load(cdir / "forecast_cot.npy", mmap_mode="r", allow_pickle=False)
        n = len(s_keys)
        if agri.shape != (n, *PRED_TAIL) or geom.shape != (n, *GEOM_TAIL) or cot.shape != (n, *COT_TAIL):
            raise ValueError(f"Fold {fold} tensor shape mismatch")
        if not (np.isfinite(agri).all() and np.isfinite(geom).all() and np.isfinite(cot).all()) or (cot < 0).any():
            raise ValueError(f"Fold {fold} tensor has invalid values")
        fold_rows[fold] = s_keys
        sources[fold] = (s_report, c_report, agri, geom, cot, r_hashes)

    maps = canonicalize_keys(canonical_keys, fold_rows, day_folds)
    a.output.mkdir(parents=True, exist_ok=False)
    outdir = a.output / "train"
    outdir.mkdir()
    n = len(canonical_keys)
    out_agri = np.lib.format.open_memmap(outdir / "pred_agri_patch.npy", mode="w+", dtype=np.float32, shape=(n, *PRED_TAIL))
    out_geom = np.lib.format.open_memmap(outdir / "future_geometry_patch.npy", mode="w+", dtype=np.float32, shape=(n, *GEOM_TAIL))
    out_cot = np.lib.format.open_memmap(outdir / "forecast_cot.npy", mode="w+", dtype=np.float32, shape=(n, *COT_TAIL))
    fold_by_row = np.full(n, -1, dtype=np.int8)
    for fold in range(5):
        _, _, agri, geom, cot, _ = sources[fold]
        for canonical_i, local_i in maps[fold].items():
            out_agri[canonical_i] = agri[local_i]
            out_geom[canonical_i] = geom[local_i]
            out_cot[canonical_i] = cot[local_i]
            fold_by_row[canonical_i] = fold
    out_agri.flush(); out_geom.flush(); out_cot.flush()
    del out_agri, out_geom, out_cot
    with (outdir / "keys.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KEY_FIELDS)
        writer.writeheader()
        writer.writerows(canonical_keys)
    with (outdir / "heldout_fold.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(("sample_index", "heldout_fold"))
        writer.writerows((i, int(v)) for i, v in enumerate(fold_by_row))
    report = {
        "state": "MERGED_HUBEI_SIMVP_FORECAST_GHI_OOF_TRAIN_BANK_TEST_CLOSED",
        "scope": "train-date held-out S->fold-specific-R forecast inputs for GHI head fitting; not validation or test skill",
        "canonical_index_sha256": EXPECTED_INDEX_SHA,
        "full_forecast_bank_closure_audit": full_bank_closure,
        "fold_assignment_sha256": sha(a.fold_assignment),
        "fold_manifest_sha256": EXPECTED_FOLD_SHA,
        "train_sequences": n,
        "oof_fold_count": 5,
        "fold_sequence_counts": {str(f): len(maps[f]) for f in range(5)},
        "train_forecast_predictions_are_oof": True,
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "test_or_buffer_rows": 0,
        "ghi_labels_read": False,
        "fold_sources": {
            str(f): {
                "heldout_fold": f,
                "simvp_bank_report_sha256": sha(fold_args[f][0] / "BANK_REPORT.json"),
                "forecast_cot_bank_report_sha256": sha(fold_args[f][1] / "BANK_REPORT.json"),
                "simvp_checkpoint_sha256": sources[f][0].get("checkpoint_sha256"),
                "simvp_normalization_sha256": sources[f][0].get("normalization_sha256"),
                "simvp_stage_manifest_sha256": sources[f][0].get("local_stage", {}).get("stage_manifest_sha256"),
                "simvp_oof_audit_sha256": sha(fold_args[f][0] / "OOF_BANK_AUDIT_20261017_V5.json"),
                "simvp_full_validation_audit_sha256": sha(val_audits[f]),
                "simvp_full_validation_index_sha256": sha(val_indices[f]),
                "fold_R_checkpoint_sha256": sources[f][1].get("fold_R_checkpoint_sha256"),
                "fold_R_source_hashes": sources[f][5],
                "oof_index_sha256": sha(fold_args[f][2]),
            } for f in range(5)
        },
        "arrays_sha256": {p.name: sha(p) for p in sorted(outdir.iterdir()) if p.suffix == ".npy"},
        "keys_sha256": sha(outdir / "keys.csv"),
        "heldout_fold_sha256": sha(outdir / "heldout_fold.csv"),
    }
    (a.output / "BANK_REPORT.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"state": report["state"], "train_sequences": n,
                      "fold_sequence_counts": report["fold_sequence_counts"],
                      "test_payloads_read": 0}, indent=2), flush=True)


if __name__ == "__main__":
    main()
