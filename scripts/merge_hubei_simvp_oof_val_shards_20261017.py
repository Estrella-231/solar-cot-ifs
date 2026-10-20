#!/usr/bin/env python3
"""Merge one fold's held-out train-date SimVP OOF validation shards.

This is intentionally separate from the train+validation merger: the OOF
index contains only held-out training dates under split='val'. It refuses
test/buffer rows, index/checkpoint/normalization/stage mismatches, missing or
duplicate sample IDs, key drift, wrong shapes, and non-finite tensors.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from hubei_forecast_time_keys import canonical_bjt_stamps

PRED_TAIL = (16, 13, 16, 16)
GEOM_TAIL = (16, 3, 16, 16)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def expected_key(row: dict[str, str], sample_index: int) -> dict[str, str]:
    paths = row["data_paths"].split("|")
    stamps = canonical_bjt_stamps(row)
    return {
        "sample_index": str(sample_index),
        "split": "val",
        "init_time_bjt": stamps[7],
        "target_times_bjt": "|".join(stamps[8:24]),
        "data_paths": row["data_paths"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True,
                        help="Frozen fold-specific OOF index; expected val-only")
    parser.add_argument("--stage-root", type=Path, required=True,
                        help="Verified local train/val payload stage used by shards")
    parser.add_argument("--shard-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="New output path; existing paths are refused")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    for path in (args.index, args.stage_root / "stage_manifest.csv"):
        if not path.is_file():
            raise FileNotFoundError(path)

    rows = read_csv(args.index)
    if not rows or any(row.get("split") != "val" for row in rows):
        raise RuntimeError("OOF index must contain only non-empty split='val' rows")
    if any(len(row.get("data_paths", "").split("|")) != 24 for row in rows):
        raise RuntimeError("Each OOF row must contain exactly 24 AGRI paths")
    index_sha = sha256(args.index)
    stage_sha = sha256(args.stage_root / "stage_manifest.csv")

    shards = []
    for root in args.shard_dir:
        report_path = root / "BANK_REPORT.json"
        if not report_path.is_file():
            raise FileNotFoundError(report_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("state") != "SIMVP_R_PATCH_TRAINVAL_FORECAST_BANK":
            raise RuntimeError(f"Unexpected shard state: {report_path}")
        if (report.get("test_payloads_read") != 0
                or report.get("split_counts", {}).get("test") != 0
                or report.get("split_counts", {}).get("buffer") != 0):
            raise RuntimeError(f"Test/buffer boundary mismatch: {report_path}")
        if report.get("index_sha256") != index_sha:
            raise RuntimeError(f"Frozen OOF index hash mismatch: {report_path}")
        local_stage = report.get("local_stage") or {}
        if (Path(local_stage.get("stage_root", "")).resolve() != args.stage_root.resolve()
                or local_stage.get("stage_manifest_sha256") != stage_sha):
            raise RuntimeError(f"Stage lineage mismatch: {report_path}")
        if report.get("sequences_by_split") != {"val": len(rows)}:
            raise RuntimeError(f"Unexpected OOF split size: {report_path}")
        shards.append((root, report))

    shard_counts = {int(report.get("shard_count", -1)) for _, report in shards}
    shard_ids = [int(report.get("shard_index", -1)) for _, report in shards]
    if len(shard_counts) != 1:
        raise RuntimeError(f"Inconsistent shard_count values: {shard_counts}")
    shard_count = next(iter(shard_counts))
    if shard_count != len(shards) or sorted(shard_ids) != list(range(shard_count)):
        raise RuntimeError(f"Incomplete/duplicate shards: {shard_ids}, expected {shard_count}")
    for field in ("checkpoint_sha256", "normalization_sha256"):
        if len({report.get(field) for _, report in shards}) != 1:
            raise RuntimeError(f"Shards disagree on {field}")

    n_rows = len(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    outdir = args.output / "val"
    outdir.mkdir()
    pred_out = np.lib.format.open_memmap(
        outdir / "pred_agri_patch.npy", mode="w+", dtype=np.float32,
        shape=(n_rows, *PRED_TAIL))
    true_out = np.lib.format.open_memmap(
        outdir / "true_agri_patch.npy", mode="w+", dtype=np.float32,
        shape=(n_rows, *PRED_TAIL))
    geom_out = np.lib.format.open_memmap(
        outdir / "future_geometry_patch.npy", mode="w+", dtype=np.float32,
        shape=(n_rows, *GEOM_TAIL))
    covered = np.zeros(n_rows, dtype=bool)
    metrics = []

    for root, report in shards:
        keys_path = root / "val" / "keys.csv"
        keys = read_csv(keys_path)
        n_shard = int(report.get("sequences_written_by_split", {}).get("val", -1))
        if len(keys) != n_shard:
            raise RuntimeError(f"Key count differs from report: {keys_path}")
        pred = np.load(root / "val" / "pred_agri_patch.npy", mmap_mode="r", allow_pickle=False)
        truth = np.load(root / "val" / "true_agri_patch.npy", mmap_mode="r", allow_pickle=False)
        geom = np.load(root / "val" / "future_geometry_patch.npy", mmap_mode="r", allow_pickle=False)
        if pred.shape != (n_shard, *PRED_TAIL) or truth.shape != pred.shape:
            raise RuntimeError(f"Unexpected AGRI shape under {root}")
        if geom.shape != (n_shard, *GEOM_TAIL):
            raise RuntimeError(f"Unexpected geometry shape under {root}")
        if not (np.isfinite(pred).all() and np.isfinite(truth).all()
                and np.isfinite(geom).all()):
            raise RuntimeError(f"Non-finite tensor under {root}")
        for local_i, key in enumerate(keys):
            sample_i = int(key["sample_index"])
            if not 0 <= sample_i < n_rows or covered[sample_i]:
                raise RuntimeError(f"Out-of-range or duplicate sample_index={sample_i}")
            if key != expected_key(rows[sample_i], sample_i):
                raise RuntimeError(f"OOF key differs from canonical index at row {sample_i}")
            pred_out[sample_i] = pred[local_i]
            true_out[sample_i] = truth[local_i]
            geom_out[sample_i] = geom[local_i]
            covered[sample_i] = True
        metric = report.get("full_field_validation_metrics", {}).get("val")
        if not isinstance(metric, dict) or int(metric.get("sequence_count", -1)) != n_shard:
            raise RuntimeError(f"Full-field metric row count mismatch: {root}")
        metrics.append(metric)

    if not covered.all():
        raise RuntimeError(f"OOF coverage missing {int((~covered).sum())} canonical rows")
    pred_out.flush(); true_out.flush(); geom_out.flush()
    with (outdir / "keys.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(expected_key(rows[0], 0)))
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow(expected_key(row, i))

    total_elements = sum(int(m["element_count"]) for m in metrics)
    metric_sequences = sum(int(m["sequence_count"]) for m in metrics)
    if metric_sequences != n_rows or total_elements <= 0:
        raise RuntimeError("Shard full-field metrics do not cover the OOF index")
    weighted_mse = sum(float(m["normalized_mse"]) * int(m["element_count"]) for m in metrics) / total_elements
    weighted_mae = sum(float(m["normalized_mae"]) * int(m["element_count"]) for m in metrics) / total_elements
    report = {
        "state": "MERGED_SIMVP_R_PATCH_OOF_VAL_BANK",
        "scope": "held-out training dates for one SimVP OOF fold; not independent test or GHI skill",
        "test_payloads_read": 0,
        "test_or_buffer_rows": 0,
        "canonical_oof_index_sha256": index_sha,
        "stage_manifest_sha256": stage_sha,
        "checkpoint_sha256": shards[0][1]["checkpoint_sha256"],
        "normalization_sha256": shards[0][1]["normalization_sha256"],
        "shard_count": shard_count,
        "shard_ids": sorted(shard_ids),
        "sequences": n_rows,
        "unique_key_coverage": int(covered.sum()),
        "agri_shape": [n_rows, *PRED_TAIL],
        "geometry_shape": [n_rows, *GEOM_TAIL],
        "full_scene_normalized_diagnostic": {
            "mse": weighted_mse, "mae": weighted_mae,
            "element_count": total_elements, "sequences": metric_sequences,
            "warning": "This full-scene diagnostic is not COT or station-GHI forecasting skill.",
        },
        "files_sha256": {
            path.name: sha256(path)
            for path in sorted(outdir.iterdir()) if path.is_file()
        },
    }
    (args.output / "BANK_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
