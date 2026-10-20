#!/usr/bin/env python3
"""Merge disjoint SimVP train/validation forecast shards with key-coverage QA.

This utility only reads the supplied train/validation index and forecast-bank
shards. It refuses test/buffer rows, duplicate/missing shard IDs, duplicate or
missing sample keys, row/key mismatches, and non-finite saved tensors.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from hubei_forecast_time_keys import canonical_bjt_stamps


PRED_SHAPE_TAIL = (16, 13, 16, 16)
GEOM_SHAPE_TAIL = (16, 3, 16, 16)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_index(path: Path) -> dict[str, list[dict[str, str]]]:
    splits: dict[str, list[dict[str, str]]] = {"train": [], "val": []}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            split = row.get("split", "")
            if split in ("test", "buffer"):
                raise RuntimeError(f"Refusing test/buffer index row: {split}")
            if split not in splits:
                raise RuntimeError(f"Unexpected split label: {split!r}")
            if len(row.get("data_paths", "").split("|")) != 24:
                raise RuntimeError("Each index row must contain exactly 24 data frame paths")
            splits[split].append(row)
    return splits


def expected_key(row: dict[str, str], sample_index: int, split: str) -> dict[str, str]:
    paths = row["data_paths"].split("|")
    stamps = canonical_bjt_stamps(row)
    return {
        "sample_index": str(sample_index),
        "split": split,
        "init_time_bjt": stamps[7],
        "target_times_bjt": "|".join(stamps[8:24]),
        "data_paths": row["data_paths"],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=Path, required=True,
                   help="Canonical frozen train/validation sequence index")
    p.add_argument("--shard-dir", type=Path, nargs="+", required=True,
                   help="One output directory from each inference shard")
    p.add_argument("--stage-report", type=Path, required=True,
                   help="Local train/validation stage report used by every shard")
    p.add_argument("--output", type=Path, required=True,
                   help="New, non-existing merged output directory")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if not a.index.is_file():
        raise FileNotFoundError(a.index)
    if a.output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {a.output}")
    if not a.shard_dir:
        raise ValueError("At least one --shard-dir is required")

    index = read_index(a.index)
    index_sha = sha256(a.index)
    stage = json.loads(a.stage_report.read_text(encoding="utf-8"))
    if stage.get("status") != "LOCAL_STAGE_COMPLETE_TRAINVAL_ONLY_TEST_CLOSED":
        raise RuntimeError("Stage report is not train/validation-only")
    if stage.get("source_index_sha256") != index_sha:
        raise RuntimeError("Stage report is not bound to the supplied canonical index")
    if stage.get("sequences") != {"train": len(index["train"]), "val": len(index["val"])}:
        raise RuntimeError("Stage report train/validation counts differ from canonical index")
    if stage.get("test_sequences") != 0 or stage.get("test_or_buffer_payloads_staged") != 0:
        raise RuntimeError("Stage contains test/buffer data")
    if sha256(Path(stage["source_index"])) != index_sha:
        raise RuntimeError("Stage source index checksum mismatch")
    if sha256(Path(stage["stage_index"])) != stage.get("stage_index_sha256"):
        raise RuntimeError("Path-rewritten stage index checksum mismatch")
    if sha256(Path(stage["stage_manifest"])) != stage.get("stage_manifest_sha256"):
        raise RuntimeError("Stage manifest checksum mismatch")
    reports = []
    for root in a.shard_dir:
        report_path = root / "BANK_REPORT.json"
        if not report_path.is_file():
            raise FileNotFoundError(report_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("state") != "SIMVP_R_PATCH_TRAINVAL_FORECAST_BANK":
            raise RuntimeError(f"Unexpected shard state in {report_path}")
        if report.get("test_payloads_read") != 0 or report.get("index_sha256") != index_sha:
            raise RuntimeError(f"Shard source/test boundary mismatch: {report_path}")
        local_stage = report.get("local_stage")
        if not isinstance(local_stage, dict):
            raise RuntimeError(f"Shard has no local-stage lineage: {report_path}")
        if (local_stage.get("stage_root") != stage.get("stage_root")
                or local_stage.get("stage_manifest_sha256") != stage.get("stage_manifest_sha256")):
            raise RuntimeError(f"Shard local-stage lineage mismatch: {report_path}")
        reports.append((root, report))

    shard_counts = {r.get("shard_count") for _, r in reports}
    shard_ids = [r.get("shard_index") for _, r in reports]
    if len(shard_counts) != 1 or None in shard_counts:
        raise RuntimeError(f"Inconsistent shard counts: {shard_counts}")
    shard_count = int(next(iter(shard_counts)))
    if shard_count != len(reports) or sorted(shard_ids) != list(range(shard_count)):
        raise RuntimeError(f"Shard set incomplete or duplicated: ids={shard_ids}, expected={shard_count}")
    for field in ("checkpoint_sha256", "normalization_sha256"):
        if len({r.get(field) for _, r in reports}) != 1:
            raise RuntimeError(f"Shards do not share {field}")

    a.output.mkdir(parents=True, exist_ok=False)
    split_summary = {}
    for split in ("train", "val"):
        expected_rows = index[split]
        n_expected = len(expected_rows)
        if n_expected == 0:
            raise RuntimeError(f"Canonical index has no {split} samples")
        outdir = a.output / split
        outdir.mkdir()
        pred_out = np.lib.format.open_memmap(
            outdir / "pred_agri_patch.npy", mode="w+", dtype=np.float32,
            shape=(n_expected, *PRED_SHAPE_TAIL))
        true_out = np.lib.format.open_memmap(
            outdir / "true_agri_patch.npy", mode="w+", dtype=np.float32,
            shape=(n_expected, *PRED_SHAPE_TAIL))
        geom_out = np.lib.format.open_memmap(
            outdir / "future_geometry_patch.npy", mode="w+", dtype=np.float32,
            shape=(n_expected, *GEOM_SHAPE_TAIL))
        covered = np.zeros(n_expected, dtype=bool)

        for root, report in reports:
            shard_keys = read_csv(root / split / "keys.csv")
            expected_shard_n = report.get("sequences_written_by_split", {}).get(split)
            if len(shard_keys) != expected_shard_n:
                raise RuntimeError(f"Shard key/array count mismatch: {root}/{split}")
            pred_src = np.load(root / split / "pred_agri_patch.npy", mmap_mode="r", allow_pickle=False)
            true_src = np.load(root / split / "true_agri_patch.npy", mmap_mode="r", allow_pickle=False)
            geom_src = np.load(root / split / "future_geometry_patch.npy", mmap_mode="r", allow_pickle=False)
            if pred_src.shape != (len(shard_keys), *PRED_SHAPE_TAIL) or true_src.shape != pred_src.shape:
                raise RuntimeError(f"Unexpected AGRI array shape in {root}/{split}")
            if geom_src.shape != (len(shard_keys), *GEOM_SHAPE_TAIL):
                raise RuntimeError(f"Unexpected geometry array shape in {root}/{split}")
            for local_i, key in enumerate(shard_keys):
                sample_i = int(key["sample_index"])
                if not 0 <= sample_i < n_expected or covered[sample_i]:
                    raise RuntimeError(f"Out-of-range or duplicate sample index: {split}/{sample_i}")
                if key != expected_key(expected_rows[sample_i], sample_i, split):
                    raise RuntimeError(f"Forecast key does not match canonical index: {split}/{sample_i}")
                p, y, g = pred_src[local_i], true_src[local_i], geom_src[local_i]
                if not (np.isfinite(p).all() and np.isfinite(y).all() and np.isfinite(g).all()):
                    raise RuntimeError(f"Non-finite forecast-bank tensor: {split}/{sample_i}")
                pred_out[sample_i] = p
                true_out[sample_i] = y
                geom_out[sample_i] = g
                covered[sample_i] = True

        if not covered.all():
            raise RuntimeError(f"Missing {split} forecast rows: {(~covered).sum()}")
        pred_out.flush(); true_out.flush(); geom_out.flush()
        with (outdir / "keys.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_index", "split", "init_time_bjt", "target_times_bjt", "data_paths"])
            writer.writeheader()
            for i, row in enumerate(expected_rows):
                writer.writerow(expected_key(row, i, split))
        split_summary[split] = {"sequences": n_expected, "unique_key_coverage": int(covered.sum())}
        del pred_out, true_out, geom_out

    final_report = {
        "state": "MERGED_SIMVP_R_PATCH_TRAINVAL_FORECAST_BANK",
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "test_or_buffer_rows": 0,
        "index_sha256": index_sha,
        "canonical_index_sha256": index_sha,
        "source_stage_report_sha256": sha256(a.stage_report),
        "source_stage_index_sha256": stage["stage_index_sha256"],
        "source_stage_manifest_sha256": stage["stage_manifest_sha256"],
        "stage_root": stage["stage_root"],
        "checkpoint_sha256": reports[0][1]["checkpoint_sha256"],
        "normalization_sha256": reports[0][1]["normalization_sha256"],
        "shard_count": shard_count,
        "shard_ids": sorted(shard_ids),
        "splits": split_summary,
        "agri_shape_per_split": ["N", *PRED_SHAPE_TAIL],
        "geometry_shape_per_split": ["N", *GEOM_SHAPE_TAIL],
        "merge_policy": "canonical index order, exact row-key equality, unique full train/val coverage",
    }
    shard_val_metrics = [
        report.get("full_field_validation_metrics", {}).get("val")
        for _, report in reports
    ]
    if all(isinstance(m, dict) for m in shard_val_metrics):
        total_elements = sum(int(m.get("element_count", 0)) for m in shard_val_metrics)
        expected_val_sequences = len(index["val"])
        metric_sequences = sum(int(m.get("sequence_count", 0)) for m in shard_val_metrics)
        if total_elements <= 0 or metric_sequences != expected_val_sequences:
            raise RuntimeError(
                "Full-field validation metric counts do not cover the canonical val split: "
                f"elements={total_elements}, sequences={metric_sequences}, expected={expected_val_sequences}"
            )
        if not all(np.isfinite(float(m.get("normalized_mse", np.nan)))
                   and np.isfinite(float(m.get("normalized_mae", np.nan)))
                   for m in shard_val_metrics):
            raise RuntimeError("Non-finite full-field validation metric in a shard report")
        final_report["full_field_validation_metrics"] = {
            "val": {
                "normalized_mse": sum(float(m["normalized_mse"]) * int(m["element_count"])
                                       for m in shard_val_metrics) / total_elements,
                "normalized_mae": sum(float(m["normalized_mae"]) * int(m["element_count"])
                                       for m in shard_val_metrics) / total_elements,
                "element_count": total_elements,
                "sequence_count": metric_sequences,
                "metric_scope": "full 256x256 output field, all 13 AGRI channels and 16 leads, frozen normalized space",
            }
        }
    (a.output / "BANK_REPORT.json").write_text(json.dumps(final_report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final_report, indent=2), flush=True)


if __name__ == "__main__":
    main()
