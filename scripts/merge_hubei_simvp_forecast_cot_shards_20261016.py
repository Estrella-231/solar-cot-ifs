#!/usr/bin/env python3
"""Merge train/validation forecast-COT inference shards with exact key QA."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


KEY_FIELDS = ("sample_index", "split", "init_time_bjt", "target_times_bjt", "data_paths")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--forecast-bank", required=True, type=Path,
                   help="Canonical merged SimVP forecast AGRI bank")
    p.add_argument("--shard-dir", required=True, nargs="+", type=Path,
                   help="One R inference output directory per shard")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {a.output}")
    source_report_path = a.forecast_bank / "BANK_REPORT.json"
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    if source_report.get("state") != "MERGED_SIMVP_R_PATCH_TRAINVAL_FORECAST_BANK" or source_report.get("test_payloads_read") != 0:
        raise ValueError("Input must be the merged train/validation SimVP forecast bank")

    reports = []
    for root in a.shard_dir:
        report_path = root / "BANK_REPORT.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("state") != "SIMVP_FORECAST_AGRI_TO_FROZEN_R_COT_TRAINVAL_SHARD":
            raise ValueError(f"Unexpected R-shard state: {report_path}")
        if report.get("test_payloads_read") != 0 or report.get("ghi_labels_read") is not False:
            raise ValueError(f"Test/GHI boundary violation: {report_path}")
        if report.get("source_forecast_bank_report_sha256") != sha256(source_report_path):
            raise ValueError(f"R shard came from a different SimVP forecast bank: {report_path}")
        reports.append((root, report))
    shard_counts = {r.get("shard_count") for _, r in reports}
    shard_ids = [r.get("shard_index") for _, r in reports]
    if len(shard_counts) != 1 or None in shard_counts:
        raise ValueError(f"Inconsistent shard counts: {shard_counts}")
    shard_count = int(next(iter(shard_counts)))
    if shard_count != len(reports) or sorted(shard_ids) != list(range(shard_count)):
        raise ValueError(f"Incomplete/duplicate shard set: {shard_ids}")
    for field in ("source_index_sha256", "source_simvp_checkpoint_sha256",
                  "source_simvp_normalization_sha256", "frozen_R_checkpoint_sha256",
                  "R_model_source_sha256", "R_norm_sha256",
                  "canonical_source_index_sha256", "source_stage_report_sha256",
                  "source_stage_index_sha256", "source_stage_manifest_sha256"):
        if len({r.get(field) for _, r in reports}) != 1:
            raise ValueError(f"Shards disagree on {field}")
    lineage_pairs = (
        ("canonical_source_index_sha256", "canonical_index_sha256"),
        ("source_stage_report_sha256", "source_stage_report_sha256"),
        ("source_stage_index_sha256", "source_stage_index_sha256"),
        ("source_stage_manifest_sha256", "source_stage_manifest_sha256"),
    )
    for shard_field, bank_field in lineage_pairs:
        if reports[0][1].get(shard_field) != source_report.get(bank_field):
            raise ValueError(f"COT shards were produced from incompatible bank lineage: {shard_field}")

    a.output.mkdir(parents=True, exist_ok=False)
    summary = {}
    cot_hashes = {}
    for split in ("train", "val"):
        src_dir = a.forecast_bank / split
        canonical_keys = read_rows(src_dir / "keys.csv")
        expected_n = len(canonical_keys)
        report_n = int(source_report.get("splits", {}).get(split, {}).get("sequences", -1))
        if expected_n != report_n or expected_n < 1:
            raise ValueError(f"Forecast-bank key count mismatch for {split}")
        outdir = a.output / split
        outdir.mkdir()
        out = np.lib.format.open_memmap(
            outdir / "forecast_cot.npy", mode="w+", dtype=np.float32,
            shape=(expected_n, 16, 1, 16, 16),
        )
        covered = np.zeros(expected_n, dtype=bool)
        for root, report in reports:
            rdir = root / split
            shard_keys = read_rows(rdir / "keys.csv")
            expected_shard_n = int(report.get("splits", {}).get(split, {}).get("sequences_written", -1))
            arr = np.load(rdir / "forecast_cot.npy", mmap_mode="r", allow_pickle=False)
            if len(shard_keys) != expected_shard_n or arr.shape != (expected_shard_n, 16, 1, 16, 16):
                raise ValueError(f"R-shard key/array shape mismatch: {rdir}")
            for j, key in enumerate(shard_keys):
                idx = int(key["sample_index"])
                if not 0 <= idx < expected_n or covered[idx]:
                    raise ValueError(f"Duplicate/out-of-range sequence key {split}/{idx}")
                if any(key.get(k) != canonical_keys[idx].get(k) for k in KEY_FIELDS):
                    raise ValueError(f"R-shard key differs from canonical SimVP bank: {split}/{idx}")
                values = np.asarray(arr[j])
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError(f"Invalid COT output at {split}/{idx}")
                out[idx] = values
                covered[idx] = True
        if not covered.all():
            raise ValueError(f"Missing R forecasts in {split}: {(~covered).sum()}")
        out.flush()
        del out
        with (outdir / "keys.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=KEY_FIELDS)
            writer.writeheader()
            writer.writerows(canonical_keys)
        path = outdir / "forecast_cot.npy"
        cot_hashes[f"{split}/forecast_cot.npy"] = sha256(path)
        summary[split] = {"sequences": expected_n, "exact_key_coverage": int(covered.sum())}

    result = {
        "state": "MERGED_SIMVP_FORECAST_COT_TRAINVAL_BANK_TEST_CLOSED",
        "test_payloads_read": 0,
        "ghi_labels_read": False,
        "source_forecast_bank_report_sha256": sha256(source_report_path),
        "source_index_sha256": reports[0][1]["source_index_sha256"],
        "canonical_source_index_sha256": reports[0][1]["canonical_source_index_sha256"],
        "source_stage_index_sha256": reports[0][1]["source_stage_index_sha256"],
        "source_stage_report_sha256": reports[0][1]["source_stage_report_sha256"],
        "source_stage_manifest_sha256": reports[0][1]["source_stage_manifest_sha256"],
        "source_simvp_checkpoint_sha256": reports[0][1]["source_simvp_checkpoint_sha256"],
        "source_simvp_normalization_sha256": reports[0][1]["source_simvp_normalization_sha256"],
        "frozen_R_checkpoint_sha256": reports[0][1]["frozen_R_checkpoint_sha256"],
        "R_model_source_sha256": reports[0][1]["R_model_source_sha256"],
        "R_norm_sha256": reports[0][1]["R_norm_sha256"],
        "shard_count": shard_count,
        "shard_ids": sorted(shard_ids),
        "splits": summary,
        "prediction_shape_per_split": ["N_sequences", 16, 1, 16, 16],
        "prediction_sha256": cot_hashes,
        "train_predictions_are_oof": False,
        "merge_policy": "canonical SimVP sequence order; exact key equality; each sequence covered once",
        "note": "Forecast AGRI -> frozen R COT. Distinct from the real-AGRI same-time COT bank.",
    }
    (a.output / "BANK_REPORT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
