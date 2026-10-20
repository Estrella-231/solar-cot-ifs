#!/usr/bin/env python3
"""Synthetic CPU witness for the train/val bank shard merger."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("merge_hubei_simvp_trainval_patch_shards_20261016.py")
SPEC = importlib.util.spec_from_file_location("hubei_bank_merge", SCRIPT)
MERGER = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MERGER)


def row_paths(i: int) -> str:
    from datetime import datetime, timedelta
    start = datetime(2025, 8, 1, 22) + timedelta(days=i)
    return "|".join('/synthetic/'+(start+timedelta(minutes=15*j)).strftime('%Y%m%d%H%M')+'.npy' for j in range(24))

def aux_paths(i: int) -> str:
    from datetime import datetime, timedelta
    start = datetime(2025, 8, 2, 6) + timedelta(days=i)
    return "|".join('/aux/'+(start+timedelta(minutes=15*j)).strftime('%Y%m%d%H%M')+'_1_aux.npz' for j in range(24))


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="hubei_bank_merge_test_") as temp:
        root = Path(temp)
        index_path = root / "index.csv"
        canonical = {"train": 7, "val": 9}
        with index_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["split", "data_paths", "aux_paths"])
            writer.writeheader()
            for split, n in canonical.items():
                for i in range(n):
                    writer.writerow({"split": split, "data_paths": row_paths(i), "aux_paths": aux_paths(i)})
        index_hash = hashlib.sha256(index_path.read_bytes()).hexdigest()
        stage_index = root / "stage_index.csv"
        stage_index.write_text("path-rewritten-index\n", encoding="utf-8")
        stage_manifest = root / "stage_manifest.csv"
        stage_manifest.write_text("kind,source,destination\n", encoding="utf-8")
        stage_report_path = root / "stage_report.json"
        stage_report_path.write_text(json.dumps({
            "status": "LOCAL_STAGE_COMPLETE_TRAINVAL_ONLY_TEST_CLOSED",
            "source_index": str(index_path), "source_index_sha256": index_hash,
            "stage_index": str(stage_index),
            "stage_index_sha256": hashlib.sha256(stage_index.read_bytes()).hexdigest(),
            "stage_manifest": str(stage_manifest),
            "stage_manifest_sha256": hashlib.sha256(stage_manifest.read_bytes()).hexdigest(),
            "stage_root": str(root / "stage"),
            "sequences": canonical, "test_sequences": 0,
            "test_or_buffer_payloads_staged": 0,
        }), encoding="utf-8")
        local_stage = {
            "stage_root": str(root / "stage"),
            "stage_manifest_sha256": hashlib.sha256(stage_manifest.read_bytes()).hexdigest(),
        }

        shard_roots = []
        for shard_id in range(4):
            shard = root / f"shard_{shard_id}"
            shard_roots.append(shard)
            report = {
                "state": "SIMVP_R_PATCH_TRAINVAL_FORECAST_BANK",
                "test_payloads_read": 0,
                "index_sha256": index_hash,
                "checkpoint_sha256": "checkpoint-test",
                "normalization_sha256": "norm-test",
                "local_stage": local_stage,
                "shard_index": shard_id,
                "shard_count": 4,
                "sequences_written_by_split": {},
                "full_field_validation_metrics": {
                    "val": {
                        "normalized_mse": float(shard_id + 1),
                        "normalized_mae": float(10 * (shard_id + 1)),
                        "element_count": len([i for i in range(canonical["val"]) if i % 4 == shard_id]),
                        "sequence_count": len([i for i in range(canonical["val"]) if i % 4 == shard_id]),
                    }
                },
            }
            for split, n in canonical.items():
                selected = [i for i in range(n) if i % 4 == shard_id]
                out = shard / split
                out.mkdir(parents=True)
                report["sequences_written_by_split"][split] = len(selected)
                with (out / "keys.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["sample_index", "split", "init_time_bjt", "target_times_bjt", "data_paths"])
                    for i in selected:
                        key = MERGER.expected_key({"data_paths": row_paths(i), "aux_paths": aux_paths(i)}, i, split)
                        writer.writerow([key[k] for k in ["sample_index", "split", "init_time_bjt", "target_times_bjt", "data_paths"]])
                pred = np.stack([np.full((16, 13, 16, 16), i, np.float32) for i in selected])
                truth = pred + 100
                geom = np.stack([np.full((16, 3, 16, 16), i, np.float32) for i in selected])
                np.save(out / "pred_agri_patch.npy", pred)
                np.save(out / "true_agri_patch.npy", truth)
                np.save(out / "future_geometry_patch.npy", geom)
            (shard / "BANK_REPORT.json").write_text(json.dumps(report), encoding="utf-8")

        output = root / "merged"
        old_argv = sys.argv
        try:
            sys.argv = [str(SCRIPT), "--index", str(index_path), "--stage-report", str(stage_report_path),
                        "--shard-dir", *map(str, shard_roots), "--output", str(output)]
            MERGER.main()
        finally:
            sys.argv = old_argv

        report = json.loads((output / "BANK_REPORT.json").read_text(encoding="utf-8"))
        assert report["state"] == "MERGED_SIMVP_R_PATCH_TRAINVAL_FORECAST_BANK"
        assert report["canonical_index_sha256"] == index_hash
        assert report["source_stage_index_sha256"] == hashlib.sha256(stage_index.read_bytes()).hexdigest()
        assert report["splits"] == {
            "train": {"sequences": 7, "unique_key_coverage": 7},
            "val": {"sequences": 9, "unique_key_coverage": 9},
        }
        assert report["full_field_validation_metrics"]["val"]["normalized_mse"] == 2.3333333333333335
        assert report["full_field_validation_metrics"]["val"]["sequence_count"] == canonical["val"]
        for split, n in canonical.items():
            p = np.load(output / split / "pred_agri_patch.npy", mmap_mode="r")
            assert p.shape == (n, 16, 13, 16, 16)
            assert [float(p[i, 0, 0, 0, 0]) for i in range(n)] == list(map(float, range(n)))
            p._mmap.close()
        print("SYNTHETIC_SHARD_MERGE_PASS")


if __name__ == "__main__":
    main()
