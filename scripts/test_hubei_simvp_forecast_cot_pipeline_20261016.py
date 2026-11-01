#!/usr/bin/env python3
"""Synthetic CPU witness for frozen-R forecast-input preparation and merge."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def load(name: str, path: Path, *, stub_torch: bool = False):
    if stub_torch:
        sys.modules.setdefault("torch", types.ModuleType("torch"))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_keys(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["sample_index", "split", "init_time_bjt", "target_times_bjt", "data_paths"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def main() -> None:
    infer = load("forecast_r_infer", HERE / "infer_hubei_cot_simvp_forecast_trainval_20261016.py", stub_torch=True)
    merge = load("forecast_cot_merge", HERE / "merge_hubei_simvp_forecast_cot_shards_20261016.py")
    packer = load("forecast_ghi_pack", HERE / "build_hubei_forecast_ghi_heads_pack_20261016.py")
    strict_index_sha = "4eddd1649e90cd1a76c48264e972561338953d791cca021b3eeadc1101d22c62"
    assert infer.EXPECTED_SIMVP_INDEX_SHA == strict_index_sha
    assert packer.EXPECTED_SIMVP_INDEX_SHA == strict_index_sha
    assert infer.EXPECTED_SEQUENCES == {"train": 4086, "val": 407}
    assert packer.EXPECTED_SEQUENCES == {"train": 4086, "val": 407}
    rows = [{"sample_index": str(i), "split": "train", "init_time_bjt": f"20260601{i:04d}",
             "target_times_bjt": "|".join(f"20260601{i:04d}{j:02d}" for j in range(16)),
             "data_paths": "synthetic"} for i in range(11)]
    selected = [[i for i, _ in infer.select_shard_rows(rows, s, 4)] for s in range(4)]
    flat = sum(selected, [])
    assert sorted(flat) == list(range(11)) and len(flat) == len(set(flat))

    with tempfile.TemporaryDirectory(prefix="hubei_stage_lineage_") as tmp:
        root = Path(tmp)
        source_index = root / "source.csv"
        stage_index = root / "stage.csv"
        stage_manifest = root / "stage_manifest.csv"
        source_index.write_text("source-path\n", encoding="utf-8")
        stage_index.write_text("localized-path\n", encoding="utf-8")
        stage_manifest.write_text("kind,source,destination\n", encoding="utf-8")
        infer.EXPECTED_SIMVP_INDEX_SHA = sha(source_index)
        packer.EXPECTED_SIMVP_INDEX_SHA = sha(source_index)
        stage_report = root / "stage_report.json"
        write_json(stage_report, {
            "status": "LOCAL_STAGE_COMPLETE_TRAINVAL_ONLY_TEST_CLOSED",
            "source_index": str(source_index), "source_index_sha256": sha(source_index),
            "stage_index": str(stage_index), "stage_index_sha256": sha(stage_index),
            "stage_manifest": str(stage_manifest), "stage_manifest_sha256": sha(stage_manifest),
            "stage_root": str(root / "stage"),
            "sequences": {"train": 4086, "val": 407}, "test_sequences": 0,
            "test_or_buffer_payloads_staged": 0,
        })
        bank_lineage = {
            "index_sha256": sha(source_index),
            "canonical_index_sha256": sha(source_index),
            "source_stage_report_sha256": sha(stage_report),
            "source_stage_index_sha256": sha(stage_index),
            "source_stage_manifest_sha256": sha(stage_manifest),
            "stage_root": str(root / "stage"),
        }
        stage = infer.validate_stage_lineage(stage_report, bank_lineage)
        assert packer.validate_stage_lineage(stage_report, bank_lineage)["source_index_sha256"] == sha(source_index)
        assert stage["source_index_sha256"] == sha(source_index)
        try:
            bad_lineage = {**bank_lineage, "source_stage_index_sha256": "wrong-stage-index"}
            infer.validate_stage_lineage(stage_report, bad_lineage)
        except ValueError:
            pass
        else:
            raise AssertionError("stage-index mismatch must be rejected")

    agri = np.zeros((2, 16, 13, 16, 16), np.float32)
    geom = np.zeros((2, 16, 3, 16, 16), np.float32)
    geom[:, :, 2] = 1.0
    geom[0, 0, 2, 0, 0] = 0.0
    x, solar = infer.prepare_r_inputs(agri, geom, np.zeros(16, np.float32), np.ones(16, np.float32))
    assert x.shape == (32, 16, 16, 16) and solar.shape == (32, 1, 16, 16)
    assert x[0, 15, 0, 0] == 0 and not solar[0, 0, 0, 0] and x[0, 15, 0, 1] == 1

    with tempfile.TemporaryDirectory(prefix="hubei_forecast_cot_merge_") as tmp:
        root = Path(tmp)
        bank_root = root / "forecast_bank"
        bank_root.mkdir()
        (bank_root / "train").mkdir(); (bank_root / "val").mkdir()
        split_keys = {}
        split_counts = {"train": 7, "val": 9}
        for split, n in split_counts.items():
            keys = []
            for i in range(n):
                keys.append({"sample_index": str(i), "split": split,
                             "init_time_bjt": f"20260601{i:04d}",
                             "target_times_bjt": "|".join(f"T{i:03d}_{j:02d}" for j in range(16)),
                             "data_paths": f"seq-{split}-{i}"})
            split_keys[split] = keys
            write_keys(bank_root / split / "keys.csv", keys)
        bank_report = {"state": "MERGED_SIMVP_R_PATCH_TRAINVAL_FORECAST_BANK", "test_payloads_read": 0,
                       "index_sha256": "synthetic-index", "checkpoint_sha256": "s-ckpt",
                       "normalization_sha256": "s-norm",
                       "canonical_index_sha256": "synthetic-canonical-index",
                       "source_stage_report_sha256": "synthetic-stage-report",
                       "source_stage_index_sha256": "synthetic-stage-index",
                       "source_stage_manifest_sha256": "synthetic-stage-manifest",
                       "stage_root": "synthetic-stage-root",
                       "splits": {s: {"sequences": n} for s, n in split_counts.items()}}
        write_json(bank_root / "BANK_REPORT.json", bank_report)
        source_report_sha = sha(bank_root / "BANK_REPORT.json")
        shards = []
        for shard_id in range(4):
            sd = root / f"r_shard_{shard_id}"; shards.append(sd)
            sd.mkdir()
            report = {"state": "SIMVP_FORECAST_AGRI_TO_FROZEN_R_COT_TRAINVAL_SHARD",
                      "test_payloads_read": 0, "ghi_labels_read": False,
                      "source_forecast_bank_report_sha256": source_report_sha,
                      "source_index_sha256": "synthetic-index",
                      "canonical_source_index_sha256": "synthetic-canonical-index",
                      "source_stage_report_sha256": "synthetic-stage-report",
                      "source_stage_index_sha256": "synthetic-stage-index",
                      "source_stage_manifest_sha256": "synthetic-stage-manifest",
                      "source_simvp_checkpoint_sha256": "s-ckpt",
                      "source_simvp_normalization_sha256": "s-norm",
                      "frozen_R_checkpoint_sha256": "r-ckpt", "R_model_source_sha256": "r-src",
                      "R_norm_sha256": "r-norm", "shard_index": shard_id, "shard_count": 4,
                      "splits": {}}
            for split, canonical in split_keys.items():
                indices = [i for i in range(len(canonical)) if i % 4 == shard_id]
                out = sd / split; out.mkdir()
                write_keys(out / "keys.csv", [canonical[i] for i in indices])
                arr = np.stack([np.full((16, 1, 16, 16), i, np.float32) for i in indices]) if indices else np.empty((0,16,1,16,16), np.float32)
                np.save(out / "forecast_cot.npy", arr, allow_pickle=False)
                report["splits"][split] = {"source_sequences": len(canonical), "sequences_written": len(indices),
                                           "shard_index": shard_id, "shard_count": 4}
            write_json(sd / "BANK_REPORT.json", report)
        output = root / "merged_cot"
        old_argv = sys.argv
        try:
            sys.argv = [str(HERE / "merge_hubei_simvp_forecast_cot_shards_20261016.py"),
                        "--forecast-bank", str(bank_root), "--shard-dir", *map(str, shards), "--output", str(output)]
            merge.main()
        finally:
            sys.argv = old_argv
        final = json.loads((output / "BANK_REPORT.json").read_text(encoding="utf-8"))
        assert final["state"] == "MERGED_SIMVP_FORECAST_COT_TRAINVAL_BANK_TEST_CLOSED"
        assert final["splits"] == {s: {"sequences": n, "exact_key_coverage": n} for s, n in split_counts.items()}
        for split, n in split_counts.items():
            arr = np.load(output / split / "forecast_cot.npy", mmap_mode="r", allow_pickle=False)
            assert arr.shape == (n, 16, 1, 16, 16)
            assert [float(arr[i, 0, 0, 0, 0]) for i in range(n)] == list(map(float, range(n)))
            arr._mmap.close()
    print("FORECAST_R_PREP_AND_MERGE_PASS")


if __name__ == "__main__":
    main()
