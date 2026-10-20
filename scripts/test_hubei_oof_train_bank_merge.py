#!/usr/bin/env python3
"""Synthetic integrity tests for the five-fold Hubei OOF merger."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta
from pathlib import Path

import merge_hubei_forecast_ghi_oof_train_bank_20261017 as merge
from merge_hubei_forecast_ghi_oof_train_bank_20261017 import (
    FULL_VAL_COUNT,
    FULL_VAL_ELEMENTS,
    EXPECTED_STAGE_MANIFEST_SHA,
    EXPECTED_R_CHANNELS,
    canonicalize_keys,
    validate_fold_r_provenance,
    validate_full_validation_audit,
    read_csv,
    validate_single_fold_oof_receipt,
    validate_closed_counters,
    validate_full_forecast_bank_closure,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row(index: int, split: str, date: str) -> dict[str, str]:
    init = datetime.strptime(date, "%Y%m%d")
    targets = "|".join((init + timedelta(minutes=15 * k)).strftime("%Y%m%d%H%M")
                       for k in range(1, 17))
    paths = "|".join(f"/stage/{(init + timedelta(minutes=15 * k)).strftime('%Y%m%d%H%M')}.npy"
                     for k in range(-7, 17))
    return {"sample_index": str(index), "split": split,
            "init_time_bjt": init.strftime("%Y%m%d%H%M"),
            "target_times_bjt": targets, "data_paths": paths}


class OOFMergeKeyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.days = [f"2025080{i}" for i in range(1, 6)]
        self.canonical = [row(i, "train", day) for i, day in enumerate(self.days)]
        self.fold_rows = {i: [row(0, "val", day)] for i, day in enumerate(self.days)}
        self.day_folds = {day: fold for fold, day in enumerate(self.days)}

    def test_exact_five_fold_coverage_maps_to_canonical_order(self) -> None:
        maps = canonicalize_keys(self.canonical, self.fold_rows, self.day_folds, expected_count=5)
        self.assertEqual({fold: sorted(mapping) for fold, mapping in maps.items()},
                         {fold: [fold] for fold in range(5)})
        self.assertEqual({fold: list(mapping.values()) for fold, mapping in maps.items()},
                         {fold: [0] for fold in range(5)})

    def test_rejects_missing_fold_coverage(self) -> None:
        incomplete = dict(self.fold_rows)
        incomplete[4] = []
        with self.assertRaisesRegex(ValueError, "cover 4/5"):
            canonicalize_keys(self.canonical, incomplete, self.day_folds, expected_count=5)

    def test_rejects_cross_fold_duplicate(self) -> None:
        duplicate = dict(self.fold_rows)
        duplicate[4] = [row(0, "val", self.days[0])]
        with self.assertRaisesRegex(ValueError, "not assigned to held-out fold"):
            canonicalize_keys(self.canonical, duplicate, self.day_folds, expected_count=5)

    def test_rejects_invalid_lead_sequence(self) -> None:
        broken = dict(self.fold_rows)
        invalid = dict(broken[0][0])
        invalid["target_times_bjt"] = invalid["target_times_bjt"].replace("202508010015", "202508010030", 1)
        broken[0] = [invalid]
        with self.assertRaisesRegex(ValueError, "target is not init"):
            canonicalize_keys(self.canonical, broken, self.day_folds, expected_count=5)


class OOFMergeReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.s_report = {
            "state": "SIMVP_R_PATCH_TRAINVAL_FORECAST_BANK",
            "split_counts": {"train": 0, "val": 823, "test": 0, "buffer": 0},
            "test_payloads_read": 0, "buffer_payloads_read": 0, "test_or_buffer_rows": 0,
            "sequences": 823,
            "checkpoint_sha256": "a" * 64,
            "normalization_sha256": "b" * 64,
            "local_stage": {"stage_manifest_sha256": EXPECTED_STAGE_MANIFEST_SHA},
        }

    def test_single_fold_receipt_binds_lineage_and_closes_buffer(self) -> None:
        receipt = {
            "state": "HUBEI_SIMVP_SINGLE_FOLD_OOF_BANK_INTEGRITY_PASS_TEST_CLOSED",
            "count": 823,
            "source_index_sha256": "c" * 64,
            "source_bank_report_sha256": "d" * 64,
            "checkpoint_sha256": "a" * 64,
            "normalization_sha256": "b" * 64,
            "shapes": {
                "pred_agri_patch": [823, 16, 13, 16, 16],
                "true_agri_patch": [823, 16, 13, 16, 16],
                "future_geometry_patch": [823, 16, 3, 16, 16],
            },
            "finite": True,
            "keys_exactly_match_ordered_oof_index": True,
            "test_payloads_read": 0,
            "buffer_payloads_read": 0,
            "test_or_buffer_rows": 0,
            "ghi_labels_read": False,
            "channel_indices": merge.EXPECTED_CHANNEL_INDICES,
            "lead_minutes": merge.EXPECTED_LEAD_MINUTES,
            "patch_origin_crop_256": [103, 118], "patch_size": 16,
            "arrays_sha256": {"pred_agri_patch": "1" * 64,
                              "true_agri_patch": "2" * 64,
                              "future_geometry_patch": "3" * 64},
        }
        with tempfile.TemporaryDirectory() as temp:
            sdir = Path(temp)
            (sdir / "val").mkdir()
            arrays = {"pred_agri_patch": "pred", "true_agri_patch": "true",
                      "future_geometry_patch": "geom"}
            for name, payload in arrays.items():
                (sdir / "val" / f"{name}.npy").write_bytes(payload.encode())
                receipt["arrays_sha256"][name] = sha(sdir / "val" / f"{name}.npy")
            validate_single_fold_oof_receipt(receipt, 0, self.s_report, "d" * 64, "c" * 64, sdir)
            receipt["arrays_sha256"]["pred_agri_patch"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "array hash mismatch"):
                validate_single_fold_oof_receipt(receipt, 0, self.s_report, "d" * 64, "c" * 64, sdir)
            receipt["arrays_sha256"]["pred_agri_patch"] = sha(sdir / "val" / "pred_agri_patch.npy")
        receipt["source_bank_report_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "does not bind"):
            with tempfile.TemporaryDirectory() as temp:
                sdir = Path(temp); (sdir / "val").mkdir()
                validate_single_fold_oof_receipt(receipt, 0, self.s_report, "d" * 64, "c" * 64, sdir)
        receipt["source_bank_report_sha256"] = "d" * 64
        receipt["buffer_payloads_read"] = 1
        with self.assertRaisesRegex(ValueError, "explicitly report"):
            with tempfile.TemporaryDirectory() as temp:
                sdir = Path(temp); (sdir / "val").mkdir()
                validate_single_fold_oof_receipt(receipt, 0, self.s_report, "d" * 64, "c" * 64, sdir)
        receipt["buffer_payloads_read"] = 0
        self.s_report["normalization_sha256"] = None
        with self.assertRaisesRegex(ValueError, "non-empty lowercase SHA-256"):
            with tempfile.TemporaryDirectory() as temp:
                sdir = Path(temp); (sdir / "val").mkdir()
                validate_single_fold_oof_receipt(receipt, 0, self.s_report, "d" * 64, "c" * 64, sdir)

    def make_val_index(self, root: Path) -> Path:
        path = root / "validation_index.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["split", "data_paths", "aux_paths"])
            writer.writeheader()
            for i in range(FULL_VAL_COUNT):
                item = row(i, "val", "20250801")
                writer.writerow({"split": "val", "data_paths": item["data_paths"],
                                 "aux_paths": "|".join(f"/stage/aux_{i}_{j}.npy" for j in range(24))})
        return path

    def make_val_shards(self, root: Path, index_sha: str,
                        counts: list[int] | None = None) -> dict[int, Path]:
        result = {}
        counts = counts or [102, 102, 102, 101]
        shard_count = len(counts)
        next_key = 0
        for shard_id, count in enumerate(counts):
            shard_dir = root / f"shard_{shard_id}"
            val_dir = shard_dir / "val"; val_dir.mkdir(parents=True)
            report = {
                "state": "SIMVP_R_PATCH_TRAINVAL_FORECAST_BANK",
                "test_payloads_read": 0, "buffer_payloads_read": 0, "test_or_buffer_rows": 0,
                "inferred_splits": ["val"], "shard_index": shard_id,
                "shard_count": shard_count,
                "channel_indices": merge.EXPECTED_CHANNEL_INDICES,
                "lead_minutes": merge.EXPECTED_LEAD_MINUTES,
                "patch_origin_crop_256": [103, 118], "patch_size": 16,
                "checkpoint_sha256": "a" * 64, "normalization_sha256": "b" * 64,
                "index_sha256": index_sha,
                "local_stage": {"stage_manifest_sha256": EXPECTED_STAGE_MANIFEST_SHA},
                "sequences_written_by_split": {"val": count},
            }
            (shard_dir / "BANK_REPORT.json").write_text(json.dumps(report), encoding="utf-8")
            for name, data in (("pred_agri_patch.npy", b"p"),
                               ("true_agri_patch.npy", b"t"),
                               ("future_geometry_patch.npy", b"g")):
                (val_dir / name).write_bytes(data * (shard_id + 1))
            with (val_dir / "keys.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["sample_index"]); writer.writeheader()
                writer.writerows({"sample_index": str(i)} for i in range(next_key, next_key + count))
            next_key += count
            result[shard_id] = shard_dir
        return result

    def make_fullval_audit(self, index_sha: str, shard_dirs: dict[int, Path]) -> dict:
        semantic = {"channel_indices": merge.EXPECTED_CHANNEL_INDICES,
                    "lead_minutes": merge.EXPECTED_LEAD_MINUTES,
                    "patch_origin_crop_256": [103, 118], "patch_size": 16}
        ordered_shards = sorted(shard_dirs)
        return {
            "status": "HUBEI_SIMVP_FOLD0_CHECKPOINT_RELOAD_VALIDATION_AUDIT_PASSED_TEST_CLOSED",
            "fold_id": 0,
            "validation_sequences_inferred": FULL_VAL_COUNT,
            "unique_validation_key_coverage": FULL_VAL_COUNT,
            "test_payloads_read": 0,
            "buffer_payloads_read": 0,
            "test_or_buffer_rows": 0,
            "checkpoint_sha256": "a" * 64,
            "normalization_sha256": "b" * 64,
            "stage_manifest_sha256": EXPECTED_STAGE_MANIFEST_SHA,
            "index_sha256": index_sha,
            **semantic,
            "full_field_validation_metrics": {
                "sequence_count": FULL_VAL_COUNT,
                "element_count": FULL_VAL_ELEMENTS,
                "normalized_mse": 0.12,
                "normalized_mae": 0.21,
            },
            "shard_count": len(ordered_shards),
            "shard_ids": ordered_shards,
            "shards": [
                {"shard_index": i, "sequence_count": n, **semantic,
                 "test_payloads_read": 0, "buffer_payloads_read": 0, "test_or_buffer_rows": 0,
                 "prediction_sha256": sha(shard_dirs[i] / "val" / "pred_agri_patch.npy"),
                 "truth_sha256": sha(shard_dirs[i] / "val" / "true_agri_patch.npy"),
                 "geometry_sha256": sha(shard_dirs[i] / "val" / "future_geometry_patch.npy"),
                 "keys_sha256": sha(shard_dirs[i] / "val" / "keys.csv"),
                 "bank_report_sha256": sha(shard_dirs[i] / "BANK_REPORT.json")}
                for i, n in ((i, len(read_csv(shard_dirs[i] / "val" / "keys.csv")))
                             for i in ordered_shards)
            ],
        }

    def test_full_validation_receipt_binds_real_index_checkpoint_and_norm(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            index_path = self.make_val_index(Path(temp))
            shard_dirs = self.make_val_shards(Path(temp), sha(index_path))
            audit = self.make_fullval_audit(sha(index_path), shard_dirs)
            validate_full_validation_audit(audit, 0, self.s_report, index_path,
                                           read_csv(index_path), {}, shard_dirs)
            for field, value in (("index_sha256", "f" * 64),
                                 ("checkpoint_sha256", None),
                                 ("normalization_sha256", None),
                                 ("fold_id", 1),
                                 ("buffer_payloads_read", 1),
                                 ("test_or_buffer_rows", 1)):
                with self.subTest(field=field):
                    altered = dict(audit, **{field: value})
                    with self.assertRaisesRegex(ValueError, "full-validation"):
                        validate_full_validation_audit(altered, 0, self.s_report, index_path,
                                                       read_csv(index_path), {}, shard_dirs)
            for mutate in (
                lambda a: a.update(channel_indices=[0]),
                lambda a: a.update(lead_minutes=[0]),
                lambda a: a.update(patch_origin_crop_256=[0, 0]),
                lambda a: a["full_field_validation_metrics"].update(normalized_mse=float("nan")),
                lambda a: a["shards"][0].pop("prediction_sha256"),
            ):
                altered = json.loads(json.dumps(audit))
                mutate(altered)
                with self.assertRaisesRegex(ValueError, "full-validation"):
                    validate_full_validation_audit(altered, 0, self.s_report, index_path,
                                                   read_csv(index_path), {}, shard_dirs)
            tampered_hash = json.loads(json.dumps(audit))
            tampered_hash["shards"][0]["prediction_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "hashes do not match actual"):
                validate_full_validation_audit(tampered_hash, 0, self.s_report, index_path,
                                               read_csv(index_path), {}, shard_dirs)

    def test_full_validation_accepts_one_shard_only_when_receipt_covers_all_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            index_path = self.make_val_index(Path(temp))
            shard_dirs = self.make_val_shards(Path(temp), sha(index_path), [FULL_VAL_COUNT])
            audit = self.make_fullval_audit(sha(index_path), shard_dirs)
            validate_full_validation_audit(audit, 0, self.s_report, index_path,
                                           read_csv(index_path), {}, shard_dirs)
            audit["shards"][0].pop("keys_sha256")
            with self.assertRaisesRegex(ValueError, "shard metadata/hash contract failed"):
                validate_full_validation_audit(audit, 0, self.s_report, index_path,
                                               read_csv(index_path), {}, shard_dirs)

    def test_explicit_test_buffer_counters_are_required(self) -> None:
        validate_closed_counters({"test_payloads_read": 0, "buffer_payloads_read": 0,
                                  "test_or_buffer_rows": 0}, "synthetic")
        for report in ({"test_payloads_read": 0, "test_or_buffer_rows": 0},
                       {"test_payloads_read": 0, "buffer_payloads_read": 1,
                        "test_or_buffer_rows": 0},
                       {"test_payloads_read": 0, "buffer_payloads_read": 0,
                        "test_or_buffer_rows": 1}):
            with self.assertRaisesRegex(ValueError, "explicitly report"):
                validate_closed_counters(report, "synthetic")

    def test_legacy_full_bank_can_close_only_from_hashed_trainval_stage_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank = root / "bank"
            (bank / "train").mkdir(parents=True)
            (bank / "val").mkdir()
            stage = root / "stage_report.json"
            stage_index = root / "stage_index.csv"
            stage_manifest = root / "stage_manifest.csv"
            canonical = root / "canonical.csv"
            report_path = bank / "BANK_REPORT.json"
            indexed_paths = [f"/stage/{i}.npy" for i in range(24)]
            aux_paths = [f"/stage/aux/{i}.npz" for i in range(24)]
            with canonical.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["split", "data_paths", "aux_paths"])
                w.writeheader()
                for split, count in (("train", 4086), ("val", 407)):
                    for _ in range(count):
                        w.writerow({"split": split, "data_paths": "|".join(indexed_paths),
                                    "aux_paths": "|".join(aux_paths)})
            stage_index.write_bytes(canonical.read_bytes())
            with stage_manifest.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["destination"]); w.writeheader()
                for path in indexed_paths + aux_paths: w.writerow({"destination": path})
            with patch.object(merge, "EXPECTED_INDEX_SHA", sha(canonical)), \
                 patch.object(merge, "EXPECTED_STAGE_MANIFEST_SHA", sha(stage_manifest)):
                stage_obj = {
                    "status": "LOCAL_STAGE_COMPLETE_TRAINVAL_ONLY_TEST_CLOSED",
                    "source_index_sha256": sha(canonical),
                    "sequences": {"train": 4086, "val": 407},
                    "test_sequences": 0,
                    "test_or_buffer_payloads_staged": 0,
                    "stage_root": "/stage",
                    "stage_manifest": str(stage_manifest),
                    "stage_manifest_sha256": sha(stage_manifest),
                    "stage_index": str(stage_index),
                    "stage_index_sha256": sha(stage_index),
                }
                stage.write_text(json.dumps(stage_obj), encoding="utf-8")
                report = {
                    "stage_root": "/stage",
                    "source_stage_report_sha256": sha(stage),
                    "source_stage_index_sha256": sha(stage_index),
                    "source_stage_manifest_sha256": sha(stage_manifest),
                    "splits": {"train": {"sequences": 4086},
                               "val": {"sequences": 407}},
                    "test_payloads_read": 0,
                }
                report_path.write_text(json.dumps(report), encoding="utf-8")
                for split, count in (("train", 4086), ("val", 407)):
                    with (bank / split / "keys.csv").open("w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=["sample_index", "split"]); w.writeheader()
                        w.writerows({"sample_index": i, "split": split} for i in range(count))
                closure = validate_full_forecast_bank_closure(
                    bank, report, report_path, stage, canonical)
                self.assertEqual(closure["splits"], {"train": 4086, "val": 407,
                                                      "test": 0, "buffer": 0})
                self.assertEqual(closure["test_or_buffer_rows"], 0)
                (bank / "buffer").mkdir()
                with self.assertRaisesRegex(ValueError, "unexpected split directory"):
                    validate_full_forecast_bank_closure(
                        bank, report, report_path, stage, canonical)

    def test_r_lineage_rehashes_sources_and_checks_fold_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_root = root / "manifests"
            input_dir = manifest_root / "input"
            fold_dir = manifest_root / "folds" / "fold_0"
            r_pack = fold_dir / "r_pack"
            run = root / "run"
            for d in (input_dir, fold_dir, r_pack, run): d.mkdir(parents=True, exist_ok=True)
            day_path = input_dir / "hubei_day_assignment_20261016.csv"
            fold_path = input_dir / "hubei_oof_train_date_fold_20261016.csv"
            raw_r_path = input_dir / "hubei_r_paths_trainval_20261016.csv"
            r_manifest = fold_dir / "r_trainval_manifest.csv"
            with day_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["date_BJT", "split"]); w.writeheader()
                w.writerows([{"date_BJT": "2025-08-01", "split": "train"},
                             {"date_BJT": "2025-08-02", "split": "train"},
                             {"date_BJT": "2025-08-03", "split": "val"}])
            with fold_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["date_BJT", "heldout_fold"]); w.writeheader()
                w.writerows([{"date_BJT": "2025-08-01", "heldout_fold": 0},
                             {"date_BJT": "2025-08-02", "heldout_fold": 1}])
            raw_fields = ["time_bjt", "split", "path_agri", "path_cpp_fixed_v2", "historical_split_read_only"]
            raw_rows = [
                {"time_bjt": "202508010900", "split": "train", "path_agri": "/a/1", "path_cpp_fixed_v2": "/c/1", "historical_split_read_only": "train"},
                {"time_bjt": "202508020900", "split": "train", "path_agri": "/a/2", "path_cpp_fixed_v2": "/c/2", "historical_split_read_only": "train"},
                {"time_bjt": "202508030900", "split": "val", "path_agri": "/a/3", "path_cpp_fixed_v2": "/c/3", "historical_split_read_only": "val"},
            ]
            def write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
                with path.open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
            write_rows(raw_r_path, raw_fields, raw_rows)
            manifest_rows = [raw_rows[1], raw_rows[2]]
            write_rows(r_manifest, raw_fields, manifest_rows)
            root_report = {"state": "HUBEI_FIVE_FOLD_S_R_MANIFESTS_BUILT_METADATA_ONLY",
                           "source_index_sha256": merge.EXPECTED_INDEX_SHA,
                           "day_assignment_sha256": sha(day_path),
                           "fold_assignment_sha256": sha(fold_path),
                           "r_path_manifest_sha256": sha(raw_r_path), "fold_count": 5}
            (manifest_root / "folds" / "root_report.json").write_text(json.dumps(root_report), encoding="utf-8")
            fold_report = {"state": "METADATA_ONLY_FOLD_MANIFESTS_PASS_TEST_BUFFER_CLOSED",
                           "fold": 0, "r_trainval_manifest_sha256": sha(r_manifest)}
            (fold_dir / "report.json").write_text(json.dumps(fold_report), encoding="utf-8")

            pack = r_pack
            model_source = root / "model.py"
            model_source.write_bytes(b"model source")
            checkpoint = run / "best.pt"; checkpoint.write_bytes(b"checkpoint")
            norm = {"fit_split": "train only", "test_used": False,
                    "input_channels": EXPECTED_R_CHANNELS}
            (pack / "norm.json").write_text(json.dumps(norm), encoding="utf-8")
            (pack / "train.npy").write_bytes(b"pack source")
            pack_rows = [{"index": str(i), "split": r["split"], "time_bjt": r["time_bjt"],
                          "path_agri": r["path_agri"], "path_cpp_fixed_v2": r["path_cpp_fixed_v2"]}
                         for i, r in enumerate(reversed(manifest_rows))]
            pack_status = {"state": "HUBEI_R_TRAIN_VAL_PACK_BUILT_TEST_CLOSED",
                           "test_cpp_payloads_read": 0, "test_agri_payloads_read": 0,
                           "source_split_sha256": sha(fold_path), "path_manifest_sha256": sha(r_manifest)}
            (pack / "pack_status.json").write_text(json.dumps(pack_status), encoding="utf-8")
            rows_path = pack / "rows.csv"
            write_rows(rows_path, ["index", "split", "time_bjt", "path_agri", "path_cpp_fixed_v2"], pack_rows)
            (run / "status.json").write_text(json.dumps({
                "state": "COMPLETE_VALIDATION_ONLY_TEST_CLOSED", "test_payloads_read": 0}), encoding="utf-8")
            pack_hashes = {"train.npy": sha(pack / "train.npy")}
            metadata = {"seed": 42, "model_source_sha256": sha(model_source),
                        "normalization": norm, "source_pack_hashes": pack_hashes,
                        "pack_status_sha256": sha(pack / "pack_status.json")}
            (run / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            (run / "audit_report.json").write_text(json.dumps({
                "state": "HUBEI_OOF_R_FOLD_COMPLETION_AUDIT_PASSED_TEST_CLOSED",
                "test_payloads_read": 0, "fold_seed": 42,
                "best_checkpoint_sha256": sha(checkpoint), "pack_sha256": pack_hashes}), encoding="utf-8")
            cot_report = {
                "fold_R_run_audit_sha256": sha(run / "audit_report.json"),
                "fold_R_checkpoint_sha256": sha(checkpoint),
                "fold_R_model_source_sha256": sha(model_source),
                "fold_R_pack_norm_sha256": sha(pack / "norm.json"),
                "fold_R_pack_rows_sha256": sha(rows_path),
            }
            with patch.object(merge, "EXPECTED_SPLIT_SHA", sha(day_path)), \
                 patch.object(merge, "EXPECTED_FOLD_SHA", sha(fold_path)), \
                 patch.object(merge, "EXPECTED_R_PATHS_SHA", sha(raw_r_path)):
                hashes = validate_fold_r_provenance(cot_report, 0, run, pack, model_source,
                                                    {"20250801"}, {"20250801": 0, "20250802": 1})
            self.assertEqual(hashes["checkpoint_sha256"], sha(checkpoint))
            bad = dict(cot_report, fold_R_pack_rows_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "actual R lineage"):
                with patch.object(merge, "EXPECTED_SPLIT_SHA", sha(day_path)), \
                     patch.object(merge, "EXPECTED_FOLD_SHA", sha(fold_path)), \
                     patch.object(merge, "EXPECTED_R_PATHS_SHA", sha(raw_r_path)):
                    validate_fold_r_provenance(bad, 0, run, pack, model_source,
                                               {"20250801"}, {"20250801": 0, "20250802": 1})
            pack_rows[0]["time_bjt"] = "202508010900"
            write_rows(rows_path, ["index", "split", "time_bjt", "path_agri", "path_cpp_fixed_v2"], pack_rows)
            bad_rows = dict(cot_report, fold_R_pack_rows_sha256=sha(rows_path))
            with self.assertRaisesRegex(ValueError, "do not exactly cover the frozen manifest keys"):
                with patch.object(merge, "EXPECTED_SPLIT_SHA", sha(day_path)), \
                     patch.object(merge, "EXPECTED_FOLD_SHA", sha(fold_path)), \
                     patch.object(merge, "EXPECTED_R_PATHS_SHA", sha(raw_r_path)):
                    validate_fold_r_provenance(bad_rows, 0, run, pack, model_source,
                                               {"20250801"}, {"20250801": 0, "20250802": 1})


if __name__ == "__main__":
    unittest.main()
