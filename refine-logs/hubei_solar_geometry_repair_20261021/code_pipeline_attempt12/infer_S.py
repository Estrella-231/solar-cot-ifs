#!/usr/bin/env python3
"""Generate frozen train/validation SimVP forecasts on the Hubei R patch only.

This is deliberately not a test runner. It reads only the frozen train/val
sequence index and stores the 16x16 region needed by the downstream R model,
not multi-gigabyte full-scene forecast cubes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from hubei_forecast_time_keys import canonical_bjt_stamps


PATCH_ROW = 103
PATCH_COL = 118
PATCH_SIZE = 16
CHANNELS = 13
GEOMETRY_CHANNELS = 3
PREDICTED_AGRI_INDICES = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
STATION_CENTER_RC_IN_PATCH = {
    "Niushou": [8, 6], "Wangzhai": [9, 8], "Mizhuang": [7, 10],
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_norm(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        out = {k: np.asarray(z[k], dtype=np.float32).copy()
               for k in ("agri_mean", "agri_std", "input_mean", "input_std")}
    if out["agri_mean"].shape != (CHANNELS,) or out["agri_std"].shape != (CHANNELS,):
        raise ValueError("Frozen train-only AGRI normalization must have 13 channels")
    if out["input_mean"].shape != (16,) or out["input_std"].shape != (16,):
        raise ValueError("Frozen train-only input normalization must have 16 channels")
    if not all(np.isfinite(v).all() for v in out.values()) or np.any(out["agri_std"] <= 0) or np.any(out["input_std"] <= 0):
        raise ValueError("Invalid/non-finite normalization values")
    return out


def select_shard_rows(rows: list[dict[str, str]], shard_index: int,
                      shard_count: int, max_sequences: int = 0) -> list[tuple[int, dict[str, str]]]:
    """Select a deterministic, disjoint subset while retaining canonical row IDs."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard-index must satisfy 0 <= shard-index < shard-count")
    if max_sequences < 0:
        raise ValueError("max_sequences must be non-negative")
    limit = min(len(rows), max_sequences) if max_sequences else len(rows)
    return [(i, row) for i, row in enumerate(rows[:limit])
            if i % shard_count == shard_index]


def remap_dataset_to_local_stage(dataset, stage_root: Path) -> dict[str, object]:
    """Use previously staged train/val payloads while keeping canonical CSV keys."""
    manifest = stage_root / "stage_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing stage manifest: {manifest}")
    by_source: dict[tuple[str, str], Path] = {}
    with manifest.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            kind = row["kind"]
            source = os.path.normpath(row["source"])
            destination = Path(row["destination"])
            by_source[(kind, source)] = destination

    remapped = 0
    checked_destinations: set[Path] = set()
    stage_dirs = {kind: (stage_root / kind).resolve() for kind in ("data", "aux")}
    for sample in dataset.samples:
        for key, kind in (("data", "data"), ("aux", "aux")):
            local_paths = []
            for source_path in sample[key]:
                source = os.path.normpath(source_path)
                destination = by_source.get((kind, source))
                if destination is None:
                    # Some frozen rows already name the local staged payload;
                    # preserve it directly instead of looking it up as a source.
                    candidate = Path(source_path)
                    try:
                        candidate.resolve().relative_to(stage_dirs[kind])
                    except ValueError as exc:
                        raise FileNotFoundError(
                            f"Stage manifest has no {kind} entry for {source}"
                        ) from exc
                    destination = candidate
                if destination not in checked_destinations and not destination.is_file():
                    raise FileNotFoundError(f"Staged payload is missing: {destination}")
                checked_destinations.add(destination)
                local_paths.append(str(destination))
                remapped += 1
            sample[key] = local_paths
    return {
        "stage_root": str(stage_root),
        "stage_manifest_sha256": sha256(manifest),
        "remapped_payload_references": remapped,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, default=Path("/home/Data_Pool_3/chenyi/GHI/SimVPv2"))
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--norm", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stage-root", type=Path, default=None,
                   help="Optional local train/val cache; canonical index keys remain unchanged")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--shard-index", type=int, default=0,
                   help="Zero-based shard index; use separate output dirs for parallel workers")
    p.add_argument("--shard-count", type=int, default=1,
                   help="Number of disjoint shards; each split is partitioned by original row index modulo this value")
    p.add_argument("--max-sequences", type=int, default=0,
                   help="Optional smoke-test cap applied independently to train and val; 0 means all")
    p.add_argument("--splits", nargs="+", choices=("train", "val"),
                   default=("train", "val"),
                   help="Splits to infer; val-only is useful for uniform checkpoint re-scoring")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if not torch.cuda.is_available() and a.device.startswith("cuda"):
        raise RuntimeError("CUDA requested but unavailable")
    if a.batch_size < 1 or a.workers < 0 or a.max_sequences < 0:
        raise ValueError("Invalid loader settings")
    if a.shard_count < 1 or not 0 <= a.shard_index < a.shard_count:
        raise ValueError("shard-index must satisfy 0 <= shard-index < shard-count")
    if not (a.repo / "openstl").is_dir():
        raise FileNotFoundError(a.repo / "openstl")
    if not a.index.is_file() or not a.norm.is_file() or not a.checkpoint.is_file():
        raise FileNotFoundError("index, norm, and checkpoint must all exist")

    sys.path.insert(0, str(a.repo))
    from openstl.datasets.dataloader_hbagri import HBAGRIDataset
    from openstl.methods.simvp import SimVP

    norm = read_norm(a.norm)
    a.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(a.device)

    def use_frozen_train_statistics(self) -> None:
        # The training dataset's stock initializer scans many source files to
        # recompute statistics. Replace that path so inference uses only the
        # recorded train-only statistics and does not trigger redundant NFS IO.
        self.mean = norm["agri_mean"].copy()
        self.std = norm["agri_std"].copy()
        self.input_mean = norm["input_mean"].copy()
        self.input_std = norm["input_std"].copy()

    HBAGRIDataset._compute_stats = use_frozen_train_statistics

    cfg = {
        "method": "simvp", "dataname": "hbagri", "data_name": "hbagri",
        "pre_seq_length": 8, "aft_seq_length": 16, "total_length": 24,
        "in_shape": [8, 16, 256, 256], "out_channels": 13,
        "model_type": "gSTA", "hid_S": 32, "hid_T": 256,
        "N_S": 4, "N_T": 8, "spatio_kernel_enc": 3,
        "spatio_kernel_dec": 3, "drop_path": 0.1, "mlp_ratio": 8.0,
        "drop": 0.0, "metrics": ["mse", "mae"],
        "steps_per_epoch": 1, "test_mean": norm["agri_mean"],
        "test_std": norm["agri_std"], "save_dir": str(a.output),
        "seed": 42, "epoch": 50, "opt": "adamw", "lr": 1e-3,
        "weight_decay": 1e-4, "sched": "cosine", "min_lr": 1e-5,
        "warmup_epoch": 5,
    }
    ckpt = torch.load(a.checkpoint, map_location="cpu")
    state = ckpt.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("Expected Lightning checkpoint with state_dict")
    module = SimVP(**cfg).to(device)
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint key mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    module.eval()

    run_splits = tuple(dict.fromkeys(a.splits))
    ds_by_split = {}
    stage_metadata = None
    if a.stage_root is not None:
        stage_metadata = {
            "stage_root": str(a.stage_root),
            "stage_manifest_sha256": sha256(a.stage_root / "stage_manifest.csv"),
            "remapped_payload_references_by_split": {},
        }
    for split in run_splits:
        ds = HBAGRIDataset(
            data_root=str(a.index.parent), pre_seq_length=8, aft_seq_length=16,
            temporal_stride=1, split=split, data_name="hbagri",
            index_csv=str(a.index),
        )
        ds.mean = norm["agri_mean"].copy()
        ds.std = norm["agri_std"].copy()
        ds.input_mean = norm["input_mean"].copy()
        ds.input_std = norm["input_std"].copy()
        if a.stage_root is not None:
            remap = remap_dataset_to_local_stage(ds, a.stage_root)
            stage_metadata["remapped_payload_references_by_split"][split] = remap["remapped_payload_references"]
        ds_by_split[split] = ds

    # Assert the supplied index contains no test/buffer rows before reading any
    # sequence payload. The CSV is the only source used by these datasets.
    with a.index.open("r", newline="", encoding="utf-8") as f:
        split_counts = {"train": 0, "val": 0, "test": 0, "buffer": 0}
        for row in csv.DictReader(f):
            if row["split"] not in split_counts:
                raise ValueError(f"Unexpected split label: {row['split']}")
            split_counts[row["split"]] += 1
    if split_counts["test"] or split_counts["buffer"]:
        raise RuntimeError(f"Refusing to read index with test/buffer rows: {split_counts}")

    rows_by_split = {}
    with a.index.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] in run_splits:
                rows_by_split.setdefault(row["split"], []).append(row)

    # Split by stable per-split row number. Workers use independent output
    # directories; the saved sample_index remains the row number in the
    # canonical (unsharded) split so outputs can be merged and audited.
    source_indices_by_split = {}
    selected_rows_by_split = {}
    for split, rows in rows_by_split.items():
        selected = select_shard_rows(rows, a.shard_index, a.shard_count, a.max_sequences)
        source_indices_by_split[split] = [i for i, _ in selected]
        selected_rows_by_split[split] = selected

    full_field_metrics = {}
    with torch.inference_mode():
        for split in run_splits:
            ds = ds_by_split[split]
            source_indices = source_indices_by_split.get(split, [])
            selected_rows = selected_rows_by_split.get(split, [])
            expected_n = len(source_indices)
            outdir = a.output / split
            outdir.mkdir()
            pred_mm = np.lib.format.open_memmap(
                outdir / "pred_agri_patch.npy", mode="w+",
                dtype=np.float32, shape=(expected_n, 16, CHANNELS, PATCH_SIZE, PATCH_SIZE),
            )
            true_mm = np.lib.format.open_memmap(
                outdir / "true_agri_patch.npy", mode="w+",
                dtype=np.float32, shape=(expected_n, 16, CHANNELS, PATCH_SIZE, PATCH_SIZE),
            )
            geom_mm = np.lib.format.open_memmap(
                outdir / "future_geometry_patch.npy", mode="w+",
                dtype=np.float32, shape=(expected_n, 16, GEOMETRY_CHANNELS, PATCH_SIZE, PATCH_SIZE),
            )
            run_ds = torch.utils.data.Subset(ds, source_indices)
            loader = DataLoader(run_ds, batch_size=a.batch_size, shuffle=False,
                                num_workers=a.workers, pin_memory=True)
            cursor = 0
            squared_error_sum = 0.0
            absolute_error_sum = 0.0
            metric_element_count = 0
            for batch_x, batch_y, batch_aux in loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_aux = batch_aux.to(device, non_blocking=True)
                pred = module(batch_x, batch_aux=batch_aux)
                # Independently record the exact full-field normalized AGRI
                # error (not only the downstream 16x16 station patch). This
                # lets us verify checkpoint selection over all validation
                # rows after distributed training's rank-local val logging.
                if split == "val":
                    residual64 = pred.to(torch.float64) - batch_y.to(device=device, dtype=torch.float64)
                    squared_error_sum += float(torch.square(residual64).sum().item())
                    absolute_error_sum += float(torch.abs(residual64).sum().item())
                    metric_element_count += int(batch_y.numel())
                pred = pred.float().cpu().numpy()
                truth = batch_y.float().cpu().numpy()
                pred = pred * norm["agri_std"][None, None, :, None, None] + norm["agri_mean"][None, None, :, None, None]
                truth = truth * norm["agri_std"][None, None, :, None, None] + norm["agri_mean"][None, None, :, None, None]
                geom_mean = norm["input_mean"][CHANNELS:][None, None, :, None, None]
                geom_std = norm["input_std"][CHANNELS:][None, None, :, None, None]
                future_geom = batch_aux[:, 8:24].float().cpu().numpy() * geom_std + geom_mean
                p = pred[:, :, :, PATCH_ROW:PATCH_ROW + PATCH_SIZE, PATCH_COL:PATCH_COL + PATCH_SIZE]
                y = truth[:, :, :, PATCH_ROW:PATCH_ROW + PATCH_SIZE, PATCH_COL:PATCH_COL + PATCH_SIZE]
                g = future_geom[:, :, :, PATCH_ROW:PATCH_ROW + PATCH_SIZE, PATCH_COL:PATCH_COL + PATCH_SIZE]
                end = cursor + len(p)
                if end > expected_n or p.shape[-2:] != (PATCH_SIZE, PATCH_SIZE) or g.shape[2:] != (GEOMETRY_CHANNELS, PATCH_SIZE, PATCH_SIZE):
                    raise RuntimeError("Unexpected validation output shape/count")
                pred_mm[cursor:end] = p
                true_mm[cursor:end] = y
                geom_mm[cursor:end] = g
                cursor = end
            pred_mm.flush(); true_mm.flush(); geom_mm.flush()
            # A shard intentionally covers only a subset of the canonical split.
            # Comparing its cursor with len(rows_by_split[split]) would reject
            # every shard_count > 1 run after inference, despite correct output.
            if cursor != expected_n or cursor != len(selected_rows):
                raise RuntimeError(
                    f"Shard index/dataloader row mismatch for {split}: "
                    f"cursor={cursor}, expected_shard_rows={expected_n}, "
                    f"selected_rows={len(selected_rows)}"
                )
            if split == "val":
                if metric_element_count == 0:
                    raise RuntimeError("Validation shard has no full-field metric elements")
                full_field_metrics[split] = {
                    "normalized_mse": squared_error_sum / metric_element_count,
                    "normalized_mae": absolute_error_sum / metric_element_count,
                    "element_count": metric_element_count,
                    "sequence_count": expected_n,
                    "metric_scope": "full 256x256 output field, all 13 AGRI channels and 16 leads, frozen normalized space",
                }

            with (outdir / "keys.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["sample_index", "split", "init_time_bjt", "target_times_bjt", "data_paths"])
                for row_out, (source_i, row) in enumerate(selected_rows):
                    paths = row["data_paths"].split("|")
                    if len(paths) != 24:
                        raise RuntimeError(f"Expected 24 frame paths at row {source_i}")
                    stamps = canonical_bjt_stamps(row)
                    writer.writerow([source_i, split, stamps[7], "|".join(stamps[8:24]), row["data_paths"]])

    metadata = {
        "state": "SIMVP_R_PATCH_TRAINVAL_FORECAST_BANK",
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "test_or_buffer_rows": 0,
        "split_counts": split_counts,
        # A validation-only rescore intentionally loads no train dataset.
        # Report only the splits actually instantiated for this invocation.
        "sequences_by_split": {s: len(ds_by_split[s]) for s in run_splits},
        "sequences_written_by_split": {
            s: len(selected_rows_by_split.get(s, [])) for s in ("train", "val")
        },
        "shard_index": a.shard_index,
        "shard_count": a.shard_count,
        "shard_policy": "original per-split index modulo shard_count; output sample_index is canonical per-split index",
        "smoke_test_cap_per_split": a.max_sequences or None,
        "inferred_splits": list(run_splits),
        "agri_tensor_shape_per_split": ["N", 16, 13, PATCH_SIZE, PATCH_SIZE],
        "future_geometry_shape_per_split": ["N", 16, 3, PATCH_SIZE, PATCH_SIZE],
        "patch_origin_crop_256": [PATCH_ROW, PATCH_COL],
        "patch_size": PATCH_SIZE,
        "station_center_rc_in_patch": STATION_CENTER_RC_IN_PATCH,
        "channel_indices": PREDICTED_AGRI_INDICES,
        "lead_minutes": list(range(15, 241, 15)),
        "initialization": "last observed input frame timestamp (BJT)",
        "time_key_source": "all 24 source data UTC filenames +8h must equal auxiliary BJT filenames; keys use auxiliary BJT",
        "checkpoint_sha256": sha256(a.checkpoint),
        "normalization_sha256": sha256(a.norm),
        "index_sha256": sha256(a.index),
        "prediction_policy": "raw denormalized SimVP output; no clipping or interpolation",
        "normalization_scope": "frozen train-only statistics",
        "local_stage": stage_metadata,
        "full_field_validation_metrics": full_field_metrics,
    }
    (a.output / "BANK_REPORT.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
