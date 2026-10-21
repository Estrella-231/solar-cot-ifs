#!/usr/bin/env python3
"""Build a train/validation-only bank of historical AGRI, geometry and frozen-R COT.

Run with torchrun on one node. Each rank owns disjoint BJT frame keys and one
scheduler-assigned GPU. Arrays are written by disjoint row ranges to shared
NumPy memmaps; no future payload is opened.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


AGRI_INDEX = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
HISTORY_OFFSETS_MIN = (105, 90, 75, 60, 45, 30, 15, 0)
PATCH_ROW, PATCH_COL, PATCH_SIZE = 103, 118, 16
COS80 = float(np.cos(np.deg2rad(80.0)))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def bjt_time(text: str) -> datetime:
    return datetime.strptime(text, "%Y%m%d%H%M")


def load_requirements(stage_path: Path, ghi_rows_path: Path):
    stage_rows = read_csv(stage_path)
    ghi_rows = read_csv(ghi_rows_path)
    stage_by_key = {r["bjt"]: r for r in stage_rows}
    if len(stage_by_key) != len(stage_rows):
        raise ValueError("stage manifest BJT keys are not unique")
    key_splits: dict[str, set[str]] = {}
    init_splits: dict[str, str] = {}
    for row in ghi_rows:
        split = row["split"]
        if split not in {"train", "val"}:
            raise ValueError(f"Refusing non-train/val GHI row: {split}")
        init = row["init_time_bjt"]
        previous = init_splits.setdefault(init, split)
        if previous != split:
            raise ValueError(f"One init crosses splits: {init}")
        t_init = bjt_time(init)
        lead = int(row["lead_minutes"])
        if lead not in range(15, 241, 15) or bjt_time(row["target_time_bjt"]) != t_init + timedelta(minutes=lead):
            raise ValueError(f"Invalid canonical target/lead row: {row}")
        for offset in HISTORY_OFFSETS_MIN:
            key = (t_init - timedelta(minutes=offset)).strftime("%Y%m%d%H%M")
            key_splits.setdefault(key, set()).add(split)
    if any(len(s) != 1 for s in key_splits.values()):
        raise ValueError("A history frame is shared across train/validation")
    missing = sorted(set(key_splits) - set(stage_by_key))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} required history keys: {missing[:10]}")
    frames = []
    for key in sorted(key_splits):
        row = stage_by_key[key]
        split = next(iter(key_splits[key]))
        if row["split"] != split:
            raise ValueError(f"History frame split mismatch at {key}: {row['split']} != {split}")
        utc_expected = (bjt_time(key) - timedelta(hours=8)).strftime("%Y%m%d%H%M")
        if row["utc"] != utc_expected or Path(row["data_path"]).stem[:12] != utc_expected:
            raise ValueError(f"BJT/UTC or data-path time mismatch at {key}")
        if Path(row["aux_path"]).name[:12] != key:
            raise ValueError(f"Auxiliary geometry time mismatch at {key}")
        frames.append({"bjt": key, "utc": utc_expected, "split": split,
                       "data_path": row["data_path"], "aux_path": row["aux_path"],
                       "data_sha256": row.get("data_sha256", ""),
                       "aux_sha256": row.get("aux_sha256", "")})
    return frames, ghi_rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage-manifest", type=Path, required=True)
    p.add_argument("--ghi-rows", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--normalization", type=Path, required=True)
    p.add_argument("--model-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-frames", type=int, default=64)
    a = p.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world < 1 or not torch.cuda.is_available():
        raise RuntimeError("History R bank requires scheduler-assigned CUDA workers")
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dist.init_process_group(backend="nccl")

    frames, ghi_rows = load_requirements(a.stage_manifest, a.ghi_rows)
    if rank == 0:
        a.output.mkdir(parents=True, exist_ok=False)
        with (a.output / "frame_keys.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["frame_index", "bjt", "utc", "split", "data_sha256", "aux_sha256"])
            w.writeheader()
            for i, row in enumerate(frames):
                w.writerow({"frame_index": i, **{k: row[k] for k in ("bjt", "utc", "split", "data_sha256", "aux_sha256")}})
        feature_mm = np.lib.format.open_memmap(
            a.output / "history_agri_geometry_16x16.npy", mode="w+", dtype=np.float32,
            shape=(len(frames), 16, PATCH_SIZE, PATCH_SIZE))
        cot_mm = np.lib.format.open_memmap(
            a.output / "history_cot_16x16.npy", mode="w+", dtype=np.float32,
            shape=(len(frames), 1, PATCH_SIZE, PATCH_SIZE))
        valid_mm = np.lib.format.open_memmap(
            a.output / "history_cot_valid_16x16.npy", mode="w+", dtype=np.uint8,
            shape=(len(frames), 1, PATCH_SIZE, PATCH_SIZE))
        feature_mm[:] = np.nan
        cot_mm[:] = np.nan
        valid_mm[:] = 0
        feature_mm.flush(); cot_mm.flush(); valid_mm.flush()
    dist.barrier()

    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("frozen_corrected_r", a.model_source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load R source {a.model_source}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.COTUNet().to(torch.device("cuda", local_rank)).eval().requires_grad_(False)
    ckpt = torch.load(a.checkpoint, map_location=f"cuda:{local_rank}", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    norm = json.loads(a.normalization.read_text(encoding="utf-8"))
    mean = np.asarray(norm["mean"], dtype=np.float32)[None, :, None, None]
    std = np.asarray(norm["std"], dtype=np.float32)[None, :, None, None]
    if mean.shape != (1, 16, 1, 1) or std.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Frozen R train-only normalization is invalid")

    feature_mm = np.load(a.output / "history_agri_geometry_16x16.npy", mmap_mode="r+", allow_pickle=False)
    cot_mm = np.load(a.output / "history_cot_16x16.npy", mmap_mode="r+", allow_pickle=False)
    valid_mm = np.load(a.output / "history_cot_valid_16x16.npy", mmap_mode="r+", allow_pickle=False)
    selected = list(range(rank, len(frames), world))
    local_count = 0
    for start in range(0, len(selected), a.batch_frames):
        ids = selected[start:start + a.batch_frames]
        raws = []
        masks = []
        for frame_index in ids:
            row = frames[frame_index]
            scene = np.load(row["data_path"], mmap_mode="r", allow_pickle=False)
            if scene.shape != (20, 256, 256):
                raise ValueError(f"Unexpected corrected scene shape at {row['bjt']}: {scene.shape}")
            sl = (slice(PATCH_ROW, PATCH_ROW + PATCH_SIZE), slice(PATCH_COL, PATCH_COL + PATCH_SIZE))
            agri = np.asarray(scene[AGRI_INDEX, sl[0], sl[1]], dtype=np.float32)
            with np.load(row["aux_path"], allow_pickle=False) as aux:
                if set(aux.files) != {"cosSOZ", "cosRAA", "day_mask"}:
                    raise ValueError(f"Unexpected geometry keys at {row['bjt']}: {aux.files}")
                geom = np.stack([np.asarray(aux[k][sl], dtype=np.float32)
                                 for k in ("cosSOZ", "cosRAA", "day_mask")])
            raw = np.concatenate((agri, geom), axis=0)
            if raw.shape != (16, 16, 16) or not np.isfinite(raw).all():
                raise ValueError(f"Bad corrected R input at {row['bjt']}")
            valid = (raw[15:16] > 0.5) & (raw[13:14] > COS80)
            raws.append(raw)
            masks.append(valid)
        raw_batch = np.stack(raws).astype(np.float32, copy=False)
        valid_batch = np.stack(masks)
        z = ((raw_batch - mean) / std).astype(np.float32, copy=False)
        xt = torch.from_numpy(np.ascontiguousarray(z)).to(torch.device("cuda", local_rank), non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            pred = torch.expm1(model(xt).float())
            pred = pred.masked_fill(~torch.from_numpy(valid_batch).to(torch.device("cuda", local_rank)), 0.0)
        result = pred.cpu().numpy().astype(np.float32, copy=False)
        if result.shape != (len(ids), 1, PATCH_SIZE, PATCH_SIZE) or not np.isfinite(result).all() or (result < 0).any():
            raise RuntimeError("Frozen R produced invalid historical COT")
        feature_mm[ids] = raw_batch
        cot_mm[ids] = result
        valid_mm[ids] = valid_batch.astype(np.uint8)
        local_count += len(ids)
        if start == 0 or (start + len(ids)) % 1000 < a.batch_frames:
            print(json.dumps({"event": "BANK_PROGRESS", "rank": rank,
                              "frames_done": min(start + len(ids), len(selected)),
                              "frames_total": len(selected), "gpu": torch.cuda.get_device_name(local_rank)}), flush=True)
    feature_mm.flush(); cot_mm.flush(); valid_mm.flush()
    count_t = torch.tensor([local_count], device=torch.device("cuda", local_rank), dtype=torch.int64)
    dist.all_reduce(count_t, op=dist.ReduceOp.SUM)
    global_count = int(count_t.item())
    if global_count != len(frames):
        raise RuntimeError(f"Distributed bank wrote {global_count} frames, expected {len(frames)}")
    dist.barrier()

    if rank == 0:
        feature_mm = np.load(a.output / "history_agri_geometry_16x16.npy", mmap_mode="r", allow_pickle=False)
        cot_mm = np.load(a.output / "history_cot_16x16.npy", mmap_mode="r", allow_pickle=False)
        valid_mm = np.load(a.output / "history_cot_valid_16x16.npy", mmap_mode="r", allow_pickle=False)
        if not np.isfinite(feature_mm).all() or not np.isfinite(cot_mm).all() or (cot_mm < 0).any():
            raise RuntimeError("Bank arrays are incomplete or invalid")
        if np.any(cot_mm[valid_mm == 0] != 0):
            raise RuntimeError("Invalid R pixels were not set to zero")
        frame_keys_path = a.output / "frame_keys.csv"
        report = {
            "state": "HISTORICAL_COT_BANK_TRAIN_VAL_COMPLETE",
            "frames": len(frames),
            "distributed_frames_written": global_count,
            "frames_by_split": {s: sum(r["split"] == s for r in frames) for s in ("train", "val")},
            "ghi_rows_used_for_key_build": len(ghi_rows),
            "history_offsets_minutes": list(HISTORY_OFFSETS_MIN),
            "feature_shape": list(feature_mm.shape),
            "cot_shape": list(cot_mm.shape),
            "valid_pixel_count": int(valid_mm.sum()),
            "cot_valid_zero_check": "PASS",
            "arrays_sha256": {
                p.name: sha256(p) for p in (
                    a.output / "history_agri_geometry_16x16.npy",
                    a.output / "history_cot_16x16.npy",
                    a.output / "history_cot_valid_16x16.npy",
                    frame_keys_path,
                )
            },
            "stage_manifest_sha256": sha256(a.stage_manifest),
            "ghi_rows_sha256": sha256(a.ghi_rows),
            "r_checkpoint_sha256": sha256(a.checkpoint),
            "r_normalization_sha256": sha256(a.normalization),
            "r_model_source_sha256": sha256(a.model_source),
            "inference": "frozen corrected R; all 16 inputs train-normalized including day_mask; COT=expm1(log1p output); valid iff day_mask>0.5 and cosSOZ>cos80deg",
            "test_payloads_read": 0,
            "buffer_payloads_read": 0,
        }
        (a.output / "bank_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
