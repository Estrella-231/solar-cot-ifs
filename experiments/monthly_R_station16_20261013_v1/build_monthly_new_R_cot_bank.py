#!/usr/bin/env python3
"""Infer frozen-R COT for the monthly train/validation target-frame table.

Inputs are observed Hunan AGRI frames and their solar-geometry sidecars. The
script writes log1p(COT) for each station's 16x16 patch, keyed by the frozen
monthly train/validation candidate table. It never reads test rows/payloads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

CHANNELS = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14)
STATIONS = ("sili", "zhujia")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def aux_path(root: Path, frame_relpath: str) -> Path:
    t = datetime.strptime(Path(frame_relpath).stem, "%Y%m%d%H%M") + timedelta(hours=8)
    directory = root / "aux_256" / t.strftime("%Y%m")
    matches = sorted(directory.glob(t.strftime("%Y%m%d%H%M") + "_*_aux.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one geometry sidecar for {t}: {matches}")
    return matches[0]


def rows_to_frames(candidate: Path):
    grouped = defaultdict(list)
    with candidate.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows or any(r["split"] not in ("train", "val") for r in rows):
        raise ValueError("candidate table must contain train/val rows only")
    for r in rows:
        grouped[r["target_time_utc"]].append(r)
    frames = []
    for stamp, group in sorted(grouped.items()):
        splits = {r["split"] for r in group}
        stations = {r["station"] for r in group}
        relpaths = {r["agri_frame_relpath"] for r in group}
        if len(splits) != 1 or stations != set(STATIONS) or len(relpaths) != 1:
            raise ValueError(f"inconsistent candidate rows at {stamp}")
        by_station = {r["station"]: r for r in group}
        coords = [tuple(map(int, by_station[s]["station_grid_rc"].split(","))) for s in STATIONS]
        if any(tuple(map(int, by_station[s]["station_patch_rc"].split(","))) != (8, 8) for s in STATIONS):
            raise ValueError(f"station center convention drift at {stamp}")
        frames.append({"split": next(iter(splits)), "target_time_utc": stamp,
                       "agri_frame_relpath": next(iter(relpaths)), "coords": coords})
    if len({(x["split"], x["target_time_utc"]) for x in frames}) != len(frames):
        raise ValueError("duplicate split/time frame key")
    return frames


def get_station_inputs(root: Path, frame: dict, simvp_mean: np.ndarray,
                       simvp_std: np.ndarray, rmean: np.ndarray, rstd: np.ndarray):
    rel = frame["agri_frame_relpath"]
    raw_path = root / "data" / rel
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    if raw.shape != (20, 256, 256) or raw.dtype != np.float32:
        raise ValueError(f"unexpected AGRI frame shape/dtype: {raw_path} {raw.shape} {raw.dtype}")
    ap = aux_path(root, rel)
    with np.load(ap, allow_pickle=False) as z:
        bits = z["agri_valid_bits"]
        geo_fields = [z[k] for k in ("cosSOZ", "cosRAA", "day_mask")]
        if bits.shape != (256, 256) or any(g.shape != (256, 256) for g in geo_fields):
            raise ValueError(f"unexpected auxiliary shapes: {ap}")

        xs, support = [], []
        for row, col in frame["coords"]:
            r0, c0 = row - 8, col - 8
            if r0 < 0 or c0 < 0 or r0 + 16 > 256 or c0 + 16 > 256:
                raise ValueError(f"station crop out of bounds: {frame['target_time_utc']} {(row, col)}")
            raw_agri = np.asarray(raw[np.asarray(CHANNELS), r0:r0 + 16, c0:c0 + 16], dtype=np.float32)
            pixel_bits = np.asarray(bits[r0:r0 + 16, c0:c0 + 16])
            valid = np.stack([((pixel_bits & (1 << k)) != 0) for k in CHANNELS]) & np.isfinite(raw_agri)
            # Match the monthly S loader: invalid AGRI values become zero after
            # train-normalization, then are returned to physical units before R.
            simvp_normed = np.where(valid, (raw_agri - simvp_mean) / simvp_std, 0.0)
            physical = simvp_normed * simvp_std + simvp_mean
            geom = np.stack([np.asarray(g[r0:r0 + 16, c0:c0 + 16], dtype=np.float32)
                             for g in geo_fields])
            x = np.concatenate((physical, geom), axis=0)
            x = (x - rmean) / rstd
            x = np.where(np.isfinite(x), x, 0.0).astype(np.float32, copy=False)
            xs.append(x)
            support.append(float(valid.mean()))
    return np.stack(xs), np.asarray(support, dtype=np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--monthly-norm", type=Path, required=True)
    p.add_argument("--r-checkpoint", type=Path, required=True)
    p.add_argument("--r-source", type=Path, required=True)
    p.add_argument("--r-norm", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-frames", type=int, default=128)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if a.output.exists():
        raise FileExistsError(f"output exists; choose a new run path: {a.output}")
    frames = rows_to_frames(a.candidate)
    if sha256(a.candidate) != "b85c2841e33cc92eda848acb183f926d49564b6c131d6418730e6b53978e2807":
        raise ValueError("candidate manifest hash differs from the audited train/val table")
    if sum(x["split"] == "train" for x in frames) != 51236 or sum(x["split"] == "val" for x in frames) != 4497:
        raise ValueError("train/validation candidate frame counts differ from the audited table")
    norm = json.loads(a.monthly_norm.read_text())
    smean, sstd = np.asarray(norm["mean"], np.float32), np.asarray(norm["std"], np.float32)
    if smean.shape != (13,) or sstd.shape != (13,) or not np.isfinite(smean).all() or (sstd <= 0).any():
        raise ValueError("monthly train normalization does not match the 13 AGRI channels")
    checkpoint = torch.load(a.r_checkpoint, map_location="cpu", weights_only=False)
    rnorm = json.loads(a.r_norm.read_text(encoding="utf-8"))
    rmean, rstd = np.asarray(rnorm["mean"], np.float32)[:, None, None], np.asarray(rnorm["std"], np.float32)[:, None, None]
    if rmean.shape != (16, 1, 1) or not np.isfinite(rmean).all() or not np.isfinite(rstd).all() or (rstd <= 0).any():
        raise ValueError("frozen R normalization is invalid")
    if sha256(a.r_checkpoint) != "342596094d1a41aa8b141972a7b4adadece179b4b9f58170f4e7f409a894f001":
        raise ValueError("new monthly R checkpoint hash differs from the audited source")

    sys.path.insert(0, str(a.r_source.parent))
    from train_station16_R import COTUNet
    if sha256(a.r_source) != "7a2e5cb441cdb5132a537cd03580017801595bdea839719d1773398d267210ce":
        raise ValueError("new monthly R implementation differs from the trained source")
    if checkpoint["metadata"].get("training_script_sha256") != sha256(a.r_source):
        raise ValueError("new R checkpoint is not bound to the supplied implementation")

    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    model = COTUNet().cuda().float().eval().requires_grad_(False)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint

    a.output.mkdir(parents=True)
    keys_path = a.output / "frame_keys.csv"
    with keys_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "split", "target_time_utc", "agri_frame_relpath", "sili_grid_rc", "zhujia_grid_rc"])
        for i, fr in enumerate(frames):
            w.writerow([i, fr["split"], fr["target_time_utc"], fr["agri_frame_relpath"],
                        ",".join(map(str, fr["coords"][0])), ",".join(map(str, fr["coords"][1]))])
    target = np.lib.format.open_memmap(a.output / "cot_log1p.npy", mode="w+",
                                      dtype=np.float32, shape=(len(frames), 2, 1, 16, 16))
    target[:] = np.nan
    target.flush()
    support = np.lib.format.open_memmap(a.output / "input_valid_fraction.npy", mode="w+",
                                       dtype=np.float32, shape=(len(frames), 2))
    support[:] = np.nan
    errors = []
    pending_x, pending_index, pending_support = [], [], []
    completed = 0

    def infer_pending():
        nonlocal completed, pending_x, pending_index, pending_support
        if not pending_x:
            return
        x = torch.from_numpy(np.stack(pending_x)).to("cuda", dtype=torch.float32, non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            pred = model(x).float().cpu().numpy()
        if not np.isfinite(pred).all():
            raise RuntimeError("nonfinite COT prediction")
        for j, idx in enumerate(pending_index):
            target[idx] = pred[2*j:2*j+2]
            support[idx] = pending_support[j]
        completed += len(pending_index)
        target.flush(); support.flush()
        (a.output / "progress.json").write_text(json.dumps({
            "state": "BUILDING_MONTHLY_REAL_AGRI_COT_BANK", "frames_complete": completed,
            "frames_total": len(frames), "frames_with_source_errors": len(errors),
            "test_payload_opened": False,
        }, indent=2) + "\n")
        pending_x, pending_index, pending_support = [], [], []
        print(f"PROGRESS {completed}/{len(frames)} missing={len(errors)}", flush=True)

    for i, fr in enumerate(frames):
        try:
            xs, sv = get_station_inputs(a.data_root, fr, smean[:, None, None], sstd[:, None, None], rmean, rstd)
        except (OSError, ValueError, KeyError) as e:
            errors.append({"index": i, "split": fr["split"], "target_time_utc": fr["target_time_utc"],
                           "agri_frame_relpath": fr["agri_frame_relpath"], "error": repr(e)})
            continue
        pending_x.extend(xs)
        pending_index.append(i)
        pending_support.append(sv)
        if len(pending_index) >= a.batch_frames:
            infer_pending()
    infer_pending()

    report = {
        "state": "COMPLETE" if not errors else "COMPLETE_WITH_UNAVAILABLE_INPUT_FRAMES",
        "split_id": "hunan_monthly_block_2d_buffer1d_seed42_v1",
        "candidate_sha256": sha256(a.candidate), "monthly_norm_sha256": sha256(a.monthly_norm),
        "R_norm_sha256": sha256(a.r_norm),
        "R_validation_audit": str(a.r_checkpoint.parent / "validation_audit_fp64.json"),
        "R_validation_checkpoint_sha256": sha256(a.r_checkpoint),
        "R_checkpoint_sha256": sha256(a.r_checkpoint), "R_source_sha256": sha256(a.r_source),
        "R_input": "13 physical AGRI channels + cosSOZ/cosRAA/day_mask; station-centered 16x16 crop",
        "R_output": "nonnegative log1p(COT), FP32 inference, no autocast",
        "layout": "[unique_trainval_target_frame, station(sili,zhujia), 1, 16, 16]",
        "frames_total": len(frames), "train_frames": sum(x["split"] == "train" for x in frames),
        "val_frames": sum(x["split"] == "val" for x in frames),
        "frames_with_source_errors": len(errors), "frames_complete": len(frames) - len(errors),
        "mean_input_valid_fraction": np.nanmean(support, axis=0).tolist(),
        "test_payload_opened": False,
        "errors": errors,
    }
    (a.output / "bank_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "errors"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
