#!/usr/bin/env python3
"""Independent key, array, and frozen-R spot audit for the history bank."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch


AGRI_INDEX = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
OFFSETS = (105, 90, 75, 60, 45, 30, 15, 0)
R0, C0, SIZE = 103, 118, 16
COS80 = float(np.cos(np.deg2rad(80.0)))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def t(key: str) -> datetime:
    return datetime.strptime(key, "%Y%m%d%H%M")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--stage-manifest", type=Path, required=True)
    p.add_argument("--ghi-rows", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--normalization", type=Path, required=True)
    p.add_argument("--model-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = json.loads((a.bank / "bank_report.json").read_text(encoding="utf-8"))
    if report.get("state") != "HISTORICAL_COT_BANK_TRAIN_VAL_COMPLETE" or report.get("test_payloads_read") != 0 or report.get("buffer_payloads_read") != 0:
        raise ValueError("Historical COT bank is not train/val-only COMPLETE")
    arrays = {name: a.bank / name for name in (
        "history_agri_geometry_16x16.npy", "history_cot_16x16.npy",
        "history_cot_valid_16x16.npy", "frame_keys.csv")}
    for name, path in arrays.items():
        if sha(path) != report["arrays_sha256"][name]:
            raise ValueError(f"Bank hash mismatch: {name}")
    keys = read_csv(arrays["frame_keys.csv"])
    if len(keys) != report["frames"] or [int(r["frame_index"]) for r in keys] != list(range(len(keys))):
        raise ValueError("Frame index table length/order mismatch")
    by_bjt = {r["bjt"]: int(r["frame_index"]) for r in keys}
    if len(by_bjt) != len(keys) or list(by_bjt) != sorted(by_bjt):
        raise ValueError("Frame keys are duplicated or not canonical sorted order")
    feature = np.load(arrays["history_agri_geometry_16x16.npy"], mmap_mode="r", allow_pickle=False)
    cot = np.load(arrays["history_cot_16x16.npy"], mmap_mode="r", allow_pickle=False)
    valid = np.load(arrays["history_cot_valid_16x16.npy"], mmap_mode="r", allow_pickle=False)
    if feature.shape != (len(keys), 16, 16, 16) or cot.shape != (len(keys), 1, 16, 16) or valid.shape != cot.shape:
        raise ValueError("History bank tensor shapes mismatch")
    if not np.isfinite(feature).all() or not np.isfinite(cot).all() or (cot < 0).any():
        raise ValueError("Nonfinite or negative bank values")
    if int(valid.sum()) != int(report["valid_pixel_count"]) or np.any(cot[valid == 0] != 0):
        raise ValueError("Validity mask and invalid-COT zeroing disagree")

    stage = {r["bjt"]: r for r in read_csv(a.stage_manifest)}
    ghi = read_csv(a.ghi_rows)
    key_split = {r["bjt"]: r["split"] for r in keys}
    required = 0
    for row in ghi:
        if row["split"] not in {"train", "val"}:
            raise ValueError(f"Non-train/val sample in GHI rows: {row['split']}")
        init = t(row["init_time_bjt"])
        if t(row["target_time_bjt"]) != init + timedelta(minutes=int(row["lead_minutes"])):
            raise ValueError("Non-canonical GHI target-time binding")
        for off in OFFSETS:
            bjt = (init - timedelta(minutes=off)).strftime("%Y%m%d%H%M")
            required += 1
            if bjt not in by_bjt or bjt not in stage or key_split[bjt] != row["split"] or stage[bjt]["split"] != row["split"]:
                raise ValueError(f"Missing or cross-split history link for {row['station']} {row['init_time_bjt']} {bjt}")

    if not torch.cuda.is_available():
        raise RuntimeError("Spot replay needs allocated CUDA")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    spec = importlib.util.spec_from_file_location("frozen_corrected_r_audit", a.model_source)
    if spec is None or spec.loader is None:
        raise ImportError(a.model_source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    device = torch.device("cuda:0")
    model = module.COTUNet().to(device).eval().requires_grad_(False)
    state = torch.load(a.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    norm = json.loads(a.normalization.read_text(encoding="utf-8"))
    mean = np.asarray(norm["mean"], np.float32)[None, :, None, None]
    std = np.asarray(norm["std"], np.float32)[None, :, None, None]
    def load_raw(frame_index: int):
        r = keys[frame_index]
        raw_scene = np.load(stage[r["bjt"]]["data_path"], mmap_mode="r", allow_pickle=False)
        sl = (slice(R0, R0 + SIZE), slice(C0, C0 + SIZE))
        agri = np.asarray(raw_scene[AGRI_INDEX, sl[0], sl[1]], dtype=np.float32)
        with np.load(stage[r["bjt"]]["aux_path"], allow_pickle=False) as aux:
            geom = np.stack([np.asarray(aux[k][sl], dtype=np.float32)
                             for k in ("cosSOZ", "cosRAA", "day_mask")])
        raw = np.concatenate((agri, geom), axis=0)
        valid_mask = (raw[15:16] > 0.5) & (raw[13:14] > COS80)
        return raw, valid_mask

    # The bank uses 64-frame autocast batches partitioned by rank-strided
    # frame indices. Replaying those exact batch shapes avoids treating normal
    # FP16 batch-kernel roundoff as a source/data mismatch.
    spot = []
    rank0_ids = np.arange(0, len(keys), 4, dtype=np.int64)
    for split in ("train", "val"):
        candidates = [int(i) for i in rank0_ids if keys[int(i)]["split"] == split]
        if not candidates:
            raise ValueError(f"No rank-0 bank frames for split={split}")
        selected_idx = candidates[0]
        pos = int(np.flatnonzero(rank0_ids == selected_idx)[0])
        batch_start = (pos // 64) * 64
        batch_ids = rank0_ids[batch_start:batch_start + 64]
        raw_masks = [load_raw(int(i)) for i in batch_ids]
        raw_batch = np.stack([x[0] for x in raw_masks]).astype(np.float32, copy=False)
        mask_batch = np.stack([x[1] for x in raw_masks])
        z = ((raw_batch - mean) / std).astype(np.float32)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            pred = torch.expm1(model(torch.from_numpy(z).to(device)).float())
            pred = pred.masked_fill(~torch.from_numpy(mask_batch).to(device), 0.0)
        replay = pred.cpu().numpy()
        cot_err = np.abs(replay.astype(np.float64) - cot[batch_ids].astype(np.float64))
        feature_err = np.abs(raw_batch.astype(np.float64) - feature[batch_ids].astype(np.float64))
        max_cot_err = float(cot_err.max())
        max_feature_err = float(feature_err.max())
        if max_cot_err > 1e-6 or max_feature_err != 0.0:
            key = keys[selected_idx]["bjt"]
            raise RuntimeError(f"Independent source replay mismatch at {split}/{key}: COT={max_cot_err}, input={max_feature_err}")
        spot.append({"split": split, "bjt": keys[selected_idx]["bjt"], "bank_index": selected_idx,
                     "rank_replayed": 0, "bank_batch_size_replayed": int(len(batch_ids)),
                     "max_abs_cot_replay": max_cot_err, "max_abs_feature_replay": max_feature_err})
    receipt = {
        "state": "PASS_HISTORY_COT_BANK_INDEPENDENT_AUDIT",
        "bank_report_sha256": sha(a.bank / "bank_report.json"),
        "frames": len(keys), "ghi_rows_checked": len(ghi),
        "history_links_checked": required,
        "train_frames": sum(r["split"] == "train" for r in keys),
        "val_frames": sum(r["split"] == "val" for r in keys),
        "train_val_only": True,
        "key_split_audit": "PASS",
        "arrays_sha256_verified": True,
        "frozen_R_source_replay": spot,
        "replay_contract": "independent frozen-R rerun of the exact rank-0 64-frame batches used by bank builder, preserving FP16 autocast batch shape",
        "r_checkpoint_sha256": sha(a.checkpoint),
        "r_normalization_sha256": sha(a.normalization),
        "r_model_source_sha256": sha(a.model_source),
        "test_payloads_read": 0, "buffer_payloads_read": 0,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
