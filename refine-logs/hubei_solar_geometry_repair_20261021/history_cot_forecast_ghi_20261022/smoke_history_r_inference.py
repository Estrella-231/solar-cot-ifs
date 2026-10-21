#!/usr/bin/env python3
"""Run one train and one validation frame through the frozen corrected Hubei R."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage-manifest", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--normalization", type=Path, required=True)
    p.add_argument("--model-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("R inference smoke requires the scheduler-assigned CUDA device")
    spec = importlib.util.spec_from_file_location("hubei_cot_repaired", a.model_source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen R implementation: {a.model_source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.COTUNet().cuda().eval()
    state = torch.load(a.checkpoint, map_location="cuda:0", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    norm = json.loads(a.normalization.read_text(encoding="utf-8"))
    mean = np.asarray(norm["mean"], dtype=np.float32)[None, :, None, None]
    std = np.asarray(norm["std"], dtype=np.float32)[None, :, None, None]
    if mean.shape != (1, 16, 1, 1) or std.shape != mean.shape or (std <= 0).any():
        raise ValueError("Frozen R normalization shape/value mismatch")
    rows = list(csv.DictReader(a.stage_manifest.open("r", newline="", encoding="utf-8-sig")))
    by_split = {}
    for row in rows:
        if row["split"] in {"train", "val"} and row["split"] not in by_split:
            by_split[row["split"]] = row
    agri_index = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
    records = []
    for split in ("train", "val"):
        row = by_split[split]
        scene = np.load(row["data_path"], mmap_mode="r", allow_pickle=False)
        if scene.shape != (20, 256, 256):
            raise ValueError(f"Expected canonical full scene [20,256,256], got {scene.shape}")
        with np.load(row["aux_path"], allow_pickle=False) as aux:
            if set(aux.files) != {"cosSOZ", "cosRAA", "day_mask"}:
                raise ValueError(f"Unexpected corrected-geometry keys: {aux.files}")
            sl = (slice(103, 119), slice(118, 134))
            agri = np.asarray(scene[agri_index, sl[0], sl[1]], dtype=np.float32)
            geometry = np.stack([np.asarray(aux[k][sl], dtype=np.float32)
                                 for k in ("cosSOZ", "cosRAA", "day_mask")])
        raw = np.concatenate((agri, geometry), axis=0)
        if raw.shape != (16, 16, 16) or not np.isfinite(raw).all():
            raise ValueError(f"Invalid corrected R tensor shape/content: {raw.shape}")
        valid = (raw[15:16] > 0.5) & (raw[13:14] > np.cos(np.deg2rad(80.0)))
        z = ((raw[None] - mean) / std).astype(np.float32)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            cot = torch.expm1(model(torch.from_numpy(z).cuda()).float())
            cot = cot.masked_fill(~torch.from_numpy(valid[None]).cuda(), 0.0)
        output = cot.cpu().numpy()
        if output.shape != (1, 1, 16, 16) or not np.isfinite(output).all() or (output < 0).any():
            raise ValueError(f"Invalid frozen R output for {split}")
        records.append({"split": split, "bjt": row["bjt"], "utc": row["utc"],
                        "input_shape": list(raw.shape), "output_shape": list(output.shape),
                        "valid_pixels": int(valid.sum()), "cot_min": float(output.min()),
                        "cot_max": float(output.max()), "data_sha256": row["data_sha256"],
                        "aux_sha256": row["aux_sha256"]})
    receipt = {"state": "PASS_FROZEN_CORRECTED_R_TRAIN_VAL_SMOKE",
               "gpu": torch.cuda.get_device_name(0), "records": records,
               "checkpoint_sha256": sha(a.checkpoint),
               "normalization_sha256": sha(a.normalization),
               "model_source_sha256": sha(a.model_source),
               "test_payloads_read": 0, "buffer_payloads_read": 0}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
