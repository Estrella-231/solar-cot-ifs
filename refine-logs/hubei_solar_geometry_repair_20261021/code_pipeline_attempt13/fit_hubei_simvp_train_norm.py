#!/usr/bin/env python3
"""Fit deterministic train-only SimVP normalization from frozen Hubei windows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

AGRI_CHANNELS = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14]
AUX_NAMES = ["cosSOZ", "cosRAA", "day_mask"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--index-csv", type=Path, required=True)
    p.add_argument("--output-npz", type=Path, required=True)
    p.add_argument("--report-json", type=Path, required=True)
    p.add_argument("--sampled-frame-csv", type=Path, required=True)
    p.add_argument("--max-frames", type=int, default=2048)
    a = p.parse_args()
    for out in (a.output_npz, a.report_json, a.sampled_frame_csv):
        if out.exists():
            raise FileExistsError(out)

    inputs: dict[str, str] = {}
    known_geometry: dict[str, str] = {}
    with a.index_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] != "train":
                continue
            data = row["data_paths"].split("|")
            aux = row["aux_paths"].split("|")
            if len(data) != 24 or len(aux) != 24:
                raise ValueError("Expected 24 aligned frames per train sequence")
            for dp, ap in zip(data[:8], aux[:8]):
                if dp in inputs and inputs[dp] != ap:
                    raise ValueError(f"Ambiguous aux mapping for {dp}")
                inputs[dp] = ap
            # The official loader supplies known solar geometry for all 16
            # future frames as well as the 8 history frames. Fit its
            # statistics on unique train-sequence timestamps, not only on
            # the history subset. AGRI statistics remain history-only.
            for dp, ap in zip(data, aux):
                if dp in known_geometry and known_geometry[dp] != ap:
                    raise ValueError(f"Ambiguous geometry mapping for {dp}")
                known_geometry[dp] = ap
    ordered = sorted(inputs.items())
    if not ordered:
        raise ValueError("No training input frames found")
    if a.max_frames < 1:
        raise ValueError("max-frames must be >=1")
    if len(ordered) > a.max_frames:
        indices = np.linspace(0, len(ordered) - 1, a.max_frames, dtype=np.int64)
        selected = [ordered[int(i)] for i in indices]
        if len({x[0] for x in selected}) != len(selected):
            raise RuntimeError("Stratified frame sample unexpectedly contains duplicates")
    else:
        selected = ordered
    geometry_ordered = sorted(known_geometry.items())
    if len(geometry_ordered) > a.max_frames:
        geometry_indices = np.linspace(0, len(geometry_ordered) - 1, a.max_frames, dtype=np.int64)
        geometry_selected = [geometry_ordered[int(i)] for i in geometry_indices]
        if len({x[0] for x in geometry_selected}) != len(geometry_selected):
            raise RuntimeError("Stratified geometry sample unexpectedly contains duplicates")
    else:
        geometry_selected = geometry_ordered

    ag_sum = np.zeros(len(AGRI_CHANNELS), np.float64)
    ag_sq = np.zeros_like(ag_sum)
    ag_count = np.zeros_like(ag_sum)
    aux_sum = np.zeros(len(AUX_NAMES), np.float64)
    aux_sq = np.zeros_like(aux_sum)
    aux_count = np.zeros_like(aux_sum)
    rows = []
    for n, (data_path, aux_path) in enumerate(selected, 1):
        scene = np.load(data_path, mmap_mode="r", allow_pickle=False)
        if scene.shape != (20, 256, 256) or scene.dtype != np.float32:
            raise ValueError(f"Unexpected scene {data_path}: {scene.shape}/{scene.dtype}")
        channels = np.asarray(scene[AGRI_CHANNELS], dtype=np.float64)
        for k, source_channel in enumerate(AGRI_CHANNELS):
            image = channels[k]
            valid = np.isfinite(image)
            if source_channel < 6:
                valid &= image != 0
            if valid.any():
                ag_sum[k] += image[valid].sum(dtype=np.float64)
                ag_sq[k] += np.square(image[valid]).sum(dtype=np.float64)
                ag_count[k] += int(valid.sum())
        rows.append({"statistic": "agri_history", "data_path": data_path, "aux_path": aux_path})
        if n % 256 == 0 or n == len(selected):
            print(f"NORM_AGRI_HISTORY_FRAMES {n}/{len(selected)}", flush=True)

    for n, (data_path, aux_path) in enumerate(geometry_selected, 1):
        with np.load(aux_path, allow_pickle=False) as z:
            aux = np.stack([np.asarray(z[name], dtype=np.float32) for name in AUX_NAMES])
        if aux.shape != (3, 256, 256) or not np.isfinite(aux).all():
            raise ValueError(f"Invalid aux sidecar {aux_path}: {aux.shape}")
        af = aux.astype(np.float64, copy=False).reshape(3, -1)
        aux_sum += af.sum(axis=1)
        aux_sq += np.square(af).sum(axis=1)
        aux_count += af.shape[1]
        rows.append({"statistic": "known_geometry_all_train_sequence_times",
                     "data_path": data_path, "aux_path": aux_path})
        if n % 256 == 0 or n == len(geometry_selected):
            print(f"NORM_GEOMETRY_FRAMES {n}/{len(geometry_selected)}", flush=True)

    if np.any(ag_count == 0) or np.any(aux_count == 0):
        raise ValueError("Empty normalization support")
    ag_mean = ag_sum / ag_count
    ag_std = np.sqrt(np.maximum(ag_sq / ag_count - ag_mean ** 2, 1e-8))
    aux_mean = aux_sum / aux_count
    aux_std = np.sqrt(np.maximum(aux_sq / aux_count - aux_mean ** 2, 1e-8))
    ag_std = np.maximum(ag_std, 1e-6)
    aux_std = np.maximum(aux_std, 1e-6)
    input_mean = np.concatenate([ag_mean, aux_mean]).astype(np.float32)
    input_std = np.concatenate([ag_std, aux_std]).astype(np.float32)
    atomic_npz(a.output_npz,
               agri_mean=ag_mean.astype(np.float32), agri_std=ag_std.astype(np.float32),
               aux_mean=aux_mean.astype(np.float32), aux_std=aux_std.astype(np.float32),
               input_mean=input_mean, input_std=input_std,
               agri_channels=np.asarray(AGRI_CHANNELS, np.int16),
               aux_names=np.asarray(AUX_NAMES, dtype="U16"))
    a.sampled_frame_csv.parent.mkdir(parents=True, exist_ok=True)
    with a.sampled_frame_csv.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["statistic", "data_path", "aux_path"])
        writer.writeheader(); writer.writerows(rows)
    report = {
        "status": "TRAIN_ONLY_NORMALIZATION_FIT",
        "index_csv": str(a.index_csv),
        "index_csv_sha256": sha256(a.index_csv),
        "sampled_frame_manifest": str(a.sampled_frame_csv),
        "sampled_frame_manifest_sha256": sha256(a.sampled_frame_csv),
        "available_unique_train_input_frames": len(ordered),
        "used_agri_history_frames": len(selected),
        "available_unique_train_geometry_frames": len(geometry_ordered),
        "used_geometry_frames": len(geometry_selected),
        "selection": "AGRI statistics use unique training history frames; known solar-geometry statistics use unique timestamps from all train sequences including future target times; both are sorted by path and evenly sampled with linspace when above max-frames; no validation/test frames",
        "agri_channel_indices": AGRI_CHANNELS,
        "aux_names": AUX_NAMES,
        "visible_zero_policy": "exclude exact zeros from C01-C06 mean/std; retain all finite pixels for other AGRI channels",
        "agri_mean": ag_mean.tolist(), "agri_std": ag_std.tolist(),
        "aux_mean": aux_mean.tolist(), "aux_std": aux_std.tolist(),
        "output_npz": str(a.output_npz), "output_npz_sha256": sha256(a.output_npz),
        "test_or_buffer_payloads_read": 0,
    }
    a.report_json.parent.mkdir(parents=True, exist_ok=True)
    a.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
