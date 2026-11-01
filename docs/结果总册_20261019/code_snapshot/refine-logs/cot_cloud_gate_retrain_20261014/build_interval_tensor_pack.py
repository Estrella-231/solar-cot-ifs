#!/usr/bin/env python3
"""Materialize same-split three-frame station patches for the interval GHI pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

CHANNELS = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14)
STATIONS = ("sili", "zhujia")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def aux_path(root: Path, rel: str) -> Path:
    t = datetime.strptime(Path(rel).stem, "%Y%m%d%H%M") + timedelta(hours=8)
    folder = root / "aux_256" / t.strftime("%Y%m")
    stem = t.strftime("%Y%m%d%H%M")
    paths = [folder / f"{stem}_{suffix}_aux.npz" for suffix in ("0", "1")]
    paths = [p for p in paths if p.is_file()]
    if len(paths) != 1:
        raise FileNotFoundError(f"expected one aux file for {rel}, found {paths}")
    return paths[0]


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    for key in ("root", "manifest", "scan_clear", "zero_flags", "data_root", "output"):
        ap.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    exp = args.root
    bank_dir = exp / "results" / "new_monthly_R_seed42_cot_bank_v1"
    pack_dir = exp / "data" / "ghi_head_pack_new_r_v1"
    bank_manifest = json.loads((bank_dir / "bank_manifest.json").read_text())
    pack_manifest = json.loads((pack_dir / "pack_manifest.json").read_text())
    if bank_manifest.get("test_payload_opened") is not False or pack_manifest.get("test_payload_opened") is not False:
        raise ValueError("test payload gate is not closed")
    if bank_manifest.get("state") != "COMPLETE" or pack_manifest.get("state") != "COMPLETE":
        raise ValueError("source feature bank/pack is incomplete")

    interval = read_csv(args.manifest)
    clear_rows = read_csv(args.scan_clear)
    if len(interval) != len(clear_rows) or len(interval) != 54461:
        raise ValueError("manifest/scan-clear row counts differ or unexpected")
    if any(r["split"] not in ("train", "val") for r in interval + clear_rows):
        raise ValueError("only train/validation rows may be materialized")
    clear_by_key = {(r["station"], r["target_time_utc"]): r for r in clear_rows}
    if len(clear_by_key) != len(clear_rows):
        raise ValueError("duplicate scan-clear key")

    pack_rows = read_csv(pack_dir / "rows.csv")
    pack_by_key = {(r["station"], r["target_time_utc"]): r for r in pack_rows}
    if len(pack_by_key) != len(pack_rows):
        raise ValueError("duplicate base GHI pack key")
    bank_rows = read_csv(bank_dir / "frame_keys.csv")
    bank_by_index = {int(r["index"]): r for r in bank_rows}
    if len(bank_by_index) != len(bank_rows):
        raise ValueError("duplicate COT bank index")

    flags = read_csv(args.zero_flags)
    flag_by_key = {(r["station"], r["target_time_utc"]): r for r in flags}
    if len(flag_by_key) != len(flags):
        raise ValueError("duplicate label quality flag key")

    records = []
    request_groups = defaultdict(list)
    for i, rec in enumerate(interval):
        key = (rec["station"], rec["target_time_utc"])
        if key not in pack_by_key or key not in flag_by_key or key not in clear_by_key:
            raise KeyError(f"missing aligned row for {key}")
        base = pack_by_key[key]
        flag = flag_by_key[key]
        cclear = clear_by_key[key]
        if rec["split"] != base["split"] or rec["split"] != flag["split"] or rec["split"] != cclear["split"]:
            raise ValueError(f"split mismatch for {key}")
        frame_indices = [int(x) for x in rec["bank_indices"].split("|")]
        if len(frame_indices) != 3:
            raise ValueError(f"non-three-frame row: {key}")
        frame_keys = [bank_by_index[x] for x in frame_indices]
        if any(f["split"] != rec["split"] for f in frame_keys):
            raise ValueError(f"cross-split context for {key}")
        weights = np.fromstring(rec["weights"], sep="|")
        offsets = np.fromstring(rec["scan_offsets_relative_target_seconds"], sep="|")
        scan_clear = np.asarray([float(cclear[f"scan_clear_{j}_wm2"]) for j in range(3)], dtype=np.float64)
        if weights.shape != (3,) or offsets.shape != (3,) or scan_clear.shape != (3,):
            raise ValueError(f"bad temporal metadata for {key}")
        if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-10):
            raise ValueError(f"invalid temporal weights for {key}")
        if not np.isfinite(scan_clear).all() or np.any(scan_clear < 0) or np.dot(weights, scan_clear) <= 0:
            raise ValueError(f"invalid scan-time clear-sky support for {key}")
        station_idx = STATIONS.index(rec["station"])
        record = {
            "split": rec["split"], "station": rec["station"], "target_time_utc": rec["target_time_utc"],
            "ghi": float(base["ghi"]), "clear": float(base["clear"]), "kt": float(base["kt"]),
            "candidate_qc1_main_mask": int(flag["candidate_qc1_main_mask"]),
            "quality_flag": flag["quality_flag"], "weights": weights.tolist(), "scan_clear": scan_clear.tolist(),
            "scan_offsets": offsets.tolist(), "bank_indices": frame_indices,
            "frame_relpaths": [f["agri_frame_relpath"] for f in frame_keys],
            "station_index": station_idx,
        }
        records.append(record)
        for j, frame in enumerate(frame_keys):
            rel = frame["agri_frame_relpath"]
            grid_rc = frame[f"{rec['station']}_grid_rc"]
            request_groups[rel].append((i, j, frame, grid_rc, station_idx))

    n = len(records)
    args.output.mkdir(parents=True)
    ag = np.lib.format.open_memmap(args.output / "agri_raw.npy", mode="w+", dtype=np.float32, shape=(n, 3, 13, 3, 3))
    av = np.lib.format.open_memmap(args.output / "agri_valid.npy", mode="w+", dtype=np.uint8, shape=(n, 3, 13, 3, 3))
    geo = np.lib.format.open_memmap(args.output / "geometry.npy", mode="w+", dtype=np.float32, shape=(n, 3, 2, 3, 3))
    cot = np.lib.format.open_memmap(args.output / "cot_log1p.npy", mode="w+", dtype=np.float32, shape=(n, 3, 1, 3, 3))
    bank_cot = np.load(bank_dir / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    bank_support = np.load(bank_dir / "input_valid_fraction.npy", mmap_mode="r", allow_pickle=False)
    loaded_frames = 0
    for rel, requests in request_groups.items():
        raw_path = args.data_root / "data" / rel
        raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
        if raw.shape != (20, 256, 256) or raw.dtype != np.float32:
            raise ValueError(f"bad AGRI source {raw_path}: shape={raw.shape}, dtype={raw.dtype}")
        aux_file = aux_path(args.data_root, rel)
        with np.load(aux_file, allow_pickle=False) as aux:
            bits_all = aux["agri_valid_bits"]
            for i, j, frame, rc_text, station_idx in requests:
                r, c = map(int, rc_text.split(","))
                r0, c0 = r - 1, c - 1
                bits = np.asarray(bits_all[r0:r0 + 3, c0:c0 + 3])
                patch = np.asarray(raw[np.asarray(CHANNELS), r0:r0 + 3, c0:c0 + 3], dtype=np.float32)
                valid = np.stack([((bits & (1 << channel)) != 0) for channel in CHANNELS]) & np.isfinite(patch)
                ag[i, j] = np.where(valid, patch, 0.0)
                av[i, j] = valid
                geo[i, j] = np.stack([aux[k][r0:r0 + 3, c0:c0 + 3] for k in ("cosSOZ", "cosRAA")]).astype(np.float32)
                bi = int(frame["index"])
                cot[i, j, 0] = np.asarray(bank_cot[bi, station_idx, 0, 7:10, 7:10], dtype=np.float32)
                if not np.isfinite(cot[i, j]).all() or not np.isfinite(geo[i, j]).all():
                    raise ValueError(f"nonfinite local input at sample={i}, frame={j}")
        loaded_frames += 1
        if loaded_frames % 2000 == 0:
            for arr in (ag, av, geo, cot):
                arr.flush()
            print(f"INTERVAL_PACK_FRAMES {loaded_frames}/{len(request_groups)}", flush=True)
    for arr in (ag, av, geo, cot):
        arr.flush()

    split = np.asarray([r["split"] for r in records])
    station = np.asarray([r["station"] for r in records])
    qc1 = np.asarray([r["candidate_qc1_main_mask"] == 1 for r in records])
    train_fit = (split == "train") & qc1
    if not train_fit.any():
        raise ValueError("QC1 train normalization support is empty")
    X = np.asarray(ag)
    V = np.asarray(av, dtype=bool)
    means, stds = [], []
    for channel in range(len(CHANNELS)):
        m = train_fit[:, None] & V[:, :, channel]
        vals = X[:, :, channel][m]
        if vals.size == 0:
            raise ValueError(f"no valid normalization pixels for AGRI channel {CHANNELS[channel]}")
        means.append(float(vals.mean(dtype=np.float64)))
        stds.append(float(max(vals.std(dtype=np.float64), 1e-6)))
    cot_values = np.asarray(cot[train_fit], dtype=np.float64)
    cmean = float(cot_values.mean())
    cstd = float(max(cot_values.std(), 1e-6))

    csv_fields = ["split", "station", "target_time_utc", "ghi", "clear", "kt", "quality_flag", "candidate_qc1_main_mask",
                  "weights", "scan_clear_0_wm2", "scan_clear_1_wm2", "scan_clear_2_wm2",
                  "scan_offset_0_s", "scan_offset_1_s", "scan_offset_2_s", "bank_indices", "frame_relpaths"]
    with (args.output / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "split": r["split"], "station": r["station"], "target_time_utc": r["target_time_utc"],
                "ghi": r["ghi"], "clear": r["clear"], "kt": r["kt"], "quality_flag": r["quality_flag"],
                "candidate_qc1_main_mask": r["candidate_qc1_main_mask"], "weights": "|".join(f"{x:.12g}" for x in r["weights"]),
                **{f"scan_clear_{j}_wm2": f"{r['scan_clear'][j]:.12g}" for j in range(3)},
                **{f"scan_offset_{j}_s": f"{r['scan_offsets'][j]:.6f}" for j in range(3)},
                "bank_indices": "|".join(map(str, r["bank_indices"])), "frame_relpaths": "|".join(r["frame_relpaths"]),
            })
    def count(sp, st, mask=None):
        m = (split == sp) & (station == st)
        if mask == "qc1": m &= qc1
        return int(m.sum())
    meta = {
        "state": "COMPLETE", "rows": n, "split_id": "hunan_monthly_block_2d_buffer1d_seed42_v1",
        "sample_key": "(station,target_time_utc)", "mode": "observed_AGRI_offline_interval_mean_diagnostic",
        "counts": {f"{sp}/{st}/{mask}": count(sp, st, mask) for sp in ("train", "val") for st in STATIONS for mask in (None, "qc1")},
        "array_shapes": {"agri_raw": list(ag.shape), "agri_valid": list(av.shape), "geometry": list(geo.shape), "cot_log1p": list(cot.shape)},
        "AGRI_channels": list(CHANNELS), "AGRI_norm_mean": means, "AGRI_norm_std": stds,
        "COT_transform": "log1p frozen monthly R bank", "COT_norm_mean": cmean, "COT_norm_std": cstd,
        "normalization_fit": "QC1 candidate train rows only, pooled over stations and three frames; shared by all ablation arms",
        "geometry": "per-frame source aux cosSOZ/cosRAA at nominal frame time; no spatial pooling",
        "input_spatial_support": "station-centered 3x3 patches, row-major positions retained",
        "COT_source_sha256": bank_manifest["R_checkpoint_sha256"], "split_manifest_sha256": sha256(args.root / "input" / "target_candidates_train_val.csv"),
        "interval_manifest_sha256": sha256(args.manifest), "scan_clear_values_sha256": sha256(args.scan_clear),
        "quality_flags_sha256": sha256(args.zero_flags), "source_pack_manifest_sha256": sha256(pack_dir / "pack_manifest.json"),
        "test_payload_opened": False, "same_split_context_only": True,
        "temporal_integration": "weights from actual scan times, C_j from pvlib Ineichen 0 m; C15 remains frozen 15-minute label denominator",
        "scan_clear_range_wm2": [float(np.min([r["scan_clear"] for r in records])), float(np.max([r["scan_clear"] for r in records]))],
        "files_sha256": {name: sha256(args.output / name) for name in ("agri_raw.npy", "agri_valid.npy", "geometry.npy", "cot_log1p.npy", "rows.csv")},
    }
    (args.output / "pack_manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
