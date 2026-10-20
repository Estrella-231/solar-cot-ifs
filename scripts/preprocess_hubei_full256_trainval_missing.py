#!/usr/bin/env python3
"""Append missing frozen train/validation 20x256x256 Hubei AGRI scenes.

Uses the official Hubei crop/grid/calibration/solar-angle implementation and
the already audited 16x16 supplement manifest only to locate source HDFs.
Original scene files are read-only; outputs go to a new isolated supplement.
No test/buffer HDF or pixel payload is opened.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import netCDF4 as nc
import numpy as np
from solar_geometry_noaa import compute_solar_angles

STAMP = re.compile(r"_(\d{14})_\d{14}_4000M", re.IGNORECASE)
EXPECTED_SPLIT_SHA = "6ef8cbc7d2bb7d1b1d3defdfcba473cb171e06db0115c8206f7c289d8bad19c6"
EXPECTED_CHANNELS = (20, 256, 256)
_WORKER_MODULE = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, array, allow_pickle=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def load_split(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assignment = {r["date_BJT"]: r["split"] for r in rows}
    if len(assignment) != len(rows) or set(assignment.values()) != {"train", "val", "test", "buffer"}:
        raise ValueError("Invalid or duplicate frozen date assignment")
    return assignment


def load_trainval_manifest(path: Path) -> dict[datetime, str]:
    out: dict[datetime, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] not in {"train", "val"}:
                continue
            stamp = row["time_bjt"]
            bjt = datetime.strptime(stamp, "%Y%m%d%H%M")
            if bjt in out:
                raise ValueError(f"Duplicate train/val BJT key: {stamp}")
            out[bjt] = row["split"]
    return out


def load_patch_sources(path: Path) -> dict[datetime, dict[str, str]]:
    out: dict[datetime, dict[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            bjt = datetime.strptime(row["time_bjt"], "%Y-%m-%d %H:%M:%S")
            if bjt in out:
                raise ValueError(f"Duplicate HDF source BJT key: {bjt}")
            out[bjt] = row
    return out


def original_data_path(data_root: Path, bjt: datetime) -> Path:
    utc = bjt - timedelta(hours=8)
    return data_root / utc.strftime("%Y") / utc.strftime("%Y%m") / utc.strftime("%Y%m%d") / f"{utc:%Y%m%d%H%M}.npy"


def supplement_path(output_root: Path, bjt: datetime) -> Path:
    utc = bjt - timedelta(hours=8)
    return output_root / utc.strftime("%Y") / utc.strftime("%Y%m") / utc.strftime("%Y%m%d") / f"{utc:%Y%m%d%H%M}.npy"


def _init_worker(processor_path: str) -> None:
    global _WORKER_MODULE
    spec = importlib.util.spec_from_file_location("official_hubei_256_worker", processor_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import official processor: {processor_path}")
    _WORKER_MODULE = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_WORKER_MODULE)
    _WORKER_MODULE.compute_solar_angles = compute_solar_angles


def _make_full_scene(hdf_path: Path, module) -> np.ndarray:
    m = STAMP.search(hdf_path.name)
    if not m:
        raise ValueError(f"Cannot parse HDF timestamp: {hdf_path}")
    utc = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    map_r, map_c = module.sub_lat_idx, module.sub_lon_idx
    if map_r.shape != (256, 256) or map_c.shape != (256, 256):
        raise ValueError(f"Official crop maps have unexpected shapes: {map_r.shape}, {map_c.shape}")

    row0, row1 = int(map_r.min()), int(map_r.max()) + 1
    col0, col1 = int(map_c.min()), int(map_c.max()) + 1
    local_r, local_c = map_r - row0, map_c - col0
    agri = np.empty((15, 256, 256), dtype=np.float32)
    with nc.Dataset(hdf_path) as fdata:
        fmt, ds, cal_group = module.detect_hdf_format(fdata)
        for i in range(1, 16):
            raw = np.asarray(ds[f"NOMChannel{i:02d}"][row0:row1, col0:col1])[local_r, local_c]
            raw = raw.astype(np.int64, copy=False)
            raw = np.where((raw == 65534) | (raw == 65535), 4096, raw)
            raw = np.clip(raw, 0, 4096)
            cal_name = f"CALChannel{i:02d}" if fmt == "group" else f"CalChannel{i:02d}"
            lut = np.append(np.asarray(cal_group[cal_name][...], dtype=np.float32), np.float32(0.0))
            if int(raw.max(initial=0)) >= len(lut):
                raise ValueError(f"Calibration LUT too short for channel {i}: {hdf_path}")
            agri[i - 1] = lut[raw]

    soz, soa, day_mask = module.compute_solar_angles(module.sub_lat, module.sub_lon, utc)
    night = day_mask == 0
    agri[:6, night] = 0.0
    soa = soa.astype(np.float32, copy=True)
    soa[night] = 0.0
    scene = np.concatenate((
        agri,
        module.sub_saa.astype(np.float32, copy=False)[None],
        module.sub_saz.astype(np.float32, copy=False)[None],
        soa[None],
        soz.astype(np.float32, copy=False)[None],
        day_mask.astype(np.float32, copy=False)[None],
    ), axis=0)
    if scene.shape != EXPECTED_CHANNELS or not np.isfinite(scene).all():
        raise ValueError(f"Invalid full scene: shape={scene.shape}, finite={np.isfinite(scene).all()}")
    return scene


def _worker(job: tuple[str, str, str, str, str]) -> dict:
    hdf_text, output_text, processor_text, bjt_text, split = job
    if _WORKER_MODULE is None:
        _init_worker(processor_text)
    hdf = Path(hdf_text)
    bjt = datetime.strptime(bjt_text, "%Y-%m-%d %H:%M:%S")
    utc = bjt - timedelta(hours=8)
    if not hdf.is_file():
        raise FileNotFoundError(hdf)
    m = STAMP.search(hdf.name)
    if not m or datetime.strptime(m.group(1), "%Y%m%d%H%M%S") != utc:
        raise ValueError(f"HDF UTC timestamp does not match BJT key: {hdf} vs {utc}")
    scene = _make_full_scene(hdf, _WORKER_MODULE)
    out = Path(output_text) / utc.strftime("%Y") / utc.strftime("%Y%m") / utc.strftime("%Y%m%d") / f"{utc:%Y%m%d%H%M}.npy"
    atomic_npy(out, scene)
    return {
        "time_bjt": bjt.strftime("%Y%m%d%H%M"),
        "time_utc_hdf_start": utc.strftime("%Y%m%d%H%M%S"),
        "split": split,
        "source_hdf": str(hdf),
        "source_size_bytes": hdf.stat().st_size,
        "path_full256": str(out),
        "output_size_bytes": out.stat().st_size,
        "output_sha256": sha256(out),
        "shape": list(scene.shape),
        "dtype": str(scene.dtype),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frozen-split", type=Path, required=True)
    p.add_argument("--trainval-manifest", type=Path, required=True)
    p.add_argument("--patch-manifest", type=Path, required=True,
                   help="Audited 16x16 manifest used only as source-HDF lookup")
    p.add_argument("--official-processor", type=Path, required=True)
    p.add_argument("--original-data-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True,
                   help="New isolated directory for missing 20x256x256 arrays")
    p.add_argument("--output-manifest", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only-bjt", help="Bounded witness YYYYMMDDHHMM; omit for full supplement")
    p.add_argument("--resume", action="store_true",
                   help="Resume only if the isolated output root has matching run metadata")
    a = p.parse_args()
    split_hash = sha256(a.frozen_split)
    if split_hash != EXPECTED_SPLIT_SHA:
        raise ValueError(f"Unexpected frozen split SHA-256: {split_hash}")
    if a.output_manifest.exists():
        raise FileExistsError(f"Output manifest already exists: {a.output_manifest}")
    metadata = {
        "frozen_split_sha256": split_hash,
        "trainval_manifest_sha256": sha256(a.trainval_manifest),
        "patch_source_manifest_sha256": sha256(a.patch_manifest),
        "official_processor_sha256": sha256(a.official_processor),
        "solar_geometry_implementation_sha256": sha256(Path(__file__).with_name("solar_geometry_noaa.py")),
        "solar_geometry_version": "NOAA_Meeus_EOT_rad_to_deg_corrected_20261021",
        "output_root": str(a.output_root),
        "only_bjt": a.only_bjt,
        "array_contract": "20x256x256 float32; official Hubei crop and channel order",
    }
    run_metadata_path = a.output_root / "run_metadata.json"
    if a.output_root.exists():
        if not a.resume or not run_metadata_path.is_file():
            raise FileExistsError("Output root exists; --resume plus matching run_metadata.json is required")
        saved_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        if saved_metadata != metadata:
            raise ValueError("Resume metadata does not match current inputs/output contract")
    else:
        a.output_root.mkdir(parents=True)
        run_metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    assignment = load_split(a.frozen_split)
    trainval = load_trainval_manifest(a.trainval_manifest)
    patch_sources = load_patch_sources(a.patch_manifest)
    jobs, already, resumed_results = [], [], []
    for bjt, split in sorted(trainval.items()):
        expected_date_split = assignment.get(bjt.strftime("%Y-%m-%d"))
        if expected_date_split != split:
            raise ValueError(f"Split mismatch for {bjt}: {split} vs frozen {expected_date_split}")
        if a.only_bjt and bjt.strftime("%Y%m%d%H%M") != a.only_bjt:
            continue
        old = original_data_path(a.original_data_root, bjt)
        if old.exists():
            already.append(bjt)
            continue
        source = patch_sources.get(bjt)
        if source is None:
            raise FileNotFoundError(f"Missing full scene and no audited HDF source mapping for {bjt}")
        hdf = Path(source["source_hdf"])
        if not hdf.is_file():
            raise FileNotFoundError(hdf)
        existing_out = supplement_path(a.output_root, bjt)
        if existing_out.exists():
            if not a.resume:
                raise FileExistsError(f"Supplement frame exists; rerun with --resume after provenance check: {existing_out}")
            arr = np.load(existing_out, mmap_mode="r", allow_pickle=False)
            if arr.shape != EXPECTED_CHANNELS or arr.dtype != np.float32 or not np.isfinite(arr).all():
                raise ValueError(f"Invalid partial frame during resume: {existing_out}")
            utc = bjt - timedelta(hours=8)
            resumed_results.append({
                "time_bjt": bjt.strftime("%Y%m%d%H%M"),
                "time_utc_hdf_start": utc.strftime("%Y%m%d%H%M%S"),
                "split": split, "source_hdf": str(hdf),
                "source_size_bytes": hdf.stat().st_size,
                "path_full256": str(existing_out), "output_size_bytes": existing_out.stat().st_size,
                "output_sha256": sha256(existing_out), "shape": list(arr.shape), "dtype": str(arr.dtype),
            })
            continue
        jobs.append((str(hdf), str(a.output_root), str(a.official_processor),
                     bjt.strftime("%Y-%m-%d %H:%M:%S"), split))

    if a.only_bjt and len(jobs) != 1:
        raise ValueError(f"Expected one missing-frame job for {a.only_bjt}, got {len(jobs)}")
    print(json.dumps({"trainval_keys": len(trainval), "already_present": len(already),
                      "missing_to_process": len(jobs), "split_sha256": split_hash,
                      "workers": a.workers}, indent=2), flush=True)
    results = list(resumed_results)
    with ProcessPoolExecutor(max_workers=max(1, a.workers), initializer=_init_worker,
                             initargs=(str(a.official_processor),)) as pool:
        futures = [pool.submit(_worker, job) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if n % 100 == 0 or n == len(futures):
                print(f"FULL256_PREPROCESSED {n}/{len(futures)}", flush=True)
    results.sort(key=lambda x: x["time_bjt"])
    a.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with a.output_manifest.open("x", encoding="utf-8", newline="") as f:
        fields = ["time_bjt", "time_utc_hdf_start", "split", "source_hdf",
                  "source_size_bytes", "path_full256", "output_size_bytes",
                  "output_sha256", "shape", "dtype"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    counts = Counter(row["split"] for row in results)
    report = {
        "status": ("BOUNDED_FULL256_WITNESS_COMPLETE" if a.only_bjt and results else
                   "FULL256_FROZEN_TRAINVAL_SUPPLEMENT_COMPLETE" if results else
                   "NO_MISSING_FRAMES"),
        "bounded_witness_only": bool(a.only_bjt),
        "witness_bjt": a.only_bjt,
        "frozen_split_sha256": split_hash,
        "trainval_manifest_sha256": sha256(a.trainval_manifest),
        "patch_source_manifest_sha256": sha256(a.patch_manifest),
        "official_processor_sha256": sha256(a.official_processor),
        "solar_geometry_implementation_sha256": sha256(Path(__file__).with_name("solar_geometry_noaa.py")),
        "solar_geometry_version": "NOAA_Meeus_EOT_rad_to_deg_corrected_20261021",
        "trainval_keys": len(trainval), "original_full_frames_present": len(already),
        "supplemented_full_frames": len(results), "supplemented_by_split": dict(counts),
        "hdf_payloads_read": len(results), "test_or_buffer_hdf_payloads_read": 0,
        "output_manifest": str(a.output_manifest), "output_manifest_sha256": sha256(a.output_manifest),
        "output_root": str(a.output_root),
        "array_contract": "official 20-channel Hubei 256x256 crop, float32, UTC-stamped, isolated append-only supplement",
        "preprocessing_contract": "FY4B_wuhan_256.py grid maps, calibration LUT, channel order, NOAA solar angles and night masking",
    }
    report_path = a.output_manifest.with_suffix(".REPORT.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
