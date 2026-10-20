#!/usr/bin/env python3
"""Build frozen train/validation Hubei 16x16 AGRI NPZs from original FY-4B HDF.

This is an append-only supplement for 2026-06-01 through 2026-07-14. It uses
the original Hubei 256-crop processor's calibration, crop, and solar-angle
contract, but reads/writes only the 16x16 station patch. HDF payloads are
opened only for frozen train/validation BJT dates; test/buffer are metadata-only.
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
PATCH_R, PATCH_C, PATCH_SIZE = 103, 118, 16
START_BJT = "2026-06-01"
END_BJT = "2026-07-14"
_WORKER_MODULE = None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
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


def load_assignments(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {r["date_BJT"]: r["split"] for r in rows}
    if len(out) != len(rows) or set(out.values()) != {"train", "val", "test", "buffer"}:
        raise ValueError("Unexpected or duplicate frozen date assignment")
    return out


def discover(root: Path, assignments: dict[str, str], bjt_start: datetime,
             bjt_end_exclusive: datetime) -> dict[str, dict[str, Path]]:
    selected: dict[str, dict[str, Path]] = {}
    for utc_day in (datetime(2026, 5, 31) + timedelta(days=i)
                    for i in range((datetime(2026, 7, 15) - datetime(2026, 5, 31)).days)):
        day_dir = root / utc_day.strftime("%Y") / utc_day.strftime("%Y%m%d")
        if not day_dir.is_dir():
            continue
        for path in day_dir.glob("FY4B*AGRI*HDF*"):
            m = STAMP.search(path.name)
            if not m:
                continue
            utc = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
            bjt = utc + timedelta(hours=8)
            if not (bjt_start <= bjt < bjt_end_exclusive):
                continue
            stamp = bjt.strftime("%Y%m%d%H%M")
            date = bjt.strftime("%Y-%m-%d")
            split = assignments.get(date)
            if split is None:
                continue
            bucket = selected.setdefault(stamp, {})
            bucket[str(path)] = path
    return selected


def choose_source(candidates: dict[str, Path]) -> tuple[Path, str]:
    # Match the original data16 lineage: real-time HDF first, historical fallback.
    realtime = [p for p in candidates.values() if "real_time_data" in p.parts]
    chosen = sorted(realtime or list(candidates.values()))[0]
    source = "realtime" if realtime else "historical_fallback"
    return chosen, source


def preprocess_one(path: Path, module, output_root: Path, bjt: datetime,
                   split: str, source_kind: str) -> dict:
    m = STAMP.search(path.name)
    if not m:
        raise ValueError(f"Cannot parse HDF filename timestamp: {path}")
    utc = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")

    # Same official 256-crop mapping, then the authorized 16x16 crop.
    rs = slice(PATCH_R, PATCH_R + PATCH_SIZE)
    cs = slice(PATCH_C, PATCH_C + PATCH_SIZE)
    map_r = module.sub_lat_idx[rs, cs]
    map_c = module.sub_lon_idx[rs, cs]
    if map_r.shape != (16, 16) or map_c.shape != (16, 16):
        raise ValueError("Official HDF index map does not contain the required 16x16 patch")

    with nc.Dataset(path) as fdata:
        fmt, ds, cal_group = module.detect_hdf_format(fdata)
        agri = np.empty((15, 16, 16), dtype=np.float32)
        # The historical implementation materialized each full-disk channel
        # before taking a 16x16 patch. Read only the bounding hyperslab around
        # the official row/column map; this is numerically identical but avoids
        # repeatedly decoding multi-million-pixel arrays for every timestamp.
        row0, row1 = int(map_r.min()), int(map_r.max()) + 1
        col0, col1 = int(map_c.min()), int(map_c.max()) + 1
        local_r = map_r - row0
        local_c = map_c - col0
        for i in range(1, 16):
            name = f"NOMChannel{i:02d}"
            cal_name = (f"CALChannel{i:02d}" if fmt == "group" else f"CalChannel{i:02d}")
            raw = np.asarray(ds[name][row0:row1, col0:col1])[local_r, local_c]
            cal = np.asarray(cal_group[cal_name][...], dtype=np.float32)
            raw = raw.astype(np.int64, copy=False)
            raw = np.where((raw == 65534) | (raw == 65535), 4096, raw)
            raw = np.clip(raw, 0, 4096)
            lut = np.append(cal, np.float32(0.0))
            if int(raw.max(initial=0)) >= len(lut):
                raise ValueError(f"Calibration LUT too short for {name} in {path.name}")
            agri[i - 1] = lut[raw]

    lat = module.sub_lat[rs, cs]
    lon = module.sub_lon[rs, cs]
    saa = module.sub_saa[rs, cs].astype(np.float32, copy=True)
    saz = module.sub_saz[rs, cs].astype(np.float32, copy=True)
    soz, soa, day_mask = module.compute_solar_angles(lat, lon, utc)
    night = day_mask == 0
    agri[:6, night] = 0.0
    soa = soa.astype(np.float32, copy=True)
    soa[night] = 0.0
    patch = np.concatenate((agri, saa[None], saz[None], soa[None],
                            soz.astype(np.float32)[None], day_mask[None]), axis=0)
    cos_soz = np.cos(np.deg2rad(soz)).astype(np.float32)
    cos_raa = np.cos(np.deg2rad(soa - saa)).astype(np.float32)
    out_path = output_root / bjt.strftime("%Y%m") / f"{bjt:%Y%m%d%H%M}.npz"
    atomic_npz(out_path, patch=patch, SAA=saa, SAZ=saz,
               SOZ=soz.astype(np.float32), SOA=soa, cosSOZ=cos_soz,
               cosRAA=cos_raa, day_mask=day_mask.astype(np.float32))
    return {
        "time_bjt": bjt.strftime("%Y-%m-%d %H:%M:%S"),
        "time_utc_hdf_start": utc.strftime("%Y-%m-%d %H:%M:%S"),
        "split": split, "source_kind": source_kind, "source_hdf": str(path),
        "source_size_bytes": path.stat().st_size, "source_mtime_ns": path.stat().st_mtime_ns,
        "path_agri": str(out_path), "output_sha256": sha256(out_path),
        "shape": list(patch.shape), "finite": bool(np.isfinite(patch).all()),
    }


def _init_worker(processor_path: str) -> None:
    global _WORKER_MODULE
    spec = importlib.util.spec_from_file_location("official_hubei_256_worker", processor_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the original Hubei preprocessor in worker")
    _WORKER_MODULE = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_WORKER_MODULE)
    _WORKER_MODULE.compute_solar_angles = compute_solar_angles


def _worker(job: tuple[str, str, str, str, str]) -> dict:
    path_text, output_root, processor_path, bjt_text, split = job
    if _WORKER_MODULE is None:
        _init_worker(processor_path)
    path = Path(path_text)
    bjt = datetime.strptime(bjt_text, "%Y-%m-%d %H:%M:%S")
    source_kind = "realtime" if "real_time_data" in path.parts else "historical_fallback"
    return preprocess_one(path, _WORKER_MODULE, Path(output_root), bjt, split, source_kind)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--frozen-split", type=Path, required=True)
    p.add_argument("--official-processor", type=Path, required=True,
                   help="Original GHI/scripts/legacy_root/FY4B_wuhan_256.py")
    p.add_argument("--historical-root", type=Path, required=True)
    p.add_argument("--realtime-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--output-manifest", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=None,
                   help="Optional bounded witness only; omit for full frozen train/val supplement")
    p.add_argument("--only-bjt", default=None,
                   help="Optional BJT timestamp YYYYMMDDHHMM for a single-frame parity witness")
    a = p.parse_args()
    if a.output_root.exists() or a.output_manifest.exists():
        raise FileExistsError("Refusing to overwrite any existing Hubei supplement")
    split_sha = sha256(a.frozen_split)
    if split_sha != "6ef8cbc7d2bb7d1b1d3defdfcba473cb171e06db0115c8206f7c289d8bad19c6":
        raise ValueError(f"Unexpected frozen split hash: {split_sha}")
    assignments = load_assignments(a.frozen_split)
    spec = importlib.util.spec_from_file_location("official_hubei_256", a.official_processor)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the original Hubei preprocessor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bjt_start = datetime.strptime(START_BJT, "%Y-%m-%d")
    bjt_end = datetime.strptime(END_BJT, "%Y-%m-%d") + timedelta(days=1)
    candidates = {}
    for root in (a.realtime_root, a.historical_root):
        for stamp, paths in discover(root, assignments, bjt_start, bjt_end).items():
            candidates.setdefault(stamp, {}).update(paths)

    by_split = Counter()
    by_source = Counter()
    records = []
    missing_expected = []
    # Expected nominal quarter-hour slots for train/val dates. Missing HDFs are
    # recorded, never synthesized; test/buffer dates are never opened.
    for date, split in sorted(assignments.items()):
        day = datetime.strptime(date, "%Y-%m-%d")
        if not (bjt_start.date() <= day.date() <= bjt_end.date() - timedelta(days=1)):
            continue
        if split not in {"train", "val"}:
            continue
        for minutes in range(0, 24 * 60, 15):
            bjt = day + timedelta(minutes=minutes)
            stamp = bjt.strftime("%Y%m%d%H%M")
            if a.only_bjt is not None and stamp != a.only_bjt:
                continue
            choices = candidates.get(stamp)
            if not choices:
                missing_expected.append({"time_bjt": bjt.isoformat(sep=" "), "split": split})
                continue
            source, kind = choose_source(choices)
            out = a.output_root / bjt.strftime("%Y%m") / f"{stamp}.npz"
            if out.exists():
                raise FileExistsError(out)
            records.append({"_job": (str(source), str(a.output_root), str(a.official_processor),
                                      bjt.strftime("%Y-%m-%d %H:%M:%S"), split),
                            "_source_kind": kind, "_split": split})

    if a.max_frames is not None:
        records = records[:a.max_frames]
    jobs = [r.pop("_job") for r in records]
    requested = records
    records = []
    with ProcessPoolExecutor(max_workers=max(1, a.workers),
                             initializer=_init_worker,
                             initargs=(str(a.official_processor),)) as pool:
        futures = {pool.submit(_worker, job): i for i, job in enumerate(jobs)}
        results = [None] * len(jobs)
        completed = 0
        for future in as_completed(futures):
            i = futures[future]
            result = future.result()
            result["source_kind"] = requested[i]["_source_kind"]
            result["split"] = requested[i]["_split"]
            results[i] = result
            completed += 1
            if completed % 100 == 0 or completed == len(jobs):
                print(f"PREPROCESSED {completed}/{len(jobs)}", flush=True)
    records = sorted(results, key=lambda r: r["time_bjt"])
    for row in records:
        by_split[row["split"]] += 1
        by_source[row["source_kind"]] += 1

    a.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with a.output_manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]) if records else ["time_bjt"])
        w.writeheader()
        w.writerows(records)
    report = {
        "state": "HUBEI_JUN01_JUL14_FROZEN_TRAINVAL_AGRI_PATCH_SUPPLEMENT_COMPLETE_TEST_CLOSED",
        "date_range_BJT": [START_BJT, END_BJT],
        "split_sha256": split_sha,
        "trainval_frames_written": len(records),
        "frames_by_split": dict(by_split), "frames_by_source": dict(by_source),
        "missing_trainval_nominal_slots": missing_expected,
        "test_and_buffer_HDF_payloads_read": 0,
        "test_and_buffer_dates_processed": 0,
        "bounded_witness_only": a.max_frames is not None,
        "workers": a.workers,
        "time_contract": "HDF filename scan-start UTC + 8 h = canonical BJT target slot",
        "pixel_contract": "official Hubei 256 crop [1112:1368,2077:2333], then [103:119,118:134]",
        "channel_contract": "15 calibrated AGRI + SAA,SAZ,SOA,SOZ,day_mask; feature keys match original data16_label readers",
        "source_precedence": "original real_time_data first; historical archive fallback; source recorded per row",
        "label_contract": "no GHI labels stored; attach separate frozen 15-minute train/val label table by BJT key",
        "manifest_sha256": sha256(a.output_manifest),
        "official_processor_sha256": sha256(a.official_processor),
        "solar_geometry_implementation_sha256": sha256(Path(__file__).with_name("solar_geometry_noaa.py")),
        "solar_geometry_version": "NOAA_Meeus_EOT_rad_to_deg_corrected_20261021",
    }
    report_path = a.output_manifest.with_name("REPORT.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
