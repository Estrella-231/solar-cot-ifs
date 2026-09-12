#!/usr/bin/env python3
"""Build one-file 16x16 AGRI+CPP+GHI samples for two Hunan PV stations.

Input AGRI filenames are nominal UTC.  The supplied station workbooks are
daily Beijing-time tables with 5-minute columns VAL0005 ... VAL2400.
For an AGRI frame at t UTC, the three GHI labels are the workbook readings at
t+8h+5min, t+8h+10min, and t+8h+15min.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import netCDF4 as nc
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/home/Data_Pool_3/chenyi/辐照度预报")
AGRI_ROOT = Path("/home/Data_Pool_3/chenyi/Auxiliary_data/HuNan_AGRI_Preprocessed")
CPP_ROOT = Path("/home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP")
OUTPUT_ROOT = Path("/home/Data_Pool_3/chenyi/Auxiliary_data/HuNan_Station16_Combined")
WORKBOOK_ROOT = PROJECT_ROOT / "data/hunan_stations_raw"
CPP_CHANNELS = ("COT", "CER", "CTH", "CLP")
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Station:
    station_id: str
    name: str
    latitude: float
    longitude: float
    agri_bounds: tuple[int, int, int, int]
    cpp_bounds: tuple[int, int, int, int]
    workbook: str


STATIONS = (
    Station(
        station_id="sili",
        name="四里",
        latitude=25.9478611111,
        longitude=112.347916667,
        agri_bounds=(155, 171, 140, 156),
        cpp_bounds=(1369, 1385, 2202, 2218),
        workbook="四里光伏.xlsx",
    ),
    Station(
        station_id="zhujia",
        name="竺家",
        latitude=29.566666667,
        longitude=112.45,
        agri_bounds=(65, 81, 142, 158),
        cpp_bounds=(1279, 1295, 2204, 2220),
        workbook="竺家光伏.xlsx",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-04-01T00:00")
    parser.add_argument("--end", default="2026-04-01T00:00")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--station",
        action="append",
        choices=[station.station_id for station in STATIONS],
        help="May be repeated; default is both stations.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_npz(path: Path, fields: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **fields)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def load_ghi(workbook_path: Path) -> tuple[dict[datetime, np.float32], dict[str, Any]]:
    frame = pd.read_excel(workbook_path, sheet_name="结果集")
    date_column = next(col for col in frame.columns if str(col).strip() == "DATA_DATE")
    value_columns = [col for col in frame.columns if re.fullmatch(r"VAL\d{4}", str(col))]
    if len(value_columns) != 288:
        raise ValueError(f"{workbook_path}: expected 288 VAL columns, found {len(value_columns)}")

    dates = pd.to_datetime(frame[date_column], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{workbook_path}: {int(dates.isna().sum())} invalid DATA_DATE values")
    if dates.duplicated().any():
        raise ValueError(f"{workbook_path}: duplicate DATA_DATE rows")

    numeric = frame[value_columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    lookup: dict[datetime, np.float32] = {}
    for row_index, base_ts in enumerate(dates.dt.to_pydatetime()):
        for column_index, column in enumerate(value_columns):
            hhmm = str(column)[3:]
            if hhmm == "2400":
                timestamp = base_ts + timedelta(days=1)
            else:
                timestamp = base_ts.replace(hour=int(hhmm[:2]), minute=int(hhmm[2:]))
            if timestamp in lookup:
                raise ValueError(f"{workbook_path}: duplicate 5-minute timestamp {timestamp}")
            lookup[timestamp] = numeric[row_index, column_index]

    values = numeric.astype(np.float64, copy=False)
    summary = {
        "path": str(workbook_path),
        "sha256": sha256(workbook_path),
        "daily_rows": int(len(frame)),
        "date_min_bjt": dates.min().strftime("%Y-%m-%d"),
        "date_max_bjt": dates.max().strftime("%Y-%m-%d"),
        "five_minute_records": int(values.size),
        "missing_values": int(np.isnan(values).sum()),
        "zero_values": int(np.sum(values == 0)),
        "negative_values": int(np.sum(values < 0)),
        "minimum": float(np.nanmin(values)),
        "maximum": float(np.nanmax(values)),
        "timezone": "Asia/Shanghai (UTC+08:00)",
        "unit": "W m-2 (inferred from IRRADIANCE field and value range)",
    }
    return lookup, summary


def discover_agri(start: datetime, end: datetime) -> list[tuple[datetime, Path]]:
    found: list[tuple[datetime, Path]] = []
    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day < end:
        directory = AGRI_ROOT / "data" / day.strftime("%Y/%Y%m/%Y%m%d")
        for path in sorted(directory.glob("*.npy")):
            try:
                timestamp = datetime.strptime(path.stem, "%Y%m%d%H%M")
            except ValueError:
                continue
            if start <= timestamp < end:
                found.append((timestamp, path))
        day += timedelta(days=1)
    return found


def cpp_path(timestamp_utc: datetime) -> Path:
    return (
        CPP_ROOT
        / timestamp_utc.strftime("%Y/%Y%m%d")
        / f"FY4B_AGRI_{timestamp_utc:%Y%m%d%H%M}00.nc"
    )


def crop_cpp(path: Path, station: Station) -> tuple[np.ndarray, np.ndarray, np.bool_]:
    if not path.exists():
        data = np.full((len(CPP_CHANNELS), 16, 16), np.nan, dtype=np.float32)
        return data, np.zeros(data.shape, dtype=np.bool_), np.bool_(False)

    r0, r1, c0, c1 = station.cpp_bounds
    channels = []
    try:
        with nc.Dataset(path) as dataset:
            for variable in CPP_CHANNELS:
                raw = dataset[variable][r0:r1, c0:c1]
                channels.append(np.ma.filled(raw, np.nan).astype(np.float32, copy=False))
    except Exception:
        data = np.full((len(CPP_CHANNELS), 16, 16), np.nan, dtype=np.float32)
        return data, np.zeros(data.shape, dtype=np.bool_), np.bool_(False)
    data = np.stack(channels, axis=0)
    if data.shape != (4, 16, 16):
        raise ValueError(f"{path}: CPP crop has shape {data.shape}")
    return data, np.isfinite(data), np.bool_(True)


def output_path(output_root: Path, station: Station, timestamp_utc: datetime) -> Path:
    return (
        output_root
        / station.station_id
        / timestamp_utc.strftime("%Y/%Y%m/%Y%m%d")
        / f"{timestamp_utc:%Y%m%d%H%M}.npz"
    )


def process_timestamp(task: tuple[Any, ...]) -> dict[str, int]:
    timestamp_utc, agri_path, output_root, stations, ghi_triplets, overwrite, grid_path = task
    result = {"written": 0, "skipped": 0, "missing_cpp": 0, "missing_ghi_values": 0}
    agri_full = np.load(agri_path, mmap_mode="r")
    if agri_full.shape != (20, 256, 256) or agri_full.dtype != np.float32:
        raise ValueError(f"{agri_path}: expected float32 (20,256,256), got {agri_full.dtype} {agri_full.shape}")
    with np.load(grid_path, allow_pickle=False) as grid:
        for station in stations:
            destination = output_path(output_root, station, timestamp_utc)
            if destination.exists() and not overwrite:
                result["skipped"] += 1
                continue

            r0, r1, c0, c1 = station.agri_bounds
            agri = np.asarray(agri_full[:, r0:r1, c0:c1], dtype=np.float32)
            if agri.shape != (20, 16, 16):
                raise ValueError(f"{agri_path}: AGRI crop has shape {agri.shape}")
            cpp, cpp_valid_mask, cpp_present = crop_cpp(cpp_path(timestamp_utc), station)
            if not cpp_present:
                result["missing_cpp"] += 1

            ghi = np.asarray(ghi_triplets[station.station_id], dtype=np.float32)
            ghi_valid = np.isfinite(ghi)
            result["missing_ghi_values"] += int((~ghi_valid).sum())
            timestamp_bjt = timestamp_utc + timedelta(hours=8)
            fields = {
                "schema_version": np.int16(SCHEMA_VERSION),
                "agri": agri,
                "cpp": cpp,
                "cpp_valid_mask": cpp_valid_mask,
                "cpp_present": cpp_present,
                "ghi_5min": ghi,
                "ghi_valid": ghi_valid,
                "irr_00_05": ghi[0],
                "irr_05_10": ghi[1],
                "irr_10_15": ghi[2],
                "timestamp_utc": np.str_(timestamp_utc.strftime("%Y-%m-%dT%H:%M:00Z")),
                "timestamp_bjt": np.str_(timestamp_bjt.strftime("%Y-%m-%dT%H:%M:00+08:00")),
                "station_id": np.str_(station.station_id),
                "station_name": np.str_(station.name),
                "station_latlon": np.asarray([station.latitude, station.longitude], dtype=np.float32),
                "agri_crop_bounds": np.asarray(station.agri_bounds, dtype=np.int16),
                "cpp_crop_bounds": np.asarray(station.cpp_bounds, dtype=np.int16),
                "grid_lat": np.asarray(grid["lat"][r0:r1, c0:c1], dtype=np.float32),
                "grid_lon": np.asarray(grid["lon"][r0:r1, c0:c1], dtype=np.float32),
            }
            atomic_npz(destination, fields)
            result["written"] += 1
    return result


def main() -> int:
    args = parse_args()
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    if end <= start:
        raise ValueError("--end must be later than --start")
    selected = tuple(s for s in STATIONS if not args.station or s.station_id in args.station)

    ghi_lookup: dict[str, dict[datetime, np.float32]] = {}
    workbook_summaries: dict[str, Any] = {}
    for station in selected:
        lookup, summary = load_ghi(WORKBOOK_ROOT / station.workbook)
        ghi_lookup[station.station_id] = lookup
        workbook_summaries[station.station_id] = summary

    frames = discover_agri(start, end)
    if args.limit is not None:
        frames = frames[: args.limit]
    grid_path = AGRI_ROOT / "grid_static.npz"
    tasks = []
    for timestamp_utc, agri_path in frames:
        timestamp_bjt = timestamp_utc + timedelta(hours=8)
        label_times = tuple(timestamp_bjt + timedelta(minutes=m) for m in (5, 10, 15))
        ghi_triplets = {
            station.station_id: tuple(
                ghi_lookup[station.station_id].get(ts, np.float32(np.nan)) for ts in label_times
            )
            for station in selected
        }
        tasks.append(
            (timestamp_utc, agri_path, args.output_root, selected, ghi_triplets, args.overwrite, grid_path)
        )

    schema = {
        "schema_version": SCHEMA_VERSION,
        "created_at_bjt": (datetime.utcnow() + timedelta(hours=8)).isoformat(timespec="seconds"),
        "time_range": {"start_utc_inclusive": start.isoformat(), "end_utc_exclusive": end.isoformat()},
        "sample_file": "uncompressed NumPy NPZ; one station and one nominal AGRI UTC time per file",
        "fields": {
            "agri": "float32 (20,16,16)",
            "cpp": "float32 (4,16,16), order COT,CER,CTH,CLP; NaN when source missing",
            "ghi_5min": "float32 (3,), BJT interval-ending readings at +5,+10,+15 min",
            "ghi_valid": "bool (3,)",
        },
        "stations": [asdict(station) for station in selected],
        "workbooks": workbook_summaries,
        "agri_root": str(AGRI_ROOT),
        "cpp_root": str(CPP_ROOT),
    }
    atomic_json(args.output_root / "schema.json", schema)

    workers = max(1, int(os.environ.get("HUNAN_STATION16_WORKERS", "4")))
    totals = {"written": 0, "skipped": 0, "missing_cpp": 0, "missing_ghi_values": 0, "failed": 0}
    print(
        f"START frames={len(tasks)} stations={len(selected)} workers={workers} "
        f"range=[{start.isoformat()},{end.isoformat()})",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(
            executor.map(process_timestamp, tasks, chunksize=1), start=1
        ):
            for key, value in result.items():
                totals[key] += value
            if index == 1 or index % 250 == 0 or index == len(tasks):
                print(f"PROGRESS frames={index}/{len(tasks)} totals={totals}", flush=True)

    summary = {
        **schema,
        "finished_at_bjt": (datetime.utcnow() + timedelta(hours=8)).isoformat(timespec="seconds"),
        "agri_frames": len(tasks),
        "stations_per_frame": len(selected),
        "workers": workers,
        "totals": totals,
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    atomic_json(args.output_root / "logs" / f"run_summary_{stamp}.json", summary)
    atomic_json(args.output_root / "latest_summary.json", summary)
    print(f"DONE totals={totals}", flush=True)
    return 0 if totals["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
