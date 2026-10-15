#!/usr/bin/env python3
"""Preprocess FY-4B AGRI full-disk HDF files for the Hunan 256x256 domain.

The historical archive and real-time archive are merged by nominal UTC time.
Historical ``*_V0001.HDF`` files have priority; real-time files fill missing
historical times.  Each selected HDF produces two compatible artifacts:

* ``data/YYYY/YYYYMM/YYYYMMDD/YYYYMMDDHHMM.npy`` (UTC), float32 (20,256,256)
* aux_256/YYYYMM/YYYYMMDDHHMM_{is_day}_aux.npz (BJT), containing the
  model geometry fields plus lossless AGRI validity, scan-time and raw QA
  metadata. See dataset_meta.json for the complete schema.

``day_mask`` uses SOZ < 85 degrees.  Visible channels C01-C06 are only zeroed
for the physical night (SOZ >= 90 degrees), so the stricter model mask does not
destroy observations in the 85-90 degree twilight interval.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from zipfile import BadZipFile

import netCDF4 as nc
import numpy as np


HISTORICAL_ROOT = Path("/home/Data_Pool/data/FY/FY4B/AGRI/4KM")
REALTIME_ROOT = Path("/home/Data_Pool/data/FY/FY4B/real_time_data/AGRI/4KM")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/Data_Pool_3/chenyi/Auxiliary_data/HuNan_AGRI_Preprocessed"
)
LAT_INDEX_PATH = Path("/home/Data_Pool/data/FY/FY4B/AGRI/LAT_4DEG_INDEX_105.npy")
LON_INDEX_PATH = Path("/home/Data_Pool/data/FY/FY4B/AGRI/LON_4DEG_INDEX_105.npy")
GEO_PATH = Path(
    "/home/Data_Pool/data/FY/FY4B/AGRI/"
    "FY4B_AGRI_SAZ_SOZ_SAA_SOA_202403180000_4km.nc"
)

# Approved Hunan crop. Python slice endpoints are exclusive.
R1, R2 = 1213, 1469
C1, C2 = 2061, 2317
HEIGHT, WIDTH = R2 - R1, C2 - C1

DAY_SOZ_LIMIT = 85.0
PHYSICAL_NIGHT_SOZ = 90.0
DEFAULT_START_UTC = "202404010000"
DEFAULT_WORKERS = int(os.environ.get("HUNAN_PREPROCESS_WORKERS", "4"))

CH_SAA, CH_SAZ, CH_SOA, CH_SOZ, CH_DAY = 15, 16, 17, 18, 19
AUX_NAMES = ("cosSOZ", "cosRAA", "day_mask")
MODEL_AGRI_INDICES = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14)
ALL_VALID_BITS = np.uint16((1 << 15) - 1)
MODEL_VALID_BITS = np.uint16(sum(1 << index for index in MODEL_AGRI_INDICES))
AUX_SCHEMA_VERSION = np.int16(2)
DATA_SCHEMA_VERSION = np.int16(1)
AUX_REQUIRED_KEYS = {
    "cosSOZ",
    "cosRAA",
    "day_mask",
    "agri_valid_bits",
    "agri_valid_mask",
    "obs_time_offset_s",
    "cal_quality_flag",
    "l1_quality_flag",
    "nav_quality_flag",
    "data_quality",
    "incomplete_scans",
    "qa_scan_flag",
    "qa_pixel_flag",
    "sun_avoidance_flag",
    "aux_schema_version",
    "data_schema_version",
    "source_type",
    "source_hdf",
    "source_size_bytes",
    "source_mtime_ns",
    "agri_valid_fraction",
    "all_channel_valid_fraction",
    "obs_time_valid_fraction",
    "obs_time_offset_min_s",
    "obs_time_offset_max_s",
}

CHANNEL_INFO = {
    "0": "AGRI C01 reflectance (0.47 um)",
    "1": "AGRI C02 reflectance (0.65 um)",
    "2": "AGRI C03 reflectance (0.825 um)",
    "3": "AGRI C04 reflectance (1.379 um)",
    "4": "AGRI C05 reflectance (1.61 um)",
    "5": "AGRI C06 reflectance (2.225 um)",
    "6": "AGRI C07 equivalent brightness temperature (3.75 um, high gain)",
    "7": "AGRI C08 equivalent brightness temperature (3.75 um, low gain)",
    "8": "AGRI C09 brightness temperature (6.25 um)",
    "9": "AGRI C10 brightness temperature (6.95 um)",
    "10": "AGRI C11 brightness temperature (7.42 um)",
    "11": "AGRI C12 brightness temperature (8.55 um)",
    "12": "AGRI C13 brightness temperature (10.8 um)",
    "13": "AGRI C14 brightness temperature (12.0 um)",
    "14": "AGRI C15 brightness temperature (13.3 um)",
    "15": "SAA sensor azimuth angle (degree, fixed)",
    "16": "SAZ sensor zenith angle (degree, fixed)",
    "17": "SOA solar azimuth angle (degree, dynamic; zero at SOZ >= 90 deg)",
    "18": "SOZ solar zenith angle (degree, dynamic)",
    "19": "day_mask (1 where SOZ < 85 deg, else 0)",
}

_TIME_RE = re.compile(r"_(20\d{12})_")
_PRODUCT_RE = re.compile(
    r"^FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_"
    r"(20\d{12})_(20\d{12})_4000M(?:_V\d{4})?\.HDF$"
)

# Loaded once before the process pool is created and inherited by forked workers.
# Workers using a spawn start method lazily load the same arrays.
_SUB_LAT_IDX: Optional[np.ndarray] = None
_SUB_LON_IDX: Optional[np.ndarray] = None
_SUB_LAT: Optional[np.ndarray] = None
_SUB_LON: Optional[np.ndarray] = None
_SUB_SAA: Optional[np.ndarray] = None
_SUB_SAZ: Optional[np.ndarray] = None


def parse_utc(text: str) -> datetime:
    """Parse YYYYMMDD[HH[MM]] using an explicit UTC convention."""
    text = text.strip()
    formats = {8: "%Y%m%d", 10: "%Y%m%d%H", 12: "%Y%m%d%H%M"}
    if len(text) not in formats:
        raise argparse.ArgumentTypeError(
            f"invalid UTC time {text!r}; expected YYYYMMDD, YYYYMMDDHH or YYYYMMDDHHMM"
        )
    try:
        return datetime.strptime(text, formats[len(text)])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def timestamp_from_name(path: Path) -> datetime:
    match = _TIME_RE.search(path.name)
    if not match:
        raise ValueError(f"cannot parse nominal UTC time from {path.name}")
    return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")


def load_static() -> Tuple[np.ndarray, ...]:
    global _SUB_LAT_IDX, _SUB_LON_IDX, _SUB_LAT, _SUB_LON, _SUB_SAA, _SUB_SAZ
    if _SUB_LAT is not None:
        return (
            _SUB_LAT_IDX,
            _SUB_LON_IDX,
            _SUB_LAT,
            _SUB_LON,
            _SUB_SAA,
            _SUB_SAZ,
        )

    lat_index = np.load(LAT_INDEX_PATH, mmap_mode="r")
    lon_index = np.load(LON_INDEX_PATH, mmap_mode="r")
    _SUB_LAT_IDX = np.asarray(lat_index[R1:R2, C1:C2], dtype=np.intp).copy()
    _SUB_LON_IDX = np.asarray(lon_index[R1:R2, C1:C2], dtype=np.intp).copy()

    with nc.Dataset(GEO_PATH) as geo:
        _SUB_LAT = np.asarray(np.ma.getdata(geo["lat"][R1:R2, C1:C2]), dtype=np.float32).copy()
        _SUB_LON = np.asarray(np.ma.getdata(geo["lon"][R1:R2, C1:C2]), dtype=np.float32).copy()
        _SUB_SAA = np.asarray(np.ma.getdata(geo["SAA"][R1:R2, C1:C2]), dtype=np.float32).copy()
        _SUB_SAZ = np.asarray(np.ma.getdata(geo["SAZ"][R1:R2, C1:C2]), dtype=np.float32).copy()

    expected = (HEIGHT, WIDTH)
    for name, array in (
        ("row_index", _SUB_LAT_IDX),
        ("col_index", _SUB_LON_IDX),
        ("lat", _SUB_LAT),
        ("lon", _SUB_LON),
        ("saa", _SUB_SAA),
        ("saz", _SUB_SAZ),
    ):
        if array.shape != expected:
            raise RuntimeError(f"static {name} shape is {array.shape}, expected {expected}")

    return (
        _SUB_LAT_IDX,
        _SUB_LON_IDX,
        _SUB_LAT,
        _SUB_LON,
        _SUB_SAA,
        _SUB_SAZ,
    )


def _julian_day(dt: datetime) -> float:
    year, month, day = dt.year, dt.month, dt.day
    hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    if month <= 2:
        year -= 1
        month += 12
    a = int(year / 100)
    b = 2 - a + int(a / 4)
    return (
        int(365.25 * (year + 4716))
        + int(30.6001 * (month + 1))
        + day
        + b
        - 1524.5
        + hour / 24.0
    )


def compute_solar_angles(
    lat: np.ndarray, lon: np.ndarray, dt_utc: datetime
) -> Tuple[np.ndarray, np.ndarray]:
    """Return solar zenith and azimuth (degrees; azimuth north-clockwise)."""
    jd = _julian_day(dt_utc)
    jc = (jd - 2451545.0) / 36525.0
    gmls = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360.0
    gmas = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    eccentricity = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)

    gmas_rad = np.deg2rad(gmas)
    center = (
        np.sin(gmas_rad) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + np.sin(2.0 * gmas_rad) * (0.019993 - 0.000101 * jc)
        + np.sin(3.0 * gmas_rad) * 0.000289
    )
    true_longitude = gmls + center
    omega = 125.04 - 1934.136 * jc
    apparent_longitude = true_longitude - 0.00569 - 0.00478 * np.sin(np.deg2rad(omega))
    mean_obliquity = 23.0 + (
        26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60.0
    ) / 60.0
    obliquity = mean_obliquity + 0.00256 * np.cos(np.deg2rad(omega))
    obliquity_rad = np.deg2rad(obliquity)
    declination = np.arcsin(
        np.sin(obliquity_rad) * np.sin(np.deg2rad(apparent_longitude))
    )

    vary = np.tan(obliquity_rad / 2.0) ** 2
    gmls_rad = np.deg2rad(gmls)
    equation_of_time = 4.0 * (
        vary * np.sin(2.0 * gmls_rad)
        - 2.0 * eccentricity * np.sin(gmas_rad)
        + 4.0
        * eccentricity
        * vary
        * np.sin(gmas_rad)
        * np.cos(2.0 * gmls_rad)
        - 0.5 * vary * vary * np.sin(4.0 * gmls_rad)
        - 1.25 * eccentricity * eccentricity * np.sin(2.0 * gmas_rad)
    )

    utc_minutes = dt_utc.hour * 60.0 + dt_utc.minute + dt_utc.second / 60.0
    true_solar_time = (utc_minutes + equation_of_time + 4.0 * lon) % 1440.0
    hour_angle = np.deg2rad(true_solar_time / 4.0 - 180.0)

    lat_rad = np.deg2rad(lat)
    cos_soz = (
        np.sin(lat_rad) * np.sin(declination)
        + np.cos(lat_rad) * np.cos(declination) * np.cos(hour_angle)
    )
    cos_soz = np.clip(cos_soz, -1.0, 1.0)
    soz = np.rad2deg(np.arccos(cos_soz))

    sin_soz = np.sqrt(np.maximum(1.0 - cos_soz**2, 1e-12))
    cos_soa = (
        np.sin(declination) - np.sin(lat_rad) * cos_soz
    ) / (np.cos(lat_rad) * sin_soz)
    cos_soa = np.clip(cos_soa, -1.0, 1.0)
    sin_soa = -np.cos(declination) * np.sin(hour_angle) / sin_soz
    sin_soa = np.clip(sin_soa, -1.0, 1.0)
    soa = np.rad2deg(np.arctan2(sin_soa, cos_soa)) % 360.0
    return soz.astype(np.float32), soa.astype(np.float32)


def make_geometry(
    obs_dt: datetime,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Compute stored geometry plus all three auxiliary fields."""
    _, _, lat, lon, saa, _ = load_static()
    soz, soa = compute_solar_angles(lat, lon, obs_dt)
    physical_night = soz >= PHYSICAL_NIGHT_SOZ
    soa_out = soa.copy()
    soa_out[physical_night] = 0.0
    day_mask = (soz < DAY_SOZ_LIMIT).astype(np.float32)
    cos_soz = np.cos(np.deg2rad(soz)).astype(np.float32)
    cos_raa = np.cos(np.deg2rad(soa_out - saa)).astype(np.float32)
    is_day = int(np.all(day_mask == 1.0))
    return soz, soa_out, day_mask, cos_soz, cos_raa, is_day


def detect_hdf_format(dataset: nc.Dataset):
    if "Data" in dataset.groups and "Calibration" in dataset.groups:
        return dataset["Data"], dataset["Calibration"], "CALChannel"
    return dataset, dataset, "CalChannel"


def _valid_range(variable, default_low: float, default_high: float) -> Tuple[float, float]:
    values = np.asarray(
        getattr(variable, "valid_range", (default_low, default_high)), dtype=np.float64
    ).reshape(-1)
    if values.size < 2 or not np.all(np.isfinite(values[:2])):
        return default_low, default_high
    return float(values[0]), float(values[1])


def _fill_value(variable):
    if "FillValue" in variable.ncattrs():
        return getattr(variable, "FillValue")
    if "_FillValue" in variable.ncattrs():
        return getattr(variable, "_FillValue")
    return None


def _quality_vector(group, name: str) -> np.ndarray:
    if name not in group.variables:
        return np.full(15, np.nan, dtype=np.float32)
    values = np.asarray(np.ma.getdata(group[name][:]), dtype=np.float32).reshape(-1)
    result = np.full(15, np.nan, dtype=np.float32)
    result[: min(15, values.size)] = values[:15]
    return result


def _scene_quality_scalar(dataset: nc.Dataset, name: str) -> np.ndarray:
    if name not in dataset.ncattrs():
        return np.asarray(-1, dtype=np.int32)
    try:
        value = np.asarray(getattr(dataset, name)).reshape(-1)[0]
        return np.asarray(int(value), dtype=np.int32)
    except (TypeError, ValueError, OverflowError):
        return np.asarray(-1, dtype=np.int32)


def _decode_obs_time(value) -> Optional[datetime]:
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if integer == 9999 or integer < 10**16:
        return None
    text = str(integer)
    if len(text) != 17:
        return None
    try:
        return datetime.strptime(text[:14], "%Y%m%d%H%M%S") + timedelta(
            milliseconds=int(text[14:17])
        )
    except ValueError:
        return None


def read_obs_time_offset(
    dataset: nc.Dataset, row_index: np.ndarray, obs_dt: datetime
) -> np.ndarray:
    """Map per-scan-line observation midpoint to each Hunan pixel."""
    obs_group = dataset.groups.get("NOMObs", dataset)
    if "NOMObsTime" not in obs_group.variables:
        return np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)
    raw = np.asarray(
        np.ma.getdata(obs_group["NOMObsTime"][:]), dtype=np.int64
    )
    if raw.ndim != 2 or raw.shape[1] < 1:
        return np.full((HEIGHT, WIDTH), np.nan, dtype=np.float32)

    offsets_by_line = np.full(raw.shape[0], np.nan, dtype=np.float64)
    for line in np.unique(row_index):
        line_number = int(line)
        if line_number < 0 or line_number >= raw.shape[0]:
            continue
        times = [
            parsed
            for parsed in (_decode_obs_time(value) for value in raw[line_number, :2])
            if parsed is not None
        ]
        if not times:
            continue
        midpoint = times[0] if len(times) == 1 else times[0] + (times[-1] - times[0]) / 2
        offsets_by_line[line_number] = (midpoint - obs_dt).total_seconds()
    return offsets_by_line[row_index].astype(np.float32)


def read_calibrated_patch(
    hdf_path: Path, obs_dt: datetime
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Read calibrated C01-C15 plus lossless source-quality auxiliaries."""
    row_index, col_index, *_ = load_static()
    patch = np.zeros((15, HEIGHT, WIDTH), dtype=np.float32)
    valid_bits = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
    with nc.Dataset(hdf_path) as hdf:
        data_group, calibration_group, calibration_prefix = detect_hdf_format(hdf)
        obs_time_offset_s = read_obs_time_offset(hdf, row_index, obs_dt)
        qa_group = hdf.groups.get("QA", hdf)
        quality = {
            "cal_quality_flag": _quality_vector(qa_group, "CalQualityFlag"),
            "l1_quality_flag": _quality_vector(qa_group, "L1QualityFlag"),
            "nav_quality_flag": _quality_vector(qa_group, "NavQualityFlag"),
            "data_quality": _scene_quality_scalar(hdf, "Data Quality"),
            "incomplete_scans": _scene_quality_scalar(hdf, "Incomplete Scans"),
            "qa_scan_flag": _scene_quality_scalar(hdf, "QA_Scan_Flag"),
            "qa_pixel_flag": _scene_quality_scalar(hdf, "QA_Pixel_Flag"),
            "sun_avoidance_flag": _scene_quality_scalar(hdf, "Sun Avoidance Flag"),
        }

        for channel in range(1, 16):
            suffix = f"{channel:02d}"
            dn_variable = data_group[f"NOMChannel{suffix}"]
            dn_raw = dn_variable[:]
            dn_full = np.asarray(np.ma.getdata(dn_raw), dtype=np.int64)
            dn_mask = np.ma.getmaskarray(dn_raw)
            dn = dn_full[row_index, col_index]
            masked = dn_mask[row_index, col_index]

            cal_variable = calibration_group[f"{calibration_prefix}{suffix}"]
            cal_raw = cal_variable[:]
            calibration = np.asarray(np.ma.getdata(cal_raw), dtype=np.float32).reshape(-1)
            cal_mask = np.ma.getmaskarray(cal_raw).reshape(-1)
            if calibration.size == 0:
                raise ValueError(f"{hdf_path.name} channel {channel:02d}: empty calibration LUT")

            nom_low, nom_high = _valid_range(dn_variable, 0.0, calibration.size - 1)
            valid = (
                (~masked)
                & (dn >= nom_low)
                & (dn <= nom_high)
                & (dn >= 0)
                & (dn < calibration.size)
            )
            nom_fill = _fill_value(dn_variable)
            if nom_fill is not None:
                valid &= dn != int(nom_fill)

            safe_dn = np.clip(dn, 0, calibration.size - 1)
            values = calibration[safe_dn]
            valid &= ~cal_mask[safe_dn]
            valid &= np.isfinite(values)
            cal_fill = _fill_value(cal_variable)
            if cal_fill is not None:
                valid &= values != float(cal_fill)
            cal_low, cal_high = _valid_range(cal_variable, -np.inf, np.inf)
            valid &= (values >= cal_low) & (values <= cal_high)

            patch[channel - 1, valid] = values[valid]
            valid_bits[valid] |= np.uint16(1 << (channel - 1))

    agri_valid_mask = (
        (valid_bits & MODEL_VALID_BITS) == MODEL_VALID_BITS
    ).astype(np.uint8)
    return patch, valid_bits, agri_valid_mask, obs_time_offset_s, quality


def data_path_for_utc(output_root: Path, obs_dt: datetime) -> Path:
    return (
        output_root
        / "data"
        / obs_dt.strftime("%Y")
        / obs_dt.strftime("%Y%m")
        / obs_dt.strftime("%Y%m%d")
        / f"{obs_dt:%Y%m%d%H%M}.npy"
    )


def aux_path_for_bjt(output_root: Path, bjt: datetime, is_day: int) -> Path:
    return output_root / "aux_256" / bjt.strftime("%Y%m") / f"{bjt:%Y%m%d%H%M}_{is_day}_aux.npz"


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")



def inspect_aux_artifact(
    path: Path,
    source_type: str,
    source_hdf: Path,
    source_size_bytes: int,
    source_mtime_ns: int,
) -> Tuple[str, Dict[str, object]]:
    """Classify aux state and read only its small indexing scalars."""
    empty: Dict[str, object] = {}
    if not path.is_file():
        return "missing", empty
    try:
        with np.load(path, allow_pickle=False) as aux:
            if not AUX_REQUIRED_KEYS.issubset(aux.files):
                return "incomplete", empty
            version = int(np.asarray(aux["aux_schema_version"]).item())
            if version < int(AUX_SCHEMA_VERSION):
                return "incomplete", empty
            data_version = int(np.asarray(aux["data_schema_version"]).item())
            if data_version != int(DATA_SCHEMA_VERSION):
                return "data_schema_mismatch", empty
            stored_source = (
                str(np.asarray(aux["source_type"]).item()),
                str(np.asarray(aux["source_hdf"]).item()),
                int(np.asarray(aux["source_size_bytes"]).item()),
                int(np.asarray(aux["source_mtime_ns"]).item()),
            )
            expected_source = (
                source_type,
                str(source_hdf),
                int(source_size_bytes),
                int(source_mtime_ns),
            )

            def optional_float(name: str) -> Optional[float]:
                value = float(np.asarray(aux[name]).item())
                return value if np.isfinite(value) else None

            summary: Dict[str, object] = {
                "agri_valid_fraction": optional_float("agri_valid_fraction"),
                "all_channel_valid_fraction": optional_float(
                    "all_channel_valid_fraction"
                ),
                "obs_time_valid_fraction": optional_float(
                    "obs_time_valid_fraction"
                ),
                "obs_time_offset_min_s": optional_float("obs_time_offset_min_s"),
                "obs_time_offset_max_s": optional_float("obs_time_offset_max_s"),
                "data_quality": int(np.asarray(aux["data_quality"]).item()),
                "incomplete_scans": int(np.asarray(aux["incomplete_scans"]).item()),
                "qa_scan_flag": int(np.asarray(aux["qa_scan_flag"]).item()),
                "qa_pixel_flag": int(np.asarray(aux["qa_pixel_flag"]).item()),
                "sun_avoidance_flag": int(
                    np.asarray(aux["sun_avoidance_flag"]).item()
                ),
                "aux_schema_version": version,
                "data_schema_version": data_version,
            }
            state = "current" if stored_source == expected_source else "source_mismatch"
            return state, summary
    except (OSError, ValueError, KeyError, EOFError, BadZipFile):
        return "corrupt", empty



def inspect_data_artifact(path: Path) -> str:
    """Validate the NPY header, dtype, shape and mmap readability."""
    if not path.is_file():
        return "missing"
    try:
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        if data.shape != (20, HEIGHT, WIDTH) or data.dtype != np.dtype(np.float32):
            return "invalid"
        _ = data[0, 0, 0]
        return "current"
    except (OSError, ValueError, EOFError):
        return "corrupt"

def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_save_aux(
    path: Path,
    cos_soz: np.ndarray,
    cos_raa: np.ndarray,
    day_mask: np.ndarray,
    agri_valid_bits: np.ndarray,
    agri_valid_mask: np.ndarray,
    obs_time_offset_s: np.ndarray,
    quality: Dict[str, np.ndarray],
    source_type: str,
    source_hdf: Path,
    source_size_bytes: int,
    source_mtime_ns: int,
) -> None:
    finite_offset = np.isfinite(obs_time_offset_s)
    offset_min = (
        float(np.min(obs_time_offset_s[finite_offset]))
        if np.any(finite_offset)
        else np.nan
    )
    offset_max = (
        float(np.max(obs_time_offset_s[finite_offset]))
        if np.any(finite_offset)
        else np.nan
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                cosSOZ=cos_soz.astype(np.float32, copy=False),
                cosRAA=cos_raa.astype(np.float32, copy=False),
                day_mask=day_mask.astype(np.float32, copy=False),
                agri_valid_bits=agri_valid_bits.astype(np.uint16, copy=False),
                agri_valid_mask=agri_valid_mask.astype(np.uint8, copy=False),
                obs_time_offset_s=obs_time_offset_s.astype(np.float32, copy=False),
                cal_quality_flag=np.asarray(
                    quality["cal_quality_flag"], dtype=np.float32
                ),
                l1_quality_flag=np.asarray(
                    quality["l1_quality_flag"], dtype=np.float32
                ),
                nav_quality_flag=np.asarray(
                    quality["nav_quality_flag"], dtype=np.float32
                ),
                data_quality=np.asarray(quality["data_quality"], dtype=np.int32),
                incomplete_scans=np.asarray(
                    quality["incomplete_scans"], dtype=np.int32
                ),
                qa_scan_flag=np.asarray(quality["qa_scan_flag"], dtype=np.int32),
                qa_pixel_flag=np.asarray(
                    quality["qa_pixel_flag"], dtype=np.int32
                ),
                sun_avoidance_flag=np.asarray(
                    quality["sun_avoidance_flag"], dtype=np.int32
                ),
                agri_valid_fraction=np.asarray(
                    np.mean(agri_valid_mask), dtype=np.float32
                ),
                all_channel_valid_fraction=np.asarray(
                    np.mean(agri_valid_bits == ALL_VALID_BITS), dtype=np.float32
                ),
                obs_time_valid_fraction=np.asarray(
                    np.mean(finite_offset), dtype=np.float32
                ),
                obs_time_offset_min_s=np.asarray(offset_min, dtype=np.float32),
                obs_time_offset_max_s=np.asarray(offset_max, dtype=np.float32),
                source_type=np.asarray(source_type),
                source_hdf=np.asarray(str(source_hdf)),
                source_size_bytes=np.asarray(source_size_bytes, dtype=np.int64),
                source_mtime_ns=np.asarray(source_mtime_ns, dtype=np.int64),
                aux_schema_version=np.asarray(AUX_SCHEMA_VERSION, dtype=np.int16),
                data_schema_version=np.asarray(DATA_SCHEMA_VERSION, dtype=np.int16),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def process_one(
    record: Tuple[str, str, str], output_root_text: str, overwrite: bool
) -> Dict[str, object]:
    """Worker entry point. record is (UTC text, source type, HDF path)."""
    started = time.monotonic()
    utc_text, source_type, hdf_text = record
    obs_dt = datetime.strptime(utc_text, "%Y%m%d%H%M%S")
    bjt = obs_dt + timedelta(hours=8)
    hdf_path = Path(hdf_text)
    output_root = Path(output_root_text)
    result: Dict[str, object] = {
        "time_utc": obs_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "time_bjt": bjt.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source_type,
        "source_hdf": str(hdf_path),
    }
    try:
        source_before = hdf_path.stat()
        result.update(
            {
                "source_size_bytes": source_before.st_size,
                "source_mtime_ns": source_before.st_mtime_ns,
            }
        )
        soz, soa_out, day_mask, cos_soz, cos_raa, is_day = make_geometry(obs_dt)
        data_path = data_path_for_utc(output_root, obs_dt)
        aux_path = aux_path_for_bjt(output_root, bjt, is_day)
        previous_data_state = inspect_data_artifact(data_path)
        previous_aux_state, previous_aux_summary = inspect_aux_artifact(
            aux_path,
            source_type,
            hdf_path,
            source_before.st_size,
            source_before.st_mtime_ns,
        )
        source_changed = previous_aux_state == "source_mismatch"
        result.update(
            {
                "is_day": is_day,
                "data_path": str(data_path.relative_to(output_root)),
                "aux_path": str(aux_path.relative_to(output_root)),
                "previous_data_state": previous_data_state,
                "previous_aux_state": previous_aux_state,
            }
        )

        # When aux provenance is unavailable or damaged, rebuild the pair.
        # This avoids attaching current-HDF QA to a data array of unknown origin.
        need_data = (
            overwrite
            or previous_data_state != "current"
            or previous_aux_state != "current"
        )
        need_aux = overwrite or previous_aux_state != "current"
        if not need_data and not need_aux:
            result.update(previous_aux_summary)
            result["status"] = "skipped"
            return result

        (
            channels,
            agri_valid_bits,
            agri_valid_mask,
            obs_time_offset_s,
            quality,
        ) = read_calibrated_patch(hdf_path, obs_dt)
        source_after = hdf_path.stat()
        if (
            source_before.st_size != source_after.st_size
            or source_before.st_mtime_ns != source_after.st_mtime_ns
        ):
            raise RuntimeError(
                "source HDF changed during read: "
                f"before=({source_before.st_size},{source_before.st_mtime_ns}) "
                f"after=({source_after.st_size},{source_after.st_mtime_ns})"
            )

        finite_offset = np.isfinite(obs_time_offset_s)
        result.update(
            {
                "agri_valid_fraction": float(np.mean(agri_valid_mask)),
                "all_channel_valid_fraction": float(
                    np.mean(agri_valid_bits == ALL_VALID_BITS)
                ),
                "obs_time_valid_fraction": float(np.mean(finite_offset)),
                "obs_time_offset_min_s": (
                    float(np.min(obs_time_offset_s[finite_offset]))
                    if np.any(finite_offset)
                    else None
                ),
                "obs_time_offset_max_s": (
                    float(np.max(obs_time_offset_s[finite_offset]))
                    if np.any(finite_offset)
                    else None
                ),
                "data_quality": int(np.asarray(quality["data_quality"]).item()),
                "incomplete_scans": int(
                    np.asarray(quality["incomplete_scans"]).item()
                ),
                "qa_scan_flag": int(np.asarray(quality["qa_scan_flag"]).item()),
                "qa_pixel_flag": int(np.asarray(quality["qa_pixel_flag"]).item()),
                "sun_avoidance_flag": int(
                    np.asarray(quality["sun_avoidance_flag"]).item()
                ),
                "aux_schema_version": int(AUX_SCHEMA_VERSION),
                "data_schema_version": int(DATA_SCHEMA_VERSION),
            }
        )

        if need_data:
            physical_night = soz >= PHYSICAL_NIGHT_SOZ
            channels[0:6, physical_night] = 0.0
            _, _, _, _, saa, saz = load_static()
            output = np.empty((20, HEIGHT, WIDTH), dtype=np.float32)
            output[0:15] = channels
            output[CH_SAA] = saa
            output[CH_SAZ] = saz
            output[CH_SOA] = soa_out
            output[CH_SOZ] = soz
            output[CH_DAY] = day_mask
            atomic_save_npy(data_path, output)

        if need_aux:
            atomic_save_aux(
                aux_path,
                cos_soz,
                cos_raa,
                day_mask,
                agri_valid_bits,
                agri_valid_mask,
                obs_time_offset_s,
                quality,
                source_type,
                hdf_path,
                source_before.st_size,
                source_before.st_mtime_ns,
            )

        if source_changed:
            result["status"] = "refreshed_source"
        elif previous_data_state == "missing" and previous_aux_state == "missing":
            result["status"] = "written"
        elif need_data and need_aux:
            result["status"] = "repaired_pair"
        elif need_data:
            result["status"] = "repaired_data"
        else:
            result["status"] = "repaired_aux"
        return result
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=8)
        return result
    finally:
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)


def _date_directories(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for year_dir in sorted(root.iterdir()):
        if not year_dir.is_dir() or not re.fullmatch(r"20\d{2}", year_dir.name):
            continue
        for date_dir in sorted(year_dir.iterdir()):
            if date_dir.is_dir() and re.fullmatch(r"20\d{6}", date_dir.name):
                yield date_dir


def collect_records(
    historical_root: Path,
    realtime_root: Path,
    start_utc: datetime,
    end_utc: Optional[datetime],
) -> List[Tuple[str, str, str]]:
    """Merge source files by UTC; historical archive has deterministic priority."""
    selected: Dict[datetime, Tuple[Tuple[int, int, str], str, Path]] = {}
    source_specs = (("realtime", realtime_root, 1), ("historical", historical_root, 2))
    for source_type, root, source_rank in source_specs:
        for date_dir in _date_directories(root):
            try:
                date_value = datetime.strptime(date_dir.name, "%Y%m%d")
            except ValueError:
                continue
            if date_value.date() < start_utc.date():
                continue
            if end_utc is not None and date_value.date() > end_utc.date():
                continue
            for path in sorted(date_dir.iterdir()):
                name = path.name
                # Only accept completed, canonical AGRI L1 products. Some
                # archive directories contain download remnants whose names
                # continue after ".HDF" with a URL query string.
                product_match = _PRODUCT_RE.fullmatch(name)
                if not path.is_file() or product_match is None:
                    continue
                try:
                    obs_dt = datetime.strptime(product_match.group(1), "%Y%m%d%H%M%S")
                except ValueError:
                    continue
                if obs_dt.second != 0 or obs_dt.minute % 15 != 0:
                    continue
                if obs_dt < start_utc or (end_utc is not None and obs_dt >= end_utc):
                    continue
                source_stat = path.stat()
                if source_stat.st_size == 0:
                    continue
                version_rank = int("_V0001.HDF" in path.name)
                priority = (source_rank, version_rank, path.name)
                current = selected.get(obs_dt)
                if current is None or priority > current[0]:
                    selected[obs_dt] = (priority, source_type, path)
    return [
        (obs_dt.strftime("%Y%m%d%H%M%S"), selected[obs_dt][1], str(selected[obs_dt][2]))
        for obs_dt in sorted(selected)
    ]


def records_from_explicit_files(paths: Sequence[Path]) -> List[Tuple[str, str, str]]:
    selected: Dict[datetime, Tuple[str, Path]] = {}
    for path in paths:
        resolved = path.resolve()
        obs_dt = timestamp_from_name(resolved)
        source = "historical" if "_V0001.HDF" in resolved.name else "realtime"
        selected[obs_dt] = (source, resolved)
    return [
        (dt.strftime("%Y%m%d%H%M%S"), selected[dt][0], str(selected[dt][1]))
        for dt in sorted(selected)
    ]


def write_static_metadata(
    output_root: Path,
    records: Sequence[Tuple[str, str, str]],
    start_utc: datetime,
    end_utc: Optional[datetime],
) -> None:
    row_index, col_index, lat, lon, saa, saz = load_static()
    output_root.mkdir(parents=True, exist_ok=True)
    grid_path = output_root / "grid_static.npz"
    temporary = _temporary_path(grid_path)
    try:
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                lat=lat,
                lon=lon,
                saa=saa,
                saz=saz,
                row_index=row_index,
                col_index=col_index,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, grid_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    atomic_write_json(output_root / "channel_info.json", CHANNEL_INFO)
    first = records[0][0] if records else None
    last = records[-1][0] if records else None
    source_counts = {
        source: sum(1 for _, item_source, _ in records if item_source == source)
        for source in ("historical", "realtime")
    }
    metadata = {
        "region": "Hunan",
        "shape": [HEIGHT, WIDTH],
        "crop": {
            "row_start": R1,
            "row_end_exclusive": R2,
            "col_start": C1,
            "col_end_exclusive": C2,
        },
        "pixel_center_extent": {
            "lon_min": float(np.nanmin(lon)),
            "lon_max": float(np.nanmax(lon)),
            "lat_min": float(np.nanmin(lat)),
            "lat_max": float(np.nanmax(lat)),
        },
        "time_conventions": {
            "data_filename": "UTC",
            "aux_filename_and_month": "BJT (UTC+8)",
            "solar_geometry_time_basis": (
                "nominal UTC from the AGRI filename; obs_time_offset_s retains "
                "the actual per-scan-line timing for optional later recomputation"
            ),
        },
        "requested_start_utc": start_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "requested_end_utc_exclusive": (
            end_utc.strftime("%Y-%m-%d %H:%M:%S") if end_utc else None
        ),
        "selected_first_utc": first,
        "selected_last_utc": last,
        "selected_count": len(records),
        "source_counts": source_counts,
        "source_precedence": "historical *_V0001.HDF, then realtime fill",
        "channels": "C01-C15,SAA,SAZ,SOA,SOZ,day_mask",
        "day_mask": "SOZ < 85 degrees",
        "visible_zeroing": "C01-C06 only where SOZ >= 90 degrees",
        "aux_schema_version": int(AUX_SCHEMA_VERSION),
        "data_schema_version": int(DATA_SCHEMA_VERSION),
        "aux_fields": {
            "cosSOZ": "cos(SOZ), computed at nominal filename UTC",
            "cosRAA": "cos(SOA-SAA), computed at nominal filename UTC",
            "day_mask": "1 where nominal-time SOZ < 85 degrees, else 0",
            "agri_valid_bits": (
                "uint16 bit0-bit14 for C01-C15; 1 means raw DN and calibrated "
                "value passed fill, mask, valid-range and LUT checks before night zeroing"
            ),
            "agri_valid_mask": (
                "uint8 source-calibration validity before visible-channel night "
                "zeroing; 1 where all 13 AGRI channels used by the current SimVP "
                "input (C01-C06,C09-C15) are valid"
            ),
            "obs_time_offset_s": (
                "actual NOMObsTime scan-line midpoint minus nominal filename UTC, "
                "seconds; NaN where source scan time is unavailable"
            ),
            "agri_valid_fraction": "cached mean of agri_valid_mask for indexing",
            "all_channel_valid_fraction": "cached fraction with all C01-C15 valid",
            "obs_time_valid_fraction": "cached finite fraction of obs_time_offset_s",
            "obs_time_offset_min_s": "cached finite minimum observation offset",
            "obs_time_offset_max_s": "cached finite maximum observation offset",
            "source_type": "historical or realtime source selected for this frame",
            "source_hdf": "absolute source HDF path used for this frame",
            "source_size_bytes": "source file size verified stable during the read",
            "source_mtime_ns": "source file mtime_ns verified stable during the read",
            "cal_quality_flag": "raw 15-element CalQualityFlag; no undocumented interpretation",
            "l1_quality_flag": "raw 15-element L1QualityFlag; no undocumented interpretation",
            "nav_quality_flag": "raw 15-element NavQualityFlag; no undocumented interpretation",
            "data_quality": "raw root Data Quality attribute; -1 if unavailable",
            "incomplete_scans": "raw root Incomplete Scans attribute; -1 if unavailable",
            "qa_scan_flag": "raw root QA_Scan_Flag attribute; -1 if unavailable",
            "qa_pixel_flag": "raw root QA_Pixel_Flag attribute; -1 if unavailable",
            "sun_avoidance_flag": "raw root Sun Avoidance Flag attribute; -1 if unavailable",
        },
        "created_bjt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    atomic_write_json(output_root / "dataset_meta.json", metadata)


def write_source_manifest(output_root: Path, records: Sequence[Tuple[str, str, str]]) -> Path:
    path = output_root / "source_manifest.csv"
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("time_utc", "time_bjt", "source", "source_hdf"))
            for utc_text, source, hdf_path in records:
                obs_dt = datetime.strptime(utc_text, "%Y%m%d%H%M%S")
                writer.writerow(
                    (
                        obs_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        (obs_dt + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"),
                        source,
                        hdf_path,
                    )
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


MANIFEST_FIELDS = (
    "time_utc",
    "time_bjt",
    "source",
    "source_hdf",
    "source_size_bytes",
    "source_mtime_ns",
    "data_path",
    "aux_path",
    "is_day",
    "previous_aux_state",
    "previous_data_state",
    "agri_valid_fraction",
    "all_channel_valid_fraction",
    "obs_time_valid_fraction",
    "obs_time_offset_min_s",
    "obs_time_offset_max_s",
    "data_quality",
    "incomplete_scans",
    "qa_scan_flag",
    "qa_pixel_flag",
    "sun_avoidance_flag",
    "aux_schema_version",
    "data_schema_version",
    "status",
    "elapsed_seconds",
    "error",
)


def run_processing(
    records: Sequence[Tuple[str, str, str]],
    output_root: Path,
    workers: int,
    overwrite: bool,
    progress_every: int,
    final_index_name: str = "file_index.csv",
) -> Tuple[Dict[str, int], Path]:
    if Path(final_index_name).name != final_index_name:
        raise ValueError("final_index_name must be a plain filename")
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = log_dir / f"preprocess_{run_stamp}.jsonl"
    partial_csv = output_root / f"file_index.partial_{run_stamp}.csv"
    counts: Dict[str, int] = {}
    started = time.monotonic()

    with jsonl_path.open("a", encoding="utf-8", buffering=1) as jsonl, partial_csv.open(
        "w", encoding="utf-8", newline="", buffering=1
    ) as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()

        def record_result(result: Dict[str, object], completed: int) -> None:
            status = str(result.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
            jsonl.write(json.dumps(result, ensure_ascii=False) + "\n")
            writer.writerow({field: result.get(field, "") for field in MANIFEST_FIELDS})
            if status == "failed":
                print(
                    f"FAILED {result.get('time_utc')} {result.get('source_hdf')}: "
                    f"{result.get('error')}",
                    flush=True,
                )
            if completed == 1 or completed % progress_every == 0 or completed == len(records):
                elapsed = max(time.monotonic() - started, 1e-6)
                rate = completed / elapsed
                eta_hours = (len(records) - completed) / rate / 3600.0 if rate else float("inf")
                print(
                    f"PROGRESS {completed}/{len(records)} rate={rate:.3f} file/s "
                    f"eta={eta_hours:.2f} h counts={json.dumps(counts, sort_keys=True)}",
                    flush=True,
                )

        if workers == 1:
            for completed, item in enumerate(records, 1):
                record_result(process_one(item, str(output_root), overwrite), completed)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(process_one, item, str(output_root), overwrite): item
                    for item in records
                }
                for completed, future in enumerate(as_completed(futures), 1):
                    try:
                        result = future.result()
                    except BaseException as exc:
                        item = futures[future]
                        result = {
                            "time_utc": item[0],
                            "source": item[1],
                            "source_hdf": item[2],
                            "status": "failed",
                            "error": f"worker failure: {type(exc).__name__}: {exc}",
                        }
                    record_result(result, completed)

    if counts.get("failed", 0) == 0:
        final_csv = output_root / final_index_name
        os.replace(partial_csv, final_csv)
        return counts, final_csv
    return counts, partial_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, default=HISTORICAL_ROOT)
    parser.add_argument("--realtime-root", type=Path, default=REALTIME_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-utc", type=parse_utc, default=parse_utc(DEFAULT_START_UTC))
    parser.add_argument("--end-utc", type=parse_utc, help="exclusive UTC endpoint; default: latest available at scan time")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true", help="rewrite existing complete artifacts")
    parser.add_argument("--dry-run", action="store_true", help="only scan and report selected sources")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="refresh static metadata and the full source manifest without processing frames",
    )
    parser.add_argument("--limit", type=int, help="process only the first N selected times (testing only)")
    parser.add_argument(
        "--hdf-file",
        action="append",
        type=Path,
        default=[],
        help="explicit HDF path; repeat to run a controlled smoke test",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be >= 1")
    if args.end_utc is not None and args.end_utc <= args.start_utc:
        raise SystemExit("--end-utc must be later than --start-utc")

    scan_started = time.monotonic()
    if args.hdf_file:
        records = records_from_explicit_files(args.hdf_file)
    else:
        records = collect_records(
            args.historical_root, args.realtime_root, args.start_utc, args.end_utc
        )
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be >= 1")
        records = records[: args.limit]

    print(f"Selected {len(records)} unique UTC times in {time.monotonic() - scan_started:.1f} s")
    if records:
        source_counts = {
            source: sum(1 for _, item_source, _ in records if item_source == source)
            for source in ("historical", "realtime")
        }
        print(f"UTC range: {records[0][0]} through {records[-1][0]}")
        print(f"Sources: {json.dumps(source_counts, sort_keys=True)}")
    if args.dry_run or not records:
        return 0

    load_static()
    write_static_metadata(args.output_root, records, args.start_utc, args.end_utc)
    source_manifest = write_source_manifest(args.output_root, records)
    print(f"Output root: {args.output_root}")
    print(f"Source manifest: {source_manifest}")
    if args.metadata_only:
        print("Metadata-only refresh complete")
        return 0
    print(f"Workers: {args.workers}; day_mask: SOZ < {DAY_SOZ_LIMIT:g} deg")
    counts, file_index = run_processing(
        records,
        args.output_root,
        args.workers,
        args.overwrite,
        args.progress_every,
    )
    print(f"Finished counts={json.dumps(counts, sort_keys=True)}")
    print(f"File index: {file_index}")
    return 1 if counts.get("failed", 0) else 0


if __name__ == "__main__":
    sys.exit(main())
