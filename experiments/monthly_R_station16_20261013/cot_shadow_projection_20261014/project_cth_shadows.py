#!/usr/bin/env python3
"""Parallax-correct official FY-4B CTH and cast cloud-top solar shadows.

Only four prespecified monthly-contract validation cases are read. Satellite
pixel geolocation comes from the hash-verified Hunan grid_static; scan times are
per-pixel NOMObsTime-derived offsets in the matching AGRI sidecars.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data"
CTH_TEXT = DATA / "ncdump_128px"
PAYLOAD = DATA / "cot_shadow_validation_case_payload.npz"
GRID = ROOT.parents[2] / "audits" / "spatial_reaudit_static_20260908" / "grid_static.npz"
GRID_SHA = "54a38fcd217a74e0def533217580b48132ef25fee40ed9ed51063e4efdd42c62"
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
EARTH_R = 6371008.8
SAT_LON_DEG = 105.0
SAT_RADIUS_M = 42164000.0  # geostationary centre-to-satellite distance
CASE_TEXTS = [
    "sili_20240419T0415Z.txt",
    "sili_20240611T0630Z.txt",
    "zhujia_20240529T0345Z.txt",
    "zhujia_20240420T0630Z.txt",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ncdump_values(text: str, name: str) -> np.ndarray:
    matches = list(re.finditer(rf"(?m)^\s*{re.escape(name)}\s*=\s*(.*?);", text, re.S))
    if not matches:
        raise ValueError(f"variable {name} not found in ncdump")
    match = matches[-1]
    vals = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", match.group(1))
    return np.asarray([float(v) for v in vals], dtype=np.float64)


def geodetic_to_ecef(lat_deg: np.ndarray, lon_deg: np.ndarray, h_m: np.ndarray | float = 0.0) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    h = np.asarray(h_m, dtype=np.float64)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    return np.stack(((n + h) * np.cos(lat) * np.cos(lon),
                     (n + h) * np.cos(lat) * np.sin(lon),
                     (n * (1.0 - WGS84_E2) + h) * np.sin(lat)), axis=-1)


def ecef_to_geodetic(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    lon = np.arctan2(y, x)
    rho = np.hypot(x, y)
    lat = np.arctan2(z, rho * (1.0 - WGS84_E2))
    for _ in range(7):
        n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
        lat = np.arctan2(z + WGS84_E2 * n * np.sin(lat), rho)
    return np.rad2deg(lat), np.rad2deg(lon)


def sun_ecef(dt_utc: np.ndarray) -> np.ndarray:
    """Approximate apparent Sun direction in ECEF (NOAA low-order solar model)."""
    jd = np.asarray([d.replace(tzinfo=timezone.utc).timestamp() / 86400.0 + 2440587.5 for d in dt_utc])
    t = (jd - 2451545.0) / 36525.0
    l0 = np.deg2rad(np.mod(280.46646 + 36000.76983 * t + 0.0003032 * t * t, 360.0))
    m = np.deg2rad(357.52911 + 35999.05029 * t - 0.0001537 * t * t)
    ecc = 0.016708634 - 0.000042037 * t - 0.0000001267 * t * t
    c = (np.sin(m) * (1.914602 - 0.004817 * t - 0.000014 * t * t)
         + np.sin(2 * m) * (0.019993 - 0.000101 * t) + np.sin(3 * m) * 0.000289)
    omega = np.deg2rad(125.04 - 1934.136 * t)
    lam = l0 + np.deg2rad(c) - np.deg2rad(0.00569) - np.deg2rad(0.00478) * np.sin(omega)
    eps = np.deg2rad(23.439291 - 0.0130042 * t + 0.00256 * np.cos(omega))
    ra = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    d = jd - 2451545.0
    gmst = np.deg2rad(np.mod(280.46061837 + 360.98564736629 * d
                             + 0.000387933 * ((jd - 2451545.0) / 36525.0) ** 2, 360.0))
    gha = ra - gmst
    return np.stack((np.cos(dec) * np.cos(gha), np.cos(dec) * np.sin(gha), np.sin(dec)), axis=-1)


def local_solar_angles(sun: np.ndarray, lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    east = np.stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)), axis=-1)
    north = np.stack((-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)), axis=-1)
    up = np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1)
    e, n, u = np.sum(sun * east, axis=-1), np.sum(sun * north, axis=-1), np.sum(sun * up, axis=-1)
    return np.rad2deg(np.arcsin(np.clip(u, -1, 1))), np.mod(np.rad2deg(np.arctan2(e, n)), 360.0)


def sun_ecef_from_local_angles(lat_deg: np.ndarray, lon_deg: np.ndarray,
                               altitude_deg: np.ndarray, azimuth_deg: np.ndarray) -> np.ndarray:
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    alt, az = np.deg2rad(altitude_deg), np.deg2rad(azimuth_deg)
    east = np.stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)), axis=-1)
    north = np.stack((-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)), axis=-1)
    up = np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1)
    return (np.cos(alt)[..., None] * np.sin(az)[..., None] * east
            + np.cos(alt)[..., None] * np.cos(az)[..., None] * north
            + np.sin(alt)[..., None] * up)


def ellipsoid_intersection(origin: np.ndarray, direction: np.ndarray, height_m: np.ndarray | float) -> np.ndarray:
    """Intersect rays with an ellipsoid shell offset by height_m."""
    h = np.asarray(height_m, dtype=np.float64)
    aa, bb = WGS84_A + h, WGS84_B + h
    inva2, invb2 = 1.0 / aa**2, 1.0 / bb**2
    A = (direction[..., 0] ** 2 + direction[..., 1] ** 2) * inva2 + direction[..., 2] ** 2 * invb2
    B = 2.0 * ((origin[..., 0] * direction[..., 0] + origin[..., 1] * direction[..., 1]) * inva2
               + origin[..., 2] * direction[..., 2] * invb2)
    C = (origin[..., 0] ** 2 + origin[..., 1] ** 2) * inva2 + origin[..., 2] ** 2 * invb2 - 1.0
    disc = np.maximum(B * B - 4 * A * C, 0.0)
    t0 = (-B - np.sqrt(disc)) / (2 * A)
    t1 = (-B + np.sqrt(disc)) / (2 * A)
    t = np.where(t0 >= 0, t0, t1)
    return origin + t[..., None] * direction


def nominal_to_cloud_top(lat: np.ndarray, lon: np.ndarray, height_m: np.ndarray) -> np.ndarray:
    sat_lon = np.deg2rad(SAT_LON_DEG)
    sat = np.array([SAT_RADIUS_M * np.cos(sat_lon), SAT_RADIUS_M * np.sin(sat_lon), 0.0])
    ground = geodetic_to_ecef(lat, lon, 0.0)
    direction = ground - sat
    p = ellipsoid_intersection(sat, direction, height_m)
    return p


def cast_shadow(cloud_top: np.ndarray, sun: np.ndarray) -> np.ndarray:
    down = -sun
    return ellipsoid_intersection(cloud_top, down, 0.0)


def local_xy(lat: np.ndarray, lon: np.ndarray, station_lat: float, station_lon: float) -> tuple[np.ndarray, np.ndarray]:
    x = EARTH_R * np.cos(np.deg2rad(station_lat)) * np.deg2rad(lon - station_lon)
    y = EARTH_R * np.deg2rad(lat - station_lat)
    return x / 1000.0, y / 1000.0


def distance_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.hypot(x, y)


def main() -> None:
    if sha256(GRID).lower() != GRID_SHA:
        raise ValueError("grid_static hash does not match the current T16 Hunan contract")
    with np.load(GRID, allow_pickle=False) as z:
        grid = {k: np.asarray(z[k]) for k in z.files}
    # Use the full-disk FY4B 105E 4-km nominal geolocation LUT. In the supplied
    # CDO hyperslab, the first array dimension follows the CTH line/row axis;
    # verify this from the known (row_index, col_index) station centre below.
    lut_files = {
        "sili": DATA / "lut_128px" / "sili_128.txt",
        "zhujia": DATA / "lut_128px" / "zhujia_128.txt",
    }
    geo_lut = {}
    for station_name, path in lut_files.items():
        t = path.read_text(encoding="utf-8")
        la = ncdump_values(t, "latitude").reshape(128, 128)
        lo = ncdump_values(t, "longitude").reshape(128, 128)
        geo_lut[station_name] = (la, lo)
    grid_key = (grid["row_index"].astype(np.int64) * 3000 + grid["col_index"].astype(np.int64)).ravel()
    grid_sort = np.argsort(grid_key)
    grid_key_sorted = grid_key[grid_sort]
    payload = np.load(PAYLOAD, allow_pickle=False)
    metadata = json.loads(PAYLOAD.with_suffix(".json").read_text(encoding="utf-8"))
    rows = metadata["case_keys"]
    records = []
    processed = []
    for i, (case_text, info) in enumerate(zip(CASE_TEXTS, rows)):
        case_t0 = time.perf_counter()
        print(f"[{i+1}/4] parsing {case_text}", flush=True)
        txt = (CTH_TEXT / case_text).read_text(encoding="utf-8")
        xrow = ncdump_values(txt, "x").astype(int)
        ycol = ncdump_values(txt, "y").astype(int)
        cth = ncdump_values(txt, "CTH")
        dqf = ncdump_values(txt, "DQF").astype(np.int32)
        shape = (len(xrow), len(ycol))
        if len(cth) != np.prod(shape) or len(dqf) != np.prod(shape):
            raise ValueError(f"array/coordinate length mismatch for {case_text}: {shape}")
        cth = cth.reshape(shape)
        dqf = dqf.reshape(shape)
        print(f"  parsed {shape}, elapsed={time.perf_counter()-case_t0:.1f}s", flush=True)
        station = info["station"]
        station_rc = (163, 148) if station == "sili" else (73, 150)
        row0 = int(grid["row_index"][station_rc])
        col0 = int(grid["col_index"][station_rc])
        ri = np.where(xrow == row0)[0]
        ci = np.where(ycol == col0)[0]
        if len(ri) != 1 or len(ci) != 1:
            raise ValueError(f"CTH crop does not contain station pixel {station} row/col={(row0, col0)}; x={xrow.min()}..{xrow.max()}, y={ycol.min()}..{ycol.max()}")
        center_index = (int(ri[0]), int(ci[0]))
        if center_index != (64, 64):
            raise ValueError(f"CTH crop center mismatch {case_text}: {center_index}")
        lat, lon = geo_lut[station]
        if lat.shape != shape or lon.shape != shape:
            raise ValueError(f"official LUT crop shape mismatch for {case_text}: {lat.shape}/{shape}")
        mapped = np.isfinite(lat) & np.isfinite(lon)
        if mapped.mean() < 0.999:
            raise ValueError(f"official LUT has incomplete geolocation for {case_text}: {mapped.mean():.3%}")

        h = cth / 1000.0
        quality = dqf & 3
        cloud_class = (dqf >> 2) & 3
        cth_valid = np.isfinite(cth) & (cth >= 1.0) & (cth <= 20000.0)
        cloud_candidate = cth_valid & (cloud_class <= 1)
        main = cloud_candidate & (quality >= 2) & mapped
        inclusive = cloud_candidate & (quality >= 1) & mapped
        ix, iy = int(station_rc[1]), int(station_rc[0])
        # Every pixel's UTC scan time is the matching L1 HDF start plus its
        # cached NOMObsTime-derived offset; no nominal 15-minute timestamp is
        # substituted for the actual station scan.
        start_utc = datetime.strptime(Path(info["source_hdf"]).name.split("_NOM_")[1][:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        # NOMObsTime is line-based: all columns on a scan line share the same
        # cached offset. Recover each full-disk row from the current 256 crop and
        # require exact within-row constancy before applying it to CTH pixels.
        offset_map = payload["obs_time_offset_s"][i]
        row_offset = {}
        for gr in xrow:
            cells = offset_map[grid["row_index"] == int(gr)]
            if cells.size == 0 or float(cells.max() - cells.min()) > 1e-3:
                raise ValueError(f"missing/nonconstant NOMObsTime scan offset for full-disk line {gr}")
            row_offset[int(gr)] = float(np.median(cells))
        off = np.repeat(np.asarray([row_offset[int(r)] for r in xrow])[:, None], len(ycol), axis=1)
        times = np.asarray([start_utc + timedelta(seconds=float(s)) for s in off.ravel()], dtype=object).reshape(shape)
        print(f"  scan times ready, elapsed={time.perf_counter()-case_t0:.1f}s", flush=True)
        sat_time = start_utc + timedelta(seconds=float(info["scan_offset_s_from_source_start_at_station"]))
        lat_m, lon_m, hm = lat[main], lon[main], h[main]
        selected_indices = np.argwhere(main)
        rr_full = xrow[selected_indices[:, 0]]
        cc_full = ycol[selected_indices[:, 1]]
        # Start with the independent ephemeris, then replace solar direction
        # with the exact sidecar cos(SOZ)/cos(RAA) geometry wherever the native
        # Hunan 256x256 crop contains the same full-disk pixel.
        print(f"  {len(rr_full)} QC-valid cloud pixels; calculating solar vectors", flush=True)
        sun = sun_ecef(times[main])
        sun_ephemeris_only = sun.copy()
        print(f"  solar vectors ready, elapsed={time.perf_counter()-case_t0:.1f}s", flush=True)
        fallback_alt, fallback_az = local_solar_angles(sun, lat_m, lon_m)
        side_cos = np.full(len(rr_full), np.nan)
        side_raa = np.full(len(rr_full), np.nan)
        side_sat_az = np.full(len(rr_full), np.nan)
        case_key = rr_full.astype(np.int64) * 3000 + cc_full.astype(np.int64)
        match_pos = np.searchsorted(grid_key_sorted, case_key)
        in_range = match_pos < len(grid_key_sorted)
        matched = np.zeros(len(case_key), dtype=bool)
        matched[in_range] = grid_key_sorted[match_pos[in_range]] == case_key[in_range]
        if matched.any():
            local_flat = grid_sort[match_pos[matched]]
            local_rr, local_cc = np.unravel_index(local_flat, grid["row_index"].shape)
            side_cos[matched] = payload["cosSOZ"][i, local_rr, local_cc]
            side_raa[matched] = payload["cosRAA"][i, local_rr, local_cc]
            side_sat_az[matched] = grid["saa"][local_rr, local_cc]
        print(f"  matched sidecar geometry, elapsed={time.perf_counter()-case_t0:.1f}s", flush=True)
        has_side = np.isfinite(side_cos) & np.isfinite(side_raa) & np.isfinite(side_sat_az)
        if has_side.any():
            delta = np.rad2deg(np.arccos(np.clip(side_raa[has_side], -1.0, 1.0)))
            az1 = np.mod(side_sat_az[has_side] + delta, 360.0)
            az2 = np.mod(side_sat_az[has_side] - delta, 360.0)
            diff1 = np.abs((az1 - fallback_az[has_side] + 180.0) % 360.0 - 180.0)
            diff2 = np.abs((az2 - fallback_az[has_side] + 180.0) % 360.0 - 180.0)
            side_az = np.where(diff1 <= diff2, az1, az2)
            side_alt = np.rad2deg(np.arcsin(np.clip(side_cos[has_side], -1.0, 1.0)))
            sun[has_side] = sun_ecef_from_local_angles(lat_m[has_side], lon_m[has_side], side_alt, side_az)
        print(f"  transforming cloud-top rays, elapsed={time.perf_counter()-case_t0:.1f}s", flush=True)
        top_corrected = nominal_to_cloud_top(lat_m, lon_m, hm * 1000.0)
        shadow_corrected = cast_shadow(top_corrected, sun)
        shadow_ephemeris_only = cast_shadow(top_corrected, sun_ephemeris_only)
        top_naive = geodetic_to_ecef(lat_m, lon_m, hm * 1000.0)
        shadow_naive = cast_shadow(top_naive, sun)
        print(f"  ray geometry ready ({len(hm)} clouds), elapsed={time.perf_counter()-case_t0:.1f}s", flush=True)
        shadow_lat, shadow_lon = ecef_to_geodetic(shadow_corrected)
        shadow_ephem_lat, shadow_ephem_lon = ecef_to_geodetic(shadow_ephemeris_only)
        top_lat, top_lon = ecef_to_geodetic(top_corrected)
        naive_lat, naive_lon = ecef_to_geodetic(shadow_naive)
        station_lat = float(lat[center_index])
        station_lon = float(lon[center_index])
        sx, sy = local_xy(shadow_lat, shadow_lon, station_lat, station_lon)
        ex, ey = local_xy(shadow_ephem_lat, shadow_ephem_lon, station_lat, station_lon)
        nx, ny = local_xy(naive_lat, naive_lon, station_lat, station_lon)
        tx, ty = local_xy(top_lat, top_lon, station_lat, station_lon)
        d_corr, d_naive = distance_xy(sx, sy), distance_xy(nx, ny)
        min_idx = int(np.argmin(d_corr))
        center_sun = sun_ecef(np.asarray([sat_time], dtype=object))
        noaa_alt, noaa_az = local_solar_angles(center_sun, np.asarray([station_lat]), np.asarray([station_lon]))
        center_offset = int(np.ravel_multi_index(center_index, shape))
        cth_center = float(cth[center_index])
        dqf_center = int(dqf[center_index])
        cots = np.asarray(payload["r_cot_3x3"][i])
        center_cos = float(payload["cosSOZ"][i, station_rc[0], station_rc[1]])
        center_raa = float(payload["cosRAA"][i, station_rc[0], station_rc[1]])
        station_sat_az = float(grid["saa"][station_rc])
        station_delta = float(np.rad2deg(np.arccos(np.clip(center_raa, -1.0, 1.0))))
        station_az_candidates = np.mod([station_sat_az + station_delta, station_sat_az - station_delta], 360.0)
        station_diff = np.abs((station_az_candidates - noaa_az[0] + 180.0) % 360.0 - 180.0)
        sun_alt_deg = float(np.rad2deg(np.arcsin(np.clip(center_cos, -1.0, 1.0))))
        sun_az_deg = float(station_az_candidates[int(np.argmin(station_diff))])
        computed_cos = float(np.sin(np.deg2rad(noaa_alt[0])))
        computed_raa = float(np.cos(np.deg2rad(noaa_az[0] - station_sat_az)))
        # Approximate ground-pixel footprint from adjacent coordinates in the
        # authoritative Hunan grid; show a 2/4/6 km center-distance sensitivity.
        nlat = float(lat[center_index[0] + 1, center_index[1]])
        elon = float(lon[center_index[0], center_index[1] + 1])
        d_north = EARTH_R * np.deg2rad(nlat - station_lat)
        d_east = EARTH_R * np.cos(np.deg2rad(station_lat)) * np.deg2rad(elon - station_lon)
        cell_radius = 0.5 * float(np.hypot(d_north, d_east)) / 1000.0

        # CTH/quality map and actual RGB scene use matched nominal image indices.
        local_x, local_y = local_xy(lat, lon, station_lat, station_lon)
        rgb = payload["agri_rgb_c03_c02_c01"][i]
        rgb_valid = payload["rgb_valid"][i].astype(bool)
        r0, c0 = station_rc
        slr, slc = slice(r0 - 32, r0 + 32), slice(c0 - 32, c0 + 32)
        rgb_crop = np.moveaxis(rgb[:, slr, slc], 0, -1)
        mask_crop = np.all(rgb_valid[:, slr, slc], axis=0)
        rgb_disp = np.clip(rgb_crop / 0.9, 0.0, 1.0) ** (1.0 / 2.2)
        rgb_lut = np.s_[center_index[0] - 32:center_index[0] + 32, center_index[1] - 32:center_index[1] + 32]
        rgb_x, rgb_y = local_xy(lat[rgb_lut], lon[rgb_lut], station_lat, station_lon)

        # Main visual is clipped to +/-75 km; metrics use every valid cloud in
        # the 128x128 CTH field, which extends to about +/-256 km.
        px, py = local_xy(lat_m, lon_m, station_lat, station_lon)
        record = {
            "case": f"{station}_{info['target_time_utc'].replace(' ', 'T')}Z",
            "station": station,
            "target_time_utc": info["target_time_utc"],
            "scan_time_utc_at_station": sat_time.isoformat().replace("+00:00", "Z"),
            "case_group": info["outcome_group"],
            "helpful_seed_count": info["helpful_seed_count"],
            "observed_ghi_wm2": info["ghi_wm2"],
            "clear_sky_ghi_wm2": info["clear_sky_ghi_wm2"],
            "R_COT_center": info["center_r_cot"],
            "A_mean_wm2": info["a_seed_mean_wm2"],
            "AC_mean_wm2": info["ac_seed_mean_wm2"],
            "delta_sse_AC_minus_A_mean": info["ac_minus_a_delta_sse_mean"],
            "station_lat": station_lat,
            "station_lon": station_lon,
            "station_full_disk_row": row0,
            "station_full_disk_col": col0,
            "CTH_at_station_m": cth_center,
            "CTH_DQF_at_station": dqf_center,
            "CTH_station_quality_low_bits": dqf_center & 3,
            "CTH_station_cloud_class_bits": (dqf_center >> 2) & 3,
            "CTH_valid_cloud_pixels_all_QC": int(cloud_candidate.sum()),
            "CTH_cloud_pixels_main_quality_ge_2": int(main.sum()),
            "CTH_cloud_pixels_inclusive_quality_ge_1": int(inclusive.sum()),
            "CTH_mean_main_km": float(np.mean(h[main])),
            "CTH_p90_main_km": float(np.percentile(h[main], 90)),
            "parallax_shift_median_km": float(np.median(np.hypot(tx - px, ty - py))),
            "parallax_shift_p90_km": float(np.percentile(np.hypot(tx - px, ty - py), 90)),
            "shadow_nearest_distance_km_main": float(d_corr[min_idx]),
            "shadow_nearest_distance_km_NOAA_ephemeris_sensitivity": float(np.min(distance_xy(ex, ey))),
            "shadow_nearest_distance_sensitivity_abs_km": float(abs(np.min(distance_xy(ex, ey)) - np.min(d_corr))),
            "shadow_nearest_distance_km_without_parallax": float(np.min(d_naive)),
            "nearest_shadow_cth_km": float(hm[min_idx]),
            "nearest_shadow_cloud_top_east_km": float(tx[min_idx]),
            "nearest_shadow_cloud_top_north_km": float(ty[min_idx]),
            "nearest_shadow_east_km": float(sx[min_idx]),
            "nearest_shadow_north_km": float(sy[min_idx]),
            # Sensitivity includes poor-but-valid retrieval quality 1. Keep CTH
            # in metres at the geometry boundary, matching the main mask.
            "nearest_shadow_distance_km_inclusive_QC": float(np.min(distance_xy(*local_xy(*ecef_to_geodetic(cast_shadow(nominal_to_cloud_top(lat[inclusive], lon[inclusive], h[inclusive] * 1000.0), sun_ecef(times[inclusive]))), station_lat, station_lon)))),
            "station_pixel_half_diagonal_km": cell_radius,
            "coverage_2km": bool(np.min(d_corr) <= 2.0),
            "coverage_4km": bool(np.min(d_corr) <= 4.0),
            "coverage_6km": bool(np.min(d_corr) <= 6.0),
            "solar_altitude_deg": sun_alt_deg,
            "solar_azimuth_deg": sun_az_deg,
            "cosSOZ_sidecar": center_cos,
            "cosSOZ_recomputed": computed_cos,
            "cosSOZ_abs_error": abs(center_cos - computed_cos),
            "cosRAA_sidecar": center_raa,
            "cosRAA_recomputed": computed_raa,
            "cosRAA_abs_error": abs(center_raa - computed_raa),
            "DQF_doc_bits_used": "bit0-1 retrieval quality; bit2-3 cloud class; higher bits excluded from mask",
            "projection_assumption": "CTH source pixels use current nominal fixed-grid row/col; each pixel top is ray-intersected from FY4B geostationary position with the WGS84 ellipsoid at CTH height, then a Sun-ray is intersected with the surface",
        }
        records.append(record)
        processed.append({
            "info": info, "rgb": rgb_disp, "rgb_mask": mask_crop,
            "rgb_x": rgb_x, "rgb_y": rgb_y,
            "cth_nominal_x": local_x, "cth_nominal_y": local_y,
            "cloud_x": px, "cloud_y": py, "cloud_h": hm,
            "top_x": tx, "top_y": ty, "shadow_x": sx, "shadow_y": sy,
            "d_corr": d_corr, "nearest": min_idx, "sun_az": sun_az_deg,
            "main_mask": main, "inclusive_mask": inclusive,
            "cth": cth, "dqf": dqf, "lat": lat, "lon": lon,
            "grid_row": xrow, "grid_col": ycol,
            "station_xy": (station_lat, station_lon), "cell_radius": cell_radius,
            "r_cot_3x3": cots,
        })

    # Store reproducible per-case geometry summary before rendering.
    with (ROOT / "shadow_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.6), constrained_layout=True)
    titles = ["Sili · 2024-04-19 · COT helps", "Sili · 2024-06-11 · COT hurts",
              "Zhujia · 2024-05-29 · COT helps", "Zhujia · 2024-04-20 · COT hurts"]
    for j, (axrgb, axgeo, case, rec, title) in enumerate(zip(axes[0], axes[1], processed, records, titles)):
        m = case["rgb_mask"]
        axrgb.scatter(case["rgb_x"][m], case["rgb_y"][m], c=case["rgb"][m], s=12, marker="s", linewidths=0)
        axrgb.scatter([0], [0], marker="+", c="#00ffff", s=75, linewidths=2.0, zorder=5)
        axrgb.set_title(title, fontsize=10, fontweight="bold")
        axrgb.set_xlim(-125, 125); axrgb.set_ylim(-125, 125); axrgb.set_aspect("equal")
        axrgb.set_xlabel("east from station (km)");
        if j == 0: axrgb.set_ylabel("north from station (km)")
        axrgb.text(0.02, 0.02, f"actual AGRI C03/C02/C01\nGHI {rec['observed_ghi_wm2']:.0f} / clear {rec['clear_sky_ghi_wm2']:.0f} W m$^{{-2}}$",
                   transform=axrgb.transAxes, fontsize=8, color="white", va="bottom",
                   bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=3))

        lim = 75.0
        cloud_show = (np.abs(case["top_x"]) <= lim) & (np.abs(case["top_y"]) <= lim)
        shadow_show = (np.abs(case["shadow_x"]) <= lim) & (np.abs(case["shadow_y"]) <= lim)
        sc = axgeo.scatter(case["top_x"][cloud_show], case["top_y"][cloud_show],
                           c=case["cloud_h"][cloud_show], s=7, cmap="magma", vmin=0, vmax=15,
                           alpha=0.82, edgecolors="none", label="parallax-corrected cloud top")
        axgeo.scatter(case["shadow_x"][shadow_show], case["shadow_y"][shadow_show],
                      c="#48c7d1", s=5, alpha=0.45, edgecolors="none", label="surface shadow point")
        n = case["nearest"]
        axgeo.plot([case["top_x"][n], case["shadow_x"][n]], [case["top_y"][n], case["shadow_y"][n]],
                   color="#00d5ff", lw=1.2, alpha=0.95)
        axgeo.scatter([case["shadow_x"][n]], [case["shadow_y"][n]], marker="o", s=42,
                      facecolors="none", edgecolors="#00d5ff", linewidths=1.6, zorder=5)
        axgeo.scatter([0], [0], marker="*", s=125, c="#ffe84a", edgecolors="black", linewidths=0.8, zorder=6)
        axgeo.add_patch(plt.Circle((0, 0), 4.0, fill=False, color="#ffe84a", lw=1.2, ls="--"))
        # Arrow points in the local anti-solar (shadow) direction.
        saz = np.deg2rad(case["sun_az"])
        dx, dy = -np.sin(saz) * 18.0, -np.cos(saz) * 18.0
        axgeo.annotate("", xy=(dx, dy), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="white", lw=1.8))
        axgeo.text(dx, dy, "shadow dir.", color="white", fontsize=7, ha="left", va="center")
        axgeo.set_xlim(-lim, lim); axgeo.set_ylim(-lim, lim); axgeo.set_aspect("equal")
        axgeo.set_xlabel("east from station (km)")
        if j == 0: axgeo.set_ylabel("north from station (km)")
        coverage = "within 4 km" if rec["coverage_4km"] else "outside 4 km"
        axgeo.text(0.02, 0.02,
                   f"nearest shadow: {rec['shadow_nearest_distance_km_main']:.1f} km ({coverage})\n"
                   f"CTH@station {rec['CTH_at_station_m']/1000:.1f} km; QC={rec['CTH_station_quality_low_bits']}\n"
                   f"sun alt {rec['solar_altitude_deg']:.1f}°; Δparallax p90 {rec['parallax_shift_p90_km']:.1f} km",
                   transform=axgeo.transAxes, fontsize=7.5, color="white", va="bottom",
                   bbox=dict(facecolor="black", alpha=0.62, edgecolor="none", pad=3))
    cbar = fig.colorbar(sc, ax=axes[1, :], shrink=0.86, pad=0.01)
    cbar.set_label("Cloud-top height (km), DQF quality ≥ 2")
    fig.suptitle("Observed FY-4B cloud top → satellite-parallax correction → solar shadow footprint\n"
                 "Four held-out validation cases from distinct dates; shadow coverage is geometric, not radiative transfer",
                 fontsize=13, fontweight="bold")
    png, pdf = ROOT / "CTH_PARALLAX_SHADOW_VALIDATION_20261014.png", ROOT / "CTH_PARALLAX_SHADOW_VALIDATION_20261014.pdf"
    fig.savefig(png, dpi=190, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    receipt = {
        "state": "FOUR_CASE_CTH_PARALLAX_AND_SOLAR_SHADOW_AUDIT_COMPLETE",
        "split": "validation only; test payloads not accessed",
        "grid_static_sha256": sha256(GRID),
        "payload_sha256": sha256(PAYLOAD),
        "cth_input_sha256": {p.name: sha256(p) for p in sorted((DATA / "official_cth").glob("*.NC"))},
        "cth_extraction_sha256": {p.name: sha256(p) for p in sorted(CTH_TEXT.glob("*.txt"))},
        "method": {
            "nominal_geolocation": "current Hunan grid_static full-disk row_index/col_index and lat/lon; SHA verified",
            "parallax": "ray from FY4B geostationary satellite through nominal surface pixel, intersected with WGS84 ellipsoid at official CTH height",
            "shadow": "per-pixel Sun ECEF vector at matching NOMObsTime; cast from parallax-corrected cloud top to WGS84 surface",
            "cloud_mask": "official CTH valid range; DQF bits 2-3 cloud or possible-cloud; main retrieval-quality bits 0-1 >= 2; quality-1 sensitivity also reported",
            "pixel_coverage": "shadow-center distance thresholds 2, 4, and 6 km; 4 km is the primary map annotation",
            "sun_model": "NOAA low-order apparent solar position; station cosSOZ/cosRAA compared against the saved physical-geometry sidecars",
            "satellite_position": "geostationary centre-to-satellite radius 42,164 km at the source file's 105E subpoint; the CTH file numeric satellite-height field is 4.2164e7 while its units attribute says km, so the value is interpreted as metres/geocentric radius for the ray geometry",
        },
        "records": records,
        "files": [png.name, pdf.name, "shadow_metrics.csv"],
    }
    (ROOT / "projection_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    compact = [{k: r[k] for k in ("case", "case_group", "CTH_at_station_m", "CTH_DQF_at_station", "parallax_shift_median_km", "parallax_shift_p90_km", "shadow_nearest_distance_km_main", "nearest_shadow_distance_km_inclusive_QC", "coverage_4km", "solar_altitude_deg", "cosSOZ_abs_error", "cosRAA_abs_error")} for r in records]
    print(json.dumps({"state": receipt["state"], "records": compact, "outputs": receipt["files"]}, indent=2))


if __name__ == "__main__":
    main()
