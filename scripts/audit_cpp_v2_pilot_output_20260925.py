"""Check a new CPP V2 file on the fixed Hunan grid without reading test data."""
import argparse
import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--cpp", type=Path, required=True)
    p.add_argument("--reference-cpp", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    with np.load(a.grid, allow_pickle=False) as grid:
        lat = np.asarray(grid["lat"])
        lon = np.asarray(grid["lon"])
    if lat.shape != lon.shape or lat.shape != (256, 256):
        raise ValueError("unexpected Hunan grid")
    with netCDF4.Dataset(a.cpp) as ds:
        sy = np.asarray(ds.variables["LAT"][:])
        sx = np.asarray(ds.variables["LON"][:])
        rows = np.abs(sy[:, None] - lat[:, 0]).argmin(axis=0)
        cols = np.abs(sx[:, None] - lon[0, :]).argmin(axis=0)
        lat_err = float(np.max(np.abs(sy[rows, None] - lat)))
        lon_err = float(np.max(np.abs(sx[None, cols] - lon)))
        if max(lat_err, lon_err) > 1e-4:
            raise ValueError((lat_err, lon_err))
        if len(set(rows)) != 256 or len(set(cols)) != 256:
            raise ValueError("nonunique grid mapping")
        y0, y1 = int(rows.min()), int(rows.max()) + 1
        x0, x1 = int(cols.min()), int(cols.max()) + 1
        data = {}
        for key in ("COT", "CLP", "QA"):
            z = np.asarray(np.ma.filled(ds.variables[key][y0:y1, x0:x1], np.nan))
            if rows[0] > rows[-1]:
                z = z[::-1]
            if cols[0] > cols[-1]:
                z = z[:, ::-1]
            data[key] = z
    cot, clp, qa = (data[k] for k in ("COT", "CLP", "QA"))
    valid = np.isfinite(cot) & (cot >= 0) & (cot <= 100)
    qa_valid = np.isfinite(qa) & np.isin(qa, (0, 1))
    report = {
        "schema": "hunan_cpp_v2_pilot_qc_v1", "test_used": False,
        "cpp": str(a.cpp), "cpp_sha256": file_sha(a.cpp),
        "grid_sha256": file_sha(a.grid),
        "max_coordinate_error_deg": {"lat": lat_err, "lon": lon_err},
        "region": {"cot_valid_fraction": float(valid.mean()),
                   "cot_mean_valid": float(cot[valid].mean()) if valid.any() else None,
                   "cot_p90_valid": float(np.percentile(cot[valid], 90)) if valid.any() else None,
                   "qa_valid_fraction": float(qa_valid.mean()),
                   "qa_previous_hour_fraction": float(np.mean(qa == 1)),
                   "clp_cloud_fraction": float(np.mean(np.isin(clp, (1, 2))))},
        "stations": {},
    }
    for name, (cy, cx) in {"sili": (163, 148), "zhujia": (73, 150)}.items():
        s = np.s_[cy-2:cy+3, cx-2:cx+3]
        report["stations"][name] = {
            "center_cot": float(cot[cy, cx]),
            "cot_5x5_mean": float(np.nanmean(cot[s])),
            "cot_5x5_max": float(np.nanmax(cot[s])),
            "qa_5x5_previous_hour_count": int(np.sum(qa[s] == 1)),
        }
    if a.reference_cpp is not None:
        with netCDF4.Dataset(a.reference_cpp) as ref:
            ref_lat = np.asarray(ref.variables["LAT"][:])
            ref_lon = np.asarray(ref.variables["LON"][:])
            if not np.allclose(ref_lat, sy) or not np.allclose(ref_lon, sx):
                raise ValueError("reference grid differs from V2 grid")
            ref_cot = np.asarray(np.ma.filled(ref.variables["COT"][y0:y1, x0:x1], np.nan))
            if rows[0] > rows[-1]:
                ref_cot = ref_cot[::-1]
            if cols[0] > cols[-1]:
                ref_cot = ref_cot[:, ::-1]
        both = valid & np.isfinite(ref_cot) & (ref_cot >= 0) & (ref_cot <= 100)
        delta = cot[both] - ref_cot[both]
        report["same_time_reference"] = {
            "cpp": str(a.reference_cpp), "cpp_sha256": file_sha(a.reference_cpp),
            "common_valid_fraction": float(both.mean()),
            "cot_mae": float(np.abs(delta).mean()) if both.any() else None,
            "cot_bias_v2_minus_reference": float(delta.mean()) if both.any() else None,
            "reference_cot_mean": float(ref_cot[both].mean()) if both.any() else None,
            "v2_cot_mean": float(cot[both].mean()) if both.any() else None,
        }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["region"]))
    print(json.dumps(report["stations"]))


if __name__ == "__main__":
    main()
