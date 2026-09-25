"""Read-only one-time witness for matching a Hunan regional grid to CPP COT.

The raw CPP file is an offline retrieval reference. Its producer provenance
and independent physical QA are outside the scope of this coordinate audit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np


STATIONS = {"sili": (163, 148), "zhujia": (73, 150)}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def region(cy, cx, size):
    if size == 256:
        return (slice(0, 256), slice(0, 256))
    return (slice(cy-size//2, cy+size//2), slice(cx-size//2, cx+size//2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--cpp", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise RuntimeError("refuse to overwrite regional CPP witness")
    with np.load(a.grid, allow_pickle=False) as grid:
        lat = np.asarray(grid["lat"])
        lon = np.asarray(grid["lon"])
    if lat.shape != (256, 256) or lon.shape != (256, 256):
        raise RuntimeError("unexpected Hunan grid shape")
    with netCDF4.Dataset(a.cpp) as nc:
        source_lat = np.asarray(np.ma.filled(nc.variables["LAT"][:], np.nan))
        source_lon = np.asarray(np.ma.filled(nc.variables["LON"][:], np.nan))
        if source_lat.ndim != 1 or source_lon.ndim != 1:
            raise RuntimeError("CPP axes must be one-dimensional")
        rows = np.abs(source_lat[:, None] - lat[:, 0][None, :]).argmin(axis=0)
        cols = np.abs(source_lon[:, None] - lon[0, :][None, :]).argmin(axis=0)
        lat_error = float(np.max(np.abs(lat-source_lat[rows, None])))
        lon_error = float(np.max(np.abs(lon-source_lon[None, cols])))
        if max(lat_error, lon_error) > 1e-4:
            raise RuntimeError(f"CPP grid mismatch: {lat_error}, {lon_error}")
        if (len(set(rows.tolist())) != 256 or len(set(cols.tolist())) != 256 or
                np.any(np.abs(np.diff(rows)) != 1) or np.any(np.abs(np.diff(cols)) != 1)):
            raise RuntimeError("regional CPP indices are not unique contiguous axes")
        raw = nc.variables["COT"][int(rows.min()):int(rows.max())+1,
                                  int(cols.min()):int(cols.max())+1]
        cot = np.asarray(np.ma.filled(raw, np.nan), dtype=np.float32)
        if rows[0] > rows[-1]:
            cot = cot[::-1]
        if cols[0] > cols[-1]:
            cot = cot[:, ::-1]
    if cot.shape != (256, 256):
        raise RuntimeError("CPP crop is not 256x256")
    report = {"state": "REGIONAL_CPP_COORDINATE_WITNESS", "test_used": False,
              "cpp_file": str(a.cpp), "cpp_sha256": sha(a.cpp),
              "grid_sha256": sha(a.grid),
              "mapping": "nearest existing CPP LAT/LON axis; no interpolation",
              "max_coordinate_error_deg": {"lat": lat_error, "lon": lon_error},
              "limitations": "single timestamp; finite/range checks are not independent CPP QA or producer binding",
              "regions": {}}
    for name, (cy, cx) in STATIONS.items():
        report["regions"][name] = {}
        for size in (16, 64, 128, 256):
            patch = cot[region(cy, cx, size)]
            valid = np.isfinite(patch) & (patch >= 0) & (patch <= 100)
            report["regions"][name][str(size)] = {
                "shape": list(patch.shape), "finite_in_range_fraction": float(valid.mean()),
                "valid_pixels": int(valid.sum())}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"coordinate_error": report["max_coordinate_error_deg"],
                      "regions": report["regions"]}))


if __name__ == "__main__":
    main()
