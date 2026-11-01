"""Read only selected new-contract validation CPP samples and summarize quality."""
import argparse
import csv
import json
from pathlib import Path

import netCDF4
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    with np.load(a.grid, allow_pickle=False) as grid:
        glat, glon = np.asarray(grid["lat"]), np.asarray(grid["lon"])
    out = []
    with a.samples.open(newline="", encoding="utf-8-sig") as stream:
        for rec in csv.DictReader(stream):
            path = Path(rec["cpp_path"])
            with netCDF4.Dataset(path) as nc:
                lat = np.asarray(np.ma.filled(nc.variables["LAT"][:], np.nan))
                lon = np.asarray(np.ma.filled(nc.variables["LON"][:], np.nan))
                if lat.ndim != 1 or lon.ndim != 1:
                    raise RuntimeError(f"expected 1D coordinate axes: {path}")
                rows = np.abs(lat[:, None] - glat[:, 0][None, :]).argmin(0)
                cols = np.abs(lon[:, None] - glon[0, :][None, :]).argmin(0)
                lat_err = float(np.max(np.abs(glat - lat[rows, None])))
                lon_err = float(np.max(np.abs(glon - lon[None, cols])))
                raw = nc.variables["COT"][rows.min():rows.max()+1, cols.min():cols.max()+1]
                cot = np.asarray(np.ma.filled(raw, np.nan), np.float32)
                if rows[0] > rows[-1]: cot = cot[::-1]
                if cols[0] > cols[-1]: cot = cot[:, ::-1]
                clp_var = nc.variables.get("CLP")
                clp_counts = None
                zero_by_clp = None
                clp_attrs = None
                if clp_var is not None:
                    clp_raw = clp_var[rows.min():rows.max()+1, cols.min():cols.max()+1]
                    clp = np.asarray(np.ma.filled(clp_raw, -999))
                    if rows[0] > rows[-1]: clp = clp[::-1]
                    if cols[0] > cols[-1]: clp = clp[:, ::-1]
                    clp_vals, clp_nums = np.unique(clp, return_counts=True)
                    clp_counts = {str(v.item()): int(n) for v, n in zip(clp_vals, clp_nums)}
                    zero_by_clp = {str(v.item()): int(np.sum((clp == v) & (cot == 0)))
                                   for v in clp_vals}
                    clp_attrs = {k: str(getattr(clp_var, k)) for k in clp_var.ncattrs()}
                finite = np.isfinite(cot)
                valid = finite & (cot >= 0) & (cot <= 100)
                summary = {"split": rec["split"], "utc": rec["utc"], "bjt": rec["bjt"],
                           "season": rec["season_bjt"], "path": str(path),
                           "variables": list(nc.variables),
                           "cot_source_shape": list(nc.variables["COT"].shape),
                           "cot_dtype": str(nc.variables["COT"].dtype),
                           "cot_attrs": {k: str(getattr(nc.variables["COT"], k))
                                         for k in nc.variables["COT"].ncattrs()},
                           "clp_attrs": clp_attrs, "clp_counts_256": clp_counts,
                           "cot_zero_counts_by_clp_256": zero_by_clp,
                           "coord_error_deg": {"lat": lat_err, "lon": lon_err},
                           "region": {"finite_fraction": float(finite.mean()),
                                      "valid_0_100_fraction": float(valid.mean()),
                                      "zero_fraction_valid": float((cot[valid] == 0).mean()) if valid.any() else None,
                                      "thick_ge30_fraction_valid": float((cot[valid] >= 30).mean()) if valid.any() else None,
                                      "mean_valid": float(cot[valid].mean()) if valid.any() else None,
                                      "p90_valid": float(np.percentile(cot[valid], 90)) if valid.any() else None},
                           "stations_5x5": {}}
                for name, (r, c) in {"sili": (163, 148), "zhujia": (73, 150)}.items():
                    patch = cot[r-2:r+3, c-2:c+3]
                    pm = np.isfinite(patch) & (patch >= 0) & (patch <= 100)
                    summary["stations_5x5"][name] = {
                        "valid_pixels": int(pm.sum()), "total_pixels": 25,
                        "mean_valid": float(patch[pm].mean()) if pm.any() else None,
                        "zero_count": int((patch[pm] == 0).sum()),
                        "thick_ge30_count": int((patch[pm] >= 30).sum())}
            out.append(summary)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"state": "SELECTED_NEW_SPLIT_VAL_CPP_PAYLOAD_SPOTCHECK",
                                    "test_used": False,
                                    "limits": "four selected seasonal files; not a full pixel-QA or producer audit",
                                    "samples": out}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"samples": len(out), "output": str(a.output),
                      "valid_fractions": [x["region"]["valid_0_100_fraction"] for x in out]}), flush=True)


if __name__ == "__main__":
    main()
