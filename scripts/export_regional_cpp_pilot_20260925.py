"""Export a small stratified raw-CPP pilot on its source host.

Only train/val manifest metadata are used. This is a data-pairing pilot, not
an independently QA-certified COT training pack.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import netCDF4
import numpy as np


def choose(values, count):
    values = sorted(values)
    if len(values) <= count:
        return values
    return [values[i] for i in np.linspace(0, len(values)-1, count, dtype=int)]


def main():
    p = argparse.ArgumentParser()
    for key in ("manifest", "grid", "cpp-root", "output"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--per-month", type=int, default=4)
    a = p.parse_args()
    if a.output.exists():
        raise RuntimeError("refuse to overwrite CPP regional pilot")
    with np.load(a.grid, allow_pickle=False) as g:
        target_lat, target_lon = np.asarray(g["lat"]), np.asarray(g["lon"])
    by_split_month = defaultdict(set)
    with a.manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["split"] not in ("train", "val"):
                continue
            for rel in row["data_relpaths"].split("|"):
                stem = Path(rel).stem
                if 0 <= int(stem[8:10]) <= 8:
                    by_split_month[(row["split"], stem[:6])].add(stem)
    a.output.mkdir(parents=True)
    summary = {"state": "REGIONAL_CPP_STRATIFIED_PILOT", "test_used": False,
               "reference": "original CPP retrieval, no independent QA/producer binding",
               "manifest_sha256": hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
               "grid_sha256": hashlib.sha256(a.grid.read_bytes()).hexdigest(),
               "per_month_requested": a.per_month, "splits": {}}
    for split in ("train", "val"):
        selected = []
        for (s, month), stems in sorted(by_split_month.items()):
            if s != split:
                continue
            available = [stem for stem in stems if
                         (a.cpp_root / stem[:4] / stem[:8] /
                          f"FY4B_AGRI_{stem}00.nc").is_file()]
            selected.extend(choose(available, a.per_month))
        if not selected:
            raise RuntimeError(f"no {split} CPP pilot frames")
        arrays, masks, records = [], [], []
        for stem in selected:
            path = a.cpp_root / stem[:4] / stem[:8] / f"FY4B_AGRI_{stem}00.nc"
            with netCDF4.Dataset(path) as nc:
                lat = np.asarray(np.ma.filled(nc.variables["LAT"][:], np.nan))
                lon = np.asarray(np.ma.filled(nc.variables["LON"][:], np.nan))
                rows = np.abs(lat[:, None]-target_lat[:, 0][None, :]).argmin(0)
                cols = np.abs(lon[:, None]-target_lon[0, :][None, :]).argmin(0)
                error = max(float(np.max(np.abs(target_lat-lat[rows, None]))),
                            float(np.max(np.abs(target_lon-lon[None, cols]))))
                if (error > 1e-4 or len(set(rows.tolist())) != 256 or
                        len(set(cols.tolist())) != 256 or
                        np.any(np.abs(np.diff(rows)) != 1) or
                        np.any(np.abs(np.diff(cols)) != 1)):
                    raise RuntimeError(f"bad CPP grid for {stem}: {error}")
                raw = nc.variables["COT"][int(rows.min()):int(rows.max())+1,
                                          int(cols.min()):int(cols.max())+1]
                cot = np.asarray(np.ma.filled(raw, np.nan), np.float32)
                if rows[0] > rows[-1]:
                    cot = cot[::-1]
                if cols[0] > cols[-1]:
                    cot = cot[:, ::-1]
            if cot.shape != (256, 256):
                raise RuntimeError(f"bad CPP crop for {stem}")
            valid = np.isfinite(cot) & (cot >= 0) & (cot <= 100)
            arrays.append(np.where(valid, cot, 0).astype(np.float32))
            masks.append(valid)
            stat = path.stat()
            records.append({"utc": stem, "cpp_file": str(path), "size_bytes": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns, "coordinate_error_deg": error,
                            "valid_fraction": float(valid.mean())})
        out = a.output / f"{split}_cpp_256.npz"
        np.savez_compressed(out, utc=np.asarray(selected, dtype="S12"),
                            cot=np.stack(arrays), valid=np.stack(masks))
        summary["splits"][split] = {
            "frames": len(selected), "utc_months": sorted({s[:6] for s in selected}),
            "mean_valid_fraction": float(np.mean([r["valid_fraction"] for r in records])),
            "archive": out.name, "archive_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "records": records}
    (a.output / "manifest.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({s: {"frames": v["frames"], "mean_valid_fraction": v["mean_valid_fraction"]}
                      for s, v in summary["splits"].items()}))


if __name__ == "__main__":
    main()
