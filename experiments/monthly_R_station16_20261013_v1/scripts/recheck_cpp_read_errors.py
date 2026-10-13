#!/usr/bin/env python3
"""Read-only retry audit of CPP files that failed during the train/val pack."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import netCDF4
import numpy as np

import pack_station16_cpp_agri as packer


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--errors", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--grid-static", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)

    with a.candidates.open(newline="", encoding="utf-8-sig") as f:
        candidate = {}
        for row in csv.DictReader(f):
            if row["split"] not in ("train", "val"):
                raise RuntimeError("refusing to inspect non-train/validation candidate")
            candidate.setdefault(row["target_time_utc"], {})[row["station"]] = tuple(
                map(int, row["station_grid_rc"].split(",")))
    with np.load(a.grid_static, allow_pickle=False) as z:
        glat, glon = np.asarray(z["lat"], np.float32), np.asarray(z["lon"], np.float32)
    with a.errors.open(newline="", encoding="utf-8-sig") as f:
        errors = list(csv.DictReader(f))
    results = []
    for row in errors:
        if row["split"] != "train":
            raise RuntimeError("unexpected validation/test source error row")
        utc, path = row["target_time_utc"], row["cpp_path"]
        item = {"target_time_utc": utc, "split": row["split"], "cpp_path": path,
                "original_error": row["error"], "retries": []}
        stations = candidate.get(utc)
        if not stations or set(stations) != set(packer.STATIONS):
            item["final_state"] = "AUDIT_KEY_MISMATCH"
            results.append(item)
            continue
        for attempt in (1, 2):
            try:
                with netCDF4.Dataset(path) as nc:
                    lat_axis, lon_axis = packer.axes(nc)
                    for station in packer.STATIONS:
                        cr, cc = stations[station]
                        sl = np.s_[cr - 8:cr + 8, cc - 8:cc + 8]
                        rr, ccols, _ = packer.cpp_index(lat_axis, lon_axis, glat[sl], glon[sl])
                        rlo, rhi = int(rr.min()), int(rr.max()) + 1
                        clo, chi = int(ccols.min()), int(ccols.max()) + 1
                        for variable in ("COT", "CLP"):
                            if variable not in nc.variables:
                                raise KeyError(f"missing {variable}")
                            block = packer.arr(nc.variables[variable],
                                               (slice(rlo, rhi), slice(clo, chi)))
                            if block.size == 0:
                                raise RuntimeError(f"empty {variable} block")
                item["retries"].append({"attempt": attempt, "state": "READ_OK"})
            except Exception as exc:  # record source evidence; never infer clear sky
                item["retries"].append({"attempt": attempt,
                                        "state": "READ_ERROR",
                                        "error": f"{type(exc).__name__}: {exc}"})
        item["final_state"] = ("REPRODUCIBLE_READ_FAILURE"
                               if all(x["state"] == "READ_ERROR" for x in item["retries"])
                               else "TRANSIENT_OR_INTERMITTENT_READ")
        results.append(item)
    report = {"state": "CPP_SOURCE_ERROR_RECHECK_COMPLETE",
              "test_payloads_read": 0, "errors_csv_sha256": sha(a.errors),
              "candidate_sha256": sha(a.candidates), "failed_frames": len(results),
              "summary": {k: sum(x["final_state"] == k for x in results)
                          for k in ("REPRODUCIBLE_READ_FAILURE", "TRANSIENT_OR_INTERMITTENT_READ",
                                    "AUDIT_KEY_MISMATCH")},
              "results": results}
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
