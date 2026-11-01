#!/usr/bin/env python3
"""Extract CPP CLP labels aligned to the frozen monthly-R train/val pack."""
import argparse, csv, hashlib, json, time
from collections import defaultdict
from pathlib import Path

import netCDF4
import numpy as np


def arr(var, key=Ellipsis):
    return np.asarray(np.ma.filled(var[key], np.nan), dtype=np.float32)


def axes(nc):
    latv, lonv = nc.variables["LAT"], nc.variables["LON"]
    lat = arr(latv, (slice(None), 0)) if len(latv.shape) == 2 else arr(latv)
    lon = arr(lonv, (0, slice(None))) if len(lonv.shape) == 2 else arr(lonv)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("CPP LAT/LON are not rectilinear")
    return lat, lon


def cpp_index(lat_axis, lon_axis, lat2d, lon2d):
    lat_row, lon_col = lat2d[:, lat2d.shape[1] // 2], lon2d[lon2d.shape[0] // 2, :]
    rr = np.abs(lat_axis[:, None] - lat_row[None, :]).argmin(0)
    cc = np.abs(lon_axis[:, None] - lon_col[None, :]).argmin(0)
    err = max(float(np.max(np.abs(lat2d - lat_axis[rr, None]))),
              float(np.max(np.abs(lon2d - lon_axis[None, cc]))))
    if err > 1e-4 or len(np.unique(rr)) != 16 or len(np.unique(cc)) != 16:
        raise ValueError(f"CPP grid mismatch: {err}")
    return rr, cc, err


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for k in ("pack", "candidates", "grid-static", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    status = json.loads((a.pack / "pack_status.json").read_text())
    if status.get("state") != "COMPLETE_MONTHLY_R_STATION16_PACK" or status.get("test_payloads_read") != 0:
        raise ValueError("input R pack is not frozen train/val-only data")
    if status.get("split_id") != "hunan_monthly_block_2d_buffer1d_seed42_v1":
        raise ValueError("unexpected split contract")
    rows = list(csv.DictReader((a.pack / "rows.csv").open(newline="", encoding="utf-8")))
    sup = [r for r in rows if r["state"] == "SUPERVISED"]
    if any(r["split"] not in ("train", "val") for r in sup):
        raise ValueError("test or non-train/val rows found")
    cot = np.load(a.pack / "cot.npy", mmap_mode="r", allow_pickle=False)
    mask = np.load(a.pack / "mask.npy", mmap_mode="r", allow_pickle=False).astype(bool)
    x = np.load(a.pack / "x_raw.npy", mmap_mode="r", allow_pickle=False)
    if x.shape != (len(sup), 16, 16, 16) or cot.shape != (len(sup), 1, 16, 16) or mask.shape != cot.shape:
        raise ValueError(f"pack arrays mismatch: {x.shape}, {cot.shape}, {mask.shape}, rows={len(sup)}")
    if not np.array_equal(np.asarray([int(r["array_index"]) for r in sup]), np.arange(len(sup))):
        raise ValueError("pack supervised rows do not align with array order")
    with np.load(a.grid_static, allow_pickle=False) as z:
        glat, glon = np.asarray(z["lat"], np.float32), np.asarray(z["lon"], np.float32)
    if glat.shape != (256, 256) or glon.shape != (256, 256):
        raise ValueError("unexpected Hunan grid")
    candidates = {(r["station"].lower(), r["target_time_utc"]): r
                  for r in csv.DictReader(a.candidates.open(newline="", encoding="utf-8-sig"))}
    groups = defaultdict(list)
    for i, r in enumerate(sup):
        key = (r["station"].lower(), r["target_time_utc"])
        c = candidates.get(key)
        if c is None or c["split"] != r["split"]:
            raise ValueError(f"missing/mismatched frozen candidate: {key}")
        if c["agri_frame_relpath"] not in r["agri_path"]:
            raise ValueError(f"AGRI path differs from candidate manifest: {key}")
        groups[r["target_time_utc"]].append((i, r, c))

    a.output.parent.mkdir(parents=True, exist_ok=True)
    partial = a.output.with_suffix(".partial.npy")
    progress_path = a.output.with_suffix(".progress.json")
    ordered_groups = sorted(groups.items())
    if partial.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress.get("pack_status_sha256") != sha(a.pack / "pack_status.json") or progress.get("groups") != len(ordered_groups):
            raise ValueError("partial extraction checkpoint belongs to a different pack")
        clp = np.load(partial, mmap_mode="r+", allow_pickle=False)
        if clp.shape != (len(sup), 1, 16, 16): raise ValueError("partial CLP shape mismatch")
        completed = int(progress["completed_groups"])
        maxerr = float(progress.get("max_grid_coordinate_error_deg", 0.0))
    elif partial.exists() or progress_path.exists():
        raise ValueError("incomplete CLP partial/checkpoint pair; inspect before removing")
    else:
        clp = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8, shape=(len(sup), 1, 16, 16))
        clp[:] = 255; clp.flush(); completed = 0; maxerr = 0.0
        progress_path.write_text(json.dumps({"completed_groups": 0, "groups": len(ordered_groups),
                                             "pack_status_sha256": sha(a.pack / "pack_status.json")}, indent=2)+"\n")
    files_read = completed
    for group_idx, (utc, group) in enumerate(ordered_groups, 1):
        if group_idx <= completed: continue
        paths = {r["cpp_path"] for _, r, _ in group}
        if len(paths) != 1:
            raise ValueError(f"timestamp maps to multiple CPP files: {utc}")
        path = Path(next(iter(paths)))
        if not path.is_file():
            raise FileNotFoundError(path)
        for attempt in range(3):
            for i, _, _ in group: clp[i] = 255
            try:
                with netCDF4.Dataset(path) as nc:
                    if "CLP" not in nc.variables: raise ValueError(f"CLP missing: {path}")
                    lat_axis, lon_axis = axes(nc)
                    for i, r, c in group:
                        cr, cc = map(int, c["station_grid_rc"].split(","))
                        sl = np.s_[cr - 8:cr + 8, cc - 8:cc + 8]
                        rr, cols, err = cpp_index(lat_axis, lon_axis, glat[sl], glon[sl])
                        maxerr = max(maxerr, err)
                        rlo, rhi = int(rr.min()), int(rr.max()) + 1
                        clo, chi = int(cols.min()), int(cols.max()) + 1
                        z = arr(nc.variables["CLP"], (slice(rlo, rhi), slice(clo, chi)))[np.ix_(rr-rlo, cols-clo)]
                        if np.diff(rr).mean() < 0: z = z[::-1]
                        if np.diff(cols).mean() < 0: z = z[:, ::-1]
                        rounded = np.rint(z)
                        valid = np.isfinite(z) & (np.abs(z-rounded) < 1e-4) & np.isin(rounded, (0, 1, 2))
                        clp[i, 0][valid] = rounded[valid].astype(np.uint8)
                break
            except Exception as exc:
                if attempt == 2: raise RuntimeError(f"CPP read failed after 3 attempts: {utc} {path}: {exc}") from exc
                time.sleep(.5 * (attempt + 1))
        files_read += 1
        if group_idx % 500 == 0 or group_idx == len(ordered_groups):
            clp.flush()
            progress_path.write_text(json.dumps({"completed_groups": group_idx, "groups": len(ordered_groups),
                                                 "pack_status_sha256": sha(a.pack / "pack_status.json"),
                                                 "max_grid_coordinate_error_deg": maxerr}, indent=2)+"\n")
        if group_idx % 500 == 0:
            print(f"PROGRESS cpp_files={files_read}/{len(groups)}", flush=True)

    common = mask[:, 0] & (clp[:, 0] != 255)
    if np.any(mask & ~np.isin(clp, (0, 1, 2))):
        raise ValueError("CLP missing under the frozen CPP/R supervision mask")
    if not common.any():
        raise RuntimeError("no common CLP/R support")
    np.savez_compressed(a.output, clp=clp, common_mask=common[:, None],
                        sample_indices=np.arange(len(sup), dtype=np.int64),
                        split=np.asarray([r["split"] for r in sup]),
                        station=np.asarray([r["station"] for r in sup]),
                        target_time_utc=np.asarray([r["target_time_utc"] for r in sup]))
    report = {
        "state": "PASS_TRAIN_VAL_CLP_EXTRACTION", "split_id": status["split_id"],
        "train_samples": sum(r["split"] == "train" for r in sup),
        "val_samples": sum(r["split"] == "val" for r in sup),
        "unique_cpp_files_read": files_read, "common_valid_pixels": int(common.sum()),
        "valid_pixels": int(mask.sum()), "max_grid_coordinate_error_deg": maxerr,
        "clp_counts_on_common_mask": {str(k): int(np.count_nonzero(common & (clp[:, 0] == k))) for k in (0,1,2)},
        "trainval_only": True, "test_payloads_read": 0,
        "candidate_sha256": sha(a.candidates), "grid_static_sha256": sha(a.grid_static),
        "R_pack_sha256": sha(a.pack / "pack_status.json"), "sidecar_sha256": sha(a.output),
        "note": "CLP 0=clear, 1/2=cloud; labels aligned to SUPERVISED rows in frozen R pack; CPP unavailable/invalid remain excluded, never mapped to clear."
    }
    a.output.with_suffix(".audit.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
