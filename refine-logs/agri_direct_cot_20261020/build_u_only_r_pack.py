#!/usr/bin/env python3
"""Build an R_U pack from archived CPP rows that intersect fold-1 U frames only.

The source arrays are memory-mapped and only selected U row indices are read.
No H, outer-validation, test, or buffer array rows are accessed by this builder.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path

import numpy as np

SOURCE_STATUS_SHA = "1b15f3e22a778f6b4d9c1546cc2d8381f14faae6c95b1350a937731efad2928b"
SOURCE_X_SHA = "06d4606623c1495d63ab015d778b5691d6f0c072e58fad79c6e559c84d450a31"
SOURCE_COT_SHA = "43dab24e9b6c5cf9b9db09f6bce16b3dede60750b312c6e808b192ec265fb63c"
SOURCE_MASK_SHA = "2614ca0b589ff3e3367f08f0e8f8ed8067226f80446f4e27f3478d5c675c340f"
SOURCE_ROWS_SHA = "b270713590f6af793b53df73d27e01f3f62ad8d7e60bb9a8eaace5c43fb2ebcb"
GRID_SHA = "54a38fcd217a74e0def533217580b48132ef25fee40ed9ed51063e4efdd42c62"
CANDIDATE_SHA = "b85c2841e33cc92eda848acb183f926d49564b6c131d6418730e6b53978e2807"
CORRECTED_GEOMETRY_SHA_EXPECTED = "04779daf8cf1f7ca06b5119b603dfe51234fb9913149f617b3d37deeda4376a5"
COS80 = float(np.cos(np.deg2rad(80.0)))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def utc_key(value: str) -> str:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d%H%M")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-pack", type=Path, required=True)
    p.add_argument("--u-frames", type=Path, required=True)
    p.add_argument("--h-frames", type=Path, required=True)
    p.add_argument("--outer-val-candidates", type=Path, required=True)
    p.add_argument("--grid-static", type=Path, required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--geometry-module", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    expected = {
        "pack_status.json": SOURCE_STATUS_SHA,
        "x_raw.npy": SOURCE_X_SHA,
        "cot.npy": SOURCE_COT_SHA,
        "mask.npy": SOURCE_MASK_SHA,
        "rows.csv": SOURCE_ROWS_SHA,
    }
    actual = {name: sha(a.source_pack / name) for name in expected}
    if actual != expected:
        raise ValueError(f"archived CPP pack hash mismatch: {actual}")
    status = json.loads((a.source_pack / "pack_status.json").read_text(encoding="utf-8"))
    if status.get("state") != "COMPLETE_MONTHLY_R_STATION16_PACK" or status.get("test_payloads_read") != 0:
        raise ValueError("source pack is not the audited train/validation-only pack")
    if status.get("split_id") != "hunan_monthly_block_2d_buffer1d_seed42_v1":
        raise ValueError("unexpected outer split")
    if sha(a.grid_static) != GRID_SHA:
        raise ValueError("Hunan grid_static hash mismatch")
    if sha(a.candidates) != CANDIDATE_SHA:
        raise ValueError("candidate manifest hash mismatch")
    if not a.outer_val_candidates.is_file():
        raise ValueError("outer validation candidate manifest is missing")
    if sha(a.geometry_module) != CORRECTED_GEOMETRY_SHA_EXPECTED:
        raise ValueError("corrected solar geometry source hash mismatch")
    spec = importlib.util.spec_from_file_location("solar_geometry_corrected", a.geometry_module)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load corrected geometry module: {a.geometry_module}")
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    compute_solar_angles = geometry.compute_solar_angles

    u_lines = [line.strip() for line in a.u_frames.read_text(encoding="utf-8").splitlines() if line.strip()]
    u_stems = {Path(line.replace("\\", "/")).stem for line in u_lines}
    if len(u_stems) != len(u_lines):
        raise ValueError("duplicate U frame path")
    h_lines = [line.strip() for line in a.h_frames.read_text(encoding="utf-8").splitlines() if line.strip()]
    h_stems = {Path(line.replace("\\", "/")).stem for line in h_lines}
    if len(h_stems) != len(h_lines) or u_stems & h_stems:
        raise ValueError("fold-1 U/H frames are duplicate or overlapping")
    outer_val_stems = set()
    with a.outer_val_candidates.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["split"] == "val":
                outer_val_stems.add(Path(r["agri_frame_relpath"].replace("\\", "/")).stem)
            elif r["split"] != "train":
                raise ValueError("outer candidate file contains unexpected split")
    if (u_stems | h_stems) & outer_val_stems:
        raise ValueError("fold-1 U/H target frames intersect outer-validation candidate frames")

    station_rc = {}
    with a.candidates.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["split"] != "train":
                continue
            key = (utc_key(r["target_time_utc"]), r["station"])
            rc = tuple(map(int, r["station_grid_rc"].split(",")))
            if key in station_rc and station_rc[key] != rc:
                raise ValueError(f"station grid index changed at {key}")
            station_rc[key] = rc

    rows = []
    group_rows = {"U": 0, "H": 0, "outer_validation": 0, "other": 0}
    with (a.source_pack / "rows.csv").open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            # Source AGRI cache path and target_time_utc are UTC; bjt is UTC+8.
            k = utc_key(r["target_time_utc"])
            if Path(r["agri_path"]).stem != k:
                raise ValueError(f"AGRI UTC path/time mismatch: {r['agri_path']} vs {r['target_time_utc']}")
            dt = datetime.strptime(r["target_time_utc"], "%Y-%m-%d %H:%M:%S")
            from datetime import timedelta
            bjt = (dt + timedelta(hours=8)).strftime("%Y%m%d%H%M")
            if r["bjt"] != bjt:
                raise ValueError(f"UTC/BJT mismatch for {k}: {r['bjt']} != {bjt}")
            if r["split"] == "train" and r["state"] == "SUPERVISED" and k in u_stems:
                group_rows["U"] += 1
                if not r["array_index"]:
                    raise ValueError("supervised U row has no source array index")
                key = (k, r["station"])
                if key not in station_rc:
                    raise ValueError(f"U CPP row absent from audited candidate station map: {key}")
                rows.append({**r, "utc_key": k, "station_grid_rc": ",".join(map(str, station_rc[key]))})
            elif r["split"] == "train" and r["state"] == "SUPERVISED" and k in h_stems:
                group_rows["H"] += 1
            elif r["split"] == "val" and r["state"] == "SUPERVISED" and k in outer_val_stems:
                group_rows["outer_validation"] += 1
            elif r["split"] == "val" and r["state"] == "SUPERVISED" and k in u_stems:
                raise ValueError("outer validation row unexpectedly intersects fold-1 U")
            elif r["state"] == "SUPERVISED":
                group_rows["other"] += 1
    if not rows:
        raise ValueError("no supervised U rows matched the archived CPP pack")
    keys = [(r["utc_key"], r["station"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate supervised U station/time key")
    if any(r["split"] != "train" for r in rows):
        raise ValueError("non-training row selected for R_U")

    indices = np.asarray([int(r["array_index"]) for r in rows], dtype=np.int64)
    xsrc = np.load(a.source_pack / "x_raw.npy", mmap_mode="r", allow_pickle=False)
    ysrc = np.load(a.source_pack / "cot.npy", mmap_mode="r", allow_pickle=False)
    msrc = np.load(a.source_pack / "mask.npy", mmap_mode="r", allow_pickle=False)
    if xsrc.shape != (16812, 16, 16, 16) or ysrc.shape != (16812, 1, 16, 16) or msrc.shape != ysrc.shape:
        raise ValueError("archived source pack array shape mismatch")
    if len(set(indices.tolist())) != len(indices) or np.any(indices < 0) or np.any(indices >= len(xsrc)):
        raise ValueError("invalid/duplicate source array index")

    with np.load(a.grid_static, allow_pickle=False) as z:
        lat = np.asarray(z["lat"], np.float32)
        lon = np.asarray(z["lon"], np.float32)
        saa = np.asarray(z["saa"], np.float32)
    if any(v.shape != (256, 256) for v in (lat, lon, saa)):
        raise ValueError("expected audited Hunan 256x256 static grids")

    a.output.mkdir(parents=True)
    xout = np.lib.format.open_memmap(a.output / "x_corrected.npy", mode="w+", dtype=np.float32,
                                     shape=(len(rows), 16, 16, 16))
    yout = np.lib.format.open_memmap(a.output / "cot.npy", mode="w+", dtype=np.float32,
                                     shape=(len(rows), 1, 16, 16))
    mout = np.lib.format.open_memmap(a.output / "mask.npy", mode="w+", dtype=np.bool_,
                                     shape=(len(rows), 1, 16, 16))
    count = {"source_valid_pixels": 0, "common_valid_pixels": 0, "excluded_by_corrected_geometry": 0,
             "exact_zero_pixels": 0, "thick_ge30_pixels": 0}
    for out_i, (src_i, r) in enumerate(zip(indices, rows)):
        x = np.asarray(xsrc[src_i], np.float32).copy()
        y = np.asarray(ysrc[src_i], np.float32)
        old_mask = np.asarray(msrc[src_i], bool)
        rc, cc = map(int, r["station_grid_rc"].split(","))
        sl = np.s_[rc - 8:rc + 8, cc - 8:cc + 8]
        if rc < 8 or cc < 8 or rc + 8 > 256 or cc + 8 > 256:
            raise ValueError(f"station crop outside Hunan grid: {r['station']} {(rc, cc)}")
        dt = datetime.strptime(r["target_time_utc"], "%Y-%m-%d %H:%M:%S")
        soz, soa = compute_solar_angles(lat[sl], lon[sl], dt)
        physical_night = soz >= 90.0
        soa = np.where(physical_night, 0.0, soa)
        cos_soz = np.cos(np.deg2rad(soz)).astype(np.float32)
        cos_raa = np.cos(np.deg2rad(soa - saa[sl])).astype(np.float32)
        day_mask = (soz < 85.0).astype(np.float32)
        x[13] = cos_soz
        x[14] = cos_raa
        x[15] = day_mask
        common = old_mask & (day_mask[None] > 0.5) & (cos_soz[None] > COS80)
        if not np.isfinite(x[:, common[0]]).all() or not np.isfinite(y[common]).all():
            raise ValueError(f"nonfinite supervised U content: {r['utc_key']} {r['station']}")
        xout[out_i] = x
        yout[out_i] = y
        mout[out_i] = common
        n0 = int(old_mask.sum())
        n1 = int(common.sum())
        count["source_valid_pixels"] += n0
        count["common_valid_pixels"] += n1
        count["excluded_by_corrected_geometry"] += n0 - n1
        count["exact_zero_pixels"] += int(((y == 0) & common).sum())
        count["thick_ge30_pixels"] += int(((y >= 30) & common).sum())
        r["source_array_index"] = r.pop("array_index")
        r["array_index"] = str(out_i)
        r["source_split"] = r.pop("split")
        r["split"] = "U_train"
        r["mask_pixels_source"] = str(n0)
        r["mask_pixels_common"] = str(n1)
    if count["common_valid_pixels"] == 0:
        raise ValueError("empty U-only common CPP support")

    fields = list(rows[0].keys())
    with (a.output / "rows_u_supervised.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    artifact_hashes = {n: sha(a.output / n) for n in ("x_corrected.npy", "cot.npy", "mask.npy", "rows_u_supervised.csv")}
    report = {
        "state": "PASS_U_ONLY_R_PACK",
        "split_id": "future_ghi_oof_5fold_future_ghi_oof_20261019_fold1",
        "source_pack_status_sha256": sha(a.source_pack / "pack_status.json"),
        "source_pack_arrays_sha256": actual,
        "u_frame_manifest_sha256": sha(a.u_frames),
        "h_frame_manifest_sha256": sha(a.h_frames),
        "outer_validation_candidate_manifest_sha256": sha(a.outer_val_candidates),
        "u_unique_frames": len(u_stems),
        "u_supervised_station_rows": len(rows),
        "u_supervised_by_station": {s: sum(r["station"] == s for r in rows) for s in sorted({r["station"] for r in rows})},
        "u_supervised_utc_dates": len({r["utc_key"][:8] for r in rows}),
        "u_supervised_bjt_dates": len({r["bjt"][:8] for r in rows}),
        "u_supervised_date_range_bjt": [min(r["bjt"][:8] for r in rows), max(r["bjt"][:8] for r in rows)],
        "h_unique_frames": len(h_stems),
        "h_supervised_station_rows": group_rows["H"],
        "outer_validation_candidate_target_frames": len(outer_val_stems),
        "outer_validation_supervised_station_rows": group_rows["outer_validation"],
        "other_supervised_station_rows": group_rows["other"],
        "u_h_frame_intersection": len(u_stems & h_stems),
        "u_outer_validation_target_intersection": len(u_stems & outer_val_stems),
        "h_outer_validation_target_intersection": len(h_stems & outer_val_stems),
        "time_contract": "source AGRI path stem and target_time_utc are UTC; row bjt is UTC+8h; no extra offset applied to UTC stem",
        "train_rows_only": True,
        "h_rows_selected": 0,
        "outer_validation_rows_selected": 0,
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "geometry": "recomputed from frozen Hunan grid_static and corrected EOT radians-to-degrees formula",
        "geometry_source_sha256": sha(a.geometry_module),
        "candidate_manifest_sha256": sha(a.candidates),
        "grid_static_sha256": sha(a.grid_static),
        "support": "archived CPP and input-valid mask AND archived old-R geometry mask AND corrected day_mask>0.5 AND corrected cosSOZ>cos80",
        "counts": count,
        "artifacts_sha256": artifact_hashes,
        "normalization": "not fitted here; trainer must fit U-only common-mask pixels",
        "training_initialization": "not performed by pack builder",
    }
    (a.output / "pack_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
