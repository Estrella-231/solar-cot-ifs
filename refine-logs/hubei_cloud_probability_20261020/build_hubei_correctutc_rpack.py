#!/usr/bin/env python3
"""Rebuild Hubei R train/validation pack using corrected UTC CPP labels.

Only the audited AGRI/geometry x_norm and train-only normalization are reused
from the historical pack. Its old COT targets and masks are never opened.
"""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-pack", required=True, type=Path)
    p.add_argument("--cpp-sidecar", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--cossoz-min", type=float, default=0.17364817766693041,
                   help="cos(80 degrees); excludes SOZ>=80 degrees")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(f"refusing to overwrite {a.output}")
    status = json.loads((a.source_pack / "pack_status.json").read_text(encoding="utf-8"))
    if status.get("state") != "HUBEI_R_TRAIN_VAL_PACK_BUILT_TEST_CLOSED":
        raise ValueError("unexpected historical input pack state")
    norm_path = a.source_pack / "norm.json"
    norm = json.loads(norm_path.read_text(encoding="utf-8"))
    if norm.get("fit_split") != "train only" or norm.get("test_used") is not False:
        raise ValueError("input normalization is not train-only")
    with (a.source_pack / "rows.csv").open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows or any(r.get("split") not in ("train", "val") for r in rows):
        raise ValueError("historical input rows are not train/val only")
    x = np.load(a.source_pack / "x_norm.npy", mmap_mode="r", allow_pickle=False)
    if x.shape != (len(rows), 16, 16, 16) or not np.isfinite(x).all():
        raise ValueError(f"unexpected normalized input shape/finite status {x.shape}")
    mean = np.asarray(norm["mean"], dtype=np.float64)
    std = np.asarray(norm["std"], dtype=np.float64)
    if mean.shape != (16,) or std.shape != (16,) or np.any(std <= 0):
        raise ValueError("invalid frozen input normalization")
    with np.load(a.cpp_sidecar, allow_pickle=False) as z:
        keys = z["time_bjt"].astype(str)
        splits = z["split"].astype(str)
        cot = np.asarray(z["cot"], dtype=np.float32)
        cm = np.asarray(z["cot_valid_mask"], dtype=bool)
        lat = np.asarray(z["lat"], dtype=np.float32)
        lon = np.asarray(z["lon"], dtype=np.float32)
    source_keys = [r["time_bjt"] for r in rows]
    if len(keys) != len(rows) or len(set(keys.tolist())) != len(keys):
        raise ValueError("CPP sidecar key count/uniqueness differs from historical AGRI input keys")
    sidecar_index = {k: i for i, k in enumerate(keys.tolist())}
    if set(source_keys) != set(sidecar_index):
        raise ValueError("CPP sidecar and historical AGRI input key sets differ")
    source_split = {r["time_bjt"]: r["split"] for r in rows}
    for j, k in enumerate(keys.tolist()):
        if splits[j] != source_split[k]:
            raise ValueError(f"CPP sidecar split mismatch at key {k}")
    # The original input pack is ordered train then validation, while the raw
    # extraction manifest may interleave split blocks. Join by unique key,
    # never by incidental row position.
    reorder = np.asarray([sidecar_index[k] for k in source_keys], dtype=np.int64)
    cot = cot[reorder]
    cm = cm[reorder]
    keys = keys[reorder]
    splits = splits[reorder]
    if cot.shape != (len(rows), 16, 16) or cm.shape != cot.shape:
        raise ValueError("CPP COT sidecar shape mismatch")
    # Reconstruct only the already normalized geometry channels. This never
    # reads the historical CPP target or its old mask.
    cossoz = np.asarray(x[:, 13], dtype=np.float64) * std[13] + mean[13]
    day = np.asarray(x[:, 15], dtype=np.float64) * std[15] + mean[15]
    if not np.isfinite(cossoz).all() or not np.isfinite(day).all():
        raise ValueError("nonfinite decoded solar geometry")
    if np.any(np.minimum(np.abs(day), np.abs(day - 1.0)) > 2e-3):
        raise ValueError("decoded day_mask is not binary within tolerance")
    if np.any(np.abs(cossoz) > 1.01):
        raise ValueError("decoded cosSOZ is outside physical range")
    geom_valid = (day > 0.5) & (cossoz > a.cossoz_min)
    mask = cm & geom_valid
    if not np.isfinite(cot[mask]).all() or np.any((cot[mask] < 0) | (cot[mask] > 100)):
        raise ValueError("valid COT values outside [0,100]")
    target = np.zeros((len(rows), 1, 16, 16), dtype=np.float32)
    target[:, 0] = np.where(mask, cot / 100.0, 0.0)
    mask_out = mask[:, None].astype(bool)

    a.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(a.source_pack / "x_norm.npy", a.output / "x_norm.npy")
    shutil.copyfile(norm_path, a.output / "norm.json")
    np.save(a.output / "target.npy", target, allow_pickle=False)
    np.save(a.output / "mask.npy", mask_out, allow_pickle=False)
    # Preserve all audited train/val rows but point the CPP lineage only to the
    # correctly UTC-matched raw NetCDF. The old target/mask files are excluded.
    cpp_rows_path = a.cpp_sidecar.parent / "rows_correct_utc.csv"
    if not cpp_rows_path.is_file():
        raise FileNotFoundError(cpp_rows_path)
    cpp_rows = list(csv.DictReader(cpp_rows_path.open(newline="", encoding="utf-8")))
    cpp_by_key = {r["time_bjt"]: r for r in cpp_rows}
    if len(cpp_rows) != len(rows) or set(cpp_by_key) != set(source_keys):
        raise ValueError("correct-UTC row ledger key-set mismatch")
    out_rows = []
    for i, r in enumerate(rows):
        cr = cpp_by_key[r["time_bjt"]]
        if cr["split"] != r["split"]:
            raise ValueError(f"correct-UTC row ledger split mismatch at {r['time_bjt']}")
        out_rows.append({"index": i, "time_bjt": r["time_bjt"], "split": r["split"],
                         "path_agri": r["path_agri"], "time_cpp_utc": cr["time_cpp_utc"],
                         "path_cpp_correct_utc": cr["path_cpp_correct_utc"],
                         "valid_cot_pixels": int(mask[i].sum())})
    with (a.output / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)
    counts = {s: {"rows": int(sum(r["split"] == s for r in rows)),
                  "supervised_rows": int(sum(r["split"] == s and mask[i].any() for i, r in enumerate(rows))),
                  "valid_pixels": int(mask[np.asarray([r["split"] == s for r in rows])].sum())}
              for s in ("train", "val")}
    pack_status = {
        "state": "HUBEI_R_TRAIN_VAL_CORRECT_UTC_PACK_BUILT_TEST_CLOSED",
        "rows": len(rows), "counts_by_split": counts,
        "source_input_pack_status_sha256": sha(a.source_pack / "pack_status.json"),
        "source_x_norm_sha256": sha(a.source_pack / "x_norm.npy"),
        "source_norm_sha256": sha(norm_path),
        "cpp_sidecar_sha256": sha(a.cpp_sidecar),
        "cpp_rows_sha256": sha(cpp_rows_path),
        "time_rule": "CPP UTC = BJT - 8 hours",
        "spatial_grid": {"lat_shape": list(lat.shape), "lon_shape": list(lon.shape),
                         "max_abs_lat": float(np.max(np.abs(lat))), "max_abs_lon": float(np.max(np.abs(lon)))},
        "target": "physical COT in [0,100], stored divided by 100; loss log1p after trainer rescales",
        "mask": "correct-UTC CPP COT valid mask AND decoded AGRI day_mask>0.5 AND cosSOZ>cos(80deg)",
        "normalization_fit": "historical train-only AGRI/geometry normalization; old CPP target and mask not read",
        "test_payloads_read": 0, "buffer_payloads_read": 0,
        "reference_status": "CPP retrieval supervision, not independent truth",
    }
    (a.output / "pack_status.json").write_text(json.dumps(pack_status, indent=2, ensure_ascii=False), encoding="utf-8")
    hashed = ("rows.csv", "x_norm.npy", "target.npy", "mask.npy", "norm.json")
    (a.output / "SHA256.json").write_text(json.dumps({n: sha(a.output / n) for n in hashed}, indent=2), encoding="utf-8")
    (a.output / "pack_report.json").write_text(json.dumps({
        "state": pack_status["state"], "counts_by_split": counts,
        "valid_clp_cot_pixels": int(mask.sum()),
        "train_geometry_valid_pixels": int(geom_valid[np.asarray([r["split"] == "train" for r in rows])].sum()),
        "val_geometry_valid_pixels": int(geom_valid[np.asarray([r["split"] == "val" for r in rows])].sum()),
        "train_valid_cot_range": [float(np.nanmin(cot[np.asarray([r["split"] == "train" for r in rows]) & np.any(mask, axis=(1,2))])),
                                  float(np.nanmax(cot[np.asarray([r["split"] == "train" for r in rows]) & np.any(mask, axis=(1,2))]))],
        "test_buffer_payloads_read": 0,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"pack": str(a.output), "pack_status": pack_status}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
