#!/usr/bin/env python3
"""Full train/val pixel-support audit of the station16 R-COT bank."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

CHANNELS = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14)
COS80 = float(np.cos(np.deg2rad(80.0)))


def aux_for(root: Path, rel: str) -> Path:
    t = datetime.strptime(Path(rel).stem, "%Y%m%d%H%M") + timedelta(hours=8)
    matches = list((root / "aux_256" / t.strftime("%Y%m")).glob(
        t.strftime("%Y%m%d%H%M") + "_*_aux.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(f"aux sidecar count={len(matches)} for {rel}")
    return matches[0]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def audit_frame(idx: int, r: dict, cot: np.ndarray, data_root: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    rel = r["agri_frame_relpath"]
    entries = []
    geometries = []
    masks = []
    with np.load(aux_for(data_root, rel), allow_pickle=False) as z:
        bits = z["agri_valid_bits"]
        cossoz = z["cosSOZ"]
        day = z["day_mask"]
        for si, station in enumerate(("sili", "zhujia")):
            row, col = map(int, r[f"{station}_grid_rc"].split(","))
            r0, c0 = row - 8, col - 8
            if r0 < 0 or c0 < 0 or r0 + 16 > 256 or c0 + 16 > 256:
                raise ValueError(f"crop out of bounds station={station} rc={(row,col)}")
            sl = np.s_[r0:r0+16, c0:c0+16]
            agri_ok = np.ones((16, 16), dtype=bool)
            for ch in CHANNELS:
                agri_ok &= (bits[sl] & (1 << ch)) != 0
            logcot = np.asarray(cot[idx, si, 0], dtype=np.float64)
            physical = np.expm1(logcot)
            finite = np.isfinite(physical)
            geo = np.stack([z[k][sl] for k in ("cosSOZ", "cosRAA", "day_mask")]).astype(np.float32)
            geo_finite = np.isfinite(geo).all(axis=0)
            reliable = agri_ok & geo_finite & (geo[2] > 0.5) & (geo[0] > COS80) & finite
            n = int(reliable.sum())
            entry = {
                "split": r["split"], "station": station,
                "month": r["target_time_utc"][:7],
                "n_reliable": n, "fraction_reliable": n / 256.0,
                "n_agri_valid_day": int((agri_ok & (day[sl] > 0.5) & (cossoz[sl] > COS80)).sum()),
                "n_zero": int((reliable & (physical == 0)).sum()),
                "n_gt100": int((reliable & (physical > 100)).sum()),
                "min": float(physical[reliable].min()) if n else None,
                "max": float(physical[reliable].max()) if n else None,
                "n_finite": int(finite.sum()), "n_total": int(finite.size),
                "n_negative": int((finite & (physical < 0)).sum()),
            }
            entries.append(entry)
            geometries.append(np.where(np.isfinite(geo), geo, 0.0))
            masks.append(reliable[None].astype(np.uint8))
            del logcot, physical, finite, reliable
    return entries, np.stack(geometries), np.stack(masks)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--pack-dir", type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads((a.bank / "bank_manifest.json").read_text())
    if manifest.get("state") != "COMPLETE" or manifest.get("test_payload_opened") is not False:
        raise ValueError("COT bank provenance/test gate failed")
    if manifest.get("split_id") != "hunan_monthly_block_2d_buffer1d_seed42_v1":
        raise ValueError("unexpected split")
    cot = np.load(a.bank / "cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    rows = list(csv.DictReader((a.bank / "frame_keys.csv").open(encoding="utf-8")))
    if cot.shape != (len(rows), 2, 1, 16, 16):
        raise ValueError(f"unexpected COT layout: {cot.shape}")
    if a.output.exists() or a.pack_dir.exists():
        raise FileExistsError("refusing to overwrite audit/pack outputs")
    a.pack_dir.mkdir(parents=True)
    geo_store = np.lib.format.open_memmap(
        a.pack_dir / "frame_geometry.npy", mode="w+", dtype=np.float32,
        shape=(len(rows), 2, 3, 16, 16))
    mask_store = np.lib.format.open_memmap(
        a.pack_dir / "frame_valid_mask.npy", mode="w+", dtype=np.uint8,
        shape=(len(rows), 2, 1, 16, 16))
    geo_store[:] = 0
    mask_store[:] = 0
    groups: dict[str, list[dict]] = defaultdict(list)
    global_finite = 0
    global_values = 0
    global_negative = 0
    global_gt100 = 0
    global_zero = 0
    failures = []
    completed = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(audit_frame, idx, r, cot, a.data_root): (idx, r)
                   for idx, r in enumerate(rows)}
        for future in as_completed(futures):
            idx, r = futures[future]
            try:
                entries, geometries, masks = future.result()
                geo_store[idx] = geometries
                mask_store[idx] = masks
                for x in entries:
                    global_finite += x["n_finite"]
                    global_values += x["n_total"]
                    global_negative += x["n_negative"]
                    global_gt100 += x["n_gt100"]
                    global_zero += x["n_zero"]
                    if x["n_finite"] != x["n_total"]:
                        failures.append({"index": idx, "station": x["station"], "kind": "nonfinite_cot"})
                    groups[(x["split"], x["station"])].append(x)
                del geometries, masks
            except Exception as exc:
                failures.append({"index": idx, "split": r.get("split"),
                                 "rel": r.get("agri_frame_relpath"), "error": repr(exc)})
                if len(failures) >= 100:
                    for f in futures:
                        f.cancel()
                    break
            completed += 1
            if completed % 5000 == 0:
                geo_store.flush(); mask_store.flush()
                print(f"PROGRESS {completed}/{len(rows)}", flush=True)

    summary = {}
    for (split, station), vals in sorted(groups.items()):
        fractions = np.asarray([x["fraction_reliable"] for x in vals], dtype=np.float64)
        mins = [x["min"] for x in vals if x["min"] is not None]
        maxs = [x["max"] for x in vals if x["max"] is not None]
        summary[f"{split}/{station}"] = {
            "frames": len(vals),
            "frames_no_daylight_reliable_pixels": int((fractions == 0).sum()),
            "valid_pixel_fraction_p50_p90": [float(x) for x in np.quantile(fractions, [0.5, 0.9])],
            "total_reliable_pixels": int(sum(x["n_reliable"] for x in vals)),
            "valid_cpp_reference_zero_pixels_in_R": int(sum(x["n_zero"] for x in vals)),
            "valid_cot_gt100_pixels": int(sum(x["n_gt100"] for x in vals)),
            "observed_valid_cot_min_max": [float(min(mins)), float(max(maxs))] if mins else None,
            "monthly": {},
        }
        by_month: dict[str, list[dict]] = defaultdict(list)
        for x in vals:
            by_month[x["month"]].append(x)
        for month, entries in sorted(by_month.items()):
            f = np.asarray([x["fraction_reliable"] for x in entries])
            summary[f"{split}/{station}"]["monthly"][month] = {
                "frames": len(entries),
                "valid_pixel_fraction_mean": float(f.mean()),
                "frames_no_daylight_reliable_pixels": int((f == 0).sum()),
                "reliable_pixels": int(sum(x["n_reliable"] for x in entries)),
            }
    result = {
        "state": "PASS_FULL_BANK_FINITE_AND_PIXEL_SUPPORT_AUDIT" if not failures else "FAIL_FULL_BANK_AUDIT",
        "split_id": manifest["split_id"],
        "R_checkpoint_sha256": manifest["R_checkpoint_sha256"],
        "candidate_sha256": manifest["candidate_sha256"],
        "frames_expected": len(rows),
        "frames_processed": len(rows) - sum(1 for x in failures if "rel" in x),
        "test_payload_opened": False,
        "mask_rule": "all 13 AGRI input bits valid AND day_mask>0.5 AND cosSOZ>cos(80deg) AND finite expm1(COT_log1p)",
        "COT_unit": "dimensionless optical depth; bank stores log1p(COT), converted once by expm1",
        "finite_fraction_all_bank_values": global_finite / max(global_values, 1),
        "negative_physical_cot_count": global_negative,
        "reliable_exact_zero_cot_count": global_zero,
        "reliable_cot_gt100_count": global_gt100,
        "groups": summary,
        "failure_count": len(failures),
        "failures": failures,
        "source_sha256": {"bank_manifest.json": sha256(a.bank / "bank_manifest.json"), "frame_keys.csv": sha256(a.bank / "frame_keys.csv")},
        "packed_arrays": {
            "frame_geometry.npy": {"shape": list(geo_store.shape), "dtype": str(geo_store.dtype), "unit_order": ["cosSOZ", "cosRAA", "day_mask"]},
            "frame_valid_mask.npy": {"shape": list(mask_store.shape), "dtype": str(mask_store.dtype), "meaning": "all 13 AGRI bits valid AND finite geometry AND day_mask>0.5 AND cosSOZ>cos80 AND finite COT"},
        },
    }
    geo_store.flush(); mask_store.flush()
    result["packed_arrays"]["frame_geometry.npy"]["sha256"] = sha256(a.pack_dir / "frame_geometry.npy")
    result["packed_arrays"]["frame_valid_mask.npy"]["sha256"] = sha256(a.pack_dir / "frame_valid_mask.npy")
    (a.pack_dir / "pack_manifest.json").write_text(json.dumps({
        "state": "COMPLETE" if not failures else "INCOMPLETE",
        "split_id": manifest["split_id"],
        "bank_frame_keys_sha256": sha256(a.bank / "frame_keys.csv"),
        "test_payload_opened": False,
        "arrays": result["packed_arrays"],
    }, indent=2) + "\n")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "groups"}, indent=2))


if __name__ == "__main__":
    main()
