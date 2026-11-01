#!/usr/bin/env python3
"""Build a train/validation-only Hubei COT forecast cache from existing banks.

Only deterministic geometry fields are read from AGRI NPZs. AGRI image channels,
GHI labels, test, and buffer payloads are never opened. CPP remains a retrieval
reference, not independent cloud truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

EXPECTED_MANIFEST_SHA = "8b1f95c2a2cf1d66cabd11539b975fe3dccd486146bf3bec17fb18e537f72183"
EXPECTED_BANK_STATE = "HUBEI_FROZEN_R_FULL_AGRI_TRAINVAL_BANK_COMPLETE_TEST_CLOSED"
EXPECTED_CPP_UTC_OFFSET_HOURS = 8
CADENCE_MINUTES = 15
HISTORY_FRAMES = 8
LEADS_MINUTES = tuple(range(15, 241, 15))
COT_MAX_TRAINING_RANGE = 100.0
COS_SZA_CUTOFF = math.cos(math.radians(80.0))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_bjt(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d%H%M")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def expected_key_split(rows: list[dict[str, str]], time_key: str = "time_bjt") -> dict[datetime, str]:
    result: dict[datetime, str] = {}
    for i, row in enumerate(rows, 2):
        t = parse_bjt(row[time_key].strip())
        split = row["split"].strip().lower()
        if split not in {"train", "val"}:
            raise ValueError(f"row {i}: non-train/val split {split!r}")
        if t in result:
            raise ValueError(f"duplicate key {t}")
        result[t] = split
    return result


def audit(args: argparse.Namespace) -> dict:
    for path in (args.agri_manifest, args.bank_rows, args.bank_array, args.bank_report,
                 args.cpp_rows, args.cpp_npz):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest_hash = sha256(args.agri_manifest)
    bank_array_hash = sha256(args.bank_array)
    bank_rows_hash = sha256(args.bank_rows)
    bank_meta = json.loads(args.bank_report.read_text(encoding="utf-8"))
    if manifest_hash != EXPECTED_MANIFEST_SHA:
        raise ValueError(f"AGRI manifest SHA mismatch: {manifest_hash}")
    if bank_meta.get("state") != EXPECTED_BANK_STATE or bank_meta.get("test_payloads_read") != 0:
        raise ValueError("R-COT bank is not frozen train/val/test-closed")
    if bank_meta.get("manifest_sha256") != manifest_hash:
        raise ValueError("bank report is not bound to the frozen AGRI manifest")
    if bank_meta.get("rows_sha256") != bank_rows_hash or bank_meta.get("prediction_sha256") != bank_array_hash:
        raise ValueError("bank rows or prediction checksum mismatch")
    if bank_meta.get("train_predictions_are_oof") is not False:
        raise ValueError("expected report of in-domain train R predictions")

    agri_rows, bank_rows, cpp_rows = map(read_rows, (args.agri_manifest, args.bank_rows, args.cpp_rows))
    agri_key = expected_key_split(agri_rows)
    bank_key = expected_key_split(bank_rows)
    if agri_key != bank_key:
        raise ValueError("frozen R bank keys/order/splits differ from AGRI manifest")
    if len(agri_rows) != 22061 or Counter(r["split"].lower() for r in agri_rows) != {"train": 20206, "val": 1855}:
        raise ValueError("unexpected frozen Hubei train/validation row counts")

    bjt_index = {parse_bjt(r["time_bjt"]): i for i, r in enumerate(bank_rows)}
    bjt_split = {parse_bjt(r["time_bjt"]): r["split"].strip().lower() for r in bank_rows}
    bank = np.load(args.bank_array, mmap_mode="r", allow_pickle=False)
    if bank.shape != (len(bank_rows), 1, 16, 16) or bank.dtype != np.float32:
        raise ValueError(f"unexpected R-COT bank shape/dtype {bank.shape}/{bank.dtype}")

    cpp_z = np.load(args.cpp_npz, allow_pickle=False)
    required_cpp = {"time_bjt", "split", "cot", "cot_valid_mask",
                    "station_local_centers", "station_names"}
    if not required_cpp.issubset(cpp_z.files):
        raise ValueError(f"CPP sidecar missing required arrays: {sorted(required_cpp - set(cpp_z.files))}")
    cpp_times = [parse_bjt(str(s)) for s in cpp_z["time_bjt"].tolist()]
    cpp_splits = [str(s).strip().lower() for s in cpp_z["split"].tolist()]
    if len(cpp_rows) != len(cpp_times):
        raise ValueError("CPP CSV and NPZ row counts differ")
    for i, (row, t, split) in enumerate(zip(cpp_rows, cpp_times, cpp_splits), 2):
        utc = datetime.strptime(row["time_cpp_utc"], "%Y%m%d%H%M%S")
        if utc != t - timedelta(hours=EXPECTED_CPP_UTC_OFFSET_HOURS):
            raise ValueError(f"CPP CSV BJT/UTC mismatch at row {i}")
        if row["time_bjt"] != t.strftime("%Y%m%d%H%M") or row["split"].strip().lower() != split:
            raise ValueError(f"CPP CSV/NPZ key or split mismatch at row {i}")
        expected_name = f"FY4B_AGRI_{utc:%Y%m%d%H%M%S}.nc"
        if Path(row["path_cpp_correct_utc"]).name != expected_name:
            raise ValueError(f"CPP source basename/UTC mismatch at row {i}")
    if len(cpp_times) != len(set(cpp_times)):
        raise ValueError("duplicate CPP time_bjt keys")
    for t, split in zip(cpp_times, cpp_splits):
        if t not in bjt_index or bjt_split[t] != split:
            raise ValueError(f"CPP key/split not bound to R bank: {t} {split}")
        if t - timedelta(hours=EXPECTED_CPP_UTC_OFFSET_HOURS) < datetime(2025, 1, 1):
            raise ValueError(f"unexpected CPP UTC conversion for {t}")
    cpp_map = {t: i for i, t in enumerate(cpp_times)}
    cpp_cot_src = cpp_z["cot"]
    cpp_mask_src = cpp_z["cot_valid_mask"]
    if cpp_cot_src.shape != (len(cpp_times), 16, 16) or cpp_mask_src.shape != cpp_cot_src.shape:
        raise ValueError("unexpected CPP station-patch target shape")
    station_names = [str(x) for x in cpp_z["station_names"].tolist()]
    station_centers = np.asarray(cpp_z["station_local_centers"], dtype=np.int64)
    if station_names != ["Niushou", "Wangzhai", "Mizhuang"]:
        raise ValueError(f"unexpected station order: {station_names}")
    if station_centers.tolist() != [[8, 6], [9, 8], [7, 10]]:
        raise ValueError(f"unexpected station pixels: {station_centers.tolist()}")

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite cache: {args.output}")
    args.output.mkdir(parents=True, exist_ok=False)
    n = len(bank_rows)
    geom_out = np.lib.format.open_memmap(args.output / "geometry.npy", mode="w+", dtype=np.float32,
                                         shape=(n, 3, 16, 16))
    cot_in_out = np.lib.format.open_memmap(args.output / "history_cot_log.npy", mode="w+", dtype=np.float32,
                                           shape=(n, 1, 16, 16))
    hist_valid_out = np.lib.format.open_memmap(args.output / "history_valid.npy", mode="w+", dtype=np.uint8,
                                               shape=(n, 1, 16, 16))
    target_out = np.lib.format.open_memmap(args.output / "cpp_cot.npy", mode="w+", dtype=np.float32,
                                           shape=(n, 1, 16, 16))
    target_mask_out = np.lib.format.open_memmap(args.output / "cpp_valid.npy", mode="w+", dtype=np.uint8,
                                                shape=(n, 1, 16, 16))
    counts: Counter[str] = Counter()
    target_out[:] = 0.0
    target_mask_out[:] = 0
    for start in range(0, n, args.chunk):
        stop = min(n, start + args.chunk)
        for i in range(start, stop):
            row = bank_rows[i]
            t = parse_bjt(row["time_bjt"])
            agri_path = Path(row["path_agri"])
            if not agri_path.is_file():
                raise FileNotFoundError(f"train/val AGRI geometry source missing: {agri_path}")
            # Read only deterministic metadata channels. Do not access NPZ 'patch'.
            with np.load(agri_path, allow_pickle=False) as z:
                if not {"cosSOZ", "cosRAA", "day_mask"}.issubset(z.files):
                    raise ValueError(f"missing geometry fields at {row['time_bjt']}")
                cosz = np.asarray(z["cosSOZ"], dtype=np.float32)
                cosraa = np.asarray(z["cosRAA"], dtype=np.float32)
                day = np.asarray(z["day_mask"], dtype=np.float32)
            if any(x.shape != (16, 16) for x in (cosz, cosraa, day)):
                raise ValueError(f"unexpected geometry shape at {row['time_bjt']}")
            if not all(np.isfinite(x).all() for x in (cosz, cosraa, day)):
                raise ValueError(f"nonfinite geometry at {row['time_bjt']}")
            if np.any((day < 0) | (day > 1)) or np.any((cosz < -1.001) | (cosz > 1.001)):
                raise ValueError(f"geometry outside expected range at {row['time_bjt']}")
            geom_out[i] = np.stack((cosz, cosraa, day), axis=0)
            pred = np.asarray(bank[i, 0], dtype=np.float32)
            if not np.isfinite(pred).all() or np.any(pred < 0):
                raise ValueError(f"nonfinite/negative R prediction at {row['time_bjt']}")
            # R was trained on COT in [0,100]. Predictions outside that support
            # are marked unavailable, not silently clipped or interpreted as data.
            hist_valid = (day > 0.5) & (cosz > COS_SZA_CUTOFF) & (pred <= COT_MAX_TRAINING_RANGE)
            hist_valid_out[i, 0] = hist_valid.astype(np.uint8)
            safe = np.where(hist_valid, pred, 0.0).astype(np.float32)
            cot_in_out[i, 0] = np.log1p(safe) / np.float32(math.log(101.0))
            counts[f"{row['split'].lower()}_R_values_gt100"] += int((pred > COT_MAX_TRAINING_RANGE).sum())
            counts[f"{row['split'].lower()}_history_valid_pixels"] += int(hist_valid.sum())
            counts[f"{row['split'].lower()}_daytime_pixels"] += int(((day > 0.5) & (cosz > COS_SZA_CUTOFF)).sum())
            if t in cpp_map:
                ci = cpp_map[t]
                ref = np.asarray(cpp_cot_src[ci], dtype=np.float32)
                raw_mask = np.asarray(cpp_mask_src[ci], dtype=bool)
                ref_valid = raw_mask & np.isfinite(ref) & (ref >= 0) & (ref <= COT_MAX_TRAINING_RANGE)
                ref_valid &= (day > 0.5) & (cosz > COS_SZA_CUTOFF)
                target_mask_out[i, 0] = ref_valid.astype(np.uint8)
                target_out[i, 0] = np.where(ref_valid, ref, 0.0).astype(np.float32)
                counts[f"{row['split'].lower()}_cpp_reference_frames"] += 1
                counts[f"{row['split'].lower()}_cpp_supervised_pixels"] += int(ref_valid.sum())
                counts[f"{row['split'].lower()}_cpp_zero_or_invalid_pixels"] += int((~ref_valid).sum())
        if stop % args.progress_every == 0 or stop == n:
            print(f"GEOMETRY_CACHE {stop}/{n}", flush=True)

    for arr in (geom_out, cot_in_out, hist_valid_out, target_out, target_mask_out):
        arr.flush()
    del geom_out, cot_in_out, hist_valid_out, target_out, target_mask_out

    # Pair every valid future CPP frame with all causal, same-split 8-frame inputs.
    pair_path = args.output / "forecast_pairs.csv"
    pair_counts: dict[str, Counter[str]] = {s: Counter() for s in ("train", "val")}
    by_time_valid = np.load(args.output / "cpp_valid.npy", mmap_mode="r", allow_pickle=False)
    with pair_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "init_index", "init_time_bjt", "lead_minutes", "target_index",
                         "target_time_bjt", "target_valid_pixels"])
        for init_t in sorted(bjt_index):
            split = bjt_split[init_t]
            hist_times = [init_t - timedelta(minutes=CADENCE_MINUTES * j) for j in range(HISTORY_FRAMES - 1, -1, -1)]
            if any(t not in bjt_index or bjt_split.get(t) != split for t in hist_times):
                continue
            init_i = bjt_index[init_t]
            for lead in LEADS_MINUTES:
                target_t = init_t + timedelta(minutes=lead)
                ti = cpp_map.get(target_t)
                if ti is None:
                    continue
                target_i = bjt_index[target_t]
                nvalid = int(by_time_valid[target_i, 0].sum())
                if nvalid <= 0:
                    continue
                writer.writerow([split, init_i, init_t.strftime("%Y%m%d%H%M"), lead,
                                 target_i, target_t.strftime("%Y%m%d%H%M"), nvalid])
                pair_counts[split]["pairs"] += 1
                pair_counts[split][f"lead_{lead}"] += 1
                pair_counts[split][f"target_date_{target_t:%Y%m%d}"] += 1

    with (args.output / "bank_rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["index", "time_bjt", "split"])
        writer.writerows([[i, r["time_bjt"], r["split"].lower()] for i, r in enumerate(bank_rows)])
    output_files = ["geometry.npy", "history_cot_log.npy", "history_valid.npy", "cpp_cot.npy",
                    "cpp_valid.npy", "bank_rows.csv", "forecast_pairs.csv"]
    report = {
        "state": "COMPLETE_HUBEI_STATION_PATCH_COT_GEOMETRY_CACHE_TRAINVAL_ONLY",
        "scope": "Shared Hubei 16x16 patch containing Niushou, Wangzhai, Mizhuang; not a 256x256 regional forecast.",
        "split": {"train": 20206, "val": 1855, "test_payloads_read": 0, "buffer_payloads_read": 0},
        "input_sources": {
            "agri_manifest_sha256": manifest_hash, "r_bank_report": str(args.bank_report),
            "r_checkpoint_sha256": bank_meta.get("frozen_R_checkpoint_sha256"),
            "r_prediction_sha256": bank_array_hash, "r_rows_sha256": bank_rows_hash,
            "cpp_rows_sha256": sha256(args.cpp_rows), "cpp_npz_sha256": sha256(args.cpp_npz),
            "r_train_predictions_are_oof": False,
        },
        "transform": {
            "history_input": "log1p(R_AGRI_COT)/log(101); history valid iff day_mask>0.5, cosSOZ>cos(80deg), finite and within R target range [0,100]",
            "out_of_range_R_predictions": "marked invalid, not clipped; count in r_values_gt100",
            "future_geometry": "cosSOZ, cosRAA, day_mask read by exact key from deterministic geometry fields only; AGRI image patch not accessed",
            "target": "CPP COT retrieval reference in physical units, supervised only where CPP valid mask, finite COT in [0,100], day_mask>0.5, and cosSOZ>cos(80deg)",
            "cpp_role": "retrieval reference, not independent cloud truth",
        },
        "stations": {"names": station_names, "local_centers_row_col": station_centers.tolist()},
        "counts": dict(counts),
        "forecast_pairs": {
            s: {"pairs": pair_counts[s]["pairs"], "unique_target_dates": len([k for k in pair_counts[s] if k.startswith("target_date_")]),
                "pairs_by_lead_minutes": {str(l): pair_counts[s][f"lead_{l}"] for l in LEADS_MINUTES}}
            for s in ("train", "val")
        },
        "files": {name: sha256(args.output / name) for name in output_files},
        "test_payloads_read": 0, "buffer_payloads_read": 0,
        "training_authorized_by_this_cache": False,
    }
    (args.output / "cache_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": report["state"], "counts": report["counts"],
                      "forecast_pairs": report["forecast_pairs"], "files": report["files"]}, indent=2), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agri-manifest", type=Path, required=True)
    p.add_argument("--bank-rows", type=Path, required=True)
    p.add_argument("--bank-array", type=Path, required=True)
    p.add_argument("--bank-report", type=Path, required=True)
    p.add_argument("--cpp-rows", type=Path, required=True)
    p.add_argument("--cpp-npz", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--progress-every", type=int, default=2048)
    args = p.parse_args()
    # The builder writes the full report atomically at the end of audit(args).
    # Keep this summary independent of the helper's return value so the completed
    # cache is not marked as failed merely because audit() returns None.
    audit(args)
    report = json.loads((args.output / "cache_report.json").read_text(encoding="utf-8"))
    print(f"CACHE_REPORT={args.output / 'cache_report.json'} state={report['state']}", flush=True)


if __name__ == "__main__":
    main()
