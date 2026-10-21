#!/usr/bin/env python3
"""Build paired train-only R-source caches for an untouched date-block holdout.

This is an exploratory Hubei 16x16 station-patch experiment. It reads only
original-split=train rows from the existing cache and the train-only five-fold
OOF R bank. The original monthly validation, test, and buffer payloads are not
indexed. CPP is a retrieval reference, not independent physical truth.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

LEADS = tuple(range(15, 241, 15))
HISTORY_MINUTES = tuple(range(-105, 1, 15))
COS80 = math.cos(math.radians(80.0))
LOG101 = math.log(101.0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_time(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y%m%d%H%M")


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def select_outer_dates(train_pair_rows: list[dict[str, str]]) -> list[str]:
    by_month: dict[str, set[str]] = defaultdict(set)
    for row in train_pair_rows:
        by_month[row["target_time_bjt"][:6]].add(row["target_time_bjt"][:8])
    selected: list[str] = []
    for month in sorted(by_month):
        dates = sorted(by_month[month])
        if not dates:
            continue
        if len(dates) == 1:
            selected.append(dates[0])
            continue
        i = (len(dates) - 1) // 3
        j = (2 * (len(dates) - 1)) // 3
        if j == i:
            j = min(len(dates) - 1, i + 1)
        selected.extend((dates[i], dates[j]))
    selected = sorted(set(selected))
    if len(selected) < 12:
        raise ValueError(f"date-stratified outer holdout too small: {len(selected)}")
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-cache", type=Path, required=True)
    ap.add_argument("--oof-bank", type=Path, required=True)
    ap.add_argument("--oof-rows", type=Path, required=True)
    ap.add_argument("--oof-manifest", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    report_path = args.base_cache / "cache_report.json"
    rows_path = args.base_cache / "bank_rows.csv"
    pairs_path = args.base_cache / "forecast_pairs.csv"
    required = [report_path, rows_path, pairs_path, args.oof_bank, args.oof_rows, args.oof_manifest]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    base_report = json.loads(report_path.read_text(encoding="utf-8"))
    if base_report.get("test_payloads_read") != 0 or base_report.get("buffer_payloads_read") != 0:
        raise ValueError("base cache does not certify test/buffer closed")
    base_rows = read_csv(rows_path)
    base_pairs = read_csv(pairs_path)
    train_rows = [r for r in base_rows if r["split"].strip().lower() == "train"]
    if len(train_rows) != 20206:
        raise ValueError(f"unexpected train frame count: {len(train_rows)}")
    train_times = [r["time_bjt"] for r in train_rows]
    if len(train_times) != len(set(train_times)):
        raise ValueError("duplicate original train timestamps")
    if any(r["split"].strip().lower() != "train" for r in train_rows):
        raise AssertionError("train-row filter invariant failed")
    train_pairs = [r for r in base_pairs if r["split"].strip().lower() == "train"]
    outer_dates = select_outer_dates(train_pairs)
    outer_set = set(outer_dates)
    all_train_dates = sorted({t[:8] for t in train_times})
    outer_dt = [dt.datetime.strptime(x, "%Y%m%d").date() for x in outer_dates]
    purged_dates = sorted(
        d for d in all_train_dates
        if d not in outer_set and any(
            abs((dt.datetime.strptime(d, "%Y%m%d").date() - h).days) <= 1 for h in outer_dt
        )
    )
    purge_set = set(purged_dates)
    assignment = {
        d: ("val" if d in outer_set else ("purge" if d in purge_set else "train"))
        for d in all_train_dates
    }
    train_date_count = sum(v == "train" for v in assignment.values())
    if train_date_count < 60:
        raise ValueError(f"too few remaining train dates after purge: {train_date_count}")

    manifest = json.loads(args.oof_manifest.read_text(encoding="utf-8"))
    oof_npz_sha = sha256(args.oof_bank)
    oof_rows_sha = sha256(args.oof_rows)
    if manifest.get("state") != "COMPLETE_TRAIN_ONLY_OOF_R_COT_BANK":
        raise ValueError("OOF bank manifest state mismatch")
    if manifest.get("outputs", {}).get(args.oof_bank.name) != oof_npz_sha:
        raise ValueError("OOF bank hash differs from its manifest")
    if manifest.get("outputs", {}).get(args.oof_rows.name) != oof_rows_sha:
        raise ValueError("OOF row manifest hash differs from its manifest")

    oof_rows = read_csv(args.oof_rows)
    oof_z = np.load(args.oof_bank, allow_pickle=False)
    oof_cot = oof_z["cot"]
    oof_day = oof_z["day_mask"]
    oof_times = [str(x) for x in oof_z["time_bjt"].tolist()]
    if oof_cot.shape != (20206, 1, 16, 16) or oof_day.shape != oof_cot.shape:
        raise ValueError(f"unexpected OOF bank shapes {oof_cot.shape}, {oof_day.shape}")
    if len(oof_rows) != len(train_rows) or len(oof_times) != len(train_rows):
        raise ValueError("OOF and original train row counts differ")
    for i, (br, orow, ot) in enumerate(zip(train_rows, oof_rows, oof_times)):
        if br["time_bjt"] != orow["time_bjt"] or br["time_bjt"] != ot:
            raise ValueError(f"OOF train key/order mismatch at {i}")
        if orow.get("split", "train").strip().lower() != "train":
            raise ValueError(f"non-train key in OOF rows at {i}")

    old_by_time = {r["time_bjt"]: int(r["index"]) for r in train_rows}
    new_split_by_time = {
        t: assignment[t[:8]] for t in train_times if assignment[t[:8]] in {"train", "val"}
    }
    selected_times = [t for t in train_times if t in new_split_by_time]
    selected_times.sort()
    if not selected_times:
        raise ValueError("no selected cache rows")
    new_by_time = {t: i for i, t in enumerate(selected_times)}
    old_index_by_time = {t: old_by_time[t] for t in selected_times}
    oof_index_by_time = {t: i for i, t in enumerate(oof_times)}
    split_by_time = new_split_by_time

    # Keep only init windows whose entire available causal history and
    # available future frame sequence stays within one date partition.
    valid_inits: set[str] = set()
    for row in train_pairs:
        init_s = row["init_time_bjt"]
        if init_s not in new_by_time:
            continue
        split = split_by_time[init_s]
        init_t = parse_time(init_s)
        hist = [(init_t + dt.timedelta(minutes=m)).strftime("%Y%m%d%H%M")
                for m in HISTORY_MINUTES]
        if any(t not in new_by_time or split_by_time[t] != split for t in hist):
            continue
        cross = False
        for lead in LEADS:
            t = (init_t + dt.timedelta(minutes=lead)).strftime("%Y%m%d%H%M")
            if t in new_by_time and split_by_time[t] != split:
                cross = True
                break
        if not cross:
            valid_inits.add(init_s)
    kept_pairs = [
        r for r in train_pairs
        if r["init_time_bjt"] in valid_inits
        and r["target_time_bjt"] in new_by_time
        and split_by_time[r["init_time_bjt"]] == split_by_time[r["target_time_bjt"]]
    ]
    if not kept_pairs:
        raise ValueError("no forecast pairs survive date-block partition")
    counts = {s: Counter() for s in ("train", "val")}
    dates_by_lead = {s: {l: set() for l in LEADS} for s in ("train", "val")}
    for r in kept_pairs:
        s, lead = split_by_time[r["init_time_bjt"]], int(r["lead_minutes"])
        counts[s][lead] += 1
        dates_by_lead[s][lead].add(r["target_time_bjt"][:8])
    for lead in LEADS:
        if counts["val"][lead] < 200 or len(dates_by_lead["val"][lead]) < 12:
            raise ValueError(
                f"outer holdout support gate failed lead={lead}: "
                f"pairs={counts['val'][lead]}, dates={len(dates_by_lead['val'][lead])}"
            )
        if counts["train"][lead] < 500:
            raise ValueError(f"fit support too small at lead={lead}: {counts['train'][lead]}")

    args.output_root.mkdir(parents=True, exist_ok=False)
    old_idx = np.asarray([old_index_by_time[t] for t in selected_times], dtype=np.int64)
    oof_idx = np.asarray([oof_index_by_time[t] for t in selected_times], dtype=np.int64)
    n = len(selected_times)
    old_geom = np.load(args.base_cache / "geometry.npy", mmap_mode="r", allow_pickle=False)
    old_full_cot = np.load(args.base_cache / "history_cot_log.npy", mmap_mode="r", allow_pickle=False)
    old_full_valid = np.load(args.base_cache / "history_valid.npy", mmap_mode="r", allow_pickle=False)
    old_cpp = np.load(args.base_cache / "cpp_cot.npy", mmap_mode="r", allow_pickle=False)
    old_cpp_valid = np.load(args.base_cache / "cpp_valid.npy", mmap_mode="r", allow_pickle=False)
    for name, arr, shape in (
        ("geometry", old_geom, (len(base_rows), 3, 16, 16)),
        ("history_cot_log", old_full_cot, (len(base_rows), 1, 16, 16)),
        ("history_valid", old_full_valid, (len(base_rows), 1, 16, 16)),
        ("cpp_cot", old_cpp, (len(base_rows), 1, 16, 16)),
        ("cpp_valid", old_cpp_valid, (len(base_rows), 1, 16, 16)),
    ):
        if arr.shape != shape:
            raise ValueError(f"unexpected source array {name}: {arr.shape}")

    rows_new = [[i, t, split_by_time[t], old_index_by_time[t]]
                for i, t in enumerate(selected_times)]
    pair_new: list[list[object]] = []
    for r in kept_pairs:
        pair_new.append([
            split_by_time[r["init_time_bjt"]],
            new_by_time[r["init_time_bjt"]],
            r["init_time_bjt"],
            int(r["lead_minutes"]),
            new_by_time[r["target_time_bjt"]],
            r["target_time_bjt"],
            int(r["target_valid_pixels"]),
        ])
    write_csv(args.output_root / "bank_rows.csv",
              ["index", "time_bjt", "split", "source_train_index"], rows_new)
    write_csv(args.output_root / "forecast_pairs.csv",
              ["split", "init_index", "init_time_bjt", "lead_minutes", "target_index",
               "target_time_bjt", "target_valid_pixels"], pair_new)

    fold_counts = Counter(int(x) for x in oof_z["heldout_fold"].tolist())
    arm_reports: dict[str, dict] = {}
    for arm in ("fullr_control", "oof_train_source"):
        cache = args.output_root / arm / "cache"
        cache.mkdir(parents=True, exist_ok=False)
        arrays = {
            "geometry": np.lib.format.open_memmap(cache / "geometry.npy", mode="w+",
                                                   dtype=np.float32, shape=(n, 3, 16, 16)),
            "history_cot_log": np.lib.format.open_memmap(cache / "history_cot_log.npy", mode="w+",
                                                          dtype=np.float32, shape=(n, 1, 16, 16)),
            "history_valid": np.lib.format.open_memmap(cache / "history_valid.npy", mode="w+",
                                                        dtype=np.uint8, shape=(n, 1, 16, 16)),
            "cpp_cot": np.lib.format.open_memmap(cache / "cpp_cot.npy", mode="w+",
                                                  dtype=np.float32, shape=(n, 1, 16, 16)),
            "cpp_valid": np.lib.format.open_memmap(cache / "cpp_valid.npy", mode="w+",
                                                    dtype=np.uint8, shape=(n, 1, 16, 16)),
        }
        oof_gt100 = oof_valid_pixels = 0
        for ni, (t, oi, fi) in enumerate(zip(selected_times, old_idx, oof_idx)):
            geom = np.asarray(old_geom[oi], dtype=np.float32)
            full_cot = np.asarray(old_full_cot[oi], dtype=np.float32)
            full_valid = np.asarray(old_full_valid[oi], dtype=np.uint8)
            cpp = np.asarray(old_cpp[oi], dtype=np.float32)
            cpp_valid = np.asarray(old_cpp_valid[oi], dtype=np.uint8)
            arrays["geometry"][ni] = geom
            arrays["cpp_cot"][ni] = cpp
            arrays["cpp_valid"][ni] = cpp_valid
            if arm == "fullr_control" or split_by_time[t] == "val":
                arrays["history_cot_log"][ni] = full_cot
                arrays["history_valid"][ni] = full_valid
            else:
                raw = np.asarray(oof_cot[fi, 0], dtype=np.float32)
                oof_day2 = np.asarray(oof_day[fi, 0], dtype=bool)
                cosz = geom[0]
                day = geom[2]
                if not np.array_equal(oof_day2, day > 0.5):
                    raise ValueError(f"OOF day mask differs from geometry day mask at {t}")
                if not np.isfinite(raw).all() or np.any(raw < 0):
                    raise ValueError(f"invalid raw OOF map at {t}")
                valid = oof_day2 & (day > 0.5) & (cosz > COS80) & (raw <= 100.0)
                safe = np.where(valid, raw, 0.0).astype(np.float32)
                arrays["history_valid"][ni, 0] = valid.astype(np.uint8)
                arrays["history_cot_log"][ni, 0] = np.log1p(safe) / np.float32(LOG101)
                oof_gt100 += int(np.count_nonzero(raw > 100.0))
                oof_valid_pixels += int(np.count_nonzero(valid))
        for arr in arrays.values():
            arr.flush()
        del arrays

        files = ["geometry.npy", "history_cot_log.npy", "history_valid.npy",
                 "cpp_cot.npy", "cpp_valid.npy", "bank_rows.csv", "forecast_pairs.csv"]
        for name in ("bank_rows.csv", "forecast_pairs.csv"):
            source = args.output_root / name
            target = cache / name
            target.write_bytes(source.read_bytes())
        arm_report = {
            "state": "COMPLETE_HUBEI_TRAIN_ONLY_OUTER_DATE_HOLDOUT_CACHE",
            "scope": "Hubei 16x16 station-patch forecast; internal train-date holdout only.",
            "arm": arm,
            "history_source": (
                "final full-R history for all train/outer-holdout rows"
                if arm == "fullr_control"
                else "five-fold OOF R-COT for fit/inner-selection train rows; final full-R for outer holdout rows"
            ),
            "target_source": "unchanged CPP retrieval reference from original train rows; not independent cloud truth",
            "test_payloads_read": 0,
            "buffer_payloads_read": 0,
            "partition": {
                "original_train_frames_seen_as_metadata": len(train_rows),
                "original_monthly_validation_payload_rows_indexed": 0,
                "selected_rows": n,
                "fit_frames": sum(v == "train" for v in split_by_time.values()),
                "outer_holdout_frames": sum(v == "val" for v in split_by_time.values()),
                "purged_original_train_dates": purged_dates,
                "outer_holdout_dates": outer_dates,
                "test_payloads_read": 0,
                "buffer_payloads_read": 0,
                "original_monthly_validation_payloads_read": 0,
            },
            "input_provenance": {
                "base_cache_report_sha256": sha256(report_path),
                "base_bank_rows_sha256": sha256(rows_path),
                "base_forecast_pairs_sha256": sha256(pairs_path),
                "oof_bank_sha256": oof_npz_sha,
                "oof_rows_sha256": oof_rows_sha,
                "oof_manifest_sha256": sha256(args.oof_manifest),
                "oof_fold_frame_counts": dict(fold_counts),
            },
            "transform": {
                "OOF": "log1p(COT)/log(101); valid iff OOF day mask, geometry day_mask>0.5, cosSOZ>cos(80deg), finite, 0<=COT<=100; invalid pixels zeroed, not clipped",
                "control": "existing frozen full-R station-patch cache feature, unchanged",
                "evaluation": "all outer-holdout history features use final full-R source in both arms",
            },
            "counts": {
                "OOF_train_values_gt100": oof_gt100 if arm == "oof_train_source" else None,
                "OOF_train_valid_pixels": oof_valid_pixels if arm == "oof_train_source" else None,
                "pairs_by_lead_minutes": {
                    s: {str(l): counts[s][l] for l in LEADS} for s in ("train", "val")
                },
                "outer_holdout_dates_by_lead": {
                    str(l): len(dates_by_lead["val"][l]) for l in LEADS
                },
            },
            "files": {},
            "new_model_fit_permitted_by_cache": False,
        }
        for name in files:
            arm_report["files"][name] = sha256(cache / name)
        (cache / "cache_report.json").write_text(
            json.dumps(arm_report, indent=2) + "\n", encoding="utf-8"
        )
        arm_reports[arm] = arm_report

    manifest_out = {
        "state": "COMPLETE_OOF_SOURCE_PAIRED_CACHE_BUILD",
        "outer_holdout_source": "original train dates only; monthly validation excluded",
        "outer_holdout_dates": outer_dates,
        "purged_dates": purged_dates,
        "train_dates_remaining": train_date_count,
        "selected_row_count": n,
        "pairs_by_lead": {
            s: {str(l): counts[s][l] for l in LEADS} for s in ("train", "val")
        },
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "official_monthly_validation_payloads_read": 0,
        "arms": {
            arm: {"cache_report_sha256": sha256(args.output_root / arm / "cache" / "cache_report.json"),
                  "files": arm_reports[arm]["files"]}
            for arm in arm_reports
        },
    }
    (args.output_root / "BUILD_MANIFEST.json").write_text(
        json.dumps(manifest_out, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest_out, indent=2), flush=True)


if __name__ == "__main__":
    main()
