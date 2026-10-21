#!/usr/bin/env python3
"""Train-only station-center COT transition and persistence diagnostic.

This reads only rows marked train from the bounded Hubei station-patch cache.
CPP is a retrieval reference, not independent physical truth. No validation,
test, buffer, AGRI, or GHI values are used as scoring inputs.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


LOG_DENOM = math.log(101.0)
LEADS = tuple(range(15, 241, 15))
CENTERS = {"Niushou": (8, 6), "Wangzhai": (9, 8), "Mizhuang": (7, 10)}
HISTORY_BIN_NAMES = ("<1", "1_to_10", "10_to_30", ">=30")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_time(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y%m%d%H%M")


def hist_bin(x: float) -> str:
    if x < 1:
        return "<1"
    if x < 10:
        return "1_to_10"
    if x < 30:
        return "10_to_30"
    return ">=30"


def stats(records: list[tuple[float, float]]) -> dict:
    if not records:
        return {"n_pixels": 0}
    h = np.asarray([x[0] for x in records], dtype=np.float64)
    y = np.asarray([x[1] for x in records], dtype=np.float64)
    e = h - y
    return {
        "n_pixels": int(len(y)),
        "mean_history_cot": float(h.mean()),
        "mean_target_cpp_cot": float(y.mean()),
        "mean_change_target_minus_history": float((y - h).mean()),
        "history_persistence_bias": float(e.mean()),
        "history_persistence_mae": float(np.abs(e).mean()),
        "history_persistence_rmse": float(np.sqrt(np.mean(e * e))),
        "target_thick_ge30_fraction": float(np.mean(y >= 30)),
    }


def summarize(records: list[dict], selector, site_equal: bool = False) -> dict:
    picked = [r for r in records if selector(r)]
    pixel_stats = stats([(r["history"], r["target"]) for r in picked])
    pixel_stats["n_init_lead_site_records"] = len(picked)
    if site_equal and picked:
        by_site = defaultdict(list)
        for r in picked:
            by_site[r["site"]].append((r["history"], r["target"]))
        ss = [stats(v) for v in by_site.values()]
        pixel_stats["equal_site_persistence_rmse"] = float(
            np.sqrt(np.mean([x["history_persistence_rmse"] ** 2 for x in ss]))
        )
        pixel_stats["equal_site_persistence_mae"] = float(
            np.mean([x["history_persistence_mae"] for x in ss])
        )
    return pixel_stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cache = a.cache

    # Only metadata is read for all bank rows; model/CPP values are indexed
    # exclusively with train timestamps and train forecast-pair indices.
    bank_rows = read_csv(cache / "bank_rows.csv")
    train_rows = [r for r in bank_rows if r["split"].lower() == "train"]
    by_time = {parse_time(r["time_bjt"]): int(r["index"]) for r in train_rows}
    if len(by_time) != len(train_rows):
        raise ValueError("duplicate train timestamps")
    train_indices = set(by_time.values())

    all_pair_rows = read_csv(cache / "forecast_pairs.csv")
    pair_rows = [r for r in all_pair_rows if r["split"].lower() == "train"]
    arrays = {
        name: np.load(cache / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in ("history_cot_log", "history_valid", "cpp_cot", "cpp_valid")
    }
    n_rows = len(train_rows)
    for name, arr in arrays.items():
        if arr.ndim != 4 or arr.shape[1:] != (1, 16, 16):
            raise ValueError(f"unexpected {name} shape {arr.shape}")
        if max(train_indices, default=-1) >= arr.shape[0]:
            raise ValueError(f"train row index outside {name} array")

    records: list[dict] = []
    missing_pairs = 0
    for row in pair_rows:
        init_t = parse_time(row["init_time_bjt"])
        target_t = parse_time(row["target_time_bjt"])
        lead = int(row["lead_minutes"])
        if lead not in LEADS or target_t - init_t != dt.timedelta(minutes=lead):
            raise ValueError(f"bad train pair: {row}")
        ii = int(row["init_index"])
        yi = int(row["target_index"])
        if by_time.get(init_t) != ii or by_time.get(target_t) != yi:
            raise ValueError(f"train key/index mismatch: {row}")
        if ii not in train_indices or yi not in train_indices:
            raise ValueError("cross-split train pair")

        hmap = np.expm1(np.asarray(arrays["history_cot_log"][ii, 0], dtype=np.float64) * LOG_DENOM)
        hvalid = np.asarray(arrays["history_valid"][ii, 0], dtype=bool)
        ymap = np.asarray(arrays["cpp_cot"][yi, 0], dtype=np.float64)
        yvalid = np.asarray(arrays["cpp_valid"][yi, 0], dtype=bool)
        common = hvalid & yvalid & np.isfinite(hmap) & np.isfinite(ymap)
        if not common.any():
            missing_pairs += 1
            continue

        for site, (r, c) in CENTERS.items():
            if common[r, c]:
                h, y = float(hmap[r, c]), float(ymap[r, c])
                records.append({"site": site, "lead": lead, "date": target_t.strftime("%Y%m%d"),
                                "history": h, "target": y, "history_bin": hist_bin(h),
                                "event": ("thick_arrival_or_growth" if y >= 30 and h < 30 else
                                          "thick_retained" if y >= 30 and h >= 30 else
                                          "thick_decay" if y < 30 and h >= 30 else "non_thick")})

    if not records:
        raise ValueError("no common train station-center support")

    site_lead = {}
    for site in CENTERS:
        site_lead[site] = {}
        for lead in LEADS:
            site_lead[site][str(lead)] = summarize(
                records, lambda x, s=site, l=lead: x["site"] == s and x["lead"] == l
            )

    history_bins = {}
    for b in HISTORY_BIN_NAMES:
        history_bins[b] = summarize(records, lambda x, k=b: x["history_bin"] == k, site_equal=True)

    lead_groups = {
        "15_60": lambda x: 15 <= x["lead"] <= 60,
        "75_120": lambda x: 75 <= x["lead"] <= 120,
        "135_180": lambda x: 135 <= x["lead"] <= 180,
        "195_240": lambda x: 195 <= x["lead"] <= 240,
    }
    lead_summary = {name: summarize(records, pred, site_equal=True)
                    for name, pred in lead_groups.items()}

    event_counts = defaultdict(int)
    event_by_lead = defaultdict(lambda: defaultdict(int))
    for x in records:
        event_counts[x["event"]] += 1
        event_by_lead[str(x["lead"])][x["event"]] += 1
    event_total = sum(event_counts.values())

    result = {
        "state": "TRAIN_ONLY_CPP_REFERENCE_COT_TRANSITION_AUDIT",
        "scope": "Hubei 16x16 station-patch center pixels; train split only",
        "target_interpretation": "CPP-retrieved COT reference, not independent physical truth",
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "validation_model_or_label_values_read": 0,
        "train_bank_rows": len(train_rows),
        "train_pair_rows": len(pair_rows),
        "train_pair_rows_with_no_common_station_center_support": missing_pairs,
        "station_center_records": len(records),
        "target_dates": sorted({x["date"] for x in records}),
        "station_centers_row_col": {k: list(v) for k, v in CENTERS.items()},
        "equal_site_persistence": summarize(records, lambda x: True, site_equal=True),
        "lead_groups_equal_site": lead_summary,
        "site_by_lead": site_lead,
        "persistence_conditioned_on_history_cot": history_bins,
        "transition_counts": {k: {"n": v, "fraction": v / event_total}
                              for k, v in sorted(event_counts.items())},
        "transition_counts_by_lead": {k: dict(v) for k, v in sorted(event_by_lead.items(), key=lambda z: int(z[0]))},
        "interpretation_limit": "Overlapping train init/lead samples are dependent; CPP is a retrieval reference. This audit describes target transitions and persistence difficulty only; it does not score a learned forecast or establish independent COT truth.",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("state", "train_bank_rows", "train_pair_rows",
                                             "station_center_records", "equal_site_persistence",
                                             "lead_groups_equal_site", "transition_counts")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
