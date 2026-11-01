#!/usr/bin/env python3
"""Freeze T16 validation cases from CPP COT only, before reading GHI values.

The candidate score is the mean absolute adjacent-frame COT change over a
common valid pixel set in a 17-frame (0..240 min) station-centered 16x16 crop.
The second case per station/season is selected by the preregistered SHA-256
rule. Missing seasons are recorded; they are never filled from another season.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

EXPECTED_GHI_LABEL_SHA256 = "a86c2425ac4e6e8259fcaa4b5b2e083ee427d16e12d4eb1b44aa87751b75988d"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def season(t: datetime) -> str:
    return "DJF" if t.month in (12, 1, 2) else "MAM" if t.month in (3, 4, 5) else "JJA" if t.month in (6, 7, 8) else "SON"


def iso_bjt(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S+08:00")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True, help="audited station16 CPP train/val pack")
    ap.add_argument("--labels", type=Path, required=True, help="train/validation GHI labels; only keys and main_mask are read")
    ap.add_argument("--rule", type=Path, required=True, help="frozen T16_EVENT_SELECTION_RULE.md")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    audit_path, rows_path = a.pack / "pack_audit.json", a.pack / "rows.csv"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("state") != "PASS_MONTHLY_R_STATION16_PACK_AUDIT" or audit.get("test_payloads_read") != 0:
        raise ValueError("CPP train/val pack audit or test gate failed")
    if audit.get("split_id") != "hunan_monthly_block_2d_buffer1d_seed42_v1":
        raise ValueError("unexpected split")
    rows = list(csv.DictReader(rows_path.open(newline="", encoding="utf-8")))
    cot = np.load(a.pack / "cot.npy", mmap_mode="r", allow_pickle=False)
    mask = np.load(a.pack / "mask.npy", mmap_mode="r", allow_pickle=False)
    packed_rows = [r for r in rows if r["state"] == "SUPERVISED"]
    expected_packed = int(audit["arrays"]["cot"]["shape"][0])
    array_indices = [int(r["array_index"]) for r in packed_rows]
    if (cot.shape != (expected_packed, 1, 16, 16) or mask.shape != cot.shape
            or len(array_indices) != expected_packed
            or sorted(array_indices) != list(range(expected_packed))):
        raise ValueError("CPP COT/mask array shape does not match audited rows")

    # Eligibility checks key coverage and the already-frozen validity mask only;
    # GHI/kt numeric values are deliberately never read during case selection.
    if sha256(a.labels) != EXPECTED_GHI_LABEL_SHA256:
        raise ValueError("GHI label table hash differs from the frozen training pack")
    ghi_keys = set()
    with a.labels.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["new_split"] != "val":
                continue
            if r["main_mask"].lower() == "true":
                ghi_keys.add((r["station"], r["target_time_utc"]))

    keyed: dict[tuple[str, datetime], int] = {}
    for r in rows:
        if r["split"] != "val" or r["state"] != "SUPERVISED":
            continue
        t = datetime.strptime(r["bjt"], "%Y%m%d%H%M")
        key = (r["station"], t)
        if key in keyed:
            raise ValueError(f"duplicate supervised station/time key: {key}")
        keyed[key] = int(r["array_index"])

    by_station_season: dict[str, dict[str, list[dict]]] = {
        s: {q: [] for q in ("DJF", "MAM", "JJA", "SON")} for s in ("sili", "zhujia")
    }
    for (station, t), _ in keyed.items():
        times = [t + timedelta(minutes=15 * k) for k in range(17)]
        if any((station, u) not in keyed for u in times):
            continue
        utc_keys = [(station, (u - timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")) for u in times]
        if any(k not in ghi_keys for k in utc_keys):
            continue
        indices = [keyed[(station, u)] for u in times]
        common = np.asarray(mask[indices, 0], dtype=bool).all(axis=0)
        n_common = int(common.sum())
        if n_common == 0:
            continue
        values = np.asarray(cot[indices, 0], dtype=np.float64)
        d = float(np.abs(np.diff(values, axis=0)[:, common]).mean(dtype=np.float64))
        sea = season(t)
        t_iso = iso_bjt(t)
        digest = hashlib.sha256(
            f"cot_value_first_20261013|{station}|{sea}|{t_iso}".encode("utf-8")
        ).hexdigest()
        by_station_season[station][sea].append({
            "station": station,
            "season": sea,
            "init_time_bjt": t_iso,
            "end_time_bjt": iso_bjt(times[-1]),
            "cadence_minutes": 15,
            "frame_count": 17,
            "D_mean_abs_adjacent_cot": d,
            "common_valid_pixels_16x16": n_common,
            "timestamps_bjt": [iso_bjt(u) for u in times],
            "deterministic_hash": digest,
            "pack_array_indices": indices,
        })

    chosen = []
    strata = {}
    for station in ("sili", "zhujia"):
        strata[station] = {}
        for sea in ("DJF", "MAM", "JJA", "SON"):
            candidates = by_station_season[station][sea]
            if not candidates:
                strata[station][sea] = {"eligible_candidates": 0, "state": "NO_COMPLETE_17_FRAME_CPP_SEQUENCE"}
                continue
            # Python's stable sort implements the registered init-time tie-break.
            change = sorted(candidates, key=lambda x: (-x["D_mean_abs_adjacent_cot"], x["init_time_bjt"]))[0]
            remainder = [x for x in candidates if x["init_time_bjt"] != change["init_time_bjt"]]
            hashed = min(remainder, key=lambda x: (x["deterministic_hash"], x["init_time_bjt"])) if remainder else None
            strata[station][sea] = {
                "eligible_candidates": len(candidates),
                "state": "SELECTED_TWO_CASES" if hashed else "ONLY_ONE_ELIGIBLE_CASE",
                "change_event": change,
                "hash_event": hashed,
            }
            chosen.append({"selection_type": "maximum_reference_COT_change", **change})
            if hashed:
                chosen.append({"selection_type": "deterministic_hash", **hashed})

    result = {
        "state": "PARTIAL_PRESELECTED_CPP_VALIDATION_CASES",
        "split_id": audit["split_id"],
        "rule_file": a.rule.name,
        "rule_sha256": sha256(a.rule),
        "selector_script_sha256": sha256(Path(__file__)),
        "source": "CPP physical COT and CPP quality/input mask; validation only; station-centered 16x16 crop",
        "selection_metric": "mean over 16 adjacent 15-minute differences of mean absolute physical-COT change on pixels valid across all 17 frames",
        "hash_rule": "SHA256(cot_value_first_20261013|station|season|init_time_bjt), init_time_bjt canonical ISO-8601 with +08:00",
        "pack_sources": {
            "pack_audit_sha256": sha256(audit_path),
            "rows_sha256": sha256(rows_path),
            "cot_array_sha256": sha256(a.pack / "cot.npy"),
            "mask_array_sha256": sha256(a.pack / "mask.npy"),
            "ghi_label_table_sha256": sha256(a.labels),
        },
        "coverage": strata,
        "selected_case_count": len(chosen),
        "requested_case_count": 16,
        "limitation": "No complete CPP-supported 17-frame validation sequences were found in DJF/MAM/SON; those strata remain missing and are not replaced across seasons.",
        "selected_cases": chosen,
        "eligibility_note": "GHI label table was read only for validation key existence and main_mask; numeric GHI/kt fields were not accessed.",
        "ghi_values_accessed": False,
        "ghi_prediction_errors_accessed": False,
        "test_payloads_read": 0,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": result["state"], "selected_case_count": len(chosen), "coverage": {
        s: {q: v["eligible_candidates"] for q, v in strata[s].items()} for s in strata
    }, "ghi_values_accessed": False, "test_payloads_read": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
