#!/usr/bin/env python3
"""Freeze error-blind, validation-only T16 cases from contiguous CPP COT sequences.

This selector reads GHI pack keys only (never GHI values or predictions).
It emits a partial manifest when the reference CPP sequence support does not
cover every station-season; unavailable strata are not silently substituted.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


SEASONS = ("DJF", "MAM", "JJA", "SON")
RULE = "T16_EVENT_SELECTION_RULE.md, 2026-10-13; hash v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bjt(utc: datetime) -> datetime:
    return utc.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=8))).replace(tzinfo=None)


def season_for(utc: datetime) -> str:
    return SEASONS[((bjt(utc).month % 12) // 3)]


def key_text(utc: datetime) -> str:
    return utc.strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    base = repo / "refine-logs/cot_value_first_20261013"
    exp = repo / "experiments/monthly_R_station16_20261013_v1"
    cot_path = base / "validation_predictions_monthly_R_seed42.npz"
    clp_path = base / "clp_reference_monthly_R_seed42.npz"
    candidate_path = base / "windows/window_1/target_candidates_train_val.csv"
    rows_path = exp / "results/ghi_head_pack_new_r_v1/rows.csv"
    pack_manifest_path = exp / "results/ghi_head_pack_new_r_v1/pack_manifest.json"
    bank_manifest_path = exp / "results/new_monthly_R_seed42_cot_bank_v1/bank_manifest.json"
    rule_path = repo / "refine-logs/cot_value_first_20261013/windows/window_3/T16_EVENT_SELECTION_RULE.md"

    pack = json.loads(pack_manifest_path.read_text(encoding="utf-8"))
    bank = json.loads(bank_manifest_path.read_text(encoding="utf-8"))
    if pack.get("state") != "COMPLETE" or pack.get("test_payload_opened") is not False:
        raise RuntimeError("GHI pack gate failed")
    if bank.get("state") != "COMPLETE" or bank.get("test_payload_opened") is not False:
        raise RuntimeError("R COT bank gate failed")
    if bank.get("frames_with_source_errors") != 0:
        raise RuntimeError("R COT bank reports source errors")

    with rows_path.open(encoding="utf-8", newline="") as f:
        # Deliberately project only key and split fields; GHI/clear/kt values
        # are not accessed by this error-blind selector.
        reader = csv.DictReader(f)
        required = {"split", "station", "target_time_utc"}
        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError("GHI pack lacks key fields")
        ghi_keys = {(r["station"].lower(), r["target_time_utc"]) for r in reader if r["split"] == "val"}

    with np.load(cot_path, allow_pickle=False) as z:
        ref = np.asarray(z["reference_cot"], dtype=np.float32)[:, 0]
        valid = np.asarray(z["mask"], dtype=bool)[:, 0]
        times_idx = np.asarray(z["sample_indices"], dtype=np.int64)
    with np.load(clp_path, allow_pickle=False) as z:
        stations = z["station"].astype(str)
        times = z["target_time_utc"].astype(str)
        clp_idx = np.asarray(z["sample_indices"], dtype=np.int64)
    if len(ref) != len(valid) or len(ref) != len(times) or len(ref) != len(stations):
        raise RuntimeError("COT/CLP metadata lengths differ")
    if not np.array_equal(times_idx, clp_idx):
        raise RuntimeError("COT and CLP extraction sample-index arrays differ")

    key_to_i = {}
    for i, (st, ts) in enumerate(zip(stations, times)):
        key = (st.lower(), ts)
        if key in key_to_i:
            raise RuntimeError(f"duplicate CPP key: {key}")
        key_to_i[key] = i
        if np.any(valid[i] & ~np.isfinite(ref[i])):
            raise RuntimeError(f"nonfinite reference COT under valid mask: {key}")
    time_to_keys = sorted({datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for _, ts in key_to_i})

    candidates: dict[tuple[str, str], list[dict]] = {(s, q): [] for s in ("sili", "zhujia") for q in SEASONS}
    for station in ("sili", "zhujia"):
        for init in time_to_keys:
            init_key = (station, key_text(init))
            if init_key not in key_to_i:
                continue
            seq_times = [init + timedelta(minutes=15 * k) for k in range(17)]
            seq_keys = [(station, key_text(t)) for t in seq_times]
            if not all(k in key_to_i for k in seq_keys):
                continue
            future = seq_keys[1:]
            if not all(k in ghi_keys for k in future):
                continue
            ids = [key_to_i[k] for k in seq_keys]
            common = np.logical_and.reduce(valid[ids], axis=0)
            if not common.any():
                continue
            cot_seq = ref[ids]
            d_value = float(np.abs(np.diff(cot_seq, axis=0)[:, common]).mean(dtype=np.float64))
            season = season_for(init)
            init_bjt = bjt(init).strftime("%Y-%m-%d %H:%M:%S")
            hash_text = f"cot_value_first_20261013|{station}|{season}|{init_bjt}"
            h = hashlib.sha256(hash_text.encode("utf-8")).hexdigest()
            candidates[(station, season)].append({
                "station": station,
                "season_BJT": season,
                "init_time_utc": key_text(init),
                "init_time_BJT": init_bjt,
                "future_target_times_utc": [key_text(t) for t in seq_times[1:]],
                "sequence_frames_including_init": len(ids),
                "reference_valid_pixels_common_to_all_17_frames": int(common.sum()),
                "D_mean_abs_COT_change": d_value,
                "candidate_hash_sha256": h,
                "sample_indices": [int(times_idx[j]) for j in ids],
            })

    selected = []
    support = {}
    for station in ("sili", "zhujia"):
        for season in SEASONS:
            pool = candidates[(station, season)]
            support[f"{station}/{season}"] = {
                "eligible_contiguous_17_frame_candidates": len(pool),
                "selected": 0,
                "reason_if_empty": "no complete CPP reference sequence with all 16 future validation GHI keys" if not pool else None,
            }
            if not pool:
                continue
            change = sorted(pool, key=lambda x: (-x["D_mean_abs_COT_change"], x["init_time_utc"], x["sample_indices"][0]))[0]
            change = dict(change, case_type="reference_cloud_change", selection_rank="maximum_D")
            selected.append(change)
            remaining = [x for x in pool if x["init_time_utc"] != change["init_time_utc"]]
            if remaining:
                hashed = sorted(remaining, key=lambda x: (x["candidate_hash_sha256"], x["init_time_utc"]))[0]
                hashed = dict(hashed, case_type="deterministic_hash", selection_rank="minimum_SHA256_after_change_case_removed")
                selected.append(hashed)
            support[f"{station}/{season}"]["selected"] = 1 + int(bool(remaining))
            support[f"{station}/{season}"]["reason_if_single_case"] = None if remaining else "only one eligible candidate; hash case unavailable"

    manifest = {
        "state": "PARTIAL_T16_FROZEN_ERROR_BLIND_VALIDATION_CASES",
        "rule": RULE,
        "scope": "Hunan monthly split validation only; no GHI values, predictions, or errors read during selection",
        "selection": {
            "candidate_sequence": "init_time, then +15..+240 min; 17 consecutive reference COT fields",
            "D": "mean over 16 adjacent frame pairs and common valid 16x16 pixels of absolute physical-COT change",
            "hash": "SHA256(cot_value_first_20261013|station|season|init_time_BJT)",
            "hash_tie_break": "ascending hash, then init_time_utc",
            "missing_frames": "excluded, never filled or interpolated",
        },
        "support_by_station_season": support,
        "selected_count": len(selected),
        "selected_cases": selected,
        "provenance": {
            "reference_cot_npz_sha256": sha256(cot_path),
            "clp_key_npz_sha256": sha256(clp_path),
            "GHI_pack_rows_sha256": sha256(rows_path),
            "GHI_pack_manifest_sha256": sha256(pack_manifest_path),
            "R_bank_manifest_sha256": sha256(bank_manifest_path),
            "target_candidate_manifest_sha256": sha256(candidate_path),
            "selection_rule_sha256": sha256(rule_path),
        },
        "test_payload_opened": False,
        "ghi_values_or_predictions_read_for_selection": False,
        "limitation": "Only JJA has contiguous CPP COT reference coverage in this validation archive; absent station-seasons are reported, not replaced. Figure generation still requires a locally materialized, hash-bound AGRI/geometry extraction for the selected validation frames.",
    }
    out = exp / "results/t16_partial_case_manifest_v1.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"state": manifest["state"], "output": str(out), "selected_count": len(selected), "support": support,
                      "selected": [{k: c[k] for k in ("station", "season_BJT", "init_time_BJT", "case_type", "D_mean_abs_COT_change", "reference_valid_pixels_common_to_all_17_frames")} for c in selected],
                      "test_payload_opened": False, "ghi_values_or_predictions_read_for_selection": False}, indent=2))


if __name__ == "__main__":
    main()
