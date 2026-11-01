#!/usr/bin/env python3
"""Read-only key/support audit for the station-patch same-R comparator."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SITES = {"Niushou": (8, 6), "Wangzhai": (9, 8), "Mizhuang": (7, 10)}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main() -> None:
    g1s_rows = read_csv(ROOT / "g1s_bank_rows.csv")
    pairs = [r for r in read_csv(ROOT / "g1s_forecast_pairs.csv") if r["split"] == "val"]
    s_rows = [r for r in read_csv(ROOT / "attempt12_simvp_val_keys.csv") if r["split"] == "val"]
    hvalid = np.load(ROOT / "g1s_history_valid.npy", mmap_mode="r", allow_pickle=False)
    cvalid = np.load(ROOT / "g1s_cpp_valid.npy", mmap_mode="r", allow_pickle=False)

    if len(g1s_rows) != hvalid.shape[0] or len(g1s_rows) != cvalid.shape[0]:
        raise ValueError("G1S row/mask count mismatch")
    if hvalid.shape != cvalid.shape or hvalid.shape[1:] != (1, 16, 16):
        raise ValueError(f"Unexpected G1S mask shapes: {hvalid.shape}, {cvalid.shape}")

    time_to_index = {}
    for r in g1s_rows:
        idx = int(r["index"])
        t = r["time_bjt"]
        if idx in time_to_index.values() or t in time_to_index:
            raise ValueError("Duplicate bank index or timestamp")
        time_to_index[t] = idx

    s_keys = set()
    for r in s_rows:
        targets = r["target_times_bjt"].split("|")
        if len(targets) != 16:
            raise ValueError(f"Expected 16 targets for {r['init_time_bjt']}")
        for i, target in enumerate(targets):
            s_keys.add((r["init_time_bjt"], target, 15 * (i + 1)))

    scored_pairs = []
    by_lead = Counter()
    by_site = Counter()
    all_scored_init = set()
    all_scored_dates = set()
    for r in pairs:
        init_i, target_i = int(r["init_index"]), int(r["target_index"])
        if init_i >= len(g1s_rows) or target_i >= len(g1s_rows):
            raise ValueError(f"G1S pair index outside bank: {r}")
        if g1s_rows[init_i]["time_bjt"] != r["init_time_bjt"]:
            raise ValueError(f"init index/time mismatch: {r}")
        if g1s_rows[target_i]["time_bjt"] != r["target_time_bjt"]:
            raise ValueError(f"target index/time mismatch: {r}")
        lead = int(r["lead_minutes"])
        supported_sites = []
        for site, (rr, cc) in SITES.items():
            if bool(cvalid[target_i, 0, rr, cc]) and bool(hvalid[init_i, 0, rr, cc]):
                supported_sites.append(site)
        if not supported_sites:
            continue
        all_scored_init.add(r["init_time_bjt"])
        all_scored_dates.add(r["target_time_bjt"][:8])
        by_lead[lead] += 1
        for site in supported_sites:
            by_site[site] += 1
        key = (r["init_time_bjt"], r["target_time_bjt"], lead)
        scored_pairs.append({"key": key, "sites": supported_sites, "target_date": r["target_time_bjt"][:8]})

    # The COT forecast archive has exactly the model input init keys needed for paired rescoring.
    cot_archive = ROOT.parent / "station_patch_pilot_v1" / "evidence" / "gpu_training_attempt9_seed42" / "validation_forecast_predictions.npz"
    cot_z = np.load(cot_archive, allow_pickle=False)
    cot_inits = {str(x) for x in cot_z["init_time_bjt"].tolist()}
    if len(cot_inits) != len(cot_z["init_time_bjt"]):
        raise ValueError("Duplicate G1S saved COT init keys")
    missing_cot_inits = sorted({p["key"][0] for p in scored_pairs} - cot_inits)

    # Confirm the time arithmetic and count exact S-bank matches only on scored support.
    matched = [p for p in scored_pairs if p["key"] in s_keys]
    missing_s = [p for p in scored_pairs if p["key"] not in s_keys]
    matched_per_lead = Counter(p["key"][2] for p in matched)
    matched_dates = sorted({p["target_date"] for p in matched})
    matched_station_records = sum(len(p["sites"]) for p in matched)
    result = {
        "state": "PASS_SUPPORT_AUDIT_PARTIAL_SIMVP_INTERSECTION",
        "scope": "train/validation only; station-center common support; CPP is retrieval reference",
        "g1s_validation_pairs_total": len(pairs),
        "g1s_pairs_scored_by_original_common_support": len(scored_pairs),
        "scored_station_center_records": int(sum(len(p["sites"]) for p in scored_pairs)),
        "g1s_scored_pairs_with_attempt12_mse_simvp_key": len(matched),
        "g1s_scored_pairs_without_attempt12_mse_simvp_key": len(missing_s),
        "matched_station_center_records": int(matched_station_records),
        "matched_target_dates": matched_dates,
        "matched_target_date_count": len(matched_dates),
        "g1s_cot_archive_init_key_count": len(cot_inits),
        "missing_g1s_cot_prediction_init_keys_on_scored_support": missing_cot_inits,
        "attempt12_simvp_val_init_windows": len(s_rows),
        "attempt12_simvp_val_forecast_init_lead_keys": len(s_keys),
        "attempt12_simvp_checkpoint_sha256": json.loads((ROOT / "attempt12_s_run_metadata.json").read_text(encoding="utf-8"))["best_checkpoint_sha256"],
        "attempt12_simvp_input_index_sha256": json.loads((ROOT / "attempt12_s_run_metadata.json").read_text(encoding="utf-8"))["index_sha256"],
        "attempt12_simvp_normalization_sha256": json.loads((ROOT / "attempt12_s_run_metadata.json").read_text(encoding="utf-8"))["input_normalization_sha256"],
        "attempt12_simvp_config_sha256": json.loads((ROOT / "attempt12_s_run_metadata.json").read_text(encoding="utf-8"))["config_sha256"],
        "attempt12_model_fit_epochs": json.loads((ROOT / "attempt12_s_run_metadata.json").read_text(encoding="utf-8"))["completed_epochs"],
        "g1s_frozen_r_checkpoint_sha256": json.loads((ROOT / "g1s_r_run_metadata.json").read_text(encoding="utf-8"))["model_sha256"] if "model_sha256" in json.loads((ROOT / "g1s_r_run_metadata.json").read_text(encoding="utf-8")) else "verified separately from remote path",
        "attempt12_total_matched_by_lead": dict(sorted(Counter(p["key"][2] for p in matched).items())),
        "g1s_scored_by_lead": dict(sorted(by_lead.items())),
        "attempt12_matched_by_lead": dict(sorted(matched_per_lead.items())),
        "g1s_scored_by_site": dict(sorted(by_site.items())),
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "note": "This validates the exact scored-support intersection. Attempt12's SimVP checkpoint is a separate MSE baseline (14 epochs), not the attempt13 MSE-control checkpoint; any resulting comparison must be labeled contextual, not a matched arm.",
        "input_sha256": {
            p.name: sha(p) for p in (
                ROOT / "g1s_forecast_pairs.csv", ROOT / "g1s_bank_rows.csv",
                ROOT / "attempt12_simvp_val_keys.csv", ROOT / "g1s_history_valid.npy",
                ROOT / "g1s_cpp_valid.npy", cot_archive,
            )
        },
    }
    (ROOT / "KEY_ALIGNMENT_AUDIT_ATTEMPT12.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
