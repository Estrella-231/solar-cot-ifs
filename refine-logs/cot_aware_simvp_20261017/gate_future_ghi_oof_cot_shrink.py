#!/usr/bin/env python3
"""Fail-closed gate for the OOF AC50 follow-up; no payload arrays are read."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(pack_path: Path, run_path: Path, audit_path: Path,
             run_output: Path, audit_output: Path):
    pack, run, audit = read(pack_path), read(run_path), read(audit_path)
    if pack.get("state") != "COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK":
        raise ValueError("OOF head pack is incomplete")
    if pack.get("train_forecast_oof") is not True or pack.get("forecast_bank_oof_fold_count") != 5:
        raise ValueError("pack does not certify five-fold OOF training forecasts")
    if pack.get("test_payload_opened") is not False or pack.get("buffer_payload_opened") is not False:
        raise ValueError("test/buffer exclusion is not certified")
    if pack.get("future_observed_agri_used_as_input") is not False or pack.get("cpp_used_as_input") is not False:
        raise ValueError("future observed AGRI/CPP exclusion is not certified")
    if run.get("state") != "COMPLETE_FUTURE_GHI_COT_SOURCE_SCREENING_PILOT":
        raise ValueError("OOF A/AH/AC run is incomplete")
    if run.get("train_forecast_oof") is not True:
        raise ValueError("baseline training inputs are not certified OOF")
    expected_pack_sha = hashlib.sha256(pack_path.read_bytes()).hexdigest()
    if run.get("pack_manifest_sha256") != expected_pack_sha:
        raise ValueError("baseline run does not use this exact OOF pack")
    if audit.get("state") != "COMPLETE_PAIRED_FUTURE_GHI_COT_SOURCE_VALIDATION_AUDIT":
        raise ValueError("paired OOF A/AH/AC audit is incomplete")
    if audit.get("train_forecast_oof") is not True or audit.get("test_payload_opened") is not False:
        raise ValueError("audit does not certify OOF input and test exclusion")
    if not audit.get("run_manifest_sha256"):
        raise ValueError("audit does not bind to the completed baseline run")
    if run_output.exists() or audit_output.exists():
        raise FileExistsError("OOF shrinkage output path already exists")

    delta = float(audit["pairwise_delta_rmse_wm2"]["AC_minus_AH"])
    ci = audit["date_block_bootstrap_95pct_ci_wm2"]["AC_minus_AH"]
    if len(ci) != 2:
        raise ValueError("invalid AC-AH bootstrap interval")
    per_station = audit["per_station_mean_seed_rmse"]
    station_pass = all(float(per_station[s]["AC"]) - float(per_station[s]["AH"]) <= 0
                       for s in ("sili", "zhujia"))
    early_deltas = []
    for station in ("sili", "zhujia"):
        rows = [r for r in audit["lead_metrics"]
                if r["station"] == station and int(r["lead_minutes"]) <= 60]
        if not rows:
            raise ValueError(f"missing +15 to +60 minute rows for {station}")
        early_deltas.append(sum(float(r["AC_rmse_wm2"]) - float(r["AH_rmse_wm2"])
                                for r in rows) / len(rows))
    early_delta = sum(early_deltas) / len(early_deltas)
    primary_pass = delta < 0 and float(ci[1]) < 0 and station_pass and early_delta <= 0
    if primary_pass:
        raise ValueError("AC-AH OOF gate passed; proceed to independent confirmation, do not run AC50")
    return {"AC_minus_AH_wm2": delta, "ci95_wm2": ci,
            "station_gate_pass": station_pass,
            "early_lead_equal_station_delta_wm2": early_delta,
            "ac50_followup_allowed": True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pack-manifest", type=Path, required=True)
    p.add_argument("--run-manifest", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--run-output", type=Path, required=True)
    p.add_argument("--audit-output", type=Path, required=True)
    a = p.parse_args()
    print("OOF_AC50_FOLLOWUP_GATE_PASS", json.dumps(evaluate(
        a.pack_manifest, a.run_manifest, a.audit, a.run_output, a.audit_output),
        sort_keys=True))


if __name__ == "__main__":
    main()
