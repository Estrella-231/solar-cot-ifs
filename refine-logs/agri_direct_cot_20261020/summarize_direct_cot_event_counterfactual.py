from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TMP = Path(r"C:\Users\29728\AppData\Local\Temp\direct_cot_rowbank_recovery_20261021_0530")
EVENT = ROOT / "results" / "direct_cot_ghi_validation_score_v1_local4060" / "event_diagnostics"
PACK = TMP / "results" / "direct_cot_single_frame_ghi_pack_v1"
IN_CSV = EVENT / "selected_event_rows_seed42.csv"
OUT_CSV = EVENT / "selected_event_rows_seed42_counterfactual.csv"
OUT_JSON = EVENT / "same_head_zero_cot_event_summary.json"

inrows = list(csv.DictReader(IN_CSV.open(newline="", encoding="utf-8-sig")))
by_station = {}
for station in ("sili", "zhujia"):
    d = PACK / station / "val"
    lines = (d / "keys.jsonl").read_text(encoding="utf-8").splitlines()
    keys = [json.loads(x) for x in lines]
    ix = {(x["init_time_utc"], x["target_time_utc"], int(x["lead_minutes"])): i for i, x in enumerate(keys)}
    cf = np.load(EVENT / f"{station}_seed42_same_head_zero_cot.npz", allow_pickle=False)
    by_station[station] = (ix, cf)

outrows = []
for r in inrows:
    station = r["station"]
    ix, cf = by_station[station]
    k = (r["init_time_utc"], r["target_time_utc"], int(r["lead_minutes"]))
    i = ix[k]
    r2 = dict(r)
    r2["same_p_xc_head_neutral_cot_ghi_wm2"] = float(cf["prediction_same_head_cot_zero"][i])
    r2["same_p_xc_head_cot_effect_wm2"] = float(cf["prediction_with_pxc_cot"][i] - cf["prediction_same_head_cot_zero"][i])
    outrows.append(r2)

with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(outrows[0]))
    w.writeheader()
    w.writerows(outrows)

summaries = {}
for r in outrows:
    summaries.setdefault(r["station"], {}).setdefault(r["case_type"], []).append(r)
result = {
    "state": "COMPLETE_DIRECT_COT_EVENT_COUNTERFACTUAL_SUMMARY",
    "scope": "validation-only; frozen P-XC seed42 head; same AGRI and geometry; predicted COT vs neutral normalized-zero COT",
    "selection_is_posthoc": True,
    "not_causal_proof": True,
    "cpp_used_as_input": False,
    "future_observed_agri_used_as_input": False,
    "test_payload_opened": False,
    "buffer_payload_opened": False,
    "cases": {},
    "enriched_csv_sha256": hashlib.sha256(OUT_CSV.read_bytes()).hexdigest(),
}
for station, cases in summaries.items():
    result["cases"][station] = {}
    for case, rr in cases.items():
        obs = np.array([float(x["ghi_obs_wm2"]) for x in rr])
        withcot = np.array([float(x["ghi_p_xc_seed42_wm2"]) for x in rr])
        neutral = np.array([float(x["same_p_xc_head_neutral_cot_ghi_wm2"]) for x in rr])
        result["cases"][station][case] = {
            "n": len(rr),
            "lead_range_min": [min(int(x["lead_minutes"]) for x in rr), max(int(x["lead_minutes"]) for x in rr)],
            "mean_prediction_shift_with_cot_minus_neutral_wm2": float(np.mean(withcot - neutral)),
            "rmse_with_predicted_cot_wm2": float(np.sqrt(np.mean((withcot - obs) ** 2))),
            "rmse_same_head_neutral_cot_wm2": float(np.sqrt(np.mean((neutral - obs) ** 2))),
            "delta_rmse_wm2": float(np.sqrt(np.mean((withcot - obs) ** 2)) - np.sqrt(np.mean((neutral - obs) ** 2))),
        }
OUT_JSON.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, allow_nan=False))
