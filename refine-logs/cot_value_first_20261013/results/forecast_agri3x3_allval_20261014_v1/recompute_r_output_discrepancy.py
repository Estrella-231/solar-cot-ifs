#!/usr/bin/env python
"""Independent NumPy recomputation of saved all-validation R-output deltas.

This measures R(predicted AGRI) - R(observed future AGRI), not physical COT
forecast error: both arrays are outputs of the same frozen retrieval model.
Only the validation NPZ bank is read; no CPP labels or test payloads are opened.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BANK = HERE / "forecast_cot_val_bank.npz"
SAVED = HERE / "allval_R_output_discrepancy_by_season_lead_fp64.csv"
OUT = HERE / "independent_R_output_discrepancy_recompute.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def season_from_month(month: int) -> str:
    return {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}[month]


def main() -> None:
    z = np.load(BANK, allow_pickle=False)
    pred = np.expm1(z["forecast_log1p_cot"].astype(np.float64))
    obs = np.expm1(z["target_agri_r_log1p_cot"].astype(np.float64))
    target_utc = z["target_time_utc"]
    if pred.shape != obs.shape or pred.shape != (3140, 2, 16, 16, 16):
        raise ValueError(f"unexpected bank shapes: {pred.shape}, {obs.shape}")
    if target_utc.shape != (3140, 16):
        raise ValueError(f"unexpected target key shape: {target_utc.shape}")

    crops = {"1x1": (slice(8, 9), slice(8, 9)),
             "3x3": (slice(7, 10), slice(7, 10)),
             "16x16": (slice(None), slice(None))}
    # group -> [count values, count windows, sum squared error, sum abs error, sum error]
    # Recompute both the legacy UTC-month grouping and the scientifically correct BJT grouping.
    acc_utc: dict[tuple[str, str, int, str], list[float]] = defaultdict(lambda: [0, 0, 0., 0., 0.])
    acc_bjt: dict[tuple[str, str, int, str], list[float]] = defaultdict(lambda: [0, 0, 0., 0., 0.])
    for i in range(pred.shape[0]):
        for lead_idx in range(16):
            lead = (lead_idx + 1) * 15
            utc_dt = datetime.fromisoformat(str(target_utc[i, lead_idx]))
            season_utc = season_from_month(utc_dt.month)
            season_bjt = season_from_month((utc_dt + timedelta(hours=8)).month)
            for station_idx, station in enumerate(("sili", "zhujia")):
                for scale, (rs, cs) in crops.items():
                    d = pred[i, station_idx, lead_idx, rs, cs] - obs[i, station_idx, lead_idx, rs, cs]
                    # Bank/profile audit proves these fields are finite before physical inverse.
                    if not np.isfinite(d).all():
                        raise ValueError(f"non-finite physical discrepancy at window={i}, lead={lead}")
                    for acc, season in ((acc_utc, season_utc), (acc_bjt, season_bjt)):
                        a = acc[(station, season, lead, scale)]
                        a[0] += int(d.size)
                        a[1] += 1
                        a[2] += float(np.sum(d * d, dtype=np.float64))
                        a[3] += float(np.abs(d).sum(dtype=np.float64))
                        a[4] += float(d.sum(dtype=np.float64))

    def make_rows(acc):
        rows = []
        for (station, season, lead, scale), a in sorted(acc.items()):
            n, n_windows, sse, sae, se = a
            rows.append({"station": station, "season": season, "lead_min": lead,
                         "scale": scale, "windows": int(n_windows), "n_values": int(n),
                         "RMSE": float(np.sqrt(sse / n)), "MAE": float(sae / n),
                         "bias_forecast_minus_target_R": float(se / n)})
        return rows

    utc_rows = make_rows(acc_utc)
    bjt_rows = make_rows(acc_bjt)

    with SAVED.open(newline="", encoding="utf-8-sig") as f:
        saved = list(csv.DictReader(f))
    saved_by_key = {(r["station"], r["season"], int(r["lead_min"]), r["scale"]): r for r in saved}
    # The existing CSV was generated from UTC month, despite later being described as BJT season.
    recomputed_by_key = {(r["station"], r["season"], r["lead_min"], r["scale"]): r for r in utc_rows}
    common = sorted(saved_by_key.keys() & recomputed_by_key.keys())
    if len(common) != len(saved) or len(common) != len(utc_rows):
        raise ValueError(f"group-key mismatch: saved={len(saved)} recomputed={len(utc_rows)} common={len(common)}")
    fields = ("windows", "n_values", "RMSE", "MAE", "bias_forecast_minus_target_R")
    max_abs = {field: 0.0 for field in fields}
    for key in common:
        for field in fields:
            x, y = float(saved_by_key[key][field]), float(recomputed_by_key[key][field])
            max_abs[field] = max(max_abs[field], abs(x - y))
    mismatch_examples = []
    for key in common:
        diffs = {}
        for field in fields:
            x, y = float(saved_by_key[key][field]), float(recomputed_by_key[key][field])
            diffs[field] = abs(x - y)
        if any(value > 1e-9 for value in diffs.values()):
            mismatch_examples.append({"key": list(key), "saved": saved_by_key[key],
                                      "recomputed": recomputed_by_key[key], "abs_difference": diffs})

    # Correct seasonal summary: use each target_time converted from UTC to BJT.
    macro = {}
    for season in ("DJF", "MAM", "JJA", "SON"):
        season_rows = [r for r in bjt_rows if r["season"] == season and r["scale"] == "3x3"]
        by_station_lead = {(r["station"], r["lead_min"]): r for r in season_rows}
        expected = 2 * 16
        if len(by_station_lead) != expected:
            raise ValueError(f"incomplete station/lead support for {season}: {len(by_station_lead)}")
        macro[season] = {
            "rmse_station_lead_equal": float(np.mean([r["RMSE"] for r in season_rows])),
            "mae_station_lead_equal": float(np.mean([r["MAE"] for r in season_rows])),
            "bias_station_lead_equal": float(np.mean([r["bias_forecast_minus_target_R"] for r in season_rows])),
        }

    result = {
        "state": "PASS_UTC_REPRODUCTION_AND_BJT_SEASON_CORRECTION" if not mismatch_examples else "MISMATCH_REQUIRES_RECONCILIATION",
        "scope": "validation-only; physical COT outputs from same frozen R; discrepancy, not true COT forecast error",
        "test_payloads_read": 0,
        "bank_sha256": sha256(BANK),
        "saved_csv_sha256": sha256(SAVED),
        "array_shape": list(pred.shape),
        "groups_compared": len(common),
        "legacy_saved_csv_season_basis": "UTC month (reproduced exactly)",
        "corrected_season_basis": "target_time converted UTC+8 to BJT",
        "max_abs_difference_vs_saved": max_abs,
        "mismatch_group_count": len(mismatch_examples),
        "first_mismatch_groups": mismatch_examples[:12],
        "three_by_three_bjt_season_equal_station_lead_summary": macro,
        "caveat": "R output response to predicted-vs-observed AGRI; observed-target AGRI is an offline proxy reference, not independent COT truth.",
    }
    with (HERE / "allval_R_output_discrepancy_by_bjt_season_fp64.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(bjt_rows[0]))
        w.writeheader()
        w.writerows(bjt_rows)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
