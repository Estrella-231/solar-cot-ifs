#!/usr/bin/env python3
"""Same-key comparison of forecast-COT and observed-future-AGRI->R oracle COT.

Only validation pack arrays are read. Values in cot_log1p.npy are unstandardized
log1p(COT); the GHI head applies the train-only normalization at load time.
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


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_keys(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def metrics(p_log: np.ndarray, r_log: np.ndarray) -> dict[str, float]:
    # Input arrays contain per-sample 3x3 COT neighborhoods in log1p space.
    p_c = np.expm1(np.clip(p_log, 0.0, 20.0))
    r_c = np.expm1(np.clip(r_log, 0.0, 20.0))
    pc, rc = p_log[:, 1, 1], r_log[:, 1, 1]
    pcc, rcc = p_c[:, 1, 1], r_c[:, 1, 1]
    sp = np.std(p_log.reshape(len(p_log), -1), axis=1)
    sr = np.std(r_log.reshape(len(r_log), -1), axis=1)
    p_anom = p_log - p_log.mean(axis=(1, 2), keepdims=True)
    r_anom = r_log - r_log.mean(axis=(1, 2), keepdims=True)
    pa, ra = p_anom.ravel(), r_anom.ravel()
    return {
        "n": int(len(p_log)),
        "center_rmse_log1p": float(np.sqrt(np.mean((pc - rc) ** 2))),
        "patch_rmse_log1p": float(np.sqrt(np.mean((p_log - r_log) ** 2))),
        "center_bias_log1p": float(np.mean(pc - rc)),
        "center_corr": float(np.corrcoef(pc, rc)[0, 1]) if len(pc) > 1 and np.std(pc) > 0 and np.std(rc) > 0 else float("nan"),
        "center_rmse_cot": float(np.sqrt(np.mean((pcc - rcc) ** 2))),
        "patch_rmse_cot": float(np.sqrt(np.mean((p_c - r_c) ** 2))),
        "center_mean_forecast_cot": float(np.mean(pcc)),
        "center_mean_oracle_cot": float(np.mean(rcc)),
        "center_std_ratio": float(np.std(pcc) / max(np.std(rcc), 1e-8)),
        "patch_spatial_contrast_ratio": float(np.mean(sp) / max(np.mean(sr), 1e-8)),
        "spatial_anomaly_corr": float(np.corrcoef(pa, ra)[0, 1]) if len(pa) > 1 and np.std(pa) > 0 and np.std(ra) > 0 else float("nan"),
        "spatial_anomaly_rmse_log1p": float(np.sqrt(np.mean((pa - ra) ** 2))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast-pack", type=Path, required=True)
    ap.add_argument("--oracle-pack", type=Path, required=True)
    ap.add_argument("--forecast-ghi-audit", type=Path, required=True)
    ap.add_argument("--oracle-ghi-audit", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    for p in (a.forecast_pack, a.oracle_pack):
        m = json.loads((p / "pack_manifest.json").read_text(encoding="utf-8"))
        if m.get("state") not in {"COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK", "COMPLETE_ORACLE_FUTURE_COT_GHI_PACK"}:
            raise ValueError(f"incomplete or unexpected pack: {p}")
        if m.get("test_payload_opened") or m.get("buffer_payload_opened"):
            raise ValueError(f"test/buffer payload flag is open: {p}")
    fkeys_path, okeys_path = a.forecast_pack / "val/keys.csv", a.oracle_pack / "val/keys.csv"
    if sha(fkeys_path) != sha(okeys_path):
        raise ValueError("validation keys differ byte-for-byte")
    keys = read_keys(fkeys_path)
    forecast = np.load(a.forecast_pack / "val/cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    oracle = np.load(a.oracle_pack / "val/cot_log1p.npy", mmap_mode="r", allow_pickle=False)
    expected = (len(keys), 3, 1, 3, 3)
    if forecast.shape != expected or oracle.shape != expected:
        raise ValueError(f"unexpected validation COT shapes: {forecast.shape} / {oracle.shape}; expected {expected}")
    if not (np.isfinite(forecast).all() and np.isfinite(oracle).all()):
        raise ValueError("non-finite COT value")

    # Slot 2 is the COT field at target_time and therefore exactly aligns with
    # the target 15-minute GHI interval end. Historical slots are excluded.
    f = np.asarray(forecast[:, 2, 0], dtype=np.float64)
    o = np.asarray(oracle[:, 2, 0], dtype=np.float64)
    leads = sorted({int(k["lead_minutes"]) for k in keys})
    stations = sorted({k["station"].lower() for k in keys})
    rows: list[dict[str, object]] = []
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    strata: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        groups[(k["station"].lower(), int(k["lead_minutes"]))].append(i)
        c = float(np.expm1(np.clip(o[i, 1, 1], 0.0, 20.0)))
        label = "near_zero_<1" if c < 1 else ("low_1_to_10" if c < 10 else "thick_ge_10")
        strata[(k["station"].lower(), int(k["lead_minutes"]), label)].append(i)

    for (st, lead), idx in sorted(groups.items()):
        ii = np.asarray(idx, dtype=np.int64)
        rows.append({"group": "station_lead", "station": st, "lead_minutes": lead,
                     "cot_stratum": "all", **metrics(f[ii], o[ii])})
    for (st, lead, label), idx in sorted(strata.items()):
        ii = np.asarray(idx, dtype=np.int64)
        rows.append({"group": "station_lead_thickness", "station": st, "lead_minutes": lead,
                     "cot_stratum": label, **metrics(f[ii], o[ii])})
    all_ids = np.arange(len(keys))
    rows.append({"group": "all", "station": "both_equal_row_weight", "lead_minutes": "all",
                 "cot_stratum": "all", **metrics(f[all_ids], o[all_ids])})

    # Combine the already paired GHI audits at station/lead resolution.
    fg = json.loads(a.forecast_ghi_audit.read_text(encoding="utf-8"))
    og = json.loads(a.oracle_ghi_audit.read_text(encoding="utf-8"))
    forecast_leads = {(r["station"].lower(), int(r["lead_minutes"])): r for r in fg["lead_metrics"]}
    oracle_leads = {(r["station"].lower(), int(r["lead_minutes"])): r for r in og["per_station_lead_metrics"]}
    ghi_rows = []
    for key in sorted(set(forecast_leads) & set(oracle_leads)):
        fr, orow = forecast_leads[key], oracle_leads[key]
        ghi_rows.append({"station": key[0], "lead_minutes": key[1],
                         "forecast_cot_AC_rmse_wm2": fr["AC_rmse_wm2"],
                         "forecast_cot_AC_minus_AH_wm2": fr["delta_AC_minus_AH_wm2"],
                         "oracle_cot_AC_rmse_wm2": orow["AC_rmse_wm2"],
                         "oracle_cot_AC_minus_AH_wm2": orow["AC_minus_AH_wm2"],
                         "descriptive_AC_rmse_gap_oracle_minus_forecast_wm2": orow["AC_rmse_wm2"] - fr["AC_rmse_wm2"]})

    a.output.mkdir(parents=True, exist_ok=True)
    metric_fields = list(rows[0].keys())
    with (a.output / "cot_forecast_vs_oracle_by_lead_and_thickness.csv").open("w", newline="", encoding="utf-8-sig") as out:
        w = csv.DictWriter(out, fieldnames=metric_fields); w.writeheader(); w.writerows(rows)
    ghi_fields = list(ghi_rows[0].keys())
    with (a.output / "ghi_forecast_vs_oracle_by_lead.csv").open("w", newline="", encoding="utf-8-sig") as out:
        w = csv.DictWriter(out, fieldnames=ghi_fields); w.writeheader(); w.writerows(ghi_rows)
    report = {
        "state": "COMPLETE_SAME_KEY_FORECAST_VS_ORACLE_COT_DIAGNOSTIC",
        "scope": "validation only; both branches use the same five-fold OOF forecast AGRI, station, GHI labels and keys; oracle COT is frozen R applied to observed future AGRI",
        "pairing": {"forecast_keys_sha256": sha(fkeys_path), "oracle_keys_sha256": sha(okeys_path),
                    "keys_byte_identical": True, "rows": len(keys), "stations": stations, "lead_minutes": leads,
                    "target_bjt_dates": len({(datetime.strptime(k['target_time_utc'], '%Y-%m-%d %H:%M:%S') + timedelta(hours=8)).date().isoformat() for k in keys})},
        "cot_measurement": "forecast and oracle cot_log1p arrays are compared at slot 2 (target time) in the central station pixel and full 3x3 patch; COT is recovered by expm1 for physical-unit metrics",
        "thickness_bins": "oracle central-pixel COT: <1, 1-<10, >=10; descriptive strata, CPP is not used",
        "forecast_pack_sha256": sha(a.forecast_pack / "pack_manifest.json"),
        "oracle_pack_sha256": sha(a.oracle_pack / "pack_manifest.json"),
        "forecast_ghi_audit_sha256": sha(a.forecast_ghi_audit), "oracle_ghi_audit_sha256": sha(a.oracle_ghi_audit),
        "overall_cot_metrics": rows[-1],
        "ghi_ac_rmse_gap_mean_wm2": float(np.mean([r["descriptive_AC_rmse_gap_oracle_minus_forecast_wm2"] for r in ghi_rows])),
        "claim_boundary": "diagnostic comparison, not independent-test evidence or deployable oracle forecast",
        "test_payload_opened": False, "buffer_payload_opened": False,
    }
    (a.output / "same_key_diagnostic.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"state": report["state"], "rows": len(keys), "overall_cot": report["overall_cot_metrics"],
                      "ghi_ac_rmse_gap_mean_wm2": report["ghi_ac_rmse_gap_mean_wm2"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
