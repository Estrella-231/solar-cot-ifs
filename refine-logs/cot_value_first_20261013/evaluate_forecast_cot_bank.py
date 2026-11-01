#!/usr/bin/env python3
"""Independent FP64 COT scoring for the frozen validation S->R bank."""
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


METHODS = {
    "S_to_R": "forecast_log1p_cot",
    "persistence_R": "persistence_log1p_cot",
    "real_target_AGRI_to_R_offline": "target_agri_r_log1p_cot",
}
SCALE_SLICES = {
    "station_pixel": (slice(8, 9), slice(8, 9)),
    "local_3x3": (slice(7, 10), slice(7, 10)),
    "patch_16x16": (slice(0, 16), slice(0, 16)),
}
STATIONS = ("sili", "zhujia")
LEADS = tuple(range(15, 241, 15))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ratio(num, den):
    return float(num / den) if den else None


def metric_row(method, station, lead, scale, truth, pred):
    e = pred - truth
    sst = np.square(truth - truth.mean()).sum(dtype=np.float64)
    sse = np.square(e).sum(dtype=np.float64)
    tp = int(((pred >= 30.0) & (truth >= 30.0)).sum())
    fp = int(((pred >= 30.0) & (truth < 30.0)).sum())
    fn = int(((pred < 30.0) & (truth >= 30.0)).sum())
    return {
        "method": method, "station": station, "lead_min": lead, "scale": scale,
        "n_pixels": int(truth.size), "rmse": float(np.sqrt(sse / truth.size)),
        "mae": float(np.abs(e).mean(dtype=np.float64)),
        "bias": float(e.mean(dtype=np.float64)),
        "r2_sse_sst": float(1.0 - sse / sst) if sst > 0 else None,
        "thick_cloud_threshold": 30.0, "tp": tp, "fp": fp, "fn": fn,
        "pod": ratio(tp, tp + fn), "far": ratio(fp, tp + fp),
        "csi": ratio(tp, tp + fp + fn),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(f"refusing to overwrite {a.output}")

    prof = json.loads(a.profile.read_text())
    if prof.get("state") != "COMPLETE_VALIDATION_S_TO_R_COT_BANK" or prof.get("test_payloads_read") != 0:
        raise RuntimeError("bank receipt is incomplete or reports test access")
    with np.load(a.bank, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    pred_log = {m: arrays[k].astype(np.float64) for m, k in METHODS.items()}
    ref = arrays["cpp_reference_cot"].astype(np.float64)
    mask = arrays["cpp_valid_mask"].astype(bool)
    keys = arrays["target_time_utc"].astype(str)
    seq_ids = arrays["sequence_ids"].astype(str)
    nwin = len(seq_ids)
    if any(x.shape != (nwin, 2, 16, 16, 16) for x in pred_log.values()):
        raise RuntimeError("prediction arrays do not match [window,station,lead,16,16]")
    if ref.shape != mask.shape or ref.shape != (nwin, 2, 16, 16, 16):
        raise RuntimeError("CPP reference/mask shapes do not match prediction bank")
    if keys.shape != (nwin, 16) or len(set(arrays["dataset_indices"].tolist())) != nwin:
        raise RuntimeError("lead keys or sequence indices are malformed/duplicated")
    if not mask.any() or not np.isfinite(ref[mask]).all():
        raise RuntimeError("CPP reference support is empty or non-finite")
    if np.isfinite(ref[~mask]).any():
        raise RuntimeError("missing CPP reference values must remain NaN outside the mask")
    for m, x in pred_log.items():
        if not np.isfinite(x).all():
            raise RuntimeError(f"non-finite log1p prediction in {m}")
    # Undo the stored log1p transform exactly once, in FP64; do not clip negative values.
    pred_cot = {m: np.expm1(x) for m, x in pred_log.items()}
    if any(not np.isfinite(x).all() for x in pred_cot.values()):
        raise RuntimeError("FP64 inverse log1p generated non-finite physical COT")

    day_keys = np.empty((nwin, 16), dtype="U10")
    for i in range(nwin):
        for j in range(16):
            target_utc = datetime.strptime(keys[i, j], "%Y-%m-%d %H:%M:%S")
            day_keys[i, j] = (target_utc + timedelta(hours=8)).date().isoformat()
            if j and datetime.strptime(keys[i, j], "%Y-%m-%d %H:%M:%S") - datetime.strptime(
                    keys[i, j - 1], "%Y-%m-%d %H:%M:%S") != timedelta(minutes=15):
                raise RuntimeError(f"noncanonical lead spacing in {seq_ids[i]}")
    unique_days = sorted(set(day_keys.ravel().tolist()))

    rows = []
    # date-block sufficient statistics for paired RMSE bootstrap
    day_sse = defaultdict(float)
    day_n = defaultdict(int)
    for scale, (rs, cs) in SCALE_SLICES.items():
        for si, station in enumerate(STATIONS):
            for lj, lead in enumerate(LEADS):
                support = mask[:, si, lj, rs, cs]
                support_pairs = int(support.reshape(nwin, -1).any(axis=1).sum())
                truth_all = ref[:, si, lj, rs, cs][support]
                for method, cot in pred_cot.items():
                    forecast = cot[:, si, lj, rs, cs][support]
                    row = metric_row(method, station, lead, scale, truth_all, forecast) if truth_all.size else {
                        "method": method, "station": station, "lead_min": lead, "scale": scale,
                        "n_pixels": 0, "rmse": None, "mae": None, "bias": None, "r2_sse_sst": None,
                        "thick_cloud_threshold": 30.0, "tp": 0, "fp": 0, "fn": 0,
                        "pod": None, "far": None, "csi": None}
                    row["n_window_station_lead"] = support_pairs
                    row["n_target_dates"] = len({day_keys[i, lj] for i in range(nwin)
                                                  if support[i].any()})
                    rows.append(row)
                    # Cluster all pixel errors by BJT target date.
                    for i in range(nwin):
                        m = support[i]
                        if not m.any():
                            continue
                        d = day_keys[i, lj]
                        e = cot[i, si, lj, rs, cs][m] - ref[i, si, lj, rs, cs][m]
                        k = (scale, method, station, lead, d)
                        day_sse[k] += float(np.square(e).sum(dtype=np.float64))
                        day_n[k] += int(e.size)

    by_key = {(r["scale"], r["method"], r["station"], r["lead_min"]): r for r in rows}
    macro_rows = []
    for scale in SCALE_SLICES:
        for method in METHODS:
            lead_scores = []
            for lead in LEADS:
                sr = [by_key[(scale, method, s, lead)] for s in STATIONS]
                if all(r["rmse"] is not None for r in sr):
                    lead_scores.append({
                        "lead_min": lead, "rmse_station_macro": float(np.mean([r["rmse"] for r in sr])),
                        "mae_station_macro": float(np.mean([r["mae"] for r in sr])),
                        "bias_station_macro": float(np.mean([r["bias"] for r in sr])),
                        "n_pixels_by_station": [r["n_pixels"] for r in sr],
                        "n_pairs_by_station": [r["n_window_station_lead"] for r in sr],
                    })
            baseline = [x["rmse_station_macro"] for x in lead_scores] if len(lead_scores) == 16 else []
            per_lead_fs = []
            for lead in LEADS:
                model_rmse = by_key[(scale, method, STATIONS[0], lead)]["rmse"]
                base = [by_key[(scale, "persistence_R", s, lead)]["rmse"] for s in STATIONS]
                if model_rmse is not None and all(v is not None for v in base):
                    # Macro-FS uses station-equal RMSE for this lead.
                    macro_model = float(np.mean([by_key[(scale, method, s, lead)]["rmse"] for s in STATIONS]))
                    macro_base = float(np.mean(base))
                    per_lead_fs.append(1.0 - macro_model / macro_base if macro_base > 0 else None)
            macro_rows.append({
                "method": method, "scale": scale, "n_leads_supported": len(lead_scores),
                "mean_lead_station_macro_rmse": float(np.mean(baseline)) if baseline else None,
                "mean_lead_station_macro_mae": float(np.mean([x["mae_station_macro"] for x in lead_scores]))
                    if len(lead_scores) == 16 else None,
                "mean_lead_station_macro_bias": float(np.mean([x["bias_station_macro"] for x in lead_scores]))
                    if len(lead_scores) == 16 else None,
                "mean_lead_macro_forecast_skill_vs_persistence": float(np.mean(per_lead_fs))
                    if len(per_lead_fs) == 16 and all(v is not None for v in per_lead_fs) else None,
                "per_lead": lead_scores,
            })

    # Paired BJT target-date block bootstrap. With <20 dates, intervals are
    # explicitly descriptive and cannot support a stable/significance claim.
    rng = np.random.default_rng(20261014)
    bootstrap = []
    for scale in SCALE_SLICES:
        base_deltas = {m: [] for m in ("S_to_R", "real_target_AGRI_to_R_offline")}
        accepted = {m: 0 for m in base_deltas}
        max_attempts = max(a.bootstrap * 50, a.bootstrap)
        attempts = 0
        while min(accepted.values(), default=0) < a.bootstrap and attempts < max_attempts:
            attempts += 1
            draw = rng.choice(unique_days, size=len(unique_days), replace=True).tolist()
            def overall(method):
                lead_macro = []
                for lead in LEADS:
                    site_rmse = []
                    for station in STATIONS:
                        ss, nn = 0.0, 0
                        for day in draw:
                            k = (scale, method, station, lead, day)
                            ss += day_sse.get(k, 0.0)
                            nn += day_n.get(k, 0)
                        if nn == 0:
                            return None
                        site_rmse.append(float(np.sqrt(ss / nn)))
                    lead_macro.append(float(np.mean(site_rmse)))
                return float(np.mean(lead_macro))
            base = overall("persistence_R")
            if base is None:
                continue
            for method in base_deltas:
                val = overall(method)
                if val is not None:
                    base_deltas[method].append(val - base)
                    accepted[method] += 1
        for method, deltas in base_deltas.items():
            ci = np.percentile(deltas, [2.5, 97.5]).tolist() if deltas else [None, None]
            bootstrap.append({
                "method_minus_persistence_mean_lead_macro_rmse": method,
                "scale": scale, "point_delta_rmse": next((
                    x["mean_lead_station_macro_rmse"] - next(y["mean_lead_station_macro_rmse"]
                    for y in macro_rows if y["method"] == "persistence_R" and y["scale"] == scale)
                    for x in macro_rows if x["method"] == method and x["scale"] == scale
                    and x["mean_lead_station_macro_rmse"] is not None), None),
                "target_dates": len(unique_days), "bootstrap_draws_requested": a.bootstrap,
                "bootstrap_draws_valid": len(deltas), "attempts": attempts,
                "ci95_percentile_descriptive": ci,
                "inference_claim_allowed": len(unique_days) >= 20,
                "status": "DESCRIPTIVE_ONLY_LT20_TARGET_DATES" if len(unique_days) < 20 else "BOOTSTRAP_COMPLETE",
            })

    thick_summary = []
    for scale in SCALE_SLICES:
        for method in METHODS:
            subset = [r for r in rows if r["scale"] == scale and r["method"] == method]
            tp = sum(r["tp"] for r in subset)
            fp = sum(r["fp"] for r in subset)
            fn = sum(r["fn"] for r in subset)
            thick_summary.append({
                "method": method, "scale": scale, "threshold_cot": 30.0,
                "n_common_pixels": sum(r["n_pixels"] for r in subset),
                "tp": tp, "fp": fp, "fn": fn,
                "pod": ratio(tp, tp + fn), "far": ratio(fp, tp + fp),
                "csi": ratio(tp, tp + fp + fn),
            })

    a.output.mkdir(parents=True)
    with (a.output / "cot_forecast_by_lead.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    result = {
        "state": "COMPLETE_VALIDATION_FP64_COT_FORECAST_SCORING",
        "split": "val_only", "test_payloads_read": 0,
        "bank_sha256": sha(a.bank), "profile_sha256": sha(a.profile),
        "n_windows": nwin, "n_unique_target_dates_bjt": len(unique_days),
        "target_dates_bjt": unique_days, "n_cpp_supported_pixels": int(mask.sum()),
        "n_complete_windows_both_stations_all_leads": int(
            mask.reshape(nwin, 2, 16, -1).any(axis=-1).all(axis=(1, 2)).sum()),
        "log1p_inverse": "exactly one FP64 expm1; predictions not clipped",
        "primary_scale": "local_3x3", "aggregation": "per-station/per-lead pooled pixels, stations equally averaged, leads equally averaged",
        "methods": list(METHODS), "threshold_cot_for_thick_cloud": 30.0,
        "macro_summary": macro_rows, "thick_cloud_summary": thick_summary,
        "paired_date_block_bootstrap": bootstrap,
        "caveat": "CPP is retrieval reference, not independent ground-truth COT. Only 8 target dates are supported; bootstrap intervals are descriptive, not evidence of stable generalization or significance.",
    }
    (a.output / "cot_forecast_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (a.output / "cot_thick_cloud_by_lead.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["method", "station", "lead_min", "scale", "n_pixels", "tp", "fp", "fn", "pod", "far", "csi"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(json.dumps({"state": result["state"], "n_windows": nwin,
                      "n_dates": len(unique_days), "n_pixels": int(mask.sum()),
                      "metrics": [{k: r[k] for k in ("method", "scale", "mean_lead_station_macro_rmse",
                                                       "mean_lead_station_macro_mae", "mean_lead_station_macro_bias",
                                                       "mean_lead_macro_forecast_skill_vs_persistence")}
                                  for r in macro_rows]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
