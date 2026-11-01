#!/usr/bin/env python3
"""Separate position/timing and thickness errors on frozen CPP events.

CPP is a retrieval reference. All scores are descriptive validation results
on four preselected JJA initializations; no test payload is read.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


def parse_utc(value: str) -> datetime:
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def metrics_one(ref: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict:
    # Inputs have [lead,row,col] axes. The caller supplies a fixed shared mask.
    if ref.shape != pred.shape or ref.shape != mask.shape:
        raise ValueError("reference, prediction and mask shapes differ")
    if not np.isfinite(ref[mask]).all() or not np.isfinite(pred[mask]).all():
        raise ValueError("non-finite value on common support")
    err = pred.astype(np.float64) - ref.astype(np.float64)
    n = int(mask.sum())
    out = {
        "n_valid_pixel_leads": n,
        "rmse": float(np.sqrt(np.sum(np.where(mask, err * err, 0.0), dtype=np.float64) / n)),
        "mae": float(np.sum(np.where(mask, np.abs(err), 0.0), dtype=np.float64) / n),
        "bias": float(np.sum(np.where(mask, err, 0.0), dtype=np.float64) / n),
    }
    loc = np.zeros_like(mask, dtype=bool)
    loc[:, 7:10, 7:10] = True
    lm = mask & loc
    le = err[lm]
    out.update({
        "local3x3_n": int(lm.sum()),
        "local3x3_rmse": float(np.sqrt(np.mean(le * le))) if le.size else None,
        "local3x3_bias": float(np.mean(le)) if le.size else None,
    })

    ref_thick = (ref >= 30.0) & mask
    pred_thick = (pred >= 30.0) & mask
    tp = int((ref_thick & pred_thick).sum())
    fp = int((~ref_thick & pred_thick & mask).sum())
    fn = int((ref_thick & ~pred_thick).sum())
    union = tp + fp + fn
    out.update({
        "cpp_thick_pixel_leads": int(ref_thick.sum()),
        "pred_thick_pixel_leads": int(pred_thick.sum()),
        "thick_tp": tp, "thick_fp": fp, "thick_fn": fn,
        "thick_csi": float(tp / union) if union else None,
        "thick_precision": float(tp / (tp + fp)) if tp + fp else None,
        "thick_recall": float(tp / (tp + fn)) if tp + fn else None,
    })

    centroids = []
    lead_csi = []
    lead_ref_thick = 0
    lead_joint_detect = 0
    overlap_err = []
    for l in range(ref.shape[0]):
        rmask = ref_thick[l]
        pmask = pred_thick[l]
        nr, npred = int(rmask.sum()), int(pmask.sum())
        if nr:
            lead_ref_thick += 1
        if nr and npred:
            lead_joint_detect += 1
            ry, rx = np.nonzero(rmask)
            py, px = np.nonzero(pmask)
            centroids.append(float(np.hypot(ry.mean() - py.mean(), rx.mean() - px.mean())))
        if nr + npred:
            lead_csi.append(float((rmask & pmask).sum() / (nr + npred - (rmask & pmask).sum())))
        both = rmask & pmask
        if both.any():
            overlap_err.extend(err[l][both].astype(np.float64).tolist())
    out.update({
        "leads_with_cpp_thick": lead_ref_thick,
        "leads_both_detect_thick": lead_joint_detect,
        "thick_centroid_shift_px_mean_when_both_detect": float(np.mean(centroids)) if centroids else None,
        "thick_centroid_shift_px_n": len(centroids),
        "thick_csi_mean_by_lead": float(np.mean(lead_csi)) if lead_csi else None,
        "co_located_thick_n_pixel_leads": len(overlap_err),
        "co_located_thick_bias": float(np.mean(overlap_err)) if overlap_err else None,
        "co_located_thick_rmse": float(np.sqrt(np.mean(np.square(overlap_err)))) if overlap_err else None,
    })

    # Arrival is a station-centred 3x3 spatial mean crossing COT=30.
    ref_series, pred_series = [], []
    for l in range(ref.shape[0]):
        lm = mask[l, 7:10, 7:10]
        ref_series.append(float(ref[l, 7:10, 7:10][lm].mean()) if lm.any() else np.nan)
        pred_series.append(float(pred[l, 7:10, 7:10][lm].mean()) if lm.any() else np.nan)
    def arrival(series):
        return next((15 * (i + 1) for i, v in enumerate(series) if np.isfinite(v) and v >= 30.0), None)
    ref_arr, pred_arr = arrival(ref_series), arrival(pred_series)
    out.update({
        "cpp_local3x3_arrival_min": ref_arr,
        "pred_local3x3_arrival_min": pred_arr,
        "local3x3_arrival_error_min_if_both_detect": (pred_arr - ref_arr) if pred_arr is not None and ref_arr is not None else None,
        "local3x3_arrival_miss": pred_arr is None and ref_arr is not None,
        "cpp_local3x3_peak_cot": float(np.nanmax(ref_series)) if np.isfinite(ref_series).any() else None,
        "pred_local3x3_peak_cot": float(np.nanmax(pred_series)) if np.isfinite(pred_series).any() else None,
        "local3x3_peak_cot_error": (float(np.nanmax(pred_series) - np.nanmax(ref_series))
                                    if np.isfinite(pred_series).any() and np.isfinite(ref_series).any() else None),
        "cpp_local3x3_peak_lead_min": (15 * (int(np.nanargmax(ref_series)) + 1)
                                       if np.isfinite(ref_series).any() else None),
        "pred_local3x3_peak_lead_min": (15 * (int(np.nanargmax(pred_series)) + 1)
                                        if np.isfinite(pred_series).any() else None),
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", type=Path, required=True)
    ap.add_argument("--field-manifest", type=Path, required=True)
    ap.add_argument("--simvp-bank", type=Path, required=True)
    ap.add_argument("--event-selection", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    fields = np.load(a.fields, allow_pickle=False)
    fmanifest = json.loads(a.field_manifest.read_text(encoding="utf-8"))
    events = json.loads(a.event_selection.read_text(encoding="utf-8"))["selected_cases"]
    if fmanifest.get("state") != "PASS_FIXED_EVENT_KEYS_EXACT" or fmanifest.get("test_payload_opened"):
        raise ValueError("event extraction manifest is not accepted or test-open state is unexpected")
    bank = np.load(a.simvp_bank, allow_pickle=False)
    simvp_log = bank["forecast_log1p_cot"]
    seq_ids = bank["sequence_ids"].astype(str)
    target_times = bank["target_time_utc"]
    recs = fmanifest["case_records"]
    if len(recs) != len(events) or len(recs) != 4:
        raise ValueError("event extraction and frozen selection counts differ")

    method_names = ("SimVP_to_R", "P_Hybrid", "P_Hybrid_amp01", "future_observed_AGRI_to_R_oracle", "Persistence_R")
    rows = []
    event_rollup = []
    csi_totals = {m: {"tp": 0, "fp": 0, "fn": 0, "ref_thick": 0, "pred_thick": 0} for m in method_names}
    pred_bank = []
    for e, (case, rec) in enumerate(zip(events, recs)):
        q = int(rec["validation_sequence_index"])
        seqid = str(rec["seq_id"])
        if seqid != str(seq_ids[q]):
            raise ValueError(f"baseline bank key mismatch for {seqid} at {q}: {seq_ids[q]}")
        init = parse_utc(rec["init_time_utc"])
        expected = [init + timedelta(minutes=15 * (j + 1)) for j in range(16)]
        got = [parse_utc(str(x)) for x in target_times[q]]
        if got != expected:
            raise ValueError(f"baseline target times mismatch for {seqid}")

        ref = np.asarray(fields["cpp_cot"][e], dtype=np.float32)
        mask = np.asarray(fields["common_valid"][e], dtype=bool)
        simvp = np.expm1(np.asarray(simvp_log[q, int(rec["station_index"])], dtype=np.float32))
        persistence = np.broadcast_to(fields["simvp_history_cot"][e], (16, 16, 16)).copy()
        methods = {
            "SimVP_to_R": simvp,
            "P_Hybrid": np.asarray(fields["p_hybrid_base_cot"][e], dtype=np.float32),
            "P_Hybrid_amp01": np.asarray(fields["p_hybrid_amp_cot"][e], dtype=np.float32),
            "future_observed_AGRI_to_R_oracle": np.asarray(fields["future_agri_R_cot"][e], dtype=np.float32),
            "Persistence_R": persistence,
        }
        if simvp.shape != ref.shape or mask.shape != ref.shape:
            raise ValueError(f"shape mismatch for {seqid}: ref={ref.shape}, simvp={simvp.shape}, mask={mask.shape}")
        event_row = {
            "station": rec["station"], "selection_type": rec["selection_type"], "init_time_bjt": rec["init_time_bjt"],
            "seq_id": seqid, "cpp_change_selection_value": rec["cpp_change_selection_value"],
            "common_valid_pixel_leads": int(mask.sum()),
        }
        for method, pred in methods.items():
            m = metrics_one(ref, pred, mask)
            row = {"station": rec["station"], "selection_type": rec["selection_type"],
                   "init_time_bjt": rec["init_time_bjt"], "seq_id": seqid, "method": method,
                   "cpp_change_selection_value": rec["cpp_change_selection_value"], **m}
            rows.append(row)
            event_row[f"{method}_rmse"] = m["rmse"]
            event_row[f"{method}_local3x3_rmse"] = m["local3x3_rmse"]
            event_row[f"{method}_thick_csi"] = m["thick_csi"]
            event_row[f"{method}_thick_centroid_shift_px"] = m["thick_centroid_shift_px_mean_when_both_detect"]
            event_row[f"{method}_both_detect_leads"] = m["leads_both_detect_thick"]
            event_row[f"{method}_arrival_min"] = m["pred_local3x3_arrival_min"]
            event_row[f"{method}_arrival_error_min"] = m["local3x3_arrival_error_min_if_both_detect"]
            event_row[f"{method}_peak_local3x3_error"] = m["local3x3_peak_cot_error"]
            for k in ("thick_tp", "thick_fp", "thick_fn", "cpp_thick_pixel_leads", "pred_thick_pixel_leads"):
                outk = {"thick_tp": "tp", "thick_fp": "fp", "thick_fn": "fn",
                        "cpp_thick_pixel_leads": "ref_thick", "pred_thick_pixel_leads": "pred_thick"}.get(k, k)
                csi_totals[method][outk] += int(m[k])
        event_rollup.append(event_row)

    # Event-equal rollup; no pixel/lead is treated as an independent sample.
    summary = {}
    for method in method_names:
        mr = [r for r in rows if r["method"] == method]
        def mean(key):
            vals = [float(r[key]) for r in mr if r.get(key) is not None and np.isfinite(r[key])]
            return float(np.mean(vals)) if vals else None
        counts = csi_totals[method]
        den = counts["tp"] + counts["fp"] + counts["fn"]
        summary[method] = {
            "event_equal_mean_patch_rmse": mean("rmse"),
            "event_equal_mean_local3x3_rmse": mean("local3x3_rmse"),
            "event_equal_mean_bias": mean("bias"),
            "pooled_thick_csi_across_4_events_and_16_leads": counts["tp"] / den if den else None,
            "pooled_thick_precision": counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else None,
            "pooled_thick_recall": counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else None,
            "pooled_counts": counts,
            "event_equal_mean_centroid_shift_px_when_both_detect": mean("thick_centroid_shift_px_mean_when_both_detect"),
            "event_equal_mean_local3x3_peak_error": mean("local3x3_peak_cot_error"),
        }
    a.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = a.out_dir / "event_method_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    rollup_path = a.out_dir / "event_rollup.csv"
    with rollup_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(event_rollup[0]))
        w.writeheader(); w.writerows(event_rollup)
    result = {
        "state": "PASS_FOUR_FROZEN_VALIDATION_EVENTS_POSITION_AMPLITUDE_DIAGNOSTIC",
        "events": event_rollup, "event_equal_summary": summary,
        "method_names": list(method_names),
        "reference": "compact CPP COT retrieval; not independent physical truth",
        "common_mask": "compact CPP common-valid AND new-R future-frame-valid",
        "thick_cloud_threshold": 30.0,
        "centroid_distance_unit": "16x16 grid pixels; conditional on CPP and method both detecting COT>=30",
        "arrival_definition": "first +15..+240 min when station-centred local 3x3 common-support mean COT reaches 30; misses retained",
        "aggregation": "per-event metrics pool that event's leads/pixels; rollup averages four event scores equally; no inferential interval",
        "coverage_limit": "four preselected JJA initializations only; CPP complete 17-frame sequences are unavailable in other seasons",
        "test_payload_opened": False,
        "source_manifest_sha256": fmanifest["event_selection_manifest_sha256"],
        "extracted_fields_sha256": fmanifest["output_sha256"],
    }
    (a.out_dir / "event_position_amplitude_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (a.out_dir / "EVENT_POSITION_AMPLITUDE_DIAGNOSTIC_20261017.md").write_text(render_report(result, rows), encoding="utf-8")
    print(json.dumps({"state": result["state"], "summary": summary, "out_dir": str(a.out_dir)}, ensure_ascii=False))


def render_report(result: dict, rows: list[dict]) -> str:
    lines = [
        "# Position, timing and thickness diagnostics on preselected COT events",
        "",
        "Date: 2026-10-17. Validation only; no test payload was opened.",
        "",
        "## Evidence and limits",
        "",
        "Four event initializations were selected before model-error inspection from the frozen CPP event manifest: one reference-change case and one deterministic-hash ordinary case for each Hunan station. All four are JJA because the source CPP product has no complete 17-frame sequences for other seasons. The new monthly validation sequence keys, station keys, and all 16 target times were checked exactly against saved model predictions and the original SimVP→R forecast bank. All methods use the same per-pixel intersection of the compact CPP validity mask and current R frame-valid mask.",
        "",
        "CPP is a retrieval reference, not independent truth. The true-future-AGRI→R curve is a hindsight retrieval diagnostic, not a deployable forecast. These four selected cases are descriptive and are not an independent sample for significance testing.",
        "",
        "## Event-equal scores",
        "",
        "RMSE and bias are in dimensionless COT units. Thick CSI pools valid pixels across the four events and 16 leads; the centroid distance is in 16×16 grid pixels and is conditional on both reference and method detecting at least one COT≥30 pixel. Arrival is the first +15…+240 min at which the local 3×3 common-support mean reaches COT 30; misses are retained in the event table.",
        "",
        "| Method | Event-equal patch RMSE | Event-equal local 3×3 RMSE | Pooled thick CSI | Event-equal centroid shift (px; both detect) | Event-equal local peak error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, s in result["event_equal_summary"].items():
        def fmt(v, n=3): return "—" if v is None else f"{v:.{n}f}"
        lines.append(f"| {method} | {fmt(s['event_equal_mean_patch_rmse'])} | {fmt(s['event_equal_mean_local3x3_rmse'])} | {fmt(s['pooled_thick_csi_across_4_events_and_16_leads'])} | {fmt(s['event_equal_mean_centroid_shift_px_when_both_detect'])} | {fmt(s['event_equal_mean_local3x3_peak_error'])} |")
    lines += ["", "## Case-level behavior", "", "See `event_rollup.csv` for each event's RMSE, thick-mask detection/centroid displacement, station-neighbourhood arrival and peak timing/amplitude. `event_method_metrics.csv` contains the per-method measurements and supports the pooled CSI counts.", "", "No unique cause is assigned from these diagnostics: a large centroid shift or arrival error points to a location/timing problem; low COT amplitude with good overlap points to thickness evolution; disagreement of future-AGRI→R from CPP indicates an R-versus-CPP reference disagreement. Mixed signatures remain possible.", "", f"Test-payload state: `test_payload_opened={str(result['test_payload_opened']).lower()}`.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
