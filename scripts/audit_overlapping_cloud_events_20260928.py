"""Deduplicate overlapping validation forecasts into station COT cloud events.

CPP is treated as a retrieval reference, not independent physical truth. The
catalog stores each exact target timestamp once after verifying overlapping
windows have identical CPP station values; forecast errors remain stratified
by lead because forecasts for one event are dependent.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/regional_station_cot_20260927"
SOURCE = RES / "sili_transport_variants_val512/sequence_events.json"
OUT = RES / "overlapping_cloud_events_20260928"
THICK = 30.0
EVENT_FLOOR = 20.0
BRIDGE_GAP_STEPS = 1
FIXED_LEADS = (15, 60, 120, 180, 240)
METHODS = ("simvp", "oracle", "global_med7", "local64_med7", "local32_med7", "local16_med7")


def finite(v):
    return v is not None and np.isfinite(float(v))


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sequences = json.loads(SOURCE.read_text(encoding="utf-8"))

    # Canonical target-time index. The reference must be invariant across all
    # forecast windows that include the same physical target timestamp.
    targets: dict[datetime, list[dict]] = defaultdict(list)
    for seq in sequences:
        init = datetime.fromisoformat(seq["init"])
        tracks = seq["station_tracks"]
        for i, cpp in enumerate(tracks["cpp_reference"]):
            if not finite(cpp):
                continue
            lead = (i + 1) * 15
            target_time = init + timedelta(minutes=lead)
            row = {"sequence": int(seq["sequence"]), "init": init.isoformat(),
                   "lead_min": lead, "cpp": float(cpp)}
            for method in METHODS:
                val = tracks.get(method, [None] * 16)[i]
                row[method] = float(val) if finite(val) else None
            targets[target_time].append(row)

    target_records = []
    max_duplicate_spread = 0.0
    duplicate_target_count = 0
    max_multiplicity = 1
    for t in sorted(targets):
        rows = targets[t]
        vals = np.asarray([r["cpp"] for r in rows], dtype=np.float64)
        spread = float(vals.max() - vals.min())
        max_duplicate_spread = max(max_duplicate_spread, spread)
        max_multiplicity = max(max_multiplicity, len(rows))
        duplicate_target_count += len(rows) > 1
        # Since checked exact equality, the first value is a canonical CPP ref.
        target_records.append({"target_time": t.isoformat(), "cpp": float(vals[0]),
                               "n_overlapping_windows": len(rows),
                               "duplicate_spread": spread})
    if max_duplicate_spread > 1e-5:
        raise RuntimeError(f"Overlapping CPP target disagreement: {max_duplicate_spread}")

    # Mark thick-cloud runs. One isolated 15-min dip below the 20-COT floor is
    # bridged only when bounded by thick-event samples; a run must contain >=30.
    times = sorted(targets)
    vals = [float(np.median([r["cpp"] for r in targets[t]])) for t in times]
    support = [bool(targets[t]) for t in times]
    event_mask = [v >= THICK for v in vals]
    # Bridge one below-floor sample if its neighbours are continuous and >=20.
    for i in range(1, len(times) - 1):
        if (not event_mask[i] and event_mask[i - 1] and event_mask[i + 1]
                and (times[i] - times[i - 1]).total_seconds() == 900
                and (times[i + 1] - times[i]).total_seconds() == 900
                and vals[i] >= EVENT_FLOOR):
            event_mask[i] = True

    segments = []
    cur = []
    for i, flag in enumerate(event_mask):
        if flag and cur and (times[i] - times[cur[-1]]).total_seconds() != 900:
            segments.append(cur); cur = []
        if flag:
            cur.append(i)
        elif cur:
            segments.append(cur); cur = []
    if cur:
        segments.append(cur)

    # Keep every distinct thick reference episode in the manifest, including
    # one-frame episodes; distinguish these from multi-frame cloud events.
    event_for_time = {}
    events = []
    for eid, inds in enumerate(segments, 1):
        event_times = [times[i] for i in inds]
        event_values = [vals[i] for i in inds]
        peak_ix = int(np.argmax(event_values))
        first, last = event_values[0], event_values[-1]
        if last - first >= 10:
            evolution = "growth"
        elif first - last >= 10:
            evolution = "decay"
        else:
            evolution = "mixed_or_quasi_steady"
        item = {"event_id": f"SILI-{eid:03d}", "start": event_times[0].isoformat(),
                "end": event_times[-1].isoformat(), "n_15min_samples": len(inds),
                "duration_min_inclusive": 15 * len(inds), "peak_cpp": float(event_values[peak_ix]),
                "peak_time": event_times[peak_ix].isoformat(),
                "cpp_at_start": float(first), "cpp_at_end": float(last),
                "evolution": evolution,
                "episode_class": "multi_frame" if len(inds) >= 2 else "single_frame",
                "target_times": [t.isoformat() for t in event_times]}
        events.append(item)
        for t in event_times:
            event_for_time[t] = item["event_id"]

    # For each event, use a single saved forecast window only if it spans the
    # entire thick core. This makes the event trajectory interpretable as one
    # forecast, rather than stitching different initializations together.
    sequence_diagnostics = []
    for ev in events:
        ets = [datetime.fromisoformat(t) for t in ev["target_times"]]
        candidates = []
        for seq in sequences:
            init = datetime.fromisoformat(seq["init"])
            target_times_seq = [init + timedelta(minutes=15 * k) for k in range(1, 17)]
            index = {t: i for i, t in enumerate(target_times_seq)}
            if all(t in index for t in ets):
                candidates.append((init, seq, index))
        if not candidates:
            sequence_diagnostics.append({"event_id": ev["event_id"], "status": "no_single_forecast_covers_event"})
            continue
        # Select the earliest initialization whose future window covers the
        # whole event; this maximizes a consistent, pre-event forecasting view.
        init, seq, index = min(candidates, key=lambda x: x[0])
        target_times_seq = [init + timedelta(minutes=15 * k) for k in range(1, 17)]
        simvp_track = np.asarray(seq["station_tracks"]["simvp"], dtype=np.float64)
        cpp = np.asarray([float(seq["station_tracks"]["cpp_reference"][index[t]]) for t in ets])
        simvp = np.asarray([float(seq["station_tracks"]["simvp"][index[t]]) for t in ets])
        oracle = np.asarray([float(seq["station_tracks"]["oracle"][index[t]]) for t in ets])
        peak = int(np.argmax(cpp)); pred_peak = int(np.argmax(simvp))
        shifts = (-30, -15, 0, 15, 30)
        common_times = [t for t in ets
                        if finite(seq["station_tracks"]["cpp_reference"][index[t]])
                        and all(t + timedelta(minutes=d) in index
                                and finite(seq["station_tracks"]["simvp"][index[t + timedelta(minutes=d)]])
                                for d in shifts)]
        shift_mae = {}
        for delta in shifts:
            diffs = [simvp_track[index[t + timedelta(minutes=delta)]] -
                     float(seq["station_tracks"]["cpp_reference"][index[t]]) for t in common_times]
            shift_mae[str(delta)] = float(np.mean(np.abs(diffs))) if diffs else None
        zero_mae = shift_mae["0"]
        best_shift = min([(int(k), v) for k, v in shift_mae.items() if v is not None], key=lambda x: x[1]) if common_times else (None, None)
        sequence_diagnostics.append({
            "event_id": ev["event_id"], "status": "single_forecast_covers_event",
            "sequence": int(seq["sequence"]), "init": init.isoformat(),
            "n_event_targets": len(ets), "cpp_peak": float(cpp[peak]),
            "simvp_at_cpp_peak": float(simvp[peak]), "oracle_at_cpp_peak": float(oracle[peak]),
            "simvp_event_peak": float(simvp[pred_peak]),
            "simvp_peak_time_error_min": (pred_peak - peak) * 15 if float(simvp.max()) >= THICK else None,
            "simvp_peak_classification": "predicted_thick_peak" if float(simvp.max()) >= THICK else "no_predicted_thick_peak",
            "simvp_cpp_mae_cot": float(np.mean(np.abs(simvp - cpp))),
            "oracle_cpp_mae_cot": float(np.mean(np.abs(oracle - cpp))),
            "simvp_oracle_mae_cot": float(np.mean(np.abs(simvp - oracle))),
            "timing_shift_diagnostic": {"same_common_targets": len(common_times),
                                         "tested_prediction_time_offsets_min": list(shifts),
                                         "mae_by_offset": shift_mae,
                                         "best_offset_min": best_shift[0],
                                         "best_shifted_mae_cot": best_shift[1],
                                         "unshifted_common_support_mae_cot": zero_mae,
                                         "interpretation": "hindsight station-series timing sensitivity only; cannot isolate spatial displacement or produce an operational correction"},
            "peak_time_warning": "Peak-time error is reported only when SimVP reaches COT>=30; otherwise amplitude/occurrence failure dominates."
        })

    # Fixed-lead predictions targeting each event. Record forecasts as rows,
    # then aggregate within each event before the cohort-level summary.
    event_lead = defaultdict(list)
    point_rows = []
    for target_time, rows in targets.items():
        event_id = event_for_time.get(target_time)
        if event_id is None:
            continue
        for r in rows:
            lead = r["lead_min"]
            errors = {m: (r[m] - r["cpp"]) if r[m] is not None else None for m in METHODS}
            rec = {"event_id": event_id, "target_time": target_time.isoformat(),
                   "sequence": r["sequence"], "init": r["init"], "lead_min": lead,
                   "cpp": r["cpp"], **{f"{m}_cot": r[m] for m in METHODS},
                   **{f"{m}_error": errors[m] for m in METHODS}}
            point_rows.append(rec)
            if lead in FIXED_LEADS:
                event_lead[(event_id, lead)].append(rec)

    lead_summaries = []
    for lead in FIXED_LEADS:
        event_stats = []
        for ev in events:
            rows = event_lead.get((ev["event_id"], lead), [])
            if not rows:
                continue
            cpp = np.array([r["cpp"] for r in rows])
            stat = {"event_id": ev["event_id"], "start": ev["start"],
                    "evolution": ev["evolution"], "n_forecast_rows": len(rows),
                    "n_unique_targets": len({r["target_time"] for r in rows}),
                    "thick_miss_rate_simvp_pred_lt10": float(np.mean([r["simvp_cot"] < 10 for r in rows]))}
            for m in METHODS:
                x = np.asarray([r[f"{m}_error"] for r in rows if r[f"{m}_error"] is not None])
                if x.size:
                    stat[f"{m}_mae"] = float(np.mean(np.abs(x)))
                    stat[f"{m}_bias"] = float(np.mean(x))
                    stat[f"{m}_rmse"] = float(np.sqrt(np.mean(x*x)))
            # Attribution is descriptive and sign-aware: total = retrieval gap
            # + forecast-AGRI-to-R gap for signed errors, not for RMSE/MAE.
            stat["oracle_cpp_mae"] = float(np.mean([abs(r["oracle_error"]) for r in rows]))
            stat["simvp_oracle_mae"] = float(np.mean([abs(r["simvp_cot"] - r["oracle_cot"])
                                                       for r in rows if r["simvp_cot"] is not None and r["oracle_cot"] is not None]))
            stat["simvp_cpp_mae"] = float(np.mean([abs(r["simvp_error"]) for r in rows if r["simvp_error"] is not None]))
            event_stats.append(stat)
        lead_summaries.append({"lead_min": lead, "n_events": len(event_stats),
                               "n_event_target_forecasts": sum(x["n_forecast_rows"] for x in event_stats),
                               "event_level_median": {
                                   key: float(np.median([x[key] for x in event_stats if key in x]))
                                   for key in ("simvp_mae", "oracle_mae", "global_med7_mae",
                                               "local64_med7_mae", "local32_med7_mae", "local16_med7_mae",
                                               "simvp_cpp_mae", "oracle_cpp_mae", "simvp_oracle_mae",
                                               "thick_miss_rate_simvp_pred_lt10")
                                   if any(key in x for x in event_stats)},
                               "per_event": event_stats})

    # Classify multiple independent dates for case review. Select only from the
    # validation cohort, requiring >=4 thick-core target times and distinct dates.
    event_by_id = {e["event_id"]: e for e in events}
    cases = []
    for lead in FIXED_LEADS:
        candidates = []
        for ev in events:
            rows = event_lead.get((ev["event_id"], lead), [])
            if len({r["target_time"] for r in rows}) < 2:
                continue
            miss = np.array([r["simvp_cot"] < 10 for r in rows if r["simvp_cot"] is not None])
            candidates.append((float(np.mean(miss)) if miss.size else 0.0, ev, rows))
        if candidates:
            score, ev, rows = max(candidates, key=lambda x: (x[0], len(x[2]), event_by_id[x[1]["event_id"]]["peak_cpp"]))
            cases.append({"event_id": ev["event_id"], "lead_min": lead, "start": ev["start"],
                          "evolution": ev["evolution"], "peak_cpp": ev["peak_cpp"],
                          "simvp_thick_miss_rate_at_lead": score,
                          "sequences": sorted({r["sequence"] for r in rows})})
    # Distinct-date representatives: highest reference peak, highest thick-miss,
    # and closest oracle-to-CPP retrieval agreement at 120 min.
    seen_dates = set()
    representatives = []
    for label, scorer in (
        ("largest_cpp_peak", lambda e: e["peak_cpp"]),
        ("largest_simvp_miss", lambda e: max((r["cpp"] - r["simvp_cot"] for r in point_rows
                                               if r["event_id"] == e["event_id"] and r["simvp_cot"] is not None), default=-1)),
        ("retrieval_agrees_but_forecast_misses", lambda e: max((abs(r["oracle_cot"]-r["cpp"]) - abs(r["simvp_cot"]-r["cpp"])
                                                                  for r in point_rows if r["event_id"] == e["event_id"]
                                                                  and r["oracle_cot"] is not None and r["simvp_cot"] is not None), default=-1))):
        ordered = sorted(events, key=scorer, reverse=True)
        pick = next((e for e in ordered if e["episode_class"] == "multi_frame"
                     and e["start"][:10] not in seen_dates), None)
        if pick:
            seen_dates.add(pick["start"][:10])
            representatives.append({"case_type": label, **pick})
    existing_rgb_case_dates = {"2025-07-09", "2025-08-05", "2025-08-14"}
    existing_rep_dates = seen_dates | existing_rgb_case_dates
    event_diag_by_id = {x["event_id"]: x for x in sequence_diagnostics}
    new_date_candidates = [x for x in sequence_diagnostics
                           if x.get("status") == "single_forecast_covers_event"
                           and x["n_event_targets"] >= 3
                           and event_by_id[x["event_id"]]["start"][:10] not in existing_rep_dates]
    if new_date_candidates:
        pick = max(new_date_candidates, key=lambda x: x["simvp_cpp_mae_cot"])
        representatives.append({"case_type": "additional_date_severe_miss",
                                **event_by_id[pick["event_id"]],
                                "single_forecast_sequence": pick["sequence"],
                                "single_forecast_init": pick["init"],
                                "single_forecast_simvp_cpp_mae_cot": pick["simvp_cpp_mae_cot"]})

    # CSV for complete event manifest and forecast rows.
    with (OUT / "event_catalog.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["event_id", "start", "end", "n_15min_samples",
                                          "duration_min_inclusive", "peak_cpp", "peak_time",
                                          "cpp_at_start", "cpp_at_end", "evolution", "episode_class"])
        w.writeheader()
        for e in events:
            w.writerow({k: e[k] for k in w.fieldnames})
    with (OUT / "event_fixed_lead_forecasts.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = list(point_rows[0]) if point_rows else []
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(point_rows)

    report = {"test_used": False, "source": str(SOURCE.relative_to(ROOT)),
              "cohort": "fixed 512-sequence Sili validation bank; CPP availability not used for selection",
              "station_pixel_rc_local_crop": [163, 148],
              "target_time_contract": "init + lead; leads +15..+240 min; BJT",
              "cpp_role": "retrieval reference only; not independent physical truth",
              "deduplication": {"sequences_with_cpp": len(sequences), "unique_cpp_target_times": len(targets),
                                "target_times_repeated_across_windows": duplicate_target_count,
                                "max_overlapping_windows": max_multiplicity,
                                "max_cpp_duplicate_spread": max_duplicate_spread,
                                "exact_duplicate_consistency_pass": max_duplicate_spread <= 1e-5},
              "event_definition": {"thick_core_cpp_ge": THICK, "one_sample_bridge_floor_cpp_ge": EVENT_FLOOR,
                                   "bridge_limit_steps": BRIDGE_GAP_STEPS,
                                   "cadence": "15 min; only exact continuous timestamps are connected",
                                   "duration_inclusive_min": "15 * number of samples; endpoint span is also retained"},
              "n_events_ge30": len(events),
              "n_multiframe_events": sum(e["episode_class"] == "multi_frame" for e in events),
              "n_events_by_evolution": {k: sum(e["evolution"] == k for e in events)
                                        for k in sorted({e["evolution"] for e in events})},
              "fixed_leads": lead_summaries, "representative_dates": representatives,
              "single_forecast_event_diagnostics": sequence_diagnostics,
              "interpretation_limits": [
                  "Overlapping forecasts for the same target are retained only for their lead-stratified forecast rows; they are not independent samples.",
                  "Event-level medians give each detected event equal weight; uncertainty must be clustered by date/event in any inferential test.",
                  "Station-point tracks can diagnose thickness and arrival timing, but cannot independently distinguish 2-D displacement from cloud-field growth without spatial maps.",
                  "CPP product errors are shared by CPP and the AGRI-to-COT retrieval comparison; oracle agreement is not independent validation."]}
    (OUT / "event_audit.json").write_text(json.dumps(json_safe(report), indent=2, allow_nan=False), encoding="utf-8")

    # Truthful event figures: CPP/station series are the full unique chronology;
    # forecast errors are equal-event medians at fixed horizons.
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout="constrained")
    dates = [datetime.fromisoformat(x["target_time"]) for x in target_records]
    cpp = [x["cpp"] for x in target_records]
    axes[0].plot(dates, cpp, color="#222222", lw=1.2, label="Unique target-time CPP COT")
    for e in events:
        if e["episode_class"] == "multi_frame":
            axes[0].axvspan(datetime.fromisoformat(e["start"]),
                            datetime.fromisoformat(e["end"]) + timedelta(minutes=15),
                            color="#D55E00", alpha=.12)
    axes[0].axhline(THICK, color="#D55E00", ls="--", lw=1, label="Thick-core threshold (30)")
    axes[0].set(ylabel="Station COT (CPP reference)", title="Sili validation chronology after exact target-time deduplication")
    axes[0].grid(alpha=.2); axes[0].legend(ncol=2, fontsize=8)
    for method, label in (("simvp_mae", "SimVP→R"), ("oracle_mae", "R(real future AGRI)"),
                          ("local64_med7_mae", "64-domain transport")):
        ys = [x["event_level_median"].get(method, np.nan) for x in lead_summaries]
        axes[1].plot(FIXED_LEADS, ys, marker="o", label=label)
    axes[1].set(xlabel="Forecast lead (min)", ylabel="Median event-level COT MAE",
                title="Errors on thick-event core; each event contributes equally")
    axes[1].grid(alpha=.2); axes[1].legend()
    fig.savefig(OUT / "event_dedup_and_errors.png", dpi=180)
    fig.savefig(OUT / "event_dedup_and_errors.pdf")
    plt.close(fig)

    # Distinct-date trajectories use one saved forecast per event, never a splice
    # across different initializations. This complements (does not replace) spatial
    # RGB maps, which are available only for the three previously archived cases.
    if representatives:
        fig, axes = plt.subplots(len(representatives), 1, figsize=(10, 3.1 * len(representatives)),
                                 sharex=True, layout="constrained")
        if len(representatives) == 1:
            axes = [axes]
        for ax, rep in zip(axes, representatives):
            eid = rep["event_id"]
            diag = event_diag_by_id.get(eid)
            if diag is None or diag.get("sequence") is None:
                ax.set_axis_off(); continue
            seq = next(s for s in sequences if int(s["sequence"]) == diag["sequence"])
            leads = np.arange(1, 17) * 15
            for method, label, style in (("cpp_reference", "CPP retrieval reference", "-o"),
                                         ("simvp", "SimVP forecast AGRI→R", "--s"),
                                         ("oracle", "R(real future AGRI), hindsight", ":^")):
                track = np.asarray([np.nan if not finite(v) else float(v)
                                    for v in seq["station_tracks"][method]], dtype=float)
                ax.plot(leads, track, style, lw=1.5, ms=3, label=label)
            core = [15 * (datetime.fromisoformat(t) - datetime.fromisoformat(diag["init"])).total_seconds() / 900
                    for t in rep["target_times"]]
            if core:
                ax.axvspan(min(core), max(core), color="#D55E00", alpha=.12, label="CPP thick-event core")
            ax.axhline(THICK, color="gray", ls="--", lw=1)
            ax.set_ylabel("Station COT")
            ax.set_title(f"{eid} · {rep['case_type']} · {rep['start'][:10]} · sequence {diag['sequence']}")
            ax.grid(alpha=.2); ax.legend(fontsize=7, ncol=4)
        axes[-1].set_xlabel("Lead from the single forecast initialization (min)")
        ax.axhline(THICK, color="gray", ls="--", lw=1)
        fig.suptitle("Different-date thick-cloud events · each panel is one uninterrupted forecast")
        fig.savefig(OUT / "representative_event_tracks.png", dpi=180)
        fig.savefig(OUT / "representative_event_tracks.pdf")
        plt.close(fig)

    print(json.dumps({"output": str(OUT), "unique_cpp_target_times": len(targets),
                      "events": len(events), "multi_frame_events": report["n_multiframe_events"],
                      "max_duplicate_spread": max_duplicate_spread,
                      "representatives": representatives}, indent=2))


if __name__ == "__main__":
    main()
