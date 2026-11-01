#!/usr/bin/env python3
"""Validation-only event decomposition for current SimVP->R and direct COT forecasts.

The event definition is fixed before reading model errors: CPP station-pixel COT >=30
for at least two consecutive, valid 15-minute targets. For each unique physical event,
select the available validation forecast initialization with the largest start lead
that still covers the complete event. No test payload is read and no model is run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def dt(value: str) -> datetime:
    d = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d.astimezone(timezone.utc)


def qtile_mean(a: np.ndarray) -> float:
    return float(np.mean(a, dtype=np.float64)) if a.size else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    root = a.project_root
    branch = root / "refine-logs/cot_retrieved_forecast_20261013"
    pack = branch / "data/sequence_cpp_pack_20261017"
    side = branch / "data/frame_sidecars_20261017_v1"
    rbank = root / "experiments/monthly_R_station16_20261013_v1/results/new_monthly_R_seed42_cot_bank_v1/cot_log1p.npy"
    forecast_bank = root / "experiments/cot_value_first_20261013/results/simvp_supervision_cot_allval_20261017_v2/mse/forecast_cot_val_bank.npz"
    run = branch / "results/fulltrain_20261017_seed42_b1024_e12"
    direct = {
        "P_R": run / "P_R_val_pred_epoch03.npy",
        "P_Hybrid": run / "P_Hybrid_val_pred_epoch09.npy",
        "P_Hybrid_amp": branch / "results/amplitude_loss_20261017_seed42_b1024_e12/P_Hybrid_val_pred_epoch11.npy",
    }
    outdir = a.output
    outdir.mkdir(parents=True, exist_ok=True)

    keys = list(csv.DictReader((pack / "sequence_keys.csv").open(encoding="utf-8")))
    keys = [r for r in keys if r["split"] == "val"]
    seqidx = np.load(pack / "sequence_indices_val.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(side / "frame_valid_mask.npy", mmap_mode="r", allow_pickle=False)
    cpp_frames = np.load(pack / "cpp_cot_physical.npy", mmap_mode="r", allow_pickle=False)
    cpp_frame_mask = np.load(pack / "cpp_common_mask.npy", mmap_mode="r", allow_pickle=False)
    rlog = np.load(rbank, mmap_mode="r", allow_pickle=False)
    pkeys = list(csv.DictReader((run / "validation_prediction_keys.csv").open(encoding="utf-8")))
    forecast = np.load(forecast_bank, allow_pickle=False)
    forecast_profile_path = forecast_bank.parent / "profile.json"
    forecast_profile = json.loads(forecast_profile_path.read_text(encoding="utf-8"))
    r_manifest_path = root / "experiments/monthly_R_station16_20261013_v1/results/new_monthly_R_seed42_cot_bank_v1/bank_manifest.json"
    r_manifest = json.loads(r_manifest_path.read_text(encoding="utf-8"))
    if (forecast_profile.get("state") != "COMPLETE_ALL_VALIDATION_S_TO_R_COT_AND_AGRI3X3_BANK" or
            forecast_profile.get("split") != "val_only" or forecast_profile.get("test_payloads_read") != 0 or
            forecast_profile.get("r_checkpoint_sha256") != r_manifest.get("R_checkpoint_sha256") or
            r_manifest.get("state") != "COMPLETE" or r_manifest.get("test_payload_opened") is not False):
        raise ValueError("SimVP->R bank or frozen-R provenance is not accepted for this validation analysis")
    f_hash = hashlib.sha256(np.ascontiguousarray(forecast["forecast_log1p_cot"]).tobytes()).hexdigest()
    if f_hash != forecast_profile.get("forecast_output_sha256"):
        raise ValueError("SimVP->R forecast array checksum does not match its complete profile")
    n = len(keys)
    if n != 3140 or seqidx.shape != (n, 24) or len(pkeys) != 2*n:
        raise ValueError(f"unexpected validation contract: n={n}, indices={seqidx.shape}, predkeys={len(pkeys)}")
    expected_shapes = {
        "forecast_log1p_cot": (n, 2, 16, 16, 16),
        "target_agri_r_log1p_cot": (n, 2, 16, 16, 16),
        "persistence_log1p_cot": (n, 2, 16, 16, 16),
        "cpp_reference_cot": (n, 2, 16, 16, 16),
        "cpp_valid_mask": (n, 2, 16, 16, 16),
        "target_time_utc": (n, 16),
        "sequence_ids": (n,),
    }
    for k, shape in expected_shapes.items():
        if forecast[k].shape != shape:
            raise ValueError(f"{k} shape {forecast[k].shape} != {shape}")
    if forecast["sequence_ids"].tolist() != [r["seq_id"] for r in keys]:
        raise ValueError("all-validation COT bank sequence order does not exactly match current sequence pack")

    # Key and date-join audit for direct forecasts; no reliance on row position alone.
    pred_row = {}
    for i, row in enumerate(pkeys):
        key = (int(row["sequence_index"]), row["station"])
        if key in pred_row:
            raise ValueError(f"duplicate direct prediction key: {key}")
        pred_row[key] = i
    station_names = ("sili", "zhujia")
    for q, row in enumerate(keys):
        for s, station in enumerate(station_names):
            j = pred_row.get((q, station))
            if j is None or dt(pkeys[j]["init_time_utc"]) != dt(row["init_time_utc"]):
                raise ValueError(f"direct forecast key mismatch: {row['seq_id']} {station}")
            actual_target_times = [dt(str(x)) for x in forecast["target_time_utc"][q]]
            expected_target_times = [dt(row["init_time_utc"]) + timedelta(minutes=15*(l+1)) for l in range(16)]
            if actual_target_times != expected_target_times:
                raise ValueError(f"target-time contract mismatch at {row['seq_id']}")

    p_arrays = {name: np.load(path, mmap_mode="r", allow_pickle=False) for name, path in direct.items()}
    for name, arr in p_arrays.items():
        if arr.shape != (2*n, 16, 1, 16, 16):
            raise ValueError(f"{name} direct prediction shape mismatch: {arr.shape}")

    stale_cpp = forecast["cpp_reference_cot"]  # decompress this NPZ member once
    pred_r = np.expm1(forecast["forecast_log1p_cot"].astype(np.float64))
    times = forecast["target_time_utc"]
    q_time_lookup = []
    target_to_sequences = [defaultdict(list) for _ in station_names]
    for q in range(n):
        qtimes = [dt(str(x)) for x in times[q]]
        qmap = {t:l for l,t in enumerate(qtimes)}
        q_time_lookup.append(qmap)
        for s in range(2):
            for lead, t in enumerate(qtimes):
                target_to_sequences[s][t].append((q, lead))

    # Verify and quarantine the legacy CPP value join in this all-validation bank.
    # It must not be used as a reference; the current compact pack is authoritative.
    max_stale_cpp_diff = 0.0
    support_count = 0
    timeline = [defaultdict(list) for _ in station_names]
    for q in range(n):
        target_frames = np.asarray(seqidx[q, 8:], dtype=np.int64)
        for s, station in enumerate(station_names):
            cv = np.asarray(cpp_frames[target_frames, s, 0], dtype=np.float32)
            cm = np.asarray(cpp_frame_mask[target_frames, s, 0], dtype=bool)
            rv = np.asarray(valid[target_frames, s, 0], dtype=bool)
            packed_cpp = np.asarray(cpp_frames[target_frames, s, 0], dtype=np.float32)
            packed_mask = np.asarray(cpp_frame_mask[target_frames, s, 0], dtype=bool)
            max_stale_cpp_diff = max(max_stale_cpp_diff, float(np.max(np.abs(
                np.asarray(stale_cpp[q,s],dtype=np.float64)-packed_cpp.astype(np.float64)))))
            if not np.array_equal(cm, packed_mask):
                raise ValueError(f"CPP compact-pack indexing failed: {keys[q]['seq_id']} {station}")
            common = cm & rv
            support_count += int(common.sum())
            for l in range(16):
                t = dt(str(times[q, l]))
                center_valid = bool(common[l, 8, 8])
                timeline[s][t].append((float(cv[l,8,8]), center_valid, q, l))

    # Collapse overlapping sequence windows to one audited target-time record. Duplicates
    # must agree in label value and support; otherwise the current time/key join is rejected.
    series = []
    for s, station in enumerate(station_names):
        ts = []
        for t, vals in timeline[s].items():
            good = [v for v in vals if v[1]]
            if len(good) > 1:
                values = np.asarray([v[0] for v in good], dtype=np.float64)
                if float(np.ptp(values)) > 1e-5:
                    raise ValueError(f"overlapping CPP labels disagree at {station} {t}: range={np.ptp(values)}")
            ts.append((t, good[0][0] if good else float("nan"), bool(good)))
        ts.sort(key=lambda x: x[0])
        series.append(ts)

    # Unique station-pixel thick-cloud events: >=2 consecutive valid 15-min target times.
    events = []
    event_id = 0
    for s, station in enumerate(station_names):
        seq = series[s]
        runs = []
        cur = []
        for item in seq:
            t, c, ok = item
            if ok and c >= 30.0 and cur and t-cur[-1][0] == timedelta(minutes=15):
                cur.append(item)
            elif ok and c >= 30.0:
                if len(cur) >= 2: runs.append(cur)
                cur = [item]
            else:
                if len(cur) >= 2: runs.append(cur)
                cur = []
        if len(cur) >= 2: runs.append(cur)
        for run0 in runs:
            event_id += 1
            start, end = run0[0][0], run0[-1][0]
            core_times = [x[0] for x in run0]
            candidates = []
            # Only inspect inits whose forecast actually contains the event start;
            # this avoids scanning all 3,140 windows for every event.
            for q, start_lead in target_to_sequences[s].get(start, []):
                row = keys[q]
                init = dt(row["init_time_utc"])
                lead_start = int((start-init).total_seconds()//60)
                lead_end = int((end-init).total_seconds()//60)
                if lead_start != 15*(start_lead+1) or lead_start < 15 or lead_end > 240:
                    continue
                t_to_l = q_time_lookup[q]
                if not all(t in t_to_l for t in core_times):
                    continue
                ff = np.asarray(seqidx[q,8:], dtype=np.int64)
                rv_center = np.asarray(valid[ff, s, 0, 8, 8], dtype=bool)
                cpp_center = np.asarray(cpp_frame_mask[ff, s, 0, 8, 8], dtype=bool)
                if not all(rv_center[t_to_l[t]] and cpp_center[t_to_l[t]] for t in core_times):
                    continue
                candidates.append((lead_start, q, t_to_l))
            if not candidates:
                continue
            # Longest-available forecast that still covers the full physical event.
            lead_start, q, t_to_l = max(candidates, key=lambda x: (x[0], keys[x[1]]["seq_id"]))
            # Include one supported pre-event target if available to score onset behavior.
            pre = start - timedelta(minutes=15)
            eval_times = ([pre] if pre in t_to_l else []) + core_times
            leads = [t_to_l[t] for t in eval_times]
            frame_ix = np.asarray(seqidx[q,8:],dtype=np.int64)
            event_frame_ix = frame_ix[leads]
            common = cpp_frame_mask[event_frame_ix,s,0,7:10,7:10].astype(bool) & valid[
                event_frame_ix,s,0,7:10,7:10].astype(bool)
            truth = np.asarray(cpp_frames[event_frame_ix,s,0,7:10,7:10],dtype=np.float64)
            methods = {
                "SimVP_to_R": pred_r[q,s,leads,7:10,7:10],
                "Observed_future_AGRI_to_R": np.expm1(np.asarray(rlog[event_frame_ix,s,0,7:10,7:10],dtype=np.float64)),
                "Persistence_R": np.broadcast_to(np.expm1(np.asarray(rlog[int(seqidx[q,7]),s,0,7:10,7:10],dtype=np.float64)), (len(leads),3,3)),
            }
            for name, arr in p_arrays.items():
                methods[name] = np.asarray(arr[pred_row[(q,station)],leads,0,7:10,7:10],dtype=np.float64)
            rec = {
                "event_id": f"{station}_{event_id:04d}", "station": station,
                "event_start_utc": start.isoformat(), "event_end_utc": end.isoformat(),
                "event_start_bjt": start.astimezone(timezone(timedelta(hours=8))).isoformat(),
                "event_end_bjt": end.astimezone(timezone(timedelta(hours=8))).isoformat(),
                "n_thick_center_frames": len(core_times), "cpp_peak_center": max(x[1] for x in run0),
                "init_time_utc": keys[q]["init_time_utc"], "seq_id": keys[q]["seq_id"],
                "event_start_lead_min": lead_start, "event_end_lead_min": int((end-dt(keys[q]["init_time_utc"])).total_seconds()//60),
                "n_eval_local_pixel_leads": int(common.sum()),
            }
            ref_event = np.asarray([next(x[1] for x in seq if x[0] == t) for t in eval_times],dtype=np.float64)
            rec["cpp_center_peak"] = float(np.nanmax(ref_event))
            rec["cpp_center_arrival_lead_min"] = lead_start
            for name, arr in methods.items():
                diff = arr-truth
                err = diff[common]
                tp = int(((arr>=30)&(truth>=30)&common).sum())
                fp = int(((arr>=30)&(truth<30)&common).sum())
                fn = int(((arr<30)&(truth>=30)&common).sum())
                den=tp+fp+fn
                # The station pixel is the centre cell of the 3x3 local crop.
                center = arr[:,1,1]
                center_arrival = next((eval_times[i] for i,x in enumerate(center) if x>=30),None)
                center_peak_i = int(np.argmax(center))
                rec[f"{name}_local3_rmse"] = float(np.sqrt(np.mean(err**2))) if err.size else float("nan")
                rec[f"{name}_local3_bias"] = float(np.mean(err)) if err.size else float("nan")
                rec[f"{name}_thick_csi"] = float(tp/den) if den else float("nan")
                rec[f"{name}_thick_tp"] = tp; rec[f"{name}_thick_fp"] = fp; rec[f"{name}_thick_fn"] = fn
                rec[f"{name}_center_peak_cot"] = float(center[center_peak_i])
                rec[f"{name}_center_peak_error"] = float(center[center_peak_i] - rec["cpp_center_peak"])
                rec[f"{name}_center_peak_lead_min"] = int((eval_times[center_peak_i]-dt(keys[q]["init_time_utc"])).total_seconds()//60)
                rec[f"{name}_center_arrival_lead_min"] = int((center_arrival-dt(keys[q]["init_time_utc"])).total_seconds()//60) if center_arrival else ""
                rec[f"{name}_center_arrival_error_min"] = int((center_arrival-start).total_seconds()//60) if center_arrival else ""
            events.append(rec)

    if not events:
        raise ValueError("no event forecast pairs found under the fixed event definition")
    fields = list(events[0])
    with (outdir/"event_forecast_pairs.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(events)
    group_summary=[]
    for station in station_names:
        subset=[e for e in events if e["station"]==station]
        for method in ("SimVP_to_R","Observed_future_AGRI_to_R","Persistence_R","P_R","P_Hybrid","P_Hybrid_amp"):
            x=[e for e in subset if np.isfinite(e[f"{method}_local3_rmse"])]
            if not x: continue
            tp=sum(e[f"{method}_thick_tp"] for e in x); fp=sum(e[f"{method}_thick_fp"] for e in x); fn=sum(e[f"{method}_thick_fn"] for e in x)
            d=tp+fp+fn
            group_summary.append({"station":station,"method":method,"n_unique_events":len(x),
                "mean_event_local3_rmse":qtile_mean(np.asarray([e[f"{method}_local3_rmse"] for e in x])),
                "median_event_local3_rmse":float(np.median([e[f"{method}_local3_rmse"] for e in x])),
                "mean_event_center_peak_error":qtile_mean(np.asarray([e[f"{method}_center_peak_error"] for e in x])),
                "thick_tp":tp,"thick_fp":fp,"thick_fn":fn,"pooled_thick_csi":tp/d if d else float("nan"),
                "arrival_misses":sum(e[f"{method}_center_arrival_lead_min"]=="" for e in x)})
    with (outdir/"event_method_summary.csv").open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(group_summary[0]));w.writeheader();w.writerows(group_summary)
    prov={"state":"PASS_FULLVAL_STATION_PIXEL_EVENT_DECOMPOSITION","split":"val_only",
        "test_payload_opened":False,"fixed_event_rule":"CPP station pixel (8,8) COT >=30 for >=2 consecutive, frame-valid 15-min targets; invalid target breaks event",
        "forecast_selection":"for each unique physical event, use the available validation initialization with the largest start lead while covering the full event",
        "event_count":len(events),"events_by_station":{s:sum(e["station"]==s for e in events) for s in station_names},
        "unique_target_timestamps_by_station":{station_names[s]:len(series[s]) for s in range(2)},
        "stale_allval_bank_cpp_join_max_absdiff_quarantined":max_stale_cpp_diff,
        "cpp_common_mask_pixels_all_sequences":support_count,
        "source_sha256":{"sequence_pack_manifest":sha256(pack/"pack_manifest.json"),
            "forecast_bank":sha256(forecast_bank),"P_R_prediction":sha256(direct["P_R"]),
            "P_Hybrid_prediction":sha256(direct["P_Hybrid"]),"P_Hybrid_amp_prediction":sha256(direct["P_Hybrid_amp"]),
            "prediction_keys":sha256(run/"validation_prediction_keys.csv"),"forecast_profile":sha256(forecast_profile_path),
            "R_bank":sha256(rbank),"R_bank_manifest":sha256(r_manifest_path),
            "frame_valid_mask":sha256(side/"frame_valid_mask.npy")},
        "interpretation_limit":"selected unique CPP-referenced station-pixel thick events, one longest-lead forecast per event; event counts are not independent confidence intervals; CPP is retrieval reference; observed-future-AGRI->R is hindsight only"}
    (outdir/"manifest.json").write_text(json.dumps(prov,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(prov,ensure_ascii=False))


if __name__=="__main__": main()
