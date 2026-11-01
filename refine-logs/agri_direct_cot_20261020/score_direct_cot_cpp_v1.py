#!/usr/bin/env python3
"""Score direct-COT H/V forecasts against the archived CPP validation reference.

CPP is a retrieval reference, not independent truth. This scorer reads only V
predictions and the explicitly supervised rows in the archived CPP train/val
pack. It reports pixel/lead metrics and paired BJT two-day-block bootstrap CIs.
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

STATIONS = ("sili", "zhujia")
ARMS = ("cot_last", "cot_simvp_r", "cot_p_c", "cot_p_xc")
ARRAYS = {"cot_last": "cot_last.npy", "cot_simvp_r": "cot_simvp_r.npy",
          "cot_p_c": "cot_p_c.npy", "cot_p_xc": "cot_p_xc.npy"}
SCALES = {"patch_16x16": (slice(0, 16), slice(0, 16)),
          "station_3x3": (slice(7, 10), slice(7, 10)),
          "center_1x1": (slice(8, 9), slice(8, 9))}
LEADS = tuple(range(15, 241, 15))
THRESHOLDS = (1.0, 30.0)
REPS = 5000
SEED = 20261020


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validation_month_blocks(path: Path) -> dict[str, str]:
    """Verify the frozen outer validation days are consecutive pairs per month."""
    rows = read_csv(path)
    val = [r["date_BJT"] for r in rows if r["split"] == "val"]
    grouped: dict[str, list[datetime]] = defaultdict(list)
    for s in val:
        d = datetime.strptime(s, "%Y-%m-%d")
        grouped[d.strftime("%Y-%m")].append(d)
    out = {}
    for month, dates in grouped.items():
        dates = sorted(set(dates))
        if len(dates) != 2 or (dates[1] - dates[0]).days != 1:
            raise ValueError(f"validation block is not exactly two consecutive dates: {month} {dates}")
        out.update({d.date().isoformat(): month for d in dates})
    if len(grouped) != 26:
        raise ValueError(f"expected 26 frozen validation blocks, found {len(grouped)}")
    return out


def cpp_reference_map(pack: Path) -> tuple[dict, dict]:
    status = read_json(pack / "pack_status.json")
    audit = read_json(pack / "pack_audit.json")
    if status.get("state") != "COMPLETE_MONTHLY_R_STATION16_PACK" or status.get("test_payloads_read") != 0:
        raise ValueError("CPP source pack status/isolation gate failed")
    if audit.get("state") != "PASS_MONTHLY_R_STATION16_PACK_AUDIT" or audit.get("test_payloads_read") != 0:
        raise ValueError("CPP source pack audit/isolation gate failed")
    if audit.get("claim_scope") != "CPP-supported daytime retrieval-reference pixels; not independent truth":
        raise ValueError("CPP claim scope is not the expected retrieval-reference scope")
    for name, file_name in (("cot", "cot.npy"), ("mask", "mask.npy")):
        if sha(pack / file_name) != audit["arrays"][name]["sha256"]:
            raise ValueError(f"CPP {file_name} does not match its immutable pack audit SHA")
    rows = read_csv(pack / "rows.csv")
    supervised = [r for r in rows if r["split"] == "val" and r["state"] == "SUPERVISED"]
    cot = np.load(pack / "cot.npy", mmap_mode="r", allow_pickle=False)
    mask = np.load(pack / "mask.npy", mmap_mode="r", allow_pickle=False)
    if cot.shape != (int(audit["arrays"]["cot"]["shape"][0]), 1, 16, 16) or mask.shape != cot.shape:
        raise ValueError(f"CPP COT/mask shape mismatch: {cot.shape}/{mask.shape}")
    ref = {}
    for r in supervised:
        idx = int(r["array_index"])
        if idx < 0 or idx >= len(cot):
            raise ValueError(f"CPP array index invalid for {r}")
        key = (r["station"].lower(), r["target_time_utc"])
        if key in ref:
            raise ValueError(f"duplicate supervised CPP station/time key {key}")
        y = np.asarray(cot[idx, 0], dtype=np.float32)
        m = np.asarray(mask[idx, 0], dtype=bool)
        if not np.isfinite(y[m]).all() or (y[m] < 0).any():
            raise ValueError(f"invalid supervised CPP values {key}")
        ref[key] = (y, m)
    return ref, {"pack_status_sha256": sha(pack / "pack_status.json"),
                 "pack_audit_sha256": sha(pack / "pack_audit.json"),
                 "rows_sha256": sha(pack / "rows.csv"), "cot_sha256": sha(pack / "cot.npy"),
                 "mask_sha256": sha(pack / "mask.npy"), "supervised_validation_keys": len(ref),
                 "claim_scope": audit["claim_scope"]}


def load_prediction_keys(inference: Path, support: Path) -> list[dict]:
    idx_rows = read_csv(inference / "V" / "index.csv")
    sup_rows = read_csv(support / "v_sequences.csv")
    sup = {r["seq_id"]: r for r in sup_rows}
    if len(sup) != len(sup_rows) or len(idx_rows) != len(sup_rows):
        raise ValueError("V index/support cardinality mismatch or duplicate sequence")
    out = []
    for i, r in enumerate(idx_rows):
        if int(r["sequence_index_within_split"]) != i or r["seq_id"] not in sup:
            raise ValueError("V prediction sequence index does not match frozen support")
        paths = sup[r["seq_id"]]["data_relpaths"].split("|")
        if len(paths) != 24:
            raise ValueError("V sequence does not have the frozen 24-frame support")
        for li, lead in enumerate(LEADS):
            stamp = Path(paths[8 + li]).stem
            target = datetime.strptime(stamp, "%Y%m%d%H%M").strftime("%Y-%m-%d %H:%M:%S")
            init = datetime.strptime(Path(paths[7]).stem, "%Y%m%d%H%M")
            target_dt = datetime.strptime(target, "%Y-%m-%d %H:%M:%S")
            if int((target_dt - init).total_seconds() // 60) != lead:
                raise ValueError(f"forecast lead/time mismatch for {r['seq_id']} lead={lead}")
            out.append({"sequence_index": i, "station": None, "lead_minutes": lead,
                        "target_time_utc": target})
    return out


def safe_metrics(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> dict:
    m = mask & np.isfinite(pred) & np.isfinite(truth)
    if not m.any():
        return {"n_pixels": 0, "rmse": None, "mae": None, "bias": None,
                "thick_pod": None, "thick_far": None, "thick_csi": None,
                "thick_bias": None, "thick_tp": 0, "thick_fp": 0, "thick_fn": 0}
    p, y = pred[m].astype(np.float64), truth[m].astype(np.float64)
    e = p - y
    thick_y, thick_p = y >= 30.0, p >= 30.0
    tp = int(np.count_nonzero(thick_y & thick_p))
    fp = int(np.count_nonzero(~thick_y & thick_p))
    fn = int(np.count_nonzero(thick_y & ~thick_p))
    my = mask & np.isfinite(pred) & np.isfinite(truth) & (truth >= 30.0)
    thick_bias = float(np.mean(pred[my].astype(np.float64) - truth[my].astype(np.float64))) if my.any() else None
    return {"n_pixels": int(m.sum()), "rmse": float(np.sqrt(np.mean(e * e))),
            "mae": float(np.mean(np.abs(e))), "bias": float(np.mean(e)),
            "thick_pod": tp / (tp + fn) if tp + fn else None,
            "thick_far": fp / (tp + fp) if tp + fp else None,
            "thick_csi": tp / (tp + fp + fn) if tp + fp + fn else None,
            "thick_bias": thick_bias, "thick_tp": tp, "thick_fp": fp, "thick_fn": fn}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inference", type=Path, required=True)
    ap.add_argument("--support", type=Path, required=True)
    ap.add_argument("--cpp-pack", type=Path, required=True)
    ap.add_argument("--day-assignment", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    receipt = read_json(a.inference / "inference_manifest.json")
    if receipt.get("state") != "COMPLETE_CAUSAL_DIRECT_COT_HV_INFERENCE":
        raise ValueError("H/V inference is incomplete")
    for field in ("future_observed_agri_read", "cpp_read", "ghi_labels_read", "test_payload_opened", "buffer_payload_opened"):
        if receipt.get(field) is not False:
            raise ValueError(f"inference isolation gate failed: {field}")
    if a.output.exists():
        raise FileExistsError(a.output)
    month_by_date = validation_month_blocks(a.day_assignment)
    ref, ref_audit = cpp_reference_map(a.cpp_pack)
    vdir = a.inference / "V"
    rows = read_csv(vdir / "index.csv")
    support = {r["seq_id"]: r for r in read_csv(a.support / "v_sequences.csv")}
    if len(support) != len(rows):
        raise ValueError("V prediction/support sequence count mismatch")
    arrays = {arm: np.load(vdir / fn, mmap_mode="r", allow_pickle=False) for arm, fn in ARRAYS.items()}
    nseq = len(rows)
    expected_shape = (nseq, 2, 16, 1, 16, 16)
    if any(arr.dtype != np.float32 or arr.shape != expected_shape for arr in arrays.values()):
        raise ValueError(f"V COT arrays do not match contract {expected_shape}")
    hit_targets, all_targets = set(), set()
    # Each matched item is one sequence/station/lead forecast, clustered by the
    # predeclared monthly two-consecutive-day V block.
    items = []
    for r in rows:
        si_seq = int(r["sequence_index_within_split"])
        seq = support.get(r["seq_id"])
        if si_seq >= nseq or seq is None:
            raise ValueError("V index references absent sequence")
        paths = seq["data_relpaths"].split("|")
        if len(paths) != 24:
            raise ValueError("V sequence frame count mismatch")
        init = datetime.strptime(Path(paths[7]).stem, "%Y%m%d%H%M")
        for si, station in enumerate(STATIONS):
            for li, lead in enumerate(LEADS):
                t = datetime.strptime(Path(paths[8 + li]).stem, "%Y%m%d%H%M")
                if (t - init).total_seconds() != lead * 60:
                    raise ValueError("V inference target time does not match canonical lead")
                key = (station, t.strftime("%Y-%m-%d %H:%M:%S"))
                all_targets.add(key)
                pair = ref.get(key)
                if pair is None:
                    continue
                y, m = pair
                if not m.any():
                    continue
                bjt_date = (t + timedelta(hours=8)).date().isoformat()
                block = month_by_date.get(bjt_date)
                if block is None:
                    raise ValueError(f"CPP-matched V target date is absent from validation blocks: {bjt_date}")
                hit_targets.add(key)
                item = {"sequence_index": si_seq, "station_index": si, "station": station,
                        "lead_minutes": lead, "lead_index": li, "target_time_utc": key[1],
                        "target_bjt_date": bjt_date, "block_id": block, "truth": y, "mask": m,
                        "predictions": {arm: np.asarray(arr[si_seq, si, li, 0], dtype=np.float32) for arm, arr in arrays.items()}}
                items.append(item)
    if not items:
        raise ValueError("no V forecasts overlap quality-controlled CPP validation support")

    # Point metrics, identically masked across methods.
    per_lead, overall, strata = {}, {}, {}
    for scale, (rs, cs) in SCALES.items():
        sub = [dict(it, truth=it["truth"][rs, cs], mask=it["mask"][rs, cs],
                    predictions={k: v[rs, cs] for k, v in it["predictions"].items()}) for it in items]
        for arm in ARMS:
            overall.setdefault(scale, {})[arm] = safe_metrics(
                np.concatenate([it["predictions"][arm].ravel() for it in sub]),
                np.concatenate([it["truth"].ravel() for it in sub]),
                np.concatenate([it["mask"].ravel() for it in sub]))
        for station in STATIONS:
            for lead in LEADS:
                ss = [it for it in sub if it["station"] == station and it["lead_minutes"] == lead]
                if not ss:
                    continue
                k = f"{scale}/{station}/+{lead}min"
                per_lead[k] = {arm: safe_metrics(
                    np.concatenate([it["predictions"][arm].ravel() for it in ss]),
                    np.concatenate([it["truth"].ravel() for it in ss]),
                    np.concatenate([it["mask"].ravel() for it in ss])) for arm in ARMS}
        for lo, hi, label in ((0., 1., "cot_0_1"), (1., 10., "cot_1_10"),
                              (10., 30., "cot_10_30"), (30., np.inf, "cot_ge30")):
            mm = [it["mask"] & (it["truth"] >= lo) & (it["truth"] < hi) for it in sub]
            strata[label] = {arm: safe_metrics(
                np.concatenate([it["predictions"][arm].ravel() for it in sub]),
                np.concatenate([it["truth"].ravel() for it in sub]),
                np.concatenate([m.ravel() for m in mm])) for arm in ARMS}

    # Paired bootstrap: resample the frozen 26 monthly contiguous two-day blocks
    # once per replicate and reuse the draw across both sites and every arm.
    rng = np.random.default_rng(SEED)
    blocks = sorted(set(month_by_date.values()))
    if len(blocks) != 26:
        raise ValueError(f"frozen validation manifest has {len(blocks)} blocks, expected 26")
    matched_blocks = sorted(set(it["block_id"] for it in items))
    block_sse_n: dict[str, dict[str, dict[str, tuple[float, int]]]] = {arm: {} for arm in ARMS}
    block_thick: dict[str, dict[str, dict[str, tuple[int, int, int, float, int]]]] = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for block in blocks:
            block_sse_n[arm][block] = {}
            block_thick[arm][block] = {}
            for station in STATIONS:
                ss = [it for it in items if it["block_id"] == block and it["station"] == station]
                if ss:
                    p = np.concatenate([it["predictions"][arm][7:10, 7:10].ravel() for it in ss]).astype(np.float64)
                    y = np.concatenate([it["truth"][7:10, 7:10].ravel() for it in ss]).astype(np.float64)
                    m = np.concatenate([it["mask"][7:10, 7:10].ravel() for it in ss]).astype(bool)
                else:
                    p = y = np.empty(0, dtype=np.float64)
                    m = np.empty(0, dtype=bool)
                m &= np.isfinite(p) & np.isfinite(y)
                e = p[m] - y[m]
                block_sse_n[arm][block][station] = (float(np.dot(e, e)), int(e.size))
                yy, pp = y[m], p[m]
                ty, py = yy >= 30., pp >= 30.
                tb = yy >= 30.
                thick_bias_sum = float(np.sum((pp[tb] - yy[tb]), dtype=np.float64))
                block_thick[arm][block][station] = (int(np.count_nonzero(ty & py)),
                    int(np.count_nonzero(~ty & py)), int(np.count_nonzero(ty & ~py)),
                    thick_bias_sum, int(tb.sum()))
    pair_names = (("cot_p_xc", "cot_p_c"), ("cot_p_xc", "cot_simvp_r"),
                  ("cot_p_xc", "cot_last"), ("cot_p_c", "cot_simvp_r"))
    diffs = {f"{c}-{b}": [] for c, b in pair_names}
    thick_diff = {f"{c}-{b}": [] for c, b in pair_names}
    thick_abs_bias_diff = {f"{c}-{b}": [] for c, b in pair_names}
    for _ in range(REPS):
        picked = rng.choice(blocks, size=len(blocks), replace=True).tolist()
        rmse_by_arm = {}
        thick_by_arm = {}
        for arm in ARMS:
            station_rmse = {}
            tp = fp = fn = bias_sum = bias_n = 0
            for station in STATIONS:
                sse = count = 0
                stp = sfp = sfn = sbias = sn = 0
                for block in picked:
                    a_sse, a_n = block_sse_n[arm][block][station]
                    sse += a_sse; count += a_n
                    t1, f1, n1, b1, bn1 = block_thick[arm][block][station]
                    stp += t1; sfp += f1; sfn += n1; sbias += b1; sn += bn1
                station_rmse[station] = np.sqrt(sse / count) if count else np.nan
                tp += stp; fp += sfp; fn += sfn; bias_sum += sbias; bias_n += sn
            rmse_by_arm[arm] = float(np.nanmean(list(station_rmse.values()))) if np.isfinite(list(station_rmse.values())).any() else np.nan
            thick_by_arm[arm] = {"csi": tp / (tp + fp + fn) if tp + fp + fn else np.nan,
                                 "abs_bias": abs(bias_sum / bias_n) if bias_n else np.nan}
        for c, b in pair_names:
            name = f"{c}-{b}"
            diffs[name].append(rmse_by_arm[c] - rmse_by_arm[b])
            thick_diff[name].append(thick_by_arm[c]["csi"] - thick_by_arm[b]["csi"])
            thick_abs_bias_diff[name].append(thick_by_arm[c]["abs_bias"] - thick_by_arm[b]["abs_bias"])

    bootstrap = {}
    station_point = {}
    for arm in ARMS:
        station_point[arm] = {}
        for station in STATIONS:
            ss = [it for it in items if it["station"] == station]
            p = np.concatenate([it["predictions"][arm][7:10, 7:10].ravel() for it in ss]).astype(np.float64)
            y = np.concatenate([it["truth"][7:10, 7:10].ravel() for it in ss]).astype(np.float64)
            m = np.concatenate([it["mask"][7:10, 7:10].ravel() for it in ss]).astype(bool)
            m &= np.isfinite(p) & np.isfinite(y)
            e = p[m] - y[m]
            station_metrics = safe_metrics(p, y, m)
            station_point[arm][station] = {"n_pixels": int(m.sum()),
                "rmse": float(np.sqrt(np.mean(e * e))) if m.any() else None,
                "thick_csi": station_metrics["thick_csi"],
                "absolute_thick_bias": abs(station_metrics["thick_bias"])
                    if station_metrics["thick_bias"] is not None else None}
    bootstrap = {}
    for c, b in pair_names:
        name = f"{c}-{b}"
        d = np.asarray(diffs[name], dtype=np.float64); d = d[np.isfinite(d)]
        dc = np.asarray(thick_diff[name], dtype=np.float64); dc = dc[np.isfinite(dc)]
        db = np.asarray(thick_abs_bias_diff[name], dtype=np.float64); db = db[np.isfinite(db)]
        if min(len(d), len(dc), len(db)) < 100:
            raise ValueError(f"too few valid bootstrap draws for {name}")
        point_rmse_c = float(np.mean([station_point[c][s]["rmse"] for s in STATIONS]))
        point_rmse_b = float(np.mean([station_point[b][s]["rmse"] for s in STATIONS]))
        point_csi_delta = overall["station_3x3"][c]["thick_csi"] - overall["station_3x3"][b]["thick_csi"]
        bias_c = overall["station_3x3"][c]["thick_bias"]
        bias_b = overall["station_3x3"][b]["thick_bias"]
        point_abs_bias_delta = (abs(bias_c) - abs(bias_b)) if bias_c is not None and bias_b is not None else None
        bootstrap[name] = {"station_equal_3x3_rmse_delta": point_rmse_c - point_rmse_b,
                           "station_equal_3x3_rmse_delta_ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))],
                           "pooled_thick_csi_delta": float(point_csi_delta),
                           "thick_csi_delta_ci95": [float(np.quantile(dc, .025)), float(np.quantile(dc, .975))],
                           "pooled_absolute_thick_bias_delta": float(point_abs_bias_delta),
                           "absolute_thick_bias_delta_ci95": [float(np.quantile(db, .025)), float(np.quantile(db, .975))],
                           "bootstrap_reps": REPS, "valid_bootstrap_reps": {"rmse": len(d), "thick_csi": len(dc), "thick_bias": len(db)},
                           "seed": SEED, "resampling_unit": "all 26 frozen monthly contiguous two-day validation blocks; target BJT month"}

    a.output.mkdir(parents=True)
    result = {"state": "COMPLETE_DIRECT_COT_CPP_REFERENCE_VALIDATION_SCORE",
              "claim_scope": "CPP is a retrieval reference, not independent truth; outer-validation development evidence only",
              "split": "hunan_monthly_block_2d_buffer1d_seed42_v1; validation only",
              "scorer_sha256": sha(Path(__file__).resolve()),
              "cpp_reference": ref_audit, "inference_manifest_sha256": sha(a.inference / "inference_manifest.json"),
              "day_assignment_sha256": sha(a.day_assignment), "prediction_index_sha256": sha(vdir / "index.csv"),
              "test_payload_opened": False, "buffer_payload_opened": False,
              "future_observed_agri_used_as_input": False, "cpp_used_as_inference_input": False,
              "matched_forecast_keys": len(items), "matched_unique_station_target_keys": len(hit_targets),
              "all_unique_station_target_keys": len(all_targets),
              "unique_station_target_key_coverage": len(hit_targets) / len(all_targets) if all_targets else None,
              "matched_bjt_blocks": len(matched_blocks), "total_frozen_bjt_blocks": len(blocks),
              "matched_blocks": matched_blocks, "overall": overall, "per_station_lead": per_lead,
              "station_equal_metrics": station_point,
              "reference_cot_strata": strata, "primary_pairwise_bootstrap": bootstrap,
              "primary_gate": {"pair": "cot_p_xc-cot_p_c", "improvement_requires": "station-equal station_3x3 CPP RMSE delta 95% CI upper < 0; thick CSI point does not decrease; absolute thick-cloud bias does not increase", "cpp_is_independent_truth": False}}
    out = a.output / "cot_cpp_reference_metrics.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"state": result["state"], "matched_forecast_keys": len(items),
                      "matched_unique_station_target_keys": len(hit_targets), "blocks": len(blocks),
                      "primary": bootstrap["cot_p_xc-cot_p_c"]}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
