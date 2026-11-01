#!/usr/bin/env python3
"""Paired validation scoring for the four causal forecast-COT GHI arms."""
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
ARMS = ("none", "simvp_r", "p_c", "p_xc")
SEEDS = (42, 43, 44)
LEADS = tuple(range(15, 241, 15))
BANDS = {"all_16_leads": lambda lead: True,
         "lead_15_60": lambda lead: 15 <= lead <= 60,
         "lead_75_120": lambda lead: 75 <= lead <= 120,
         "lead_135_240": lambda lead: 135 <= lead <= 240}
PAIRWISE = (("p_xc", "none"), ("p_xc", "simvp_r"), ("p_xc", "p_c"))
REPS = 5000
SEED = 20261020
HIGH_GHI_THRESHOLD = 800.0


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_bytes(raw: np.ndarray) -> tuple[bytes, list[dict]]:
    data = np.asarray(raw, dtype=np.uint8).tobytes()
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line]
    return data, rows


def val_blocks(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_month: dict[str, list[datetime]] = defaultdict(list)
    for r in rows:
        if r["split"] == "val":
            d = datetime.strptime(r["date_BJT"], "%Y-%m-%d")
            by_month[d.strftime("%Y-%m")].append(d)
    out = {}
    for month, ds in by_month.items():
        ds = sorted(set(ds))
        if len(ds) != 2 or (ds[1] - ds[0]).days != 1:
            raise ValueError(f"validation block is not a consecutive two-day pair: {month} {ds}")
        out.update({d.date().isoformat(): month for d in ds})
    if len(by_month) != 26:
        raise ValueError(f"expected 26 frozen monthly validation blocks; found {len(by_month)}")
    return out


def parse_target(row: dict) -> tuple[datetime, str]:
    t = datetime.strptime(row["target_time_utc"], "%Y-%m-%d %H:%M:%S")
    date = (t + timedelta(hours=8)).date().isoformat()
    return t, date


def metric(pred: np.ndarray, truth: np.ndarray) -> dict:
    if pred.shape != truth.shape or pred.ndim != 1 or not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("GHI prediction/truth vectors must be paired finite vectors")
    if len(truth) == 0:
        return {"n": 0, "rmse_wm2": None, "mae_wm2": None, "bias_wm2": None,
                "high_ghi_threshold_wm2": HIGH_GHI_THRESHOLD, "high_ghi_n": 0,
                "high_ghi_rmse_wm2": None, "high_ghi_bias_wm2": None,
                "high_ghi_underestimate_fraction": None}
    err = pred.astype(np.float64) - truth.astype(np.float64)
    high = truth >= HIGH_GHI_THRESHOLD
    return {"n": int(len(truth)), "rmse_wm2": float(np.sqrt(np.mean(err * err))),
            "mae_wm2": float(np.mean(np.abs(err))), "bias_wm2": float(np.mean(err)),
            "high_ghi_threshold_wm2": HIGH_GHI_THRESHOLD,
            "high_ghi_n": int(high.sum()),
            "high_ghi_rmse_wm2": float(np.sqrt(np.mean(err[high] ** 2))) if high.any() else None,
            "high_ghi_bias_wm2": float(np.mean(err[high])) if high.any() else None,
            "high_ghi_underestimate_fraction": float(np.mean(err[high] < 0)) if high.any() else None}


def block_weighted_rmse(draw_weights: np.ndarray, block_sse: np.ndarray,
                        block_n: np.ndarray) -> np.ndarray:
    """RMSE for with-replacement block draws; repeated blocks keep their weight."""
    numerator = draw_weights @ block_sse
    denominator = draw_weights @ block_n
    out = np.full(np.shape(numerator), np.nan, dtype=np.float64)
    valid = denominator > 0
    out[valid] = np.sqrt(numerator[valid] / denominator[valid])
    return out


def load_station(station: str, worker: Path, pack: Path, pack_sha: str) -> dict:
    mpath = worker / "run_manifest.json"
    m = read_json(mpath)
    if m.get("state") != "COMPLETE_DIRECT_COT_SINGLE_FRAME_GHI_HEADS":
        raise ValueError(f"incomplete GHI worker {station}")
    if m.get("stations") != [station] or m.get("arms") != list(ARMS) or m.get("seeds") != list(SEEDS):
        raise ValueError(f"GHI worker contract mismatch for {station}")
    if m.get("epochs") != 30 or m.get("effective_batch") != 512 or m.get("pack_manifest_sha256") != pack_sha:
        raise ValueError(f"fixed budget or pack hash mismatch for {station}")
    for field in ("test_payload_opened", "buffer_payload_opened", "future_observed_agri_used_as_input", "cpp_used_as_input"):
        if m.get(field) is not False:
            raise ValueError(f"GHI worker isolation failed: {station}/{field}")
    expected_keys = (pack / station / "val" / "keys.jsonl").read_bytes()
    result = {}
    for arm in ARMS:
        for seed in SEEDS:
            run = worker / station / arm / f"seed{seed}"
            metrics = read_json(run / "metrics.json")
            with np.load(run / "validation_predictions.npz", allow_pickle=False) as z:
                pred = np.asarray(z["prediction_ghi_wm2"], np.float64)
                truth = np.asarray(z["truth_ghi_wm2"], np.float64)
                key_bytes, keys = read_jsonl_bytes(z["keys_jsonl_bytes"])
            if key_bytes != expected_keys or pred.shape != truth.shape or len(keys) != len(truth):
                raise ValueError(f"validation predictions do not match exact frozen V keys: {station}/{arm}/{seed}")
            if metrics.get("prediction_sha256") != sha(run / "validation_predictions.npz"):
                raise ValueError(f"prediction SHA mismatch: {station}/{arm}/{seed}")
            if metrics.get("checkpoint_sha256") != sha(run / "epoch_030.pt"):
                raise ValueError(f"checkpoint SHA mismatch: {station}/{arm}/{seed}")
            if metrics.get("n_val") != len(truth) or metrics.get("epochs") != 30 or metrics.get("updates", 0) <= 0:
                raise ValueError(f"training receipt mismatch: {station}/{arm}/{seed}")
            result[(arm, seed)] = {"pred": pred, "truth": truth, "keys": keys,
                                   "metrics": metrics, "run_dir": str(run)}
    # One paired initialization per station and seed across the four arms.
    for seed in SEEDS:
        hashes = [m["initial_state_sha256"].get(f"{station}/{seed}/{arm}") for arm in ARMS]
        if not hashes[0] or len(set(hashes)) != 1:
            raise ValueError(f"paired initial-state hashes differ for {station}/seed{seed}: {hashes}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--sili-worker", type=Path, required=True)
    ap.add_argument("--zhujia-worker", type=Path, required=True)
    ap.add_argument("--day-assignment", type=Path, required=True)
    ap.add_argument("--cpp-pack", type=Path, required=True)
    ap.add_argument("--exclusions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    pm = read_json(a.pack / "pack_manifest.json")
    if pm.get("state") != "COMPLETE_DIRECT_COT_SINGLE_FRAME_GHI_PACK":
        raise ValueError("forecast-COT GHI pack incomplete")
    for f in ("test_payload_opened", "buffer_payload_opened", "future_observed_agri_used_as_input", "cpp_used_as_input"):
        if pm.get(f) is not False:
            raise ValueError(f"GHI pack isolation failed: {f}")
    pack_sha = sha(a.pack / "pack_manifest.json")
    workers = {"sili": load_station("sili", a.sili_worker, a.pack, pack_sha),
               "zhujia": load_station("zhujia", a.zhujia_worker, a.pack, pack_sha)}
    date_to_block = val_blocks(a.day_assignment)
    exclusion_obj = read_json(a.exclusions)
    if exclusion_obj.get("date_timezone") != "Asia/Shanghai":
        raise ValueError("station-day exclusion timezone mismatch")
    exclusions = {(r["station"].lower(), r["bjt_date"]) for r in exclusion_obj["exclusions"]}
    cpp_status = read_json(a.cpp_pack / "pack_status.json")
    cpp_audit = read_json(a.cpp_pack / "pack_audit.json")
    if cpp_status.get("state") != "COMPLETE_MONTHLY_R_STATION16_PACK" or cpp_status.get("test_payloads_read") != 0:
        raise ValueError("CPP reference pack status/isolation gate failed")
    if cpp_audit.get("state") != "PASS_MONTHLY_R_STATION16_PACK_AUDIT" or cpp_audit.get("test_payloads_read") != 0:
        raise ValueError("CPP reference pack audit/isolation gate failed")
    if cpp_audit.get("claim_scope") != "CPP-supported daytime retrieval-reference pixels; not independent truth":
        raise ValueError("CPP reference claim scope mismatch")
    for name, filename in (("cot", "cot.npy"), ("mask", "mask.npy")):
        if sha(a.cpp_pack / filename) != cpp_audit["arrays"][name]["sha256"]:
            raise ValueError(f"CPP {filename} SHA mismatch against immutable audit")
    key_meta = {}
    cpp_meta = {}
    cpp_rows = []
    with (a.cpp_pack / "rows.csv").open(newline="", encoding="utf-8") as f:
        cpp_rows = list(csv.DictReader(f))
    # np.load .npy returns arrays, not an NpzFile; use memory maps.
    cpp_cot = np.load(a.cpp_pack / "cot.npy", mmap_mode="r", allow_pickle=False)
    cpp_mask = np.load(a.cpp_pack / "mask.npy", mmap_mode="r", allow_pickle=False)
    ref = {}
    for r in cpp_rows:
        if r["split"] == "val" and r["state"] == "SUPERVISED":
            k = (r["station"].lower(), r["target_time_utc"])
            if k in ref:
                raise ValueError(f"duplicate CPP station/time key {k}")
            idx = int(r["array_index"])
            ref[k] = (float(cpp_cot[idx, 0, 8, 8]), bool(cpp_mask[idx, 0, 8, 8]))

    # Common keys/truth and frozen BJT block assignment.
    for station in STATIONS:
        ref_arm = workers[station][("none", SEEDS[0])]
        keys = ref_arm["keys"]
        y = ref_arm["truth"]
        if len(keys) != len(y):
            raise ValueError(f"key/label count mismatch {station}")
        dates, blocks, leads = [], [], []
        for i, row in enumerate(keys):
            if row.get("station", "").lower() != station:
                raise ValueError(f"station mismatch in saved key {station}/{row}")
            _, date = parse_target(row)
            block = date_to_block.get(date)
            if block is None:
                raise ValueError(f"V target date not in frozen date blocks: {date}")
            if (station, date) in exclusions:
                raise ValueError(f"excluded station-day remains in forecast GHI validation keys: {station}/{date}")
            lead = int(row["lead_minutes"])
            if lead not in LEADS:
                raise ValueError(f"invalid lead in saved key: {lead}")
            dates.append(date); blocks.append(block); leads.append(lead)
        # Check each complete method vector once, outside the per-row loop.
        # Doing this per row would repeatedly compare whole 30k-row lists.
        for arm in ARMS:
            for seed in SEEDS:
                other = workers[station][(arm, seed)]
                if other["keys"] != keys or not np.array_equal(other["truth"], y):
                    raise ValueError(f"methods do not share exact validation keys/truth: {station}/{arm}/{seed}")
        # Record one CPP center-pixel thickness label where available; missing CPP
        # remains in the main GHI scores and only leaves CPP-stratified tables.
        cpp_bin = []
        for row in keys:
            pair = ref.get((station, row["target_time_utc"]))
            if pair is None or not pair[1]:
                cpp_bin.append(None)
            else:
                c = pair[0]
                cpp_bin.append("cot_0_1" if c < 1 else "cot_1_10" if c < 10 else "cot_10_30" if c < 30 else "cot_ge30")
        key_meta[station] = {"dates": np.asarray(dates), "blocks": np.asarray(blocks),
                             "leads": np.asarray(leads), "cpp_bin": cpp_bin}
    all_blocks = sorted(set(date_to_block.values()))
    if len(all_blocks) != 26:
        raise ValueError("frozen validation block count changed")

    # Exact point estimates for every station/arm/seed and lead band.
    detail = {}
    for station in STATIONS:
        km = key_meta[station]
        detail[station] = {}
        for arm in ARMS:
            detail[station][arm] = {}
            for seed in SEEDS:
                rec = workers[station][(arm, seed)]
                detail[station][arm][str(seed)] = {"all": metric(rec["pred"], rec["truth"])}
                for band, selector in BANDS.items():
                    sel = np.asarray([selector(x) for x in km["leads"]], dtype=bool)
                    detail[station][arm][str(seed)][band] = metric(rec["pred"][sel], rec["truth"][sel])
                high = rec["truth"] >= HIGH_GHI_THRESHOLD
                detail[station][arm][str(seed)]["high_ghi_ge_800"] = metric(rec["pred"][high], rec["truth"][high]) if high.any() else None
                by_cpp = {}
                for cat in ("cot_0_1", "cot_1_10", "cot_10_30", "cot_ge30"):
                    sel = np.asarray([x == cat for x in km["cpp_bin"]], dtype=bool)
                    by_cpp[cat] = metric(rec["pred"][sel], rec["truth"][sel]) if sel.any() else {"n": 0, "rmse_wm2": None, "mae_wm2": None, "bias_wm2": None}
                by_cpp["cpp_missing_or_invalid"] = int(sum(x is None for x in km["cpp_bin"]))
                detail[station][arm][str(seed)]["by_cpp_center_bin"] = by_cpp

    rng = np.random.default_rng(SEED)
    draw_blocks = [rng.choice(all_blocks, size=len(all_blocks), replace=True).tolist() for _ in range(REPS)]
    block_to_col = {b: i for i, b in enumerate(all_blocks)}
    draw_weights = np.zeros((REPS, len(all_blocks)), dtype=np.int16)
    for i, sampled in enumerate(draw_blocks):
        for block in sampled:
            draw_weights[i, block_to_col[block]] += 1

    # Pre-aggregate squared errors and counts. This keeps bootstrap duplicates
    # as multiplicities and avoids rescanning all validation rows per draw.
    block_stats = {}
    for station in STATIONS:
        km = key_meta[station]
        row_block = np.asarray([block_to_col[b] for b in km["blocks"]], dtype=np.int64)
        for band in ("all_16_leads", "lead_15_60"):
            row_band = np.ones(len(row_block), dtype=bool) if band == "all_16_leads" else np.asarray(
                [BANDS[band](x) for x in km["leads"]], dtype=bool)
            use_block = row_block[row_band]
            for arm in ARMS:
                for seed in SEEDS:
                    rec = workers[station][(arm, seed)]
                    error = rec["pred"].astype(np.float64) - rec["truth"].astype(np.float64)
                    block_sse = np.bincount(use_block, weights=error[row_band] ** 2,
                                            minlength=len(all_blocks)).astype(np.float64)
                    block_n = np.bincount(use_block, minlength=len(all_blocks)).astype(np.float64)
                    block_stats[(station, band, arm, seed)] = (block_sse, block_n)

    contrasts = {}
    for cand, base in PAIRWISE:
        name = f"{cand}-{base}"
        # Point and paired bootstrap distributions: each station and seed is
        # scored separately; station and seed are averaged with equal weight.
        point_by_station = {}
        point_by_seed = {}
        for station in STATIONS:
            point_by_station[station] = float(np.mean([
                detail[station][cand][str(seed)]["all"]["rmse_wm2"] -
                detail[station][base][str(seed)]["all"]["rmse_wm2"] for seed in SEEDS]))
        for seed in SEEDS:
            point_by_seed[str(seed)] = float(np.mean([
                detail[station][cand][str(seed)]["all"]["rmse_wm2"] -
                detail[station][base][str(seed)]["all"]["rmse_wm2"] for station in STATIONS]))
        point_macro = float(np.mean(list(point_by_station.values())))
        point_short_site = {}
        for station in STATIONS:
            point_short_site[station] = float(np.mean([
                detail[station][cand][str(seed)]["lead_15_60"]["rmse_wm2"] -
                detail[station][base][str(seed)]["lead_15_60"]["rmse_wm2"] for seed in SEEDS]))
        point_short = float(np.mean(list(point_short_site.values())))

        draws_station, draws_short_site = {}, {}
        for station in STATIONS:
            for band, target in (("all_16_leads", draws_station), ("lead_15_60", draws_short_site)):
                per_seed_delta = []
                for seed in SEEDS:
                    c_sse, c_n = block_stats[(station, band, cand, seed)]
                    b_sse, b_n = block_stats[(station, band, base, seed)]
                    per_seed_delta.append(block_weighted_rmse(draw_weights, c_sse, c_n) -
                                          block_weighted_rmse(draw_weights, b_sse, b_n))
                stack = np.stack(per_seed_delta, axis=0)
                count = np.isfinite(stack).sum(axis=0)
                station_delta = np.full(REPS, np.nan, dtype=np.float64)
                valid = count > 0
                station_delta[valid] = np.nansum(stack[:, valid], axis=0) / count[valid]
                target[station] = station_delta
        stack_macro = np.stack([draws_station[s] for s in STATIONS], axis=0)
        valid_macro = np.isfinite(stack_macro).all(axis=0)
        draws_macro = np.full(REPS, np.nan, dtype=np.float64)
        draws_macro[valid_macro] = stack_macro[:, valid_macro].mean(axis=0)
        stack_short = np.stack([draws_short_site[s] for s in STATIONS], axis=0)
        valid_short = np.isfinite(stack_short).all(axis=0)
        draws_short = np.full(REPS, np.nan, dtype=np.float64)
        draws_short[valid_short] = stack_short[:, valid_short].mean(axis=0)
        dm = draws_macro[np.isfinite(draws_macro)]
        ds = draws_short[np.isfinite(draws_short)]
        station_ci = {}
        for station in STATIONS:
            q = draws_station[station][np.isfinite(draws_station[station])]
            station_ci[station] = {"point_rmse_delta_wm2": point_by_station[station],
                                   "ci95": [float(np.quantile(q, .025)), float(np.quantile(q, .975))]}
        short_station_ci = {}
        for station in STATIONS:
            q = draws_short_site[station][np.isfinite(draws_short_site[station])]
            short_station_ci[station] = {"point_rmse_delta_wm2": point_short_site[station],
                                         "ci95": [float(np.quantile(q, .025)), float(np.quantile(q, .975))]}

        # Leave-one-validation-block-out direction and paired 95th-percentile
        # absolute-error movement; neither changes the preselected models.
        lobo = []
        for omitted in all_blocks:
            station_seed = []
            for station in STATIONS:
                km = key_meta[station]
                keep = km["blocks"] != omitted
                for seed in SEEDS:
                    pc = workers[station][(cand, seed)]["pred"][keep]
                    pb = workers[station][(base, seed)]["pred"][keep]
                    yy = workers[station][(base, seed)]["truth"][keep]
                    station_seed.append(float(np.sqrt(np.mean((pc - yy) ** 2)) - np.sqrt(np.mean((pb - yy) ** 2))))
            lobo.append(float(np.mean(station_seed)))
        p95_delta = []
        for station in STATIONS:
            for seed in SEEDS:
                yc = workers[station][(cand, seed)]["truth"]
                ec = np.abs(workers[station][(cand, seed)]["pred"] - yc)
                eb = np.abs(workers[station][(base, seed)]["pred"] - yc)
                p95_delta.append(float(np.quantile(ec, .95) - np.quantile(eb, .95)))
        seed_wins = sum(point_by_seed[str(seed)] < 0 for seed in SEEDS)
        contrasts[name] = {"point_station_equal_mean_rmse_delta_wm2": point_macro,
                           "paired_date_block_bootstrap_ci95_wm2": [float(np.quantile(dm, .025)), float(np.quantile(dm, .975))],
                           "per_station": station_ci,
                           "lead_15_60_point_delta_wm2": point_short,
                           "lead_15_60_ci95_wm2": [float(np.quantile(ds, .025)), float(np.quantile(ds, .975))],
                           "lead_15_60_per_station": short_station_ci,
                           "per_seed_station_equal_delta_wm2": point_by_seed,
                           "seeds_improved": int(seed_wins), "seeds_total": len(SEEDS),
                           "leave_one_block_out_delta_range_wm2": [float(min(lobo)), float(max(lobo))],
                           "leave_one_block_out_all_negative": bool(all(x < 0 for x in lobo)),
                           "p95_abs_error_delta_mean_over_station_seed_wm2": float(np.mean(p95_delta)),
                           "p95_abs_error_delta_by_station_seed_wm2": p95_delta,
                           "valid_bootstrap_reps": int(len(dm)), "bootstrap_reps": REPS,
                           "bootstrap_seed": SEED, "resampling_unit": "26 frozen monthly contiguous two-day validation blocks by target BJT date",
                           "bootstrap_method": "with replacement; repeated block selections retain multiplicity"}

    # Explicit CPP reference coverage: missing/invalid CPP rows stay in all-GHI metrics.
    cpp_counts = {s: {"n_val_rows": len(key_meta[s]["cpp_bin"]),
                      "cpp_center_valid_rows": sum(x is not None for x in key_meta[s]["cpp_bin"]),
                      "cpp_center_missing_or_invalid_rows": sum(x is None for x in key_meta[s]["cpp_bin"])} for s in STATIONS}
    gates = {}
    for name in ("p_xc-none", "p_xc-simvp_r"):
        c = contrasts[name]
        gates[name] = {"primary_mean_gain_and_ci_pass": c["point_station_equal_mean_rmse_delta_wm2"] < 0 and c["paired_date_block_bootstrap_ci95_wm2"][1] < 0,
                       "no_station_point_degradation": all(v["point_rmse_delta_wm2"] <= 0 for v in c["per_station"].values()),
                       "no_short_lead_point_degradation": c["lead_15_60_point_delta_wm2"] <= 0,
                       "at_least_two_seeds_improve": c["seeds_improved"] >= 2,
                       "leave_one_block_out_direction_pass": c["leave_one_block_out_all_negative"],
                       "p95_absolute_error_guard_pass": c["p95_abs_error_delta_mean_over_station_seed_wm2"] <= 5.0}
        gates[name]["all_predeclared_ghi_gates_pass"] = all(gates[name].values())

    result = {"state": "COMPLETE_DIRECT_COT_GHI_VALIDATION_SCORE",
              "claim_scope": "outer-validation development evidence; not independent test; COT inputs are predicted without future observed AGRI/CPP",
              "split": "hunan_monthly_block_2d_buffer1d_seed42_v1; H-trained heads, V-only score",
              "pack_manifest_sha256": pack_sha,
              "sili_worker_manifest_sha256": sha(a.sili_worker / "run_manifest.json"),
              "zhujia_worker_manifest_sha256": sha(a.zhujia_worker / "run_manifest.json"),
              "day_assignment_sha256": sha(a.day_assignment),
              "exclusion_policy_sha256": sha(a.exclusions),
              "cpp_pack_audit_sha256": sha(a.cpp_pack / "pack_audit.json"),
              "bootstrap_implementation": "preaggregated block SSE/count weighted by multinomial resample counts; exact with-replacement multiplicity",
              "test_payload_opened": False, "buffer_payload_opened": False,
              "future_observed_agri_used_as_input": False, "cpp_used_as_input": False,
              "cpp_center_reference_coverage": cpp_counts,
              "point_metrics": detail, "primary_contrasts": contrasts, "predeclared_gates": gates}
    a.output.mkdir(parents=True)
    out = a.output / "direct_cot_ghi_metrics.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"state": result["state"], "contrasts": {
        k: {"point": v["point_station_equal_mean_rmse_delta_wm2"],
            "ci95": v["paired_date_block_bootstrap_ci95_wm2"],
            "gates": gates.get(k)} for k, v in contrasts.items()}}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
