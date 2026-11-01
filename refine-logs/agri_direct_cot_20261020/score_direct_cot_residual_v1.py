from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TMP = Path(r"C:\Users\29728\AppData\Local\Temp\direct_cot_rowbank_recovery_20261021_0530")
PACK = TMP / "results" / "direct_cot_single_frame_ghi_pack_v1"
BASE = ROOT / "results" / "direct_cot_single_frame_ghi_heads_v1_local4060"
RESIDUAL = ROOT / "results" / "direct_cot_ghi_residual_v1_local4060"
DAY = TMP / "day_assignment.csv"
OUT = RESIDUAL / "score" / "direct_cot_residual_metrics.json"
STATIONS = ("sili", "zhujia")
SEEDS = (42, 43, 44)
BLOCKS_REPS = 5000
BOOT_SEED = 20261020

by_month = defaultdict(list)
with DAY.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row["split"] == "val":
            d = datetime.strptime(row["date_BJT"], "%Y-%m-%d").date()
            by_month[d.strftime("%Y-%m")].append(d)
block_for_date = {}
for month, dates in by_month.items():
    dates = sorted(set(dates))
    if len(dates) != 2 or (dates[1] - dates[0]).days != 1:
        raise ValueError(f"invalid frozen validation block {month}: {dates}")
    for date in dates:
        block_for_date[date.isoformat()] = month
if len(by_month) != 26:
    raise ValueError(f"expected 26 validation blocks, found {len(by_month)}")
block_ids = sorted(by_month)

data = {}
for station in STATIONS:
    packdir = PACK / station / "val"
    keys = (packdir / "keys.jsonl").read_bytes()
    rows = [json.loads(x) for x in keys.decode("utf-8").splitlines() if x]
    truth = np.load(packdir / "ghi.npy", mmap_mode="r", allow_pickle=False).astype(np.float64)
    dates = [(datetime.strptime(r["target_time_utc"], "%Y-%m-%d %H:%M:%S") + timedelta(hours=8)).date().isoformat() for r in rows]
    block = np.array([block_for_date[d] for d in dates])
    lead = np.array([int(r["lead_minutes"]) for r in rows])
    if len(truth) != len(rows):
        raise ValueError(f"pack row mismatch {station}")
    data[station] = {"truth": truth, "block": block, "lead": lead,
                     "keys_sha256": hashlib.sha256(keys).hexdigest(), "methods": {}}
    for seed in SEEDS:
        cfile = RESIDUAL / station / f"seed{seed}" / "validation_predictions.npz"
        c = np.load(cfile, allow_pickle=False)
        if np.asarray(c["keys_jsonl_bytes"], dtype=np.uint8).tobytes() != keys:
            raise ValueError(f"candidate key mismatch {station}/{seed}")
        if not np.array_equal(np.asarray(c["truth_ghi_wm2"], dtype=np.float64), truth):
            raise ValueError(f"candidate truth mismatch {station}/{seed}")
        data[station]["methods"][("residual", seed)] = np.asarray(c["prediction_ghi_wm2"], dtype=np.float64)
        for arm in ("none", "simvp_r", "p_c", "p_xc"):
            pfile = BASE / f"{station}_worker" / station / arm / f"seed{seed}" / "validation_predictions.npz"
            p = np.load(pfile, allow_pickle=False)
            if np.asarray(p["keys_jsonl_bytes"], dtype=np.uint8).tobytes() != keys:
                raise ValueError(f"baseline key mismatch {station}/{arm}/{seed}")
            if not np.array_equal(np.asarray(p["truth_ghi_wm2"], dtype=np.float64), truth):
                raise ValueError(f"baseline truth mismatch {station}/{arm}/{seed}")
            data[station]["methods"][(arm, seed)] = np.asarray(p["prediction_ghi_wm2"], dtype=np.float64)

rng = np.random.default_rng(BOOT_SEED)
draws = rng.integers(0, len(block_ids), size=(BLOCKS_REPS, len(block_ids)))
weight_draws = np.stack([np.bincount(row, minlength=len(block_ids)) for row in draws]).astype(np.float64)
contrasts = (("residual", "none"), ("residual", "simvp_r"), ("residual", "p_xc"))
bands = {"all_16_leads": {s: np.ones(len(data[s]["truth"]), dtype=bool) for s in STATIONS},
         "lead_15_60": {s: (data[s]["lead"] >= 15) & (data[s]["lead"] <= 60) for s in STATIONS}}
stats = {}

def block_sse_counts(d, pred, mask):
    err2 = (pred - d["truth"]) ** 2
    sse = np.zeros(len(block_ids), dtype=np.float64)
    count = np.zeros(len(block_ids), dtype=np.float64)
    for i, b in enumerate(block_ids):
        m = (d["block"] == b) & mask
        sse[i] = err2[m].sum()
        count[i] = np.count_nonzero(m)
    return sse, count

for band, masks in bands.items():
    stats[band] = {}
    comp = {}
    for station in STATIONS:
        d = data[station]
        for seed in SEEDS:
            for method in ("residual", "none", "simvp_r", "p_xc"):
                comp[(station, method, seed)] = block_sse_counts(d, d["methods"][(method, seed)], masks[station])
    for cand, base in contrasts:
        per_station = {}
        per_seed = {}
        point_pairs = []
        for station in STATIONS:
            deltas = []
            for seed in SEEDS:
                a, n = comp[(station, cand, seed)]
                b, n0 = comp[(station, base, seed)]
                if not np.array_equal(n, n0):
                    raise ValueError("paired sample counts differ")
                delta = np.sqrt(a.sum() / n.sum()) - np.sqrt(b.sum() / n.sum())
                deltas.append(float(delta))
                point_pairs.append(float(delta))
            per_station[station] = {"seed_rmse_deltas_wm2": deltas, "point_rmse_delta_wm2": float(np.mean(deltas))}
        for seed in SEEDS:
            deltas = []
            for station in STATIONS:
                a, n = comp[(station, cand, seed)]
                b, _ = comp[(station, base, seed)]
                deltas.append(float(np.sqrt(a.sum()/n.sum()) - np.sqrt(b.sum()/n.sum())))
            per_seed[str(seed)] = float(np.mean(deltas))
        point = float(np.mean(point_pairs))
        boot = np.zeros(BLOCKS_REPS, dtype=np.float64)
        for j, w in enumerate(weight_draws):
            ds = []
            for station in STATIONS:
                for seed in SEEDS:
                    a, n = comp[(station, cand, seed)]
                    b, _ = comp[(station, base, seed)]
                    ds.append(float(np.sqrt((a @ w)/(n @ w)) - np.sqrt((b @ w)/(n @ w))))
            boot[j] = np.mean(ds)
        lobo = []
        for omitted in range(len(block_ids)):
            w = np.ones(len(block_ids), dtype=np.float64)
            w[omitted] = 0.0
            ds = []
            for station in STATIONS:
                for seed in SEEDS:
                    a, n = comp[(station, cand, seed)]
                    b, _ = comp[(station, base, seed)]
                    ds.append(float(np.sqrt((a @ w)/(n @ w)) - np.sqrt((b @ w)/(n @ w))))
            lobo.append(float(np.mean(ds)))
        stats[band][f"{cand}_minus_{base}"] = {
            "station_equal_seed_equal_point_delta_rmse_wm2": point,
            "paired_date_block_bootstrap_ci95_wm2": [float(x) for x in np.quantile(boot, [.025, .975])],
            "per_station": per_station, "per_seed_region_equal": per_seed,
            "seeds_improved": int(sum(v < 0 for v in per_seed.values())), "seeds_total": len(SEEDS),
            "leave_one_block_out_all_negative": bool(all(x < 0 for x in lobo)),
            "leave_one_block_out_min_max_wm2": [float(min(lobo)), float(max(lobo))],
        }

p95_by_contrast = {}
for cand, base in contrasts[:2]:
    ds = []
    for station in STATIONS:
        for seed in SEEDS:
            truth = data[station]["truth"]
            ec = np.abs(data[station]["methods"][(cand, seed)] - truth)
            eb = np.abs(data[station]["methods"][(base, seed)] - truth)
            ds.append(float(np.quantile(ec, .95) - np.quantile(eb, .95)))
    p95_by_contrast[f"{cand}_minus_{base}"] = float(np.mean(ds))

receipt = json.loads((RESIDUAL / "run_manifest.json").read_text(encoding="utf-8"))
predeclared = {}
for key in ("residual_minus_none", "residual_minus_simvp_r"):
    allm = stats["all_16_leads"][key]
    short = stats["lead_15_60"][key]
    predeclared[key] = {
        "full_region_delta_negative_ci_upper_below_zero": allm["station_equal_seed_equal_point_delta_rmse_wm2"] < 0 and allm["paired_date_block_bootstrap_ci95_wm2"][1] < 0,
        "short_region_delta_negative_ci_upper_below_zero": short["station_equal_seed_equal_point_delta_rmse_wm2"] < 0 and short["paired_date_block_bootstrap_ci95_wm2"][1] < 0,
        "both_station_full_point_non_degrading": all(v["point_rmse_delta_wm2"] <= 0 for v in allm["per_station"].values()),
        "both_station_short_point_non_degrading": all(v["point_rmse_delta_wm2"] <= 0 for v in short["per_station"].values()),
        "at_least_two_of_three_seed_directions_improve_full": allm["seeds_improved"] >= 2,
        "at_least_two_of_three_seed_directions_improve_short": short["seeds_improved"] >= 2,
        "leave_one_block_out_all_negative_full": allm["leave_one_block_out_all_negative"],
        "leave_one_block_out_all_negative_short": short["leave_one_block_out_all_negative"],
        "p95_absolute_error_guard_pass": p95_by_contrast[key] <= 5.0,
    }
    predeclared[key]["all_predeclared_gates_pass"] = all(predeclared[key].values())

score = {
    "state": "COMPLETE_DIRECT_COT_RESIDUAL_VALIDATION_SCORE",
    "claim_scope": "outer-validation development evidence; no independent-test claim",
    "candidate_run_manifest_sha256": hashlib.sha256((RESIDUAL / "run_manifest.json").read_bytes()).hexdigest(),
    "pack_manifest_sha256": json.loads((RESIDUAL / "run_manifest.json").read_text(encoding="utf-8"))["pack_manifest_sha256"],
    "day_assignment_sha256": hashlib.sha256(DAY.read_bytes()).hexdigest(),
    "bootstrap": {"repetitions": BLOCKS_REPS, "seed": BOOT_SEED, "block_count": len(block_ids), "resampling_unit": "frozen monthly consecutive two-day validation blocks with replacement; multiplicity retained"},
    "contrasts": stats,
    "p95_absolute_error_delta_vs_base_wm2": p95_by_contrast,
    "predeclared_gates": predeclared,
    "test_payload_opened": False,
    "buffer_payload_opened": False,
    "future_observed_agri_used_as_input": False,
    "cpp_used_as_input": False,
    "keys_sha256": {s: data[s]["keys_sha256"] for s in STATIONS},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(score, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps({"state": score["state"], "contrasts": stats, "p95": p95_by_contrast, "gates": predeclared}, indent=2, allow_nan=False))
