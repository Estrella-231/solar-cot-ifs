"""Paired validation audit for the observed-future-AGRI→R COT ceiling diagnostic."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


STATIONS = ("sili", "zhujia")
ARMS = ("A", "AH", "AC")
CONTRASTS = (("AC", "AH"), ("AC", "A"), ("AH", "A"))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_prediction(path: Path) -> dict:
    metrics_path = path.parent / "metrics.json"
    checkpoint_path = path.parent / "best.pt"
    if not metrics_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing metrics/checkpoint next to {path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("predictions_sha256") != sha(path) or metrics.get("checkpoint_sha256") != sha(checkpoint_path):
        raise ValueError(f"hash mismatch for {path}")
    if metrics.get("oracle_only") is not True or metrics.get("test_payload_opened") is not False:
        raise ValueError(f"invalid provenance flags for {path}")
    with np.load(path, allow_pickle=False) as z:
        pred = np.asarray(z["prediction_ghi_wm2"], dtype=np.float64)
        truth = np.asarray(z["truth_ghi_wm2"], dtype=np.float64)
        keys = [json.loads(str(x)) for x in z["keys"]]
    if pred.ndim != 1 or truth.shape != pred.shape or len(keys) != len(pred):
        raise ValueError(f"invalid prediction arrays in {path}")
    if not (np.isfinite(pred).all() and np.isfinite(truth).all()):
        raise ValueError(f"non-finite prediction in {path}")
    canon = [(x["station"], x["init_time_utc"], x["target_time_utc"],
              int(x["lead_minutes"]), int(x["ghi_row_index"])) for x in keys]
    if len(set(canon)) != len(canon):
        raise ValueError(f"duplicate sample keys in {path}")
    target_days = np.asarray([
        (datetime.strptime(k[2], "%Y-%m-%d %H:%M:%S") + timedelta(hours=8)).strftime("%Y-%m-%d")
        for k in canon
    ])
    return {"pred": pred, "truth": truth, "keys": canon, "days": target_days,
            "leads": np.asarray([k[3] for k in canon], dtype=np.int16)}


def rmse(err: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(err))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261020)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pack_manifest_path = args.pack / "pack_manifest.json"
    pack_manifest = json.loads(pack_manifest_path.read_text(encoding="utf-8"))
    if pack_manifest.get("state") != "COMPLETE_ORACLE_FUTURE_COT_GHI_PACK":
        raise ValueError("oracle pack is not complete")
    if pack_manifest.get("oracle_only") is not True or pack_manifest.get("deployable") is not False:
        raise ValueError("oracle must be marked diagnostic-only")
    if pack_manifest.get("future_observed_agri_used_as_input") is not True:
        raise ValueError("oracle input provenance is not declared")
    if pack_manifest.get("future_cpp_used_as_input") is not False:
        raise ValueError("CPP cannot be an oracle GHI input")
    if pack_manifest.get("test_payload_opened") is not False or pack_manifest.get("buffer_payload_opened") is not False:
        raise ValueError("test/buffer exclusion is not certified")
    pack_sha = sha(pack_manifest_path)

    data = {}
    for seed in args.seeds:
        seed_root = args.run_root / f"seed{seed}"
        sm = json.loads((seed_root / "seed_manifest.json").read_text(encoding="utf-8"))
        if sm.get("state") != "COMPLETE_ORACLE_FUTURE_COT_GHI_SEED" or sm.get("seed") != seed:
            raise ValueError(f"seed{seed} is incomplete")
        if sm.get("pack_manifest_sha256") != pack_sha or sm.get("oracle_only") is not True:
            raise ValueError(f"seed{seed} pack/provenance mismatch")
        declared = {(x.get("station"), x.get("seed"), x.get("arm")) for x in sm.get("runs", [])}
        expected = {(station, seed, arm) for station in STATIONS for arm in ARMS}
        if declared != expected or len(sm.get("runs", [])) != len(expected):
            raise ValueError(f"seed{seed} arm/station coverage mismatch")
        for station in STATIONS:
            pair = {}
            for arm in ARMS:
                path = seed_root / station / arm / "validation_predictions.npz"
                pair[arm] = load_prediction(path)
            base = pair["A"]
            for arm in ARMS[1:]:
                if pair[arm]["keys"] != base["keys"] or not np.array_equal(pair[arm]["truth"], base["truth"]):
                    raise ValueError(f"paired keys/truth mismatch seed{seed}/{station}/{arm}")
            if any(k[0] != station for k in base["keys"]):
                raise ValueError(f"station mismatch seed{seed}/{station}")
            data[(seed, station)] = pair

    # Exact cross-seed coverage and targets are mandatory.
    for station in STATIONS:
        ref = data[(args.seeds[0], station)]["A"]
        for seed in args.seeds[1:]:
            for arm in ARMS:
                item = data[(seed, station)][arm]
                if item["keys"] != ref["keys"] or not np.array_equal(item["truth"], ref["truth"]):
                    raise ValueError(f"cross-seed keys/truth mismatch seed{seed}/{station}/{arm}")

    days = sorted(set(data[(args.seeds[0], "sili")]["A"]["days"]) |
                  set(data[(args.seeds[0], "zhujia")]["A"]["days"]))
    if len(days) != 52:
        raise ValueError(f"expected the frozen 52 target BJT dates, got {len(days)}")
    day_index = {d: i for i, d in enumerate(days)}
    metric_rows = []
    by_date = {}
    for seed in args.seeds:
        for station in STATIONS:
            base = data[(seed, station)]["A"]
            ids = np.asarray([day_index[d] for d in base["days"]], dtype=np.int32)
            for arm in ARMS:
                item = data[(seed, station)][arm]
                err = item["pred"] - item["truth"]
                metric_rows.append({"seed": seed, "station": station, "arm": arm,
                    "n": len(err), "rmse_wm2": rmse(err),
                    "mae_wm2": float(np.mean(np.abs(err))), "bias_wm2": float(np.mean(err)),
                    "first_hour_rmse_wm2": rmse(err[item["leads"] <= 60]),
                    "two_to_four_hour_rmse_wm2": rmse(err[item["leads"] > 120])})
                by_date[(seed, station, arm)] = (
                    np.bincount(ids, weights=np.square(err), minlength=len(days)),
                    np.bincount(ids, minlength=len(days)).astype(np.float64))

    per_station = {}
    equal_station = {}
    for station in STATIONS:
        per_station[station] = {}
        for arm in ARMS:
            vals = [next(r["rmse_wm2"] for r in metric_rows
                         if r["seed"] == seed and r["station"] == station and r["arm"] == arm)
                    for seed in args.seeds]
            per_station[station][arm] = float(np.mean(vals))
    for arm in ARMS:
        equal_station[arm] = float(np.mean([per_station[s][arm] for s in STATIONS]))

    rng = np.random.default_rng(args.bootstrap_seed)
    boot = {f"{l}_minus_{r}": np.empty(args.bootstrap, dtype=np.float64) for l, r in CONTRASTS}
    for b in range(args.bootstrap):
        sampled = rng.integers(0, len(days), size=len(days))
        multiplicity = np.bincount(sampled, minlength=len(days)).astype(np.float64)
        for left, right in CONTRASTS:
            station_deltas = []
            for station in STATIONS:
                seed_deltas = []
                for seed in args.seeds:
                    sse_l, n_l = by_date[(seed, station, left)]
                    sse_r, n_r = by_date[(seed, station, right)]
                    seed_deltas.append(float(np.sqrt(multiplicity @ sse_l / (multiplicity @ n_l)) -
                                             np.sqrt(multiplicity @ sse_r / (multiplicity @ n_r))))
                station_deltas.append(float(np.mean(seed_deltas)))
            boot[f"{left}_minus_{right}"][b] = float(np.mean(station_deltas))
    cis = {k: np.quantile(v, [0.025, 0.975]).tolist() for k, v in boot.items()}

    lead_rows = []
    for station in STATIONS:
        leads = data[(args.seeds[0], station)]["A"]["leads"]
        for lead in sorted(set(leads.tolist())):
            row = {"station": station, "lead_minutes": int(lead)}
            for arm in ARMS:
                scores = []
                for seed in args.seeds:
                    item = data[(seed, station)][arm]
                    err = item["pred"] - item["truth"]
                    scores.append(rmse(err[item["leads"] == lead]))
                row[f"{arm}_rmse_wm2"] = float(np.mean(scores))
            row["AC_minus_AH_wm2"] = row["AC_rmse_wm2"] - row["AH_rmse_wm2"]
            row["AC_minus_A_wm2"] = row["AC_rmse_wm2"] - row["A_rmse_wm2"]
            lead_rows.append(row)

    report = {
        "state": "COMPLETE_PAIRED_FUTURE_COT_ORACLE_VALIDATION_AUDIT",
        "interpretation": "oracle-only information ceiling; observed future AGRI-derived COT is unavailable to deployable forecasts",
        "primary_metric": "equal-station mean of seed-averaged validation GHI RMSE; negative delta favors the left arm",
        "seeds": args.seeds, "stations": list(STATIONS), "target_bjt_date_blocks": len(days),
        "bootstrap_replicates": args.bootstrap, "bootstrap_seed": args.bootstrap_seed,
        "arms": {"A": "no COT", "AH": "observed-history COT only",
                 "AC": "history COT plus observed-future-AGRI→frozen-R future COT oracle"},
        "equal_station_mean_seed_rmse_wm2": equal_station,
        "per_station_mean_seed_rmse_wm2": per_station,
        "pairwise_delta_rmse_wm2": {f"{l}_minus_{r}": equal_station[l] - equal_station[r] for l, r in CONTRASTS},
        "date_block_bootstrap_95pct_ci_wm2": cis,
        "raw_metrics": metric_rows, "per_station_lead_metrics": lead_rows,
        "first_hour_guard": {s: {arm: float(np.mean([
            next(r["first_hour_rmse_wm2"] for r in metric_rows if r["seed"] == seed and r["station"] == s and r["arm"] == arm)
            for seed in args.seeds])) for arm in ARMS} for s in STATIONS},
        "paired_key_rule": "exact (station, init, target, lead, GHI row) and truth equality across arms and seeds",
        "oracle_only": True, "deployable": False,
        "future_observed_agri_used_as_input": True, "future_cpp_used_as_input": False,
        "test_payload_opened": False, "buffer_payload_opened": False,
        "pack_manifest_sha256": pack_sha,
        "claim_boundary": "cannot be reported as operational forecast gain; a positive ceiling only localizes headroom to forecast-COT quality/alignment, while a null ceiling shifts attention to GHI fusion/representation/labels",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0]))
        writer.writeheader(); writer.writerows(metric_rows)
    print(json.dumps({k: report[k] for k in ("state", "equal_station_mean_seed_rmse_wm2",
        "pairwise_delta_rmse_wm2", "date_block_bootstrap_95pct_ci_wm2",
        "target_bjt_date_blocks", "oracle_only", "test_payload_opened")}, indent=2))


if __name__ == "__main__":
    main()
