#!/usr/bin/env python3
"""Combine three independent seed-level GPU inference audits into one receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rmse(pred: np.ndarray, truth: np.ndarray, weight: np.ndarray | None = None) -> float:
    err2 = (pred - truth) ** 2
    return float(np.sqrt(np.mean(err2) if weight is None else np.sum(weight * err2) / np.sum(weight)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-output-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=20261020)
    a = p.parse_args()
    if not a.seed_output_root.is_dir() or not a.output.is_dir():
        raise FileNotFoundError("seed staging/output root missing")

    runs: list[dict[str, object]] = []
    preds: list[dict[str, np.ndarray | str | int]] = []
    dates_all: set[str] = set()
    reference_keys: dict[str, np.ndarray] = {}
    for seed in (42, 43, 44):
        root = a.seed_output_root / f"seed{seed}"
        audit_path = root / "counterfactual_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("state") != "COMPLETE_FROZEN_FORECAST_HEAD_ORACLE_COT_INFERENCE_COUNTERFACTUAL":
            raise ValueError(f"incomplete seed audit: {audit_path}")
        if audit.get("seeds") != [seed] or audit.get("test_payload_opened") is not False or audit.get("buffer_payload_opened") is not False:
            raise ValueError(f"seed/provenance gate failed: {audit_path}")
        for station in ("sili", "zhujia"):
            pred_path = root / "predictions" / f"{station}_seed{seed}.npz"
            with np.load(pred_path, allow_pickle=False) as z:
                keys = z["keys"].astype(str)
                forecast = np.asarray(z["forecast"], dtype=np.float64)
                oracle = np.asarray(z["oracle_cot_substitution"], dtype=np.float64)
                truth = np.asarray(z["truth"], dtype=np.float64)
            if not (len(keys) == len(forecast) == len(oracle) == len(truth)):
                raise ValueError(f"prediction length mismatch: {pred_path}")
            if station in reference_keys and not np.array_equal(keys, reference_keys[station]):
                raise ValueError(f"paired validation keys differ across seeds: {pred_path}")
            reference_keys.setdefault(station, keys.copy())
            target_dates: list[str] = []
            leads: list[int] = []
            for s in keys:
                k = json.loads(s)
                dt = datetime.strptime(k["target_time_utc"], "%Y-%m-%d %H:%M:%S") + timedelta(hours=8)
                target_dates.append(dt.date().isoformat())
                leads.append(int(k["lead_minutes"]))
            ds = np.asarray(target_dates)
            lead = np.asarray(leads)
            dates_all.update(ds.tolist())
            delta = rmse(oracle, truth) - rmse(forecast, truth)
            runs.append({"station": station, "seed": seed, "n": len(truth),
                         "forecast_cot_rmse_wm2": rmse(forecast, truth),
                         "oracle_cot_substitution_rmse_wm2": rmse(oracle, truth),
                         "delta_oracle_substitution_minus_forecast_cot_wm2": delta,
                         "first_hour_forecast_rmse_wm2": rmse(forecast[lead <= 60], truth[lead <= 60]),
                         "first_hour_oracle_substitution_rmse_wm2": rmse(oracle[lead <= 60], truth[lead <= 60]),
                         "late_forecast_rmse_wm2": rmse(forecast[lead >= 120], truth[lead >= 120]),
                         "late_oracle_substitution_rmse_wm2": rmse(oracle[lead >= 120], truth[lead >= 120]),
                         "checkpoint_sha256": audit["forecast_AC_head_source_sha256"][f"{station}/seed{seed}"]})
            preds.append({"station": station, "seed": seed, "keys": keys, "dates": ds,
                          "forecast": forecast, "oracle": oracle, "truth": truth})

    dates = sorted(dates_all)
    date_to_idx = {d: i for i, d in enumerate(dates)}
    rng = np.random.default_rng(a.bootstrap_seed)
    boot = np.empty(a.bootstrap, dtype=np.float64)
    date_idx = [np.asarray([date_to_idx[d] for d in r["dates"]], dtype=np.int32) for r in preds]
    for b in range(a.bootstrap):
        counts = rng.multinomial(len(dates), np.full(len(dates), 1.0 / len(dates)))
        vals = []
        for r, di in zip(preds, date_idx):
            w = counts[di].astype(np.float64)
            if w.sum() > 0:
                vals.append(rmse(r["oracle"], r["truth"], w) - rmse(r["forecast"], r["truth"], w))
        boot[b] = float(np.mean(vals))
    point = float(np.mean([r["delta_oracle_substitution_minus_forecast_cot_wm2"] for r in runs]))
    out_pred = a.output / "predictions"
    out_pred.mkdir(exist_ok=True)
    for r in preds:
        np.savez_compressed(out_pred / f"{r['station']}_seed{r['seed']}.npz",
                            keys=r["keys"], forecast=r["forecast"],
                            oracle_cot_substitution=r["oracle"], truth=r["truth"])
    result = {"state": "COMPLETE_FROZEN_FORECAST_HEAD_ORACLE_COT_INFERENCE_COUNTERFACTUAL",
              "scope": "validation-only inference substitution; frozen forecast-COT GHI heads; future COT replaced with observed-future-AGRI to frozen-R oracle COT",
              "primary_metric": "equal mean over station×seed of paired GHI RMSE difference; negative favors oracle-COT substitution",
              "delta_oracle_substitution_minus_forecast_cot_rmse_wm2": point,
              "target_bjt_date_block_bootstrap_95pct_ci_wm2": np.quantile(boot, [0.025, 0.975]).tolist(),
              "target_bjt_dates": dates, "stations": ["sili", "zhujia"], "seeds": [42, 43, 44],
              "rows_per_station": {s: int(next(len(r["keys"]) for r in preds if r["station"] == s)) for s in ("sili", "zhujia")},
              "runs": runs, "oracle_only": True, "deployable": False,
              "test_payload_opened": False, "buffer_payload_opened": False,
              "parallel_gpu_seed_outputs": {str(seed): sha(a.seed_output_root / f"seed{seed}" / "counterfactual_audit.json") for seed in (42, 43, 44)},
              "aggregation_script_sha256": sha(Path(__file__))}
    (a.output / "counterfactual_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("state", "delta_oracle_substitution_minus_forecast_cot_rmse_wm2", "target_bjt_date_block_bootstrap_95pct_ci_wm2", "runs", "test_payload_opened")}, indent=2))


if __name__ == "__main__":
    main()
