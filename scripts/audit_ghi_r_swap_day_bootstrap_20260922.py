"""Day-block paired uncertainty for the fixed-head GHI RMSE differences."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main(root, run, repetitions, seed):
    with (root / "data/head_pack_trainval_20260908_v1/rows.csv").open(newline="") as stream:
        day = np.array([row["init_time_bjt"][:10] for row in csv.DictReader(stream)])
    if len(day) != 637902:
        raise RuntimeError("head pack row count drift")
    rng = np.random.default_rng(seed)
    report = {"state": "COMPLETE_FIXED_GHI_R_SWAP_DAY_BOOTSTRAP", "test_used": False,
              "unit": "BJT initialization calendar day, preserving all same-day station leads",
              "repetitions": repetitions, "seed": seed, "stations": {}}
    for station in ("Sili", "Zhujia"):
        report["stations"][station] = {}
        for group in ("AC_H", "AC_ST"):
            files = {}
            for scenario in ("original_fp32", "continued_fp32"):
                with np.load(run / f"{station}_{group}_{scenario}_validation_predictions.npz", allow_pickle=False) as source:
                    files[scenario] = {key: source[key] for key in ("pack_row", "pred_kt", "observed_ghi", "clear_sky_ghi")}
            old, new = files["original_fp32"], files["continued_fp32"]
            for key in ("pack_row", "observed_ghi", "clear_sky_ghi"):
                if not np.array_equal(old[key], new[key]):
                    raise RuntimeError("unpaired GHI input: " + key)
            valid_days, inverse = np.unique(day[old["pack_row"]], return_inverse=True)
            errors = {}
            for scenario, values in files.items():
                error = values["pred_kt"].astype(np.float64) * old["clear_sky_ghi"].astype(np.float64) - old["observed_ghi"].astype(np.float64)
                errors[scenario] = np.bincount(inverse, weights=error ** 2, minlength=len(valid_days))
            count = np.bincount(inverse, minlength=len(valid_days))
            if len(valid_days) < 30 or int(count.sum()) != len(old["pack_row"]):
                raise RuntimeError("invalid day block membership")
            baseline = np.sqrt(errors["original_fp32"].sum() / count.sum())
            candidate = np.sqrt(errors["continued_fp32"].sum() / count.sum())
            draws = rng.integers(0, len(valid_days), size=(repetitions, len(valid_days)))
            n = count[draws].sum(axis=1)
            delta = np.sqrt(errors["continued_fp32"][draws].sum(axis=1) / n) - np.sqrt(errors["original_fp32"][draws].sum(axis=1) / n)
            report["stations"][station][group] = {"days": len(valid_days), "rows": len(old["pack_row"]),
                "old_rmse_wm2": float(baseline), "new_rmse_wm2": float(candidate),
                "delta_new_minus_old_wm2": float(candidate - baseline),
                "day_bootstrap_95pct_interval_wm2": np.quantile(delta, [.025, .975]).tolist(),
                "bootstrap_positive_fraction": float((delta > 0).mean())}
    output = run / "day_block_bootstrap.json"
    if output.exists():
        raise RuntimeError("refuse to overwrite bootstrap report")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.root.resolve(), args.run.resolve(), args.repetitions, args.seed)
