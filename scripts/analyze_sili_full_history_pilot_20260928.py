"""Analyze the matched Sili full-history COT pilot and paired day blocks."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def bootstrap_rmse(snapshot_a, method_a, snapshot_b, method_b, scope, regime, reps=5000, seed=20260928):
    days = sorted(set(snapshot_a["day_sums"]) & set(snapshot_b["day_sums"]))
    def vectors(snapshot, method):
        sums = [snapshot["day_sums"][day][scope][method]["_by_regime"][regime] for day in days]
        return np.asarray([x["sq"] for x in sums], dtype=np.float64), np.asarray([x["n"] for x in sums], dtype=np.float64)
    sq_a, n_a = vectors(snapshot_a, method_a)
    sq_b, n_b = vectors(snapshot_b, method_b)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(days), size=(reps, len(days)))
    a = np.sqrt(sq_a[draws].sum(axis=1) / n_a[draws].sum(axis=1))
    b = np.sqrt(sq_b[draws].sum(axis=1) / n_b[draws].sum(axis=1))
    delta = a - b
    return {"days": len(days), "resamples": reps, "seed": seed,
            "ci95_rmse_difference": [float(x) for x in np.quantile(delta, [.025, .975])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    complete = json.loads((args.run / "training_complete.json").read_text(encoding="utf-8"))
    if complete.get("state") != "COMPLETE_SILI_FULL_HISTORY_PILOT" or complete.get("test_used") is not False:
        raise RuntimeError("Unexpected or test-contaminated run")
    profiles = json.loads((args.run / "batch_profile.json").read_text(encoding="utf-8"))
    arms = {}
    for arm in ("short_history", "full_history"):
        summary = json.loads((args.run / arm / "training_complete.json").read_text(encoding="utf-8"))
        snaps = read_lines(args.run / arm / "day_metrics.jsonl")
        epochs = read_lines(args.run / arm / "epochs.jsonl")
        best = min(epochs, key=lambda row: row["metrics"]["patch"]["learned"]["thick_rmse"])
        best_snap = next(x for x in snaps if x["update"] == best["update"])
        step0 = next(x for x in snaps if x["update"] == 0)
        selected_snap = best_snap if summary["selected_candidate"] != "step0_zero_residual_transport" else step0
        arms[arm] = {"summary": summary, "snapshots": snaps, "epochs": epochs,
                     "best_learned": best, "best_learned_snapshot": best_snap,
                     "step0": step0, "selected": selected_snap}

    step0_transport = arms["short_history"]["step0"]["metrics"]["patch"]["transport"]
    out = {
        "state": "COMPLETE_SILI_FULL_HISTORY_ANALYSIS",
        "test_used": False,
        "pbs_job": "211083.tc6000",
        "pbs_exit_status": 0,
        "reference": "CPP COT retrieval; not independent cloud truth",
        "support": "CPP-valid intersected with seven-pair C13 transport in-domain support",
        "overlapping_validation_windows": True,
        "matched_input_init_max_abs_diff": complete["input_function_initialization_max_abs_diff"],
        "batch_profile": profiles,
        "transport_baseline_patch": step0_transport,
        "arms": {},
        "paired_day_bootstrap_95pct": {},
        "limitations": [
            "Checkpoint selection and these intervals reuse the fixed validation bank; intervals are descriptive and post-selection.",
            "Initialization-day blocks reduce within-day overlap dependence, but sequences can overlap across adjacent days.",
            "CPP is a satellite retrieval reference, not independent in-situ cloud truth.",
            "No GHI head or test split was used.",
        ],
    }
    for arm, item in arms.items():
        metrics = item["selected"]["metrics"]
        bestm = item["best_learned"]["metrics"]
        out["arms"][arm] = {
            "selected_candidate": item["summary"]["selected_candidate"],
            "selected_update": item["summary"]["selected_update"],
            "updates_run": item["summary"]["updates"],
            "selected_metrics": metrics,
            "best_unselected_learned_update_by_patch_thick_rmse": item["best_learned"]["update"],
            "best_unselected_learned_patch": bestm["patch"]["learned"],
            "best_unselected_learned_station": bestm["station"]["learned"],
            "best_unselected_learned_patch_thick_rmse_delta_vs_transport": bestm["patch"]["learned"]["thick_rmse"] - step0_transport["thick_rmse"],
        }
        for scope in ("patch", "station"):
            out["paired_day_bootstrap_95pct"][f"{arm}_best_learned_minus_transport_{scope}_thick"] = bootstrap_rmse(
                item["best_learned_snapshot"], "learned", item["best_learned_snapshot"], "transport", scope, "thick")
    full = arms["full_history"]["best_learned_snapshot"]
    short = arms["short_history"]["best_learned_snapshot"]
    for scope in ("patch", "station"):
        out["paired_day_bootstrap_95pct"][f"full_best_learned_minus_short_best_learned_{scope}_thick"] = bootstrap_rmse(
            full, "learned", short, "learned", scope, "thick")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "analysis.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    colors = {"short_history": "#657f8c", "full_history": "#159487"}
    labels = {"short_history": "Last COT + two-frame trend", "full_history": "All eight history maps"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for arm, item in arms.items():
        epochs = item["epochs"]
        x = [0] + [row["update"] for row in epochs]
        thick = [item["step0"]["metrics"]["patch"]["transport"]["thick_rmse"]] + [row["metrics"]["patch"]["learned"]["thick_rmse"] for row in epochs]
        all_rmse = [item["step0"]["metrics"]["patch"]["transport"]["rmse"]] + [row["metrics"]["patch"]["learned"]["rmse"] for row in epochs]
        axes[0].plot(x, thick, marker="o", ms=3.5, lw=1.8, color=colors[arm], label=labels[arm])
        axes[1].plot(x, all_rmse, marker="o", ms=3.5, lw=1.8, color=colors[arm], label=labels[arm])
    axes[0].axhline(step0_transport["thick_rmse"], color="#222222", ls="--", label="Step-0 transport")
    axes[1].axhline(step0_transport["rmse"], color="#222222", ls="--", label="Step-0 transport")
    axes[0].set_title("Patch RMSE on CPP COT ≥ 30")
    axes[1].set_title("Patch RMSE on all common-support pixels")
    for ax in axes:
        ax.set_xlabel("Optimizer update")
        ax.set_ylabel("COT RMSE")
        ax.grid(alpha=.25)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Sili COT history input ablation — fixed validation bank")
    fig.tight_layout()
    fig.savefig(args.output / "history_ablation_validation_curves.png", dpi=180)
    fig.savefig(args.output / "history_ablation_validation_curves.pdf")
    plt.close(fig)
    print(json.dumps({arm: {"selected": item["summary"]["selected_candidate"],
                            "best_unselected_update": item["best_learned"]["update"],
                            "best_patch_thick_rmse": item["best_learned"]["metrics"]["patch"]["learned"]["thick_rmse"]}
                      for arm, item in arms.items()}, indent=2))


if __name__ == "__main__":
    main()
