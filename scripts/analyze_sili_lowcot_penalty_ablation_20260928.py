"""Summarize the paired Sili low-COT penalty ablation without reopening test."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    summary = load_json(args.run / "training_complete.json")
    if summary.get("state") != "COMPLETE_SILI_LOW_COT_PENALTY_ABLATION" or summary.get("test_used") is not False:
        raise RuntimeError("Unexpected or test-contaminated run")

    arms = {}
    for name in ("penalty_on_seed42", "penalty_off_seed42"):
        directory = args.run / name
        complete = load_json(directory / "training_complete.json")
        if complete.get("state") != "COMPLETE_SILI_LOW_COT_PENALTY_ABLATION_ARM" or complete.get("test_used") is not False:
            raise RuntimeError(f"Invalid arm: {name}")
        rows = [json.loads(line) for line in (directory / "epochs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != complete["epochs_run"]:
            raise RuntimeError(f"Epoch log count mismatch for {name}")
        best = min(rows, key=lambda r: r["val"]["learned"]["thick_rmse"])
        arms[name] = {"complete": complete, "epochs": rows, "best_learned_epoch": best}

    base = arms["penalty_off_seed42"]["complete"]["step0_val"]["transport"]
    out = {
        "state": "COMPLETE_SILI_LOW_COT_PENALTY_ABLATION_ANALYSIS",
        "test_used": False,
        "source": "fixed 512/512 validation bank; overlapping windows; CPP retrieval reference",
        "pbs_exit_status": 0,
        "baseline_transport": base,
        "arms": {},
        "decision": "FAIL_ADVANCE_GATE_BOTH_ARMS_SELECT_STEP0",
        "uncertainty_limit": "Per-epoch per-sample predictions were not saved; no paired day-bootstrap interval can be computed for trained epoch checkpoints. The predeclared selection gate fails on point estimate because every learned checkpoint has higher thick-COT RMSE than step 0.",
    }
    for name, item in arms.items():
        complete = item["complete"]
        best = item["best_learned_epoch"]
        metrics = best["val"]["learned"]
        out["arms"][name] = {
            "extra_lowcot_false_cot_penalty": complete["extra_lowcot_false_cot_penalty"],
            "epochs_run": complete["epochs_run"],
            "selected_candidate": complete["selected_candidate"],
            "selected_thick_rmse": complete["selected_thick_rmse"],
            "best_learned_epoch_by_thick_rmse": best["epoch"],
            "best_learned_metrics": metrics,
            "best_learned_thick_rmse_minus_transport": metrics["thick_rmse"] - base["thick_rmse"],
            "epoch_metrics": [{
                "epoch": r["epoch"],
                "thick_rmse": r["val"]["learned"]["thick_rmse"],
                "all_rmse": r["val"]["learned"]["rmse"],
                "clear_false10_rate": r["val"]["learned"]["clear_false10_rate"],
                "thick_misses_pred_lt10": r["val"]["learned"]["thick_misses_pred_lt10"],
            } for r in item["epochs"]],
        }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "ablation_analysis.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    colors = {"penalty_on_seed42": "#7B61A8", "penalty_off_seed42": "#158F83"}
    labels = {"penalty_on_seed42": "Penalty on", "penalty_off_seed42": "Penalty off"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1))
    for name, item in arms.items():
        rows = item["epochs"]
        x = [r["epoch"] for r in rows]
        axes[0].plot(x, [r["val"]["learned"]["thick_rmse"] for r in rows], marker="o", lw=2,
                     color=colors[name], label=labels[name])
        axes[1].plot(x, [r["val"]["learned"]["rmse"] for r in rows], marker="o", lw=2,
                     color=colors[name], label=labels[name])
    axes[0].axhline(base["thick_rmse"], color="#333333", ls="--", label="Pure transport / step 0")
    axes[1].axhline(base["rmse"], color="#333333", ls="--", label="Pure transport / step 0")
    axes[0].set_title("Thick-reference COT RMSE (COT ≥ 30)")
    axes[1].set_title("All-valid COT RMSE")
    for ax in axes:
        ax.set_xlabel("Training epoch")
        ax.set_ylabel("COT RMSE")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("Sili fixed-validation loss ablation — both selected step 0")
    fig.tight_layout()
    fig.savefig(args.output / "penalty_ablation_epochs.png", dpi=180)
    fig.savefig(args.output / "penalty_ablation_epochs.pdf")
    plt.close(fig)
    print(json.dumps({k: {"selected": v["selected_candidate"], "best_learned_epoch": v["best_learned_epoch_by_thick_rmse"],
                          "best_learned_thick_rmse": v["best_learned_metrics"]["thick_rmse"],
                          "delta_vs_transport": v["best_learned_thick_rmse_minus_transport"]}
                      for k, v in out["arms"].items()}, indent=2))


if __name__ == "__main__":
    main()
