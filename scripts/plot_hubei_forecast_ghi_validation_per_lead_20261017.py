#!/usr/bin/env python3
"""Plot preregistered Hubei FA/FAC validation RMSE by lead.

Reads only the already audited per-lead validation CSV.  No test/buffer payloads
are opened here.  The figure is descriptive: three seeds, not a test claim.
"""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "refine-logs" / "hubei_cot_independent_20261014"
DEFAULT_INPUT = BASE / "audit_receipts" / "forecast_ghi_three_seed_normreload_20261017" / "three_seed_per_lead_metrics.csv"
DEFAULT_OUT = BASE / "audit_receipts" / "forecast_ghi_three_seed_normreload_20261017" / "figures"
STATIONS = ["Niushou", "Wangzhai", "Mizhuang"]
ARMS = ["FA", "FAC"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    input_csv = args.input_csv
    out_dir = args.output_dir
    rows = list(csv.DictReader(input_csv.open(encoding="utf-8-sig", newline="")))
    metrics: dict[tuple[str, int, int, str], float] = {}
    for row in rows:
        key = (row["station"], int(row["seed"]), int(row["lead_minutes"]), row["arm"])
        metric = float(row["rmse_wm2"])
        if key in metrics or not np.isfinite(metric):
            raise ValueError(f"duplicate or non-finite metric: {key}")
        metrics[key] = metric

    leads = sorted({lead for _, _, lead, _ in metrics})
    seeds = sorted({int(row["seed"]) for row in rows})
    if len(seeds) != 3 or leads != list(range(15, 241, 15)):
        raise ValueError(f"unexpected validation support: seeds={seeds}, leads={leads}")

    # Aggregate in the predeclared order: equal station weights within seed,
    # then equal seed weights. Keep seed values for descriptive range bars.
    arm_seed = {arm: np.empty((len(seeds), len(leads))) for arm in ARMS}
    station_delta = {station: np.empty(len(leads)) for station in STATIONS}
    for li, lead in enumerate(leads):
        for si, seed in enumerate(seeds):
            for arm in ARMS:
                cells = [
                    float(row["rmse_wm2"])
                    for row in rows
                    if row["seed"] == str(seed)
                    and int(row["lead_minutes"]) == lead
                    and row["arm"] == arm
                ]
                if len(cells) != len(STATIONS):
                    raise ValueError(f"incomplete support for {seed=} {lead=} {arm=}")
                arm_seed[arm][si, li] = float(np.mean(cells))
        for station in STATIONS:
            diffs = [
                metrics[(station, seed, lead, "FAC")] - metrics[(station, seed, lead, "FA")]
                for seed in seeds
            ]
            station_delta[station][li] = float(np.mean(diffs))

    colors = {"FA": "#2166AC", "FAC": "#B35806"}
    station_colors = {"Niushou": "#1B9E77", "Wangzhai": "#7570B3", "Mizhuang": "#D95F02"}
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)

    ax = axes[0]
    for arm in ARMS:
        mean = arm_seed[arm].mean(axis=0)
        lo = arm_seed[arm].min(axis=0)
        hi = arm_seed[arm].max(axis=0)
        ax.plot(leads, mean, marker="o", ms=3.2, lw=2.0, color=colors[arm], label=arm)
        ax.fill_between(leads, lo, hi, color=colors[arm], alpha=0.14, linewidth=0)
    ax.set_xlabel("Forecast lead (min)")
    ax.set_ylabel("Validation RMSE (W m$^{-2}$)")
    ax.set_xlim(12, 243)
    ax.set_xticks(leads[::2])
    ax.grid(axis="both", color="#D9D9D9", linewidth=0.65, alpha=0.8)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.text(0.98, 0.97, "(a)", transform=ax.transAxes, va="top", ha="right", fontweight="bold")

    ax = axes[1]
    for station in STATIONS:
        ax.plot(
            leads,
            station_delta[station],
            marker="o",
            ms=2.8,
            lw=1.35,
            color=station_colors[station],
            label=station,
        )
    overall_delta = (arm_seed["FAC"] - arm_seed["FA"]).mean(axis=0)
    ax.plot(leads, overall_delta, color="#222222", marker="s", ms=3.4, lw=2.1, label="Three-station mean")
    ax.axhline(0.0, color="#555555", lw=1.0, ls="--")
    ax.set_xlabel("Forecast lead (min)")
    ax.set_ylabel("RMSE difference: FAC − FA (W m$^{-2}$)")
    ax.set_xlim(12, 243)
    ax.set_xticks(leads[::2])
    ax.grid(axis="both", color="#D9D9D9", linewidth=0.65, alpha=0.8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(0.98, 0.97, "(b)", transform=ax.transAxes, va="top", ha="right", fontweight="bold")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "validation_per_lead_FA_FAC.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "validation_per_lead_FA_FAC.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    def sha256(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    audit = {
        "state": "HUBEI_FORECAST_GHI_VALIDATION_FIGURE_PROVENANCE_TEST_CLOSED",
        "input_scope": "validation-only saved per-lead metrics; no test/buffer payloads read",
        "input_csv": str(input_csv.resolve()),
        "input_csv_sha256": sha256(input_csv),
        "plot_script": str(Path(__file__).resolve()),
        "plot_script_sha256": sha256(Path(__file__).resolve()),
        "outputs": {
            name: {"path": str((out_dir / name).resolve()), "sha256": sha256(out_dir / name)}
            for name in ("validation_per_lead_FA_FAC.pdf", "validation_per_lead_FA_FAC.png")
        },
        "seeds": seeds,
        "leads_minutes": leads,
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "shading": "min-max range across three seeds; not confidence interval",
    }
    (out_dir / "FIGURE_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"input={input_csv}")
    print(f"support=validation only; seeds={seeds}; leads={leads[0]}..{leads[-1]}; test_payloads_read=0")
    print(f"overall_mean_delta={overall_delta.mean():.6f} W/m^2")
    print(f"pdf={out_dir / 'validation_per_lead_FA_FAC.pdf'}")
    print(f"png={out_dir / 'validation_per_lead_FA_FAC.png'}")
    print(f"figure_audit={out_dir / 'FIGURE_AUDIT.json'}")


if __name__ == "__main__":
    main()
