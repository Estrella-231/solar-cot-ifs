#!/usr/bin/env python3
"""Plot a mechanically selected, CPP-supported validation COT forecast case."""
import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    with np.load(a.bank, allow_pickle=False) as z:
        seq = z["sequence_ids"].astype(str)
        keys = z["target_time_utc"].astype(str)
        mask = z["cpp_valid_mask"].astype(bool)
        ref = z["cpp_reference_cot"].astype(np.float64)
        methods = {
            "S→R forecast": np.expm1(z["forecast_log1p_cot"].astype(np.float64)),
            "Persistence-R": np.expm1(z["persistence_log1p_cot"].astype(np.float64)),
            "Future observed AGRI→R (offline)": np.expm1(z["target_agri_r_log1p_cot"].astype(np.float64)),
        }
    # Predeclared rule for this diagnostic: earliest BJT target date with CPP
    # support at every lead for both stations, then lexical sequence ID tie-break.
    complete = []
    for i in range(len(seq)):
        if mask[i, :, :, 7:10, 7:10].reshape(2, 16, -1).any(axis=-1).all():
            first = datetime.strptime(keys[i, 0], "%Y-%m-%d %H:%M:%S") + timedelta(hours=8)
            complete.append((first.date().isoformat(), seq[i], i))
    if not complete:
        raise RuntimeError("no window has CPP support at all 16 leads for both stations")
    complete.sort()
    chosen_date, chosen_seq, wi = complete[0]
    output_rows = []
    per_station_metrics = {}
    color_style = {
        "CPP reference": ("#000000", "--", "s"),
        "S→R forecast": ("#0072B2", "-", "o"),
        "Persistence-R": ("#D55E00", ":", "^"),
        "Future observed AGRI→R (offline)": ("#666666", "-.", "D"),
    }
    series = {"CPP reference": []}
    series.update({name: [] for name in methods})
    for s, station in enumerate(("Sili", "Zhujia")):
        station_metrics = {}
        for name in series:
            station_metrics[name] = {"sse": 0.0, "sae": 0.0, "n": 0}
        for l in range(16):
            mm = mask[wi, s, l, 7:10, 7:10]
            if not mm.any():
                raise RuntimeError(f"chosen sequence lost 3x3 common support: {station}, lead {15*(l+1)}")
            truth = ref[wi, s, l, 7:10, 7:10][mm]
            values = {"CPP reference": truth}
            for name, arr in methods.items():
                values[name] = arr[wi, s, l, 7:10, 7:10][mm]
            for name, vals in values.items():
                series[name].append((s, l, float(vals.mean())))
                if name != "CPP reference":
                    err = vals - truth
                    q = station_metrics[name]
                    q["sse"] += float(np.square(err).sum(dtype=np.float64))
                    q["sae"] += float(np.abs(err).sum(dtype=np.float64))
                    q["n"] += int(err.size)
                output_rows.append({
                    "sequence_id": chosen_seq, "target_date_bjt": chosen_date,
                    "station": station, "lead_min": 15 * (l + 1), "series": name,
                    "n_common_3x3_pixels": int(mm.sum()), "mean_cot": float(vals.mean()),
                })
        per_station_metrics[station] = {
            name: {"rmse_cot": math.sqrt(v["sse"] / v["n"]),
                  "mae_cot": v["sae"] / v["n"], "n_pixels": v["n"]}
            for name, v in station_metrics.items() if v["n"]
        }

    a.output.mkdir(parents=True)
    leads = np.arange(15, 241, 15)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharex=True, sharey=True, layout="constrained")
    for s, (ax, station) in enumerate(zip(axes, ("Sili", "Zhujia"))):
        station_max = 0.0
        for name, points in series.items():
            vals = np.array([v for ps, _, v in points if ps == s], dtype=np.float64)
            if len(vals) != 16 or not np.isfinite(vals).all():
                raise RuntimeError(f"bad plotted series length/finite check: {station}/{name}")
            c, ls, marker = color_style[name]
            ax.plot(leads, vals, color=c, linestyle=ls, marker=marker,
                    linewidth=1.9 if name == "S→R forecast" else 1.6,
                    markersize=4.2, label=name)
            station_max = max(station_max, float(vals.max()))
        rmse_text = "\n".join(
            f"{k}: RMSE {v['rmse_cot']:.2f}"
            for k, v in per_station_metrics[station].items()
        )
        ax.text(0.02, 0.97, rmse_text, transform=ax.transAxes, va="top", ha="left",
                fontsize=8.2, bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"})
        ax.set_title(station)
        ax.set_xlabel("Forecast lead (min)")
        ax.set_xticks(np.arange(15, 241, 30))
        ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.7)
    ymax = max(max(v for _, _, v in points) for points in series.values())
    for ax in axes:
        ax.set_ylim(0, max(10, math.ceil(ymax / 10) * 10))
    axes[0].set_ylabel("COT (dimensionless)")
    fig.suptitle(f"Validation COT forecast case | {chosen_date} BJT | diagnostic example only\n"
                 "CPP is a retrieval reference; future-AGRI→R is offline, not deployable")
    axes[0].legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=7.2)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(a.output / f"cot_forecast_case_{chosen_date}.{ext}", dpi=300,
                    facecolor="white", bbox_inches=None)
    plt.close(fig)
    with (a.output / "cot_forecast_case_values.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        w.writeheader(); w.writerows(output_rows)
    receipt = {
        "state": "COMPLETE_PREDECLARED_VALIDATION_DIAGNOSTIC_CASE",
        "split": "validation_only", "test_payloads_read": 0,
        "selection_rule": "earliest BJT target date among windows with CPP support at each of 16 leads for both stations; lexical sequence_id tie-break",
        "eligible_complete_windows": len(complete), "selected_sequence_id": chosen_seq,
        "selected_target_date_bjt": chosen_date, "bank_sha256": sha(a.bank),
        "scale": "3x3 station neighborhood; per lead mean uses only CPP-common valid pixels",
        "metrics": per_station_metrics,
        "caveat": "One diagnostic case, not representative evidence. CPP is a retrieval reference rather than independent truth. Future observed AGRI→R is an offline retrieval reference, not a deployable forecast.",
    }
    (a.output / "case_selection_and_metrics.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
