"""Paired event-level COT/GHI audit for the frozen regional-R GHI pilot.

Uses saved validation predictions only. Events were selected earlier from Sili
COT trajectories and are diagnostic examples, not a representative cohort.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "results/regional_station_cot_20260927/ghi_pilot"
OUT = ROOT / "results/regional_station_cot_20260927/ghi_pilot_event_audit_20260928"
EVENT_SOURCE = OUT / "cot_event_sources.json"
EVENTS = [
    {"sequence": 2311, "row_start": 592497, "name": "Cloud dissipation", "date": "2025-08-05", "sili_event": "dissipating cloud"},
    {"sequence": 2869, "row_start": 604931, "name": "Moving and thickening cloud", "date": "2025-08-14", "sili_event": "moving, thickening cloud"},
    {"sequence": 534, "row_start": 550820, "name": "Missed cloud growth", "date": "2025-07-09", "sili_event": "missed cloud growth"},
]
ARMS = ("zero", "forecast", "oracle")
STATIONS = ((0, "Sili"), (1, "Zhujia"))
ARM_LABEL = {"zero": "No COT", "forecast": "Forecast COT", "oracle": "Oracle COT"}
ARM_COLOR = {"zero": "#4c78a8", "forecast": "#f58518", "oracle": "#54a24b"}


def load_predictions():
    packs = {}
    for arm in ARMS:
        z = np.load(PILOT / f"{arm}_validation_predictions.npz", allow_pickle=False)
        packs[arm] = {k: z[k] for k in z.files}
    ref = packs["zero"]
    for arm in ARMS[1:]:
        cur = packs[arm]
        for key in ("row_ids", "station", "lead"):
            if not np.array_equal(ref[key], cur[key]):
                raise ValueError(f"paired-key mismatch: {arm}.{key}")
        if not np.array_equal(ref["true_ghi"], cur["true_ghi"]):
            raise ValueError(f"paired target mismatch: {arm}.true_ghi")
    if len(np.unique(ref["row_ids"])) != len(ref["row_ids"]):
        raise ValueError("duplicate validation row ids")
    lookup = {int(row): i for i, row in enumerate(ref["row_ids"])}
    return packs, lookup


def rmse(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(x * x))) if x.size else float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    packs, row_lookup = load_predictions()
    source = json.loads(EVENT_SOURCE.read_text(encoding="utf-8-sig"))
    rows = []
    figure_data = []

    for event in EVENTS:
        seq = str(event["sequence"])
        if seq not in source:
            raise ValueError(f"missing COT event source for sequence {seq}")
        src = source[seq]
        station_data = {}
        for st, station_name in STATIONS:
            ids = [event["row_start"] + st * 16 + lead for lead in range(16)]
            missing = [rid for rid in ids if rid not in row_lookup]
            if missing:
                raise ValueError(f"sequence {seq}/{station_name}: missing GHI row ids {missing}")
            ix = np.array([row_lookup[rid] for rid in ids], dtype=int)
            if not np.all(packs["zero"]["station"][ix] == st):
                raise ValueError(f"station key mismatch for {seq}/{station_name}")
            if not np.array_equal(packs["zero"]["lead"][ix], np.arange(16)):
                raise ValueError(f"lead key mismatch for {seq}/{station_name}")
            c = src["stations"][station_name]
            valid_cot = np.asarray(c["valid"], dtype=bool) & np.isfinite(c["cpp"])
            cpp = np.asarray(c["cpp"], dtype=float)
            fcot = np.asarray(c["forecast"], dtype=float)
            ocot = np.asarray(c["oracle"], dtype=float)
            if not (cpp.shape == fcot.shape == ocot.shape == (16,)):
                raise ValueError(f"COT lead shape mismatch for {seq}/{station_name}")
            truth = packs["zero"]["true_ghi"][ix]
            pred = {a: packs[a]["pred_ghi"][ix] for a in ARMS}
            station_data[station_name] = {"leads": np.arange(15, 241, 15), "cpp": cpp,
                "cot_valid": valid_cot, "forecast_cot": fcot, "oracle_cot": ocot,
                "true_ghi": truth, "pred_ghi": pred, "row_ids": ids}
            cot_rmse_forecast = rmse((fcot - cpp)[valid_cot])
            cot_rmse_oracle = rmse((ocot - cpp)[valid_cot])
            for arm in ARMS:
                err = pred[arm] - truth
                rows.append({
                    "sequence": event["sequence"], "event": event["name"], "date_BJT": event["date"],
                    "station": station_name, "arm": arm, "n_leads": len(err),
                    "GHI_RMSE_Wm2": rmse(err), "GHI_MAE_Wm2": float(np.mean(np.abs(err))),
                    "GHI_bias_Wm2": float(np.mean(err)),
                    "delta_RMSE_vs_noCOT_Wm2": rmse(err) - rmse(pred["zero"] - truth),
                    "COT_RMSE_vs_CPP_forecastCOT": cot_rmse_forecast,
                    "COT_RMSE_vs_CPP_oracleCOT": cot_rmse_oracle,
                    "COT_valid_leads": int(valid_cot.sum()),
                    "row_id_first": ids[0], "row_id_last": ids[-1],
                })
        figure_data.append((event, station_data))

    # Add paired COT-to-GHI error-change diagnostics per event/site/lead.
    for row in rows:
        ev, st = str(row["sequence"]), row["station"]
        sd = next(x[1][st] for x in figure_data if str(x[0]["sequence"]) == ev)
        valid = sd["cot_valid"]
        # Repeated on all arm rows by design so a single table row contains both COT endpoints.
        row["forecastCOT_RMSE_vs_CPP"] = rmse((sd["forecast_cot"] - sd["cpp"])[valid])
        row["oracleCOT_RMSE_vs_CPP"] = rmse((sd["oracle_cot"] - sd["cpp"])[valid])

    csv_path = OUT / "event_station_arm_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    summary = {
        "scope": "three preselected Sili event sequences matched exactly to the saved 128-sequence validation GHI pilot; diagnostic cases only",
        "test_used": False,
        "R": "frozen regional R256; pilot cache checkpoint SHA verified against retrained R256",
        "GHI_arms": "three independently trained pilot heads; same validation row_ids, targets, station and lead; oracle arm uses actual future AGRI -> same R and is nondeployable",
        "COT_metrics": "physical COT compared with CPP retrieval reference on valid station-pixel leads; CPP is not independent physical truth",
        "event_initialization_BJT": {
            str(e["sequence"]): (datetime.strptime(source[str(e["sequence"])] ["bjt_start"], "%Y-%m-%d %H:%M:%S")
                                  + timedelta(minutes=105)).strftime("%Y-%m-%d %H:%M:%S")
            for e, _ in figure_data
        },
        "results": rows,
        "interpretation_limits": [
            "Events were selected for cloud evolution patterns and do not estimate cohort-wide performance.",
            "Arms were trained independently; forecast/oracle differences combine COT input and optimization/learned-head differences, so they are not a causal COT-only intervention.",
            "Oracle uses actual future AGRI and cannot be deployed; it is not true COT.",
            "Lead points within an event are temporally dependent; no event-level significance test is claimed.",
        ],
    }
    (OUT / "event_station_arm_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3 events x (COT/GHI for each of two stations): shared lead axis and units per column.
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 8, "figure.titlesize": 15})
    fig, axes = plt.subplots(3, 4, figsize=(17, 10), sharex=True, constrained_layout=True)
    columns = [("Sili", "cot"), ("Sili", "ghi"), ("Zhujia", "cot"), ("Zhujia", "ghi")]
    for r, (event, sd) in enumerate(figure_data):
        for cidx, (station, kind) in enumerate(columns):
            ax = axes[r, cidx]
            d = sd[station]; x = d["leads"]
            if kind == "cot":
                ax.plot(x, d["cpp"], color="#222222", lw=1.8, marker="o", ms=3.4, label="CPP reference")
                ax.plot(x, d["forecast_cot"], color=ARM_COLOR["forecast"], lw=1.7, marker="s", ms=3.2, label="Forecast AGRI → R")
                ax.plot(x, d["oracle_cot"], color=ARM_COLOR["oracle"], lw=1.7, marker="^", ms=3.2, label="Real future AGRI → R (oracle)")
                ax.set_ylabel("COT")
                ax.set_ylim(bottom=0)
            else:
                ax.plot(x, d["true_ghi"], color="#111111", lw=2.0, marker="o", ms=3.4, label="Observed 15-min GHI")
                for arm in ARMS:
                    ax.plot(x, d["pred_ghi"][arm], color=ARM_COLOR[arm], lw=1.45,
                            ls="--" if arm == "oracle" else "-", marker={"zero":"s","forecast":"^","oracle":"D"}[arm],
                            ms=3.0, label=ARM_LABEL[arm])
                ax.set_ylabel("GHI (W m$^{-2}$)")
                ax.set_ylim(bottom=0)
            ax.grid(True, alpha=.24, linewidth=.6)
            ax.set_xlim(15, 240)
            ax.set_xticks([15, 60, 120, 180, 240])
            if r == 0:
                ax.set_title(f"{station}: {'COT trajectory' if kind == 'cot' else 'GHI response'}")
            if cidx == 0:
                ax.text(-.27, .5, f"{event['date']}\n{event['name']}", transform=ax.transAxes,
                        ha="right", va="center", fontsize=9, weight="bold")
            if r == 2:
                ax.set_xlabel("Forecast lead from initialization (minutes)")
            if r == 0:
                ax.legend(loc="best", frameon=True, ncol=1)
    fig.suptitle("Matched COT and GHI trajectories for three selected validation events\n"
                 "same station/init/target rows · R256 · oracle uses non-deployable real future AGRI")
    fig.savefig(OUT / "matched_event_cot_ghi_trajectories.png", dpi=220)
    fig.savefig(OUT / "matched_event_cot_ghi_trajectories.pdf")
    plt.close(fig)
    print(json.dumps({"events": len(EVENTS), "rows": len(rows), "figure": str(OUT / "matched_event_cot_ghi_trajectories.png"),
                      "metrics": str(csv_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
