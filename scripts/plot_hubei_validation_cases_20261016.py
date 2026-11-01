#!/usr/bin/env python3
"""Render preselected Hubei validation cases from saved A/AC/AP predictions.

Validation only. Case dates are frozen independently of model errors. Lines are
15-min station GHI estimates; ribbons show +/- one standard deviation across
three training seeds (descriptive seed spread, not a confidence interval).
Missing 15-min slots remain gaps. No test/buffer payloads are read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
ARMS = ("A", "AC", "AP")
SEEDS = (42, 43, 44)
COLORS = {"Observed": "#222222", "A": "#0072B2", "AC": "#D55E00", "AP": "#009E73"}
STYLES = {"A": "--", "AC": "-", "AP": ":"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_stamp(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d%H%M")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True, help="Hubei experiment root")
    p.add_argument("--case-manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    root, cases_path, out = args.root.resolve(), args.case_manifest.resolve(), args.out_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    rows_path = root / "pack" / "rows.csv"
    ghi_path = root / "pack" / "ghi_wm2.npy"
    rows = list(csv.DictReader(rows_path.open("r", encoding="utf-8", newline="")))
    truth_all = np.load(ghi_path, mmap_mode="r")
    if len(rows) != len(truth_all):
        raise ValueError("rows/label length mismatch")
    if not rows or not {"index", "split", "station", "time_bjt", "ghi_wm2"}.issubset(rows[0]):
        raise ValueError("Unexpected pack rows schema")
    if any(r["split"] not in {"train", "val"} for r in rows):
        raise ValueError("Pack unexpectedly contains test/buffer rows")

    cases = [r for r in csv.DictReader(cases_path.open("r", encoding="utf-8-sig", newline=""))
             if r.get("date_BJT")]
    if not cases:
        raise ValueError("No frozen cases")
    case_keys = {(r["station"], r["date_BJT"], r["case_type"]) for r in cases}
    if len(case_keys) != len(cases) or any(r["station"] not in STATIONS for r in cases):
        raise ValueError("Duplicate or invalid station/date/case in manifest")

    val_by_station: dict[str, list[tuple[int, dict[str, str]]]] = {s: [] for s in STATIONS}
    for i, r in enumerate(rows):
        if r["split"] == "val" and r["station"] in STATIONS:
            val_by_station[r["station"]].append((i, r))
    pred: dict[tuple[int, str, str], np.ndarray] = {}
    for station in STATIONS:
        expected_n = len(val_by_station[station])
        for seed in SEEDS:
            for arm in ARMS:
                f = root / "runs" / f"seed{seed}" / station / f"{arm}_val_prediction_ghi_wm2.npy"
                a = np.asarray(np.load(f, mmap_mode="r"), dtype=np.float64).reshape(-1)
                if a.size != expected_n or not np.isfinite(a).all():
                    raise ValueError(f"Invalid prediction length/values: {f}")
                pred[(seed, station, arm)] = a

    case_rows = []
    plot_max = 0.0
    prepared: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for c in cases:
        station, date, case_type = c["station"], c["date_BJT"], c["case_type"]
        local_rows = [(i, r) for i, r in val_by_station[station]
                      if r["time_bjt"][:8] == date.replace("-", "")]
        if not local_rows:
            raise ValueError(f"No validation rows for frozen case {station}/{date}")
        local_rows.sort(key=lambda x: x[1]["time_bjt"])
        n_expected = int(c["valid_daylight_targets"])
        if len(local_rows) != n_expected:
            raise ValueError(f"Case support changed for {station}/{date}: {len(local_rows)} != {n_expected}")
        records = []
        for local_pos, (global_i, row) in enumerate(local_rows):
            stamp = parse_stamp(row["time_bjt"])
            y = float(truth_all[global_i])
            rec: dict[str, object] = {"time": stamp, "truth": y}
            plot_max = max(plot_max, y)
            for arm in ARMS:
                vals = np.asarray([pred[(seed, station, arm)][
                    next(j for j, (_, rr) in enumerate(val_by_station[station]) if rr["index"] == row["index"])
                ] for seed in SEEDS], dtype=np.float64)
                rec[arm] = float(vals.mean())
                rec[f"{arm}_sd"] = float(vals.std(ddof=1))
                rec[f"{arm}_seed_values"] = vals
                plot_max = max(plot_max, float(vals.max()))
            records.append(rec)
            case_rows.append({"station": station, "date_BJT": date, "case_type": case_type,
                              "time_BJT": stamp.isoformat(sep=" "), "truth_GHI_Wm2": y,
                              **{arm: rec[arm] for arm in ARMS},
                              **{f"{arm}_seed_sd_Wm2": rec[f"{arm}_sd"] for arm in ARMS},
                              "n_seeds": len(SEEDS), "uncertainty":"sample SD across three seeds; not CI"})
        prepared[(station, date, case_type)] = records

    out.mkdir(parents=True)
    with (out / "validation_case_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        fields = list(case_rows[0])
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(case_rows)

    max_y = max(100.0, np.ceil(plot_max / 100.0) * 100.0)
    rendered = []
    for station in STATIONS:
        station_cases = [c for c in cases if c["station"] == station]
        fig, axes = plt.subplots(4, 2, figsize=(11.7, 8.3), sharey=True)
        fig.subplots_adjust(left=0.075, right=0.992, top=0.90, bottom=0.16,
                            hspace=0.42, wspace=0.08)
        for ax, c in zip(axes.flat, station_cases):
            records = prepared[(station, c["date_BJT"], c["case_type"])]
            start, stop = records[0]["time"], records[-1]["time"]
            grid = []
            t = start
            while t <= stop:
                grid.append(t); t += timedelta(minutes=15)
            index = {r["time"]: r for r in records}
            y = np.asarray([index[x]["truth"] if x in index else np.nan for x in grid])
            ax.plot(grid, y, color=COLORS["Observed"], lw=1.7, marker="o", ms=2.8,
                    label="Observed GHI", zorder=4)
            for arm in ARMS:
                m = np.asarray([index[x][arm] if x in index else np.nan for x in grid])
                s = np.asarray([index[x][f"{arm}_sd"] if x in index else np.nan for x in grid])
                ax.plot(grid, m, color=COLORS[arm], ls=STYLES[arm], lw=1.35,
                        marker="." if arm != "AC" else "s", ms=3, label=f"{arm} (mean, 3 seeds)")
                ax.fill_between(grid, m-s, m+s, color=COLORS[arm], alpha=0.10, linewidth=0)
            typ = "GHI ramp selected from observations" if "largest_observed" in c["case_type"] else "fixed-hash date"
            ax.set_title(f"{c['date_BJT']} | {typ} | n={len(records)}", fontsize=8)
            ax.set_ylim(0, max_y)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
            ax.grid(True, color="#dddddd", lw=0.45)
            ax.tick_params(axis="x", labelrotation=25, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
        for ax in axes.flat[len(station_cases):]:
            ax.set_visible(False)
        fig.suptitle(f"{station}: preselected validation cases | 15-min station GHI", fontsize=12, y=0.985)
        axes[-1, 0].set_xlabel("Target time (BJT)")
        axes[-1, 1].set_xlabel("Target time (BJT)")
        for ax in axes[:, 0]:
            ax.set_ylabel("GHI (W m$^{-2}$)")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955),
                   ncol=4, frameon=False, fontsize=8)
        fig.text(0.5, 0.018,
                 "Cases selected without model errors; ramp means observed 60-min GHI range, not a weather label. "
                 "Shading is +/-1 sample SD across seeds, not a confidence interval. Missing slots remain gaps.",
                 ha="center", va="bottom", fontsize=6.5)
        fig.savefig(out / f"{station}_validation_cases.pdf")
        fig.savefig(out / f"{station}_validation_cases.png", dpi=220)
        plt.close(fig)
        rendered.append(station)

    output_files = [out / f"{s}_validation_cases.{ext}" for s in STATIONS for ext in ("pdf", "png")]
    output_files.append(out / "validation_case_predictions.csv")
    report = {
        "state": "COMPLETE_DESCRIPTIVE_VALIDATION_CASE_FIGURES",
        "scope": "frozen validation cases and saved A/AC/AP predictions only",
        "case_manifest_sha256": sha256(cases_path),
        "pack_rows_sha256": sha256(rows_path),
        "pack_ghi_sha256": sha256(ghi_path),
        "prediction_files": {f"seed{seed}/{station}/{arm}": sha256(root / "runs" / f"seed{seed}" / station / f"{arm}_val_prediction_ghi_wm2.npy")
                             for seed in SEEDS for station in STATIONS for arm in ARMS},
        "n_cases": len(cases), "rendered_stations": rendered,
        "test_or_buffer_payloads_read": 0,
        "case_selection_used_model_errors": False,
        "weather_class_inferred": False,
        "uncertainty": "sample SD across three training seeds; descriptive only, not CI",
        "missing_slots": "retained as gaps on the 15-min target grid",
        "outputs": [f"{s}_validation_cases.{ext}" for s in STATIONS for ext in ("pdf", "png")] + ["validation_case_predictions.csv"],
        "output_sha256": {f.name: sha256(f) for f in output_files},
        "input_paths": {"root": str(root), "case_manifest": str(cases_path)},
    }
    (out / "FIGURE_AUDIT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"state": report["state"], "n_cases": len(cases), "out_dir": str(out),
                      "test_or_buffer_payloads_read": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
