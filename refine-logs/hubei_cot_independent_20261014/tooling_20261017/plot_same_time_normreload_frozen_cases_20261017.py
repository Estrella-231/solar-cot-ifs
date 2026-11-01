#!/usr/bin/env python3
"""Render frozen, error-independent Hubei validation cases for normreload runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
ARMS = ("A", "AC", "AP")
SEEDS = (42, 43, 44)
COLORS = {"Observed": "#222222", "A": "#0072B2", "AC": "#D55E00", "AP": "#009E73"}
STYLES = {"A": "--", "AC": "-", "AP": ":"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-dir", type=Path, required=True)
    ap.add_argument("--seed-root", type=Path, required=True)
    ap.add_argument("--case-manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")

    pack_report_path = args.pack_dir / "pack_report.json"
    pack_report = json.loads(pack_report_path.read_text(encoding="utf-8"))
    if pack_report.get("test_and_buffer_values_read") != 0:
        raise ValueError("pack is not certified train/validation-only")
    rows_path = args.pack_dir / "rows.csv"
    with rows_path.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if any(r["split"] not in {"train", "val"} for r in rows):
        raise ValueError("pack unexpectedly contains test/buffer rows")
    ghi = np.load(args.pack_dir / "ghi_wm2.npy", mmap_mode="r")
    if len(rows) != len(ghi):
        raise ValueError("pack rows/GHI length mismatch")
    val_by_station: dict[str, list[dict[str, str]]] = {s: [] for s in STATIONS}
    key_to_ghi = {}
    for i, row in enumerate(rows):
        if row["split"] != "val" or row["station"] not in STATIONS:
            continue
        key = (row["station"], row["time_bjt"])
        if key in key_to_ghi:
            raise ValueError(f"duplicate validation key: {key}")
        val_by_station[row["station"]].append(row)
        key_to_ghi[key] = float(ghi[i])

    cases_path = args.case_manifest.resolve()
    with cases_path.open("r", newline="", encoding="utf-8-sig") as f:
        cases = [r for r in csv.DictReader(f) if r.get("date_BJT")]
    if len(cases) != 24 or any(r["station"] not in STATIONS for r in cases):
        raise ValueError("expected the frozen 24-case manifest")
    if len({(r["station"], r["date_BJT"], r["case_type"]) for r in cases}) != len(cases):
        raise ValueError("duplicate frozen case")

    predictions: dict[tuple[int, str, str], dict[str, float]] = {}
    pred_hashes = {}
    for seed in SEEDS:
        seed_dir = args.seed_root / f"same_time_seed{seed}_normreload_v1"
        run = json.loads((seed_dir / "run_metadata.json").read_text(encoding="utf-8"))
        if run.get("test_payloads_read") != 0:
            raise ValueError(f"seed {seed} records test reads")
        for station in STATIONS:
            expected = [(station, row["time_bjt"]) for row in val_by_station[station]]
            for arm in ARMS:
                bundle_path = seed_dir / station / f"{arm}_validation_predictions_with_keys.npz"
                z = np.load(bundle_path, allow_pickle=False)
                got = list(zip(z["station"].astype(str).tolist(), z["target_time_bjt"].astype(str).tolist()))
                if got != expected:
                    raise ValueError(f"key/order mismatch: {seed}/{station}/{arm}")
                truth = np.asarray(z["truth_ghi_wm2"], dtype=np.float64)
                pred = np.asarray(z["pred_ghi_wm2"], dtype=np.float64)
                if not np.isfinite(truth).all() or not np.isfinite(pred).all():
                    raise ValueError(f"nonfinite prediction: {bundle_path}")
                pack_truth = np.asarray([key_to_ghi[k] for k in expected], dtype=np.float64)
                if not np.allclose(truth, pack_truth, rtol=0.0, atol=1e-4):
                    raise ValueError(f"truth mismatch with frozen train/validation pack: {seed}/{station}/{arm}")
                predictions[(seed, station, arm)] = {k: float(v) for k, v in zip((r["time_bjt"] for r in val_by_station[station]), pred)}
                pred_hashes[f"seed{seed}/{station}/{arm}"] = sha256(bundle_path)

    prepared = {}
    case_rows = []
    ymax_source = 100.0
    for case in cases:
        station, date = case["station"], case["date_BJT"]
        day_rows = [r for r in val_by_station[station] if r["time_bjt"][:8] == date.replace("-", "")]
        day_rows.sort(key=lambda r: r["time_bjt"])
        if len(day_rows) != int(case["valid_daylight_targets"]):
            raise ValueError(f"frozen case support changed: {station}/{date}")
        records = []
        for row in day_rows:
            stamp = datetime.strptime(row["time_bjt"], "%Y%m%d%H%M")
            y = key_to_ghi[(station, row["time_bjt"])]
            rec: dict[str, object] = {"time": stamp, "truth": y}
            ymax_source = max(ymax_source, y)
            for arm in ARMS:
                vals = np.asarray([predictions[(seed, station, arm)][row["time_bjt"]] for seed in SEEDS], dtype=np.float64)
                rec[arm] = float(vals.mean())
                rec[f"{arm}_sd"] = float(vals.std(ddof=1))
                ymax_source = max(ymax_source, float(vals.max()))
            records.append(rec)
            case_rows.append({"station": station, "date_BJT": date, "case_type": case["case_type"], "time_BJT": stamp.isoformat(sep=" "), "truth_GHI_Wm2": y,
                              **{arm: rec[arm] for arm in ARMS}, **{f"{arm}_seed_sd_Wm2": rec[f"{arm}_sd"] for arm in ARMS},
                              "seed_spread": "sample SD across seeds; not CI"})
        prepared[(station, date, case["case_type"])] = records

    out.mkdir(parents=True)
    with (out / "validation_case_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(case_rows[0]))
        writer.writeheader(); writer.writerows(case_rows)
    y_max = float(np.ceil(ymax_source / 100.0) * 100.0)
    for station in STATIONS:
        station_cases = [c for c in cases if c["station"] == station]
        fig, axes = plt.subplots(4, 2, figsize=(11.7, 8.3), sharey=True)
        fig.subplots_adjust(left=0.075, right=0.992, top=0.90, bottom=0.16, hspace=0.42, wspace=0.08)
        for ax, case in zip(axes.flat, station_cases):
            records = prepared[(station, case["date_BJT"], case["case_type"])]
            start, stop = records[0]["time"], records[-1]["time"]
            grid = []
            t = start
            while t <= stop:
                grid.append(t); t += timedelta(minutes=15)
            by_time = {r["time"]: r for r in records}
            truth = np.asarray([by_time[t]["truth"] if t in by_time else np.nan for t in grid])
            ax.plot(grid, truth, color=COLORS["Observed"], lw=1.7, marker="o", ms=2.8, label="Observed GHI", zorder=4)
            for arm in ARMS:
                mean = np.asarray([by_time[t][arm] if t in by_time else np.nan for t in grid])
                sd = np.asarray([by_time[t][f"{arm}_sd"] if t in by_time else np.nan for t in grid])
                ax.plot(grid, mean, color=COLORS[arm], ls=STYLES[arm], lw=1.35, marker=".", ms=3, label=f"{arm} (mean, 3 seeds)")
                ax.fill_between(grid, mean-sd, mean+sd, color=COLORS[arm], alpha=0.10, linewidth=0)
            label = "GHI ramp selected from observations" if "largest_observed" in case["case_type"] else "fixed-hash date"
            ax.set_title(f"{case['date_BJT']} | {label} | n={len(records)}", fontsize=8)
            ax.set_ylim(0, y_max)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
            ax.grid(True, color="#dddddd", lw=0.45)
            ax.tick_params(axis="x", labelrotation=25, labelsize=7); ax.tick_params(axis="y", labelsize=7)
        for ax in axes.flat[len(station_cases):]: ax.set_visible(False)
        fig.suptitle(f"{station}: preselected validation cases | 15-min station GHI", fontsize=12, y=0.985)
        axes[-1, 0].set_xlabel("Target time (BJT)"); axes[-1, 1].set_xlabel("Target time (BJT)")
        for ax in axes[:, 0]: ax.set_ylabel("GHI (W m$^{-2}$)")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=4, frameon=False, fontsize=8)
        fig.text(0.5, 0.018, "Cases selected without model errors; ramp means observed 60-min GHI range, not a weather label. Shading is ±1 sample SD across seeds, not a CI. Missing slots remain gaps.", ha="center", va="bottom", fontsize=6.5)
        fig.savefig(out / f"{station}_validation_cases.pdf"); fig.savefig(out / f"{station}_validation_cases.png", dpi=220); plt.close(fig)

    outputs = [out / f"{s}_validation_cases.{ext}" for s in STATIONS for ext in ("pdf", "png")] + [out / "validation_case_predictions.csv"]
    report = {
        "state": "COMPLETE_NORMRELOAD_FROZEN_VALIDATION_CASE_FIGURES_TEST_CLOSED",
        "scope": "24 preselected same-time validation cases; new A/AC/AP normreload saved predictions",
        "case_manifest_sha256": sha256(cases_path), "pack_report_sha256": sha256(pack_report_path),
        "rows_sha256": sha256(rows_path), "ghi_sha256": sha256(args.pack_dir / "ghi_wm2.npy"),
        "prediction_bundle_sha256": pred_hashes, "case_selection_used_model_errors": False,
        "weather_class_inferred": False, "test_payloads_read": 0, "buffer_payloads_read": 0,
        "uncertainty": "sample SD across 3 seeds, not confidence interval",
        "missing_slots": "remain gaps on 15-min grid",
        "outputs": {f.name: sha256(f) for f in outputs},
        "plot_script_sha256": sha256(Path(__file__).resolve()),
    }
    (out / "FIGURE_AUDIT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"state": report["state"], "cases": len(cases), "test_payloads_read": 0, "out_dir": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
