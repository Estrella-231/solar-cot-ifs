#!/usr/bin/env python3
"""Re-render preselected Hubei same-time validation GHI cases from audited NPZs."""
from __future__ import annotations

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


ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = ROOT / "audit_receipts"
CASE_MANIFEST = ROOT / "validation_case_selection_20261016_v1" / "validation_case_manifest.csv"
PRIMARY_AUDIT = RECEIPTS / "same_time_three_seeds_recheck" / "independent_three_seed_validation_audit.json"
OUT = ROOT / "validation_cases_b2_20261017_rebuilt_v2"
SEED_DIRS = {
    42: RECEIPTS / "same_time_seed42_independent_recheck",
    43: RECEIPTS / "same_time_seed43_44_independent_recheck" / "seed43",
    44: RECEIPTS / "same_time_seed43_44_independent_recheck" / "seed44",
}
STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
ARMS = ("A", "AC", "AP")
SEEDS = (42, 43, 44)
SEASON_ORDER = ("winter_DJF", "spring_MAM", "summer_JJA", "autumn_SON")
CASE_ORDER = ("largest_observed_GHI_60min_range", "fixed_hash_date")
COLORS = {"Observed": "#222222", "A": "#0072B2", "AC": "#D55E00", "AP": "#009E73"}
STYLES = {"A": "--", "AC": "-", "AP": ":"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bundle_path(seed: int, station: str, arm: str) -> Path:
    base = SEED_DIRS[seed] / station
    if not base.is_dir() and station == "Niushou":
        base = SEED_DIRS[seed]
    return base / f"{arm}_validation_predictions_with_keys.npz"


def load_bundles() -> tuple[dict, dict]:
    series: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    hashes = {}
    for station in STATIONS:
        canonical_times = None
        canonical_truth = None
        for seed in SEEDS:
            for arm in ARMS:
                path = bundle_path(seed, station, arm)
                hashes[f"seed{seed}/{station}/{arm}"] = sha256(path)
                with np.load(path, allow_pickle=False) as z:
                    times = z["target_time_bjt"].astype(str).tolist()
                    station_ids = z["station"].astype(str)
                    truth = z["truth_ghi_wm2"].astype(np.float64)
                    pred = z["pred_ghi_wm2"].astype(np.float64)
                if not np.all(station_ids == station) or not np.isfinite(truth).all() or not np.isfinite(pred).all():
                    raise ValueError(f"invalid station or non-finite values in {path}")
                if canonical_times is None:
                    canonical_times, canonical_truth = times, truth
                elif times != canonical_times or not np.array_equal(truth, canonical_truth):
                    raise ValueError(f"non-identical validation keys/truth across arms/seeds: {station}/{seed}/{arm}")
                for t, y, p in zip(times, truth, pred):
                    key = (station, t)
                    series.setdefault(key, {"truth": {}, "pred": {}})
                    series[key]["truth"][f"seed{seed}/{arm}"] = float(y)
                    series[key]["pred"][f"seed{seed}/{arm}"] = float(p)
    return series, hashes


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    primary = json.loads(PRIMARY_AUDIT.read_text(encoding="utf-8"))
    if primary["state"] != "HUBEI_SAME_TIME_THREE_SEED_VALIDATION_AUDIT_COMPLETE_TEST_CLOSED" or primary["test_payloads_read"] != 0:
        raise ValueError("parent validation audit is not closed/test-free")
    with CASE_MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        cases = [r for r in csv.DictReader(f) if r.get("date_BJT")]
    if len(cases) != 24 or len({(r["station"], r["date_BJT"], r["case_type"]) for r in cases}) != len(cases):
        raise ValueError("expected 24 unique preselected validation cases")
    if any(r["weather_class"] != "unverified_no_independent_CPP_CLP_label" for r in cases):
        raise ValueError("case manifest does not preserve the no-weather-label boundary")

    series, prediction_hashes = load_bundles()
    case_by_station: dict[str, dict[tuple[str, str], dict]] = {s: {} for s in STATIONS}
    for c in cases:
        case_by_station[c["station"]][(c["season"], c["case_type"])] = c

    csv_rows = []
    OUT.mkdir(parents=True)
    rendered = []
    for station in STATIONS:
        matrix = []
        max_y = 100.0
        for season in SEASON_ORDER:
            row_cases = []
            for case_type in CASE_ORDER:
                c = case_by_station[station].get((season, case_type))
                row_cases.append(c)
                if c:
                    date_key = c["date_BJT"].replace("-", "")
                    available = sorted(t for (s, t) in series if s == station and t.startswith(date_key))
                    if len(available) != int(c["valid_daylight_targets"]):
                        raise ValueError(f"case support mismatch {station}/{c['date_BJT']}: {len(available)}")
                    for t in available:
                        item = series[(station, t)]
                        max_y = max(max_y, *(item["truth"].values()), *(item["pred"].values()))
            matrix.append(row_cases)

        fig, axes = plt.subplots(4, 2, figsize=(11.7, 8.3), sharex=False, sharey=True)
        fig.subplots_adjust(left=0.075, right=0.99, top=0.855, bottom=0.13, hspace=0.39, wspace=0.08)
        for ri, season in enumerate(SEASON_ORDER):
            for ci, case_type in enumerate(CASE_ORDER):
                ax = axes[ri, ci]
                c = matrix[ri][ci]
                if not c:
                    ax.set_visible(False)
                    continue
                date_key = c["date_BJT"].replace("-", "")
                available = sorted(t for (s, t) in series if s == station and t.startswith(date_key))
                parsed = [datetime.strptime(t, "%Y%m%d%H%M") for t in available]
                start, stop = parsed[0], parsed[-1]
                grid, cur = [], start
                while cur <= stop:
                    grid.append(cur)
                    cur += timedelta(minutes=15)
                lookup = {datetime.strptime(t, "%Y%m%d%H%M"): series[(station, t)] for t in available}
                truth_y, means, spreads = [], {a: [] for a in ARMS}, {a: [] for a in ARMS}
                for tick in grid:
                    it = lookup.get(tick)
                    truth_y.append(np.nan if it is None else next(iter(it["truth"].values())))
                    for arm in ARMS:
                        values = [] if it is None else [it["pred"][f"seed{seed}/{arm}"] for seed in SEEDS]
                        means[arm].append(np.nan if not values else float(np.mean(values)))
                        spreads[arm].append(np.nan if not values else float(np.std(values, ddof=1)))
                    row = {"station": station, "season": season, "date_BJT": c["date_BJT"], "case_type": case_type,
                           "target_time_BJT": tick.strftime("%Y-%m-%d %H:%M:%S"), "slot_present": it is not None,
                           "truth_GHI_Wm2": truth_y[-1]}
                    for arm in ARMS:
                        row[f"{arm}_mean_3seeds_Wm2"] = means[arm][-1]
                        row[f"{arm}_sample_sd_3seeds_Wm2"] = spreads[arm][-1]
                        for seed in SEEDS:
                            row[f"{arm}_seed{seed}_Wm2"] = np.nan if it is None else it["pred"][f"seed{seed}/{arm}"]
                    csv_rows.append(row)

                ax.plot(grid, truth_y, color=COLORS["Observed"], lw=1.8, marker="o", ms=2.7,
                        label="Observed GHI", zorder=4)
                for arm in ARMS:
                    m, sd = np.asarray(means[arm]), np.asarray(spreads[arm])
                    ax.plot(grid, m, color=COLORS[arm], ls=STYLES[arm], lw=1.35,
                            marker="." if arm != "AC" else "s", ms=3.1,
                            label=f"{arm} (mean, 3 seeds)")
                    ax.fill_between(grid, m - sd, m + sd, color=COLORS[arm], alpha=0.10, linewidth=0)
                label = "Observed 60-min change" if "largest_observed" in case_type else "Fixed-hash date"
                ax.set_title(f"{c['date_BJT']} | {label} | n={len(available)}", fontsize=8)
                ax.set_ylim(0, np.ceil(max_y / 100.0) * 100.0)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
                ax.grid(True, color="#dddddd", lw=0.45)
                ax.tick_params(axis="x", labelrotation=25, labelsize=7)
                ax.tick_params(axis="y", labelsize=7)
                if ci == 0:
                    ax.set_ylabel("GHI (W m$^{-2}$)")
                if ri == 3:
                    ax.set_xlabel("Target time (BJT)")
        fig.suptitle(f"{station}: preselected validation examples | 15-min station GHI", fontsize=12, y=0.985)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94),
                   ncol=4, frameon=False, fontsize=8)
        fig.text(0.5, 0.015,
                 "Validation illustrations only; observed-change selection is not a cloud/weather label. "
                 "Shading is +/-1 sample SD across seeds (not a CI); missing 15-min slots remain gaps.",
                 ha="center", va="bottom", fontsize=7)
        for ext, kwargs in (("pdf", {}), ("png", {"dpi": 220})):
            fig.savefig(OUT / f"{station}_validation_cases.{ext}", **kwargs)
        plt.close(fig)
        rendered.append(station)

    csv_path = OUT / "validation_case_predictions.csv"
    if csv_rows:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    outputs = [OUT / f"{s}_validation_cases.{ext}" for s in STATIONS for ext in ("pdf", "png")] + [csv_path]
    report = {
        "state": "COMPLETE_REBUILT_VALIDATION_GHI_LINE_ILLUSTRATIONS_FROM_AUDITED_THREE_SEED_PREDICTIONS",
        "scope": "GHI-only same-time validation illustrations; not the full AGRI/COT case analysis, not test, not forecast skill, not weather classes",
        "case_manifest_sha256": sha256(CASE_MANIFEST),
        "parent_validation_audit_sha256": sha256(PRIMARY_AUDIT),
        "frozen_pack_sha256": primary["contract"]["pack_sha256"],
        "prediction_bundle_sha256": prediction_hashes,
        "n_cases": len(cases), "rendered_stations": rendered,
        "case_selection_used_model_errors": False,
        "weather_class_inferred": False,
        "uncertainty": "sample SD across three training seeds; not a confidence interval",
        "missing_target_slots": "rendered as gaps on 15-min target grid",
        "test_payloads_read": 0, "buffer_payloads_read": 0,
        "cloud_and_image_panels": "not included: matching image sidecars are not in the local audited receipt set; independent cloud labels unavailable",
        "outputs": [p.name for p in outputs],
        "output_sha256": {p.name: sha256(p) for p in outputs},
    }
    (OUT / "FIGURE_AUDIT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": report["state"], "out_dir": str(OUT), "cases": len(cases),
                      "test_payloads_read": 0, "buffer_payloads_read": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
