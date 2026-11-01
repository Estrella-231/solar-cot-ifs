"""Current observed-trained three-frame GHI heads, regional DyDiff forecast replay.

Recomputes FP64 station/seed RMSE and 5,000 paired BJT target-date bootstrap
draws from immutable saved validation predictions; verifies every plotted
quantity against the original evaluation JSON. No source rows are filtered.
Run from any directory with Python, NumPy, pandas, Matplotlib and PyMuPDF.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SOURCES = {
    "Hubei": ROOT / "refine-logs/hubei_threeframe_dydiff_forecast_20261102/recovered/evaluation",
    "Hunan": ROOT / "refine-logs/fulltrain_forecast_cot_20261101/observed_trained_heads_evaluation_v1",
}
SEED = 20261102
BOOTSTRAPS = 5000


def read_source(region, directory):
    keys = pd.read_csv(directory / "keys.csv", dtype={"target_time_bjt": str, "init_time_bjt": str})
    stored = json.loads((directory / "RESULTS.json").read_text(encoding="utf-8"))
    assert stored["state"] == "COMPLETE" and stored["test_payloads_read"] == 0
    with np.load(directory / "predictions.npz", allow_pickle=False) as z:
        truth = np.asarray(z["truth"], dtype=np.float64)
        seeds = z["seeds"].tolist()
        if region == "Hubei":
            pred = np.asarray(z["prediction"], dtype=np.float64)
            target = pd.to_datetime(keys.target_time_bjt, format="%Y%m%d%H%M")
            init = pd.to_datetime(keys.init_time_bjt, format="%Y%m%d%H%M")
            station_order = ["Niushou", "Wangzhai", "Mizhuang"]
        else:
            pred = np.stack([z["prediction_A"].T, z["prediction_AC"].T], axis=-1).astype(np.float64)
            target = pd.to_datetime(keys.target_time_utc) + pd.Timedelta(hours=8)
            init = pd.to_datetime(keys.init_time_utc) + pd.Timedelta(hours=8)
            station_order = ["sili", "zhujia"]
            assert set(keys["split"]) == {"val"}
    assert pred.shape == (len(keys), 3, 2) and len(truth) == len(keys)
    assert seeds == [42, 43, 44]
    assert np.isfinite(pred).all() and np.isfinite(truth).all()
    assert np.array_equal(((target - init).dt.total_seconds() / 60).to_numpy(), keys.lead_minutes.to_numpy())
    canonical = pd.DataFrame({"station": keys.station, "init": init, "target": target})
    assert not canonical.duplicated().any()
    return {
        "region": region, "truth": truth, "prediction": pred,
        "station": keys.station.to_numpy(), "station_order": station_order,
        "dates": target.dt.strftime("%Y-%m-%d").to_numpy(),
        "leads": keys.lead_minutes.to_numpy(), "stored": stored,
        "first_target_bjt": str(target.min()), "last_target_bjt": str(target.max()),
    }


def score(source, mask):
    dates = source["dates"][mask]
    days = np.unique(dates)
    weights = np.random.default_rng(SEED).multinomial(
        len(days), np.full(len(days), 1 / len(days)), size=BOOTSTRAPS
    ).astype(np.float64)
    points, boot_points, counts_by_station = [], [], {}
    for station in source["station_order"]:
        use = mask & (source["station"] == station)
        assert use.any()
        counts_by_station[station] = int(use.sum())
        day_index = np.searchsorted(days, source["dates"][use])
        counts = np.bincount(day_index, minlength=len(days))
        denominator = weights @ counts
        # Do not let an absent resampled station silently become zero error.
        assert (denominator > 0).all()
        for seed_index in range(3):
            error_squared = (source["prediction"][use, seed_index] - source["truth"][use, None]) ** 2
            points.append(np.sqrt(error_squared.mean(axis=0)))
            daily_sse = np.column_stack([
                np.bincount(day_index, weights=error_squared[:, arm], minlength=len(days))
                for arm in range(2)
            ])
            boot_points.append(np.sqrt((weights @ daily_sse) / denominator[:, None]))
    point = np.mean(points, axis=0)
    resampled = np.mean(boot_points, axis=0)
    interval = np.quantile(resampled[:, 1] - resampled[:, 0], [0.025, 0.975])
    return {
        "region": source["region"], "no_cot_rmse": float(point[0]),
        "with_cot_rmse": float(point[1]), "delta_rmse": float(point[1] - point[0]),
        "reduction_percent": float(100 * (1 - point[1] / point[0])),
        "delta_ci95_lower": float(interval[0]), "delta_ci95_upper": float(interval[1]),
        "records": int(mask.sum()), "target_bjt_dates": len(days),
        "records_by_station": counts_by_station,
    }


def verify(recomputed, stored):
    no = stored.get("A_rmse", stored.get("no_cot_rmse"))
    yes = stored.get("AC_rmse", stored.get("cot_rmse"))
    calculated = np.array([recomputed["no_cot_rmse"], recomputed["with_cot_rmse"],
                           recomputed["delta_ci95_lower"], recomputed["delta_ci95_upper"]])
    previous = np.array([no, yes, *stored["delta_ci95"]])
    difference = np.abs(calculated - previous)
    assert difference.max() < 1e-9, (recomputed, previous, difference)
    assert recomputed["records"] == stored["rows"]
    assert recomputed["target_bjt_dates"] == stored.get("target_dates", stored.get("target_bjt_dates"))
    return float(difference.max())


def plot(records):
    df = pd.DataFrame(records)
    style = {
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.labelsize": 10,
        "axes.titlesize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9.5, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.8, "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.2), sharex=True, sharey="row")
        fig.subplots_adjust(left=0.105, right=0.985, bottom=0.135, top=0.86, wspace=0.11, hspace=0.16)
        for col, region in enumerate(("Hubei", "Hunan")):
            d = df[df.region == region].sort_values("lead_minutes")
            x = d.lead_minutes.to_numpy()
            top, bottom = axes[:, col]
            for ax in (top, bottom):
                ax.axvspan(7.5, 37.5, color="#777777", alpha=0.11, linewidth=0, zorder=0)
                ax.axvline(37.5, color="#888888", linestyle=":", linewidth=0.8, zorder=1)
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="y", color="#dddddd", linewidth=0.55, zorder=0)
                ax.set_xlim(7.5, 247.5)
                ax.set_xticks([15, 60, 120, 180, 240])
            top.plot(x, d.no_cot_rmse, "o-", color="#0072B2", lw=1.6, ms=3.3, label="Without COT", zorder=3)
            top.plot(x, d.with_cot_rmse, "s--", color="#D55E00", lw=1.6, ms=3.1, label="With predicted COT", zorder=3)
            top.set_title(f"({chr(97 + col)})  {region}", loc="left", pad=7, fontweight="bold")
            bottom.fill_between(x, d.delta_ci95_lower.to_numpy(), d.delta_ci95_upper.to_numpy(), color="#6B4C9A", alpha=0.18, linewidth=0, label="95% date-block interval", zorder=2)
            bottom.plot(x, d.delta_rmse, "o-", color="#6B4C9A", lw=1.6, ms=3.3, zorder=3)
            bottom.axhline(0, color="#555555", lw=0.9, zorder=3)
            bottom.text(0.025, 0.965, f"({chr(99 + col)})", transform=bottom.transAxes, va="top", fontweight="bold", fontsize=10)
        top_values = df[["no_cot_rmse", "with_cot_rmse"]].to_numpy()
        low = np.floor(top_values.min() / 10) * 10
        high = np.ceil(top_values.max() / 10) * 10
        axes[0, 0].set_ylim(low - 3, high + 3)
        interval_values = df[["delta_ci95_lower", "delta_ci95_upper"]].to_numpy()
        lo = np.floor(min(0.0, interval_values.min()) / 2) * 2
        hi = np.ceil(max(0.0, interval_values.max()) / 2) * 2
        axes[1, 0].set_ylim(lo - 0.5, hi + 0.5)
        axes[0, 0].set_ylabel(r"GHI RMSE (W m$^{-2}$)")
        axes[1, 0].set_ylabel(r"RMSE difference (W m$^{-2}$)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.985), frameon=False, ncol=2, handlelength=2.7, columnspacing=2.5)
        fig.text(0.55, 0.055, "Forecast lead from history end (min)", ha="center", fontsize=10)
        fig.text(0.55, 0.010, "Gray band: inputs include observed history.  Difference: with COT minus without COT.", ha="center", fontsize=7.8)
        fig.savefig(OUT / "regional_forecast_lead.pdf", facecolor="white")
        fig.savefig(OUT / "regional_forecast_lead.png", dpi=360, facecolor="white")
        plt.close(fig)


def main():
    records, summaries, audits = [], [], []
    for region, directory in SOURCES.items():
        source = read_source(region, directory)
        q = score(source, np.ones(len(source["truth"]), dtype=bool))
        error = verify(q, source["stored"]["comparisons"]["full"])
        summaries.append({**q, "first_target_bjt": source["first_target_bjt"], "last_target_bjt": source["last_target_bjt"]})
        region_errors = [error]
        for lead in range(15, 241, 15):
            q = {"lead_minutes": lead, **score(source, source["leads"] == lead)}
            previous = (source["stored"]["comparisons"][f"lead_{lead}"] if region == "Hubei"
                        else next(v for v in source["stored"]["lead_metrics"] if v["lead_minutes"] == lead))
            region_errors.append(verify(q, previous))
            records.append(q)
        audits.append({
            "region": region, "source": str(directory.relative_to(ROOT)),
            "files_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                             for name in ("RESULTS.json", "predictions.npz", "keys.csv")},
            "max_absolute_difference_from_saved_point_and_ci": max(region_errors),
        })
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "records_by_station"} for r in records])
    df.to_csv(OUT / "lead_metrics.csv", index=False, float_format="%.12g")
    result = {"status": "PASS", "scope": "current regional DyDiff validation; observed-trained frozen GHI models",
              "bootstrap_repeats": BOOTSTRAPS, "bootstrap_seed": SEED,
              "bootstrap_unit": "BJT target date; shared resamples across methods, stations and seeds",
              "weighting": "RMSE within each station and each seed, then equal weights",
              "test_payloads_read": 0, "smoothing": "none", "subsampling": "none",
              "summaries": summaries, "sources": audits, "lead_metrics": records,
              "figure_size_inches": [7.0, 5.2], "png_dpi": 360,
              "versions": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__}}
    (OUT / "RECHECK.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    plot(records)
    print(json.dumps({"status": "PASS", "summaries": summaries, "max_error": max(a["max_absolute_difference_from_saved_point_and_ci"] for a in audits)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
