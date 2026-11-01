from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BINS = np.array([0.0, 0.1, 5.0, 30.0, 100.0001])
BIN_LABELS = ["<0.1", "0.1–5", "5–30", "≥30"]
BOOTSTRAPS = 2000
SEED = 20261017

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "mathtext.fontset": "stix",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def _hunan_data():
    base = ROOT / "refine-logs/cot_value_first_20261013"
    rows = ROOT / "refine-logs/cot_value_first_20261013/results/trainval_data_20261013/cpp_support_v1/cpp_file_support_by_frame.csv"
    pred = np.load(base / "validation_predictions_monthly_R_seed42.npz")
    keys = np.load(base / "clp_reference_monthly_R_seed42.npz")
    yhat = np.expm1(pred["pred_log1p_cot"].astype(np.float64))[:, 0]
    y = pred["reference_cot"].astype(np.float64)[:, 0]
    mask = pred["mask"][:, 0].astype(bool) & keys["common_mask"][:, 0].astype(bool)
    stations = keys["station"].astype(str)
    bjt = {
        r["target_time_utc"]: r["target_time_bjt"]
        for r in csv.DictReader(rows.open(encoding="utf-8-sig", newline=""))
        if r["split"] == "val"
    }
    times = keys["target_time_utc"].astype(str)
    assert all(t in bjt for t in times)
    dates = np.array([bjt[t][:10] for t in times])
    assert np.array_equal(pred["sample_indices"], keys["sample_indices"])
    return yhat, y, mask, dates, stations, "two station-centred 16×16 patches"


def _hubei_data():
    run = ROOT / "experiments/hubei_r_runtime_gate_20261015_v1/runs/hubei_R_station16_fixedv2_v1_seed42"
    yhat = np.load(run / "val_prediction_cot.npy").astype(np.float64)[:, 0]
    y = np.load(run / "val_reference_cpp_cot.npy").astype(np.float64)[:, 0]
    mask = np.load(run / "val_valid_mask.npy").astype(bool)[:, 0]
    rows = list(csv.DictReader((run / "val_keys.csv").open(encoding="utf-8-sig", newline="")))
    dates = np.array([r["time_bjt"][:8] for r in rows])
    assert len(dates) == len(yhat)
    audit = json.loads((run / "independent_validation_fp64_audit.json").read_text(encoding="utf-8"))
    assert audit["test_payloads_read"] == 0 and int(mask.sum()) == audit["metrics"]["valid_cpp_pixels"]
    return yhat, y, mask, dates, None, "shared 16×16 three-station patch"


def _metric(yhat, y, mask):
    e = yhat - y
    return {
        "n": int(mask.sum()),
        "rmse": float(np.sqrt(np.mean(e[mask] ** 2))),
        "mae": float(np.mean(np.abs(e[mask]))),
        "bias": float(np.mean(e[mask])),
    }


def _strata(yhat, y, mask, dates, stations):
    error = yhat - y
    rng = np.random.default_rng(SEED)
    unique_dates = np.unique(dates)
    date_index = np.searchsorted(unique_dates, dates)
    station_values = np.unique(stations) if stations is not None else np.array(["region"])
    station_index = np.searchsorted(station_values, stations) if stations is not None else np.zeros(len(dates), dtype=int)
    multiplicity = rng.multinomial(
        len(unique_dates), np.full(len(unique_dates), 1.0 / len(unique_dates)), size=BOOTSTRAPS
    )
    out = []
    for i, (lo, hi) in enumerate(zip(BINS[:-1], BINS[1:])):
        band = mask & (y >= lo) & (y < hi)
        sums = np.zeros((len(unique_dates), len(station_values)), dtype=np.float64)
        counts = np.zeros_like(sums)
        for d in range(len(unique_dates)):
            rows = date_index == d
            for s in range(len(station_values)):
                row_mask = rows & (station_index == s)
                if row_mask.any():
                    pix_mask = band[row_mask]
                    sums[d, s] = error[row_mask][pix_mask].sum()
                    counts[d, s] = pix_mask.sum()
        pooled_sum = sums.sum(axis=0)
        pooled_count = counts.sum(axis=0)
        valid_station = pooled_count > 0
        point = float(np.mean(pooled_sum[valid_station] / pooled_count[valid_station])) if valid_station.any() else np.nan
        n = int(pooled_count.sum())
        sampled_sums = multiplicity @ sums
        sampled_counts = multiplicity @ counts
        station_means = np.divide(
            sampled_sums, sampled_counts, out=np.full_like(sampled_sums, np.nan),
            where=sampled_counts > 0,
        )
        valid_draw_station = np.isfinite(station_means)
        draws = np.divide(
            np.nansum(station_means, axis=1), valid_draw_station.sum(axis=1),
            out=np.full(BOOTSTRAPS, np.nan), where=valid_draw_station.sum(axis=1) > 0,
        )
        good = draws[np.isfinite(draws)]
        lower, upper = np.quantile(good, [0.025, 0.975]) if len(good) else (np.nan, np.nan)
        n_dates = int(np.sum(counts.sum(axis=1) > 0))
        out.append({
            "bin": BIN_LABELS[i],
        "lower": float(lo),
        "upper": float(min(hi, 100.0)),
            "n_pixels": n,
            "n_dates": n_dates,
            "bias": point,
            "ci_low": float(lower),
            "ci_high": float(upper),
        })
    return out


def _draw_region(ax_scatter, ax_bias, data, title, color):
    yhat, y, mask, dates, stations, support_desc = data
    yt, yr = yhat[mask], y[mask]
    hist = ax_scatter.hist2d(
        yr, yt, bins=70, range=[[0, 100], [0, 100]],
        norm=matplotlib.colors.LogNorm(vmin=1), cmap="magma",
    )
    ax_scatter.plot([0, 100], [0, 100], color="white", lw=0.9, ls="--", alpha=0.95)
    ax_scatter.set(
        xlim=(0, 100), ylim=(0, 100), xlabel="CPP reference COT",
        ylabel="Retrieved COT", title=title,
    )
    metric = _metric(yhat, y, mask)
    ax_scatter.text(
        0.04, 0.96,
        f"N = {metric['n']:,} pixels\nPooled RMSE = {metric['rmse']:.2f}; bias = {metric['bias']:+.2f}\n{support_desc}",
        transform=ax_scatter.transAxes, va="top", ha="left", fontsize=7,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.8", "alpha": 0.92},
    )
    ax_scatter.set_aspect("equal", adjustable="box")
    strata = _strata(yhat, y, mask, dates, stations)
    xs = np.arange(len(strata))
    vals = np.array([x["bias"] for x in strata])
    lo = vals - np.array([x["ci_low"] for x in strata])
    hi = np.array([x["ci_high"] for x in strata]) - vals
    ax_bias.errorbar(xs, vals, yerr=np.vstack([lo, hi]), marker="o", lw=1.2, ms=4,
                     color=color, capsize=2.5, elinewidth=0.9)
    ax_bias.axhline(0, color="0.25", lw=0.8)
    ax_bias.set_xticks(xs, [f"{x['bin']}\n{x['n_pixels'] / 1000:.1f}k px\n{x['n_dates']} dates" for x in strata])
    ax_bias.tick_params(axis="x", labelsize=6.5, pad=2)
    ax_bias.set_ylabel("Bias (retrieved − CPP), COT")
    ax_bias.set_xlabel("CPP reference thickness bin")
    ax_bias.set_title("Thickness-conditioned retrieval bias")
    ax_bias.grid(axis="y", color="0.88", lw=0.6)
    return metric, strata, support_desc, int(len(np.unique(dates))), hist[3]


def main():
    hn = _hunan_data()
    hb = _hubei_data()
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 6.0), constrained_layout=True)
    summary = {}
    summary["Hunan"], hunan_strata, hn_support, hn_dates, hn_mesh = _draw_region(
        axes[0, 0], axes[1, 0], hn, "Hunan validation", "#277da1"
    )
    summary["Hubei"], hubei_strata, hb_support, hb_dates, _ = _draw_region(
        axes[0, 1], axes[1, 1], hb, "Hubei validation", "#d95f02"
    )
    for ax in axes[1]:
        ax.set_ylim(-25, 25)
    cbar = fig.colorbar(hn_mesh, ax=list(axes[0]), shrink=0.86, pad=0.025)
    cbar.set_label("Pixel count per bin (log scale)", fontsize=7.5)
    fig.suptitle("COT retrieval against CPP reference", fontsize=11, y=1.02)
    fig.text(
        0.5, -0.015,
        "CPP is a retrieval reference, not independent physical truth. Intervals are date-cluster bootstrap (2,000 draws); bins are descriptive and unadjusted.",
        ha="center", va="top", fontsize=7.3,
    )
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / f"fig3_retrieval_quality.{ext}")
    plt.close(fig)

    summary["metadata"] = {
        "reference": "FY-4B CPP retrieval, not independent truth",
        "common_mask_only": True,
        "Hunan_support": hn_support,
        "Hubei_support": hb_support,
        "Hunan_unique_BJT_dates": hn_dates,
        "Hubei_unique_BJT_dates": hb_dates,
        "bootstrap_unit": "BJT target date from audited CPP support manifest; paired pixels retained within date; station means equally weighted within Hunan",
        "bootstrap_draws": BOOTSTRAPS,
        "seed": SEED,
        "test_payloads_read": 0,
        "sources_sha256": {
            "hunan_prediction_and_cpp": hashlib.sha256(
                (ROOT / "refine-logs/cot_value_first_20261013/validation_predictions_monthly_R_seed42.npz").read_bytes()
            ).hexdigest(),
            "hunan_reference_keys": hashlib.sha256(
                (ROOT / "refine-logs/cot_value_first_20261013/clp_reference_monthly_R_seed42.npz").read_bytes()
            ).hexdigest(),
            "hunan_bjt_time_mapping": hashlib.sha256(
                (ROOT / "refine-logs/cot_value_first_20261013/results/trainval_data_20261013/cpp_support_v1/cpp_file_support_by_frame.csv").read_bytes()
            ).hexdigest(),
            "hubei_prediction": hashlib.sha256(
                (ROOT / "experiments/hubei_r_runtime_gate_20261015_v1/runs/hubei_R_station16_fixedv2_v1_seed42/val_prediction_cot.npy").read_bytes()
            ).hexdigest(),
            "hubei_cpp_reference": hashlib.sha256(
                (ROOT / "experiments/hubei_r_runtime_gate_20261015_v1/runs/hubei_R_station16_fixedv2_v1_seed42/val_reference_cpp_cot.npy").read_bytes()
            ).hexdigest(),
            "hubei_mask": hashlib.sha256(
                (ROOT / "experiments/hubei_r_runtime_gate_20261015_v1/runs/hubei_R_station16_fixedv2_v1_seed42/val_valid_mask.npy").read_bytes()
            ).hexdigest(),
            "hubei_keys": hashlib.sha256(
                (ROOT / "experiments/hubei_r_runtime_gate_20261015_v1/runs/hubei_R_station16_fixedv2_v1_seed42/val_keys.csv").read_bytes()
            ).hexdigest(),
            "hubei_audit": hashlib.sha256(
                (ROOT / "experiments/hubei_r_runtime_gate_20261015_v1/runs/hubei_R_station16_fixedv2_v1_seed42/independent_validation_fp64_audit.json").read_bytes()
            ).hexdigest(),
        },
        "strata_Hunan": hunan_strata,
        "strata_Hubei": hubei_strata,
    }
    (OUT / "fig3_retrieval_quality_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (OUT / "fig3_retrieval_quality_strata.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["region", "bin", "lower", "upper", "n_pixels", "n_dates", "bias", "ci_low", "ci_high"]
        )
        writer.writeheader()
        for region, rows in (("Hunan", hunan_strata), ("Hubei", hubei_strata)):
            for row in rows:
                writer.writerow({"region": region, **row})
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
