from __future__ import annotations

import csv
import hashlib
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SEED = 20261017
N_BOOT = 5000
STATIONS = ["Sili", "Zhujia"]
METHODS = {
    "original_simvp": "forecast_log1p_cot",
    "mse_finetune": "forecast_log1p_cot",
    "temporal_change_finetune": "forecast_log1p_cot",
    "spatial_gradient_finetune": "forecast_log1p_cot",
    "observed_future_agri_to_R_offline": "target_agri_r_log1p_cot",
    "last_frame_persistence": "persistence_log1p_cot",
}


def rmse(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def date_key(s: str) -> str:
    dt = datetime.strptime(s.replace("T", " "), "%Y-%m-%d %H:%M:%S") + timedelta(hours=8)
    return dt.strftime("%Y-%m-%d")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True,
                        help="Experiment results directory containing original and R07 banks")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    results_root = args.results_root.resolve()
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    run = results_root / "simvp_supervision_cot_allval_20261017_v2"
    banks = {
        "original_simvp": results_root / "forecast_agri3x3_allval_20261014_v1" / "forecast_cot_val_bank.npz",
        "mse_finetune": run / "mse" / "forecast_cot_val_bank.npz",
        "temporal_change_finetune": run / "temporal" / "forecast_cot_val_bank.npz",
        "spatial_gradient_finetune": run / "spatial" / "forecast_cot_val_bank.npz",
    }
    # The current source-keyed all-validation CPP reference is stored in each R07 arm.
    ref_bank = banks["mse_finetune"]
    zref = np.load(ref_bank, allow_pickle=False)
    ref = zref["cpp_reference_cot"].astype(np.float32)
    valid = zref["cpp_valid_mask"].astype(bool)
    times = zref["target_time_utc"].astype(str)
    if ref.shape != valid.shape or ref.shape[2] != 16:
        raise ValueError(f"unexpected COT/mask shape: {ref.shape}, {valid.shape}")
    if len(np.unique(zref["sequence_ids"])) != ref.shape[0]:
        raise ValueError("duplicate sequence IDs in source bank")
    # Align every candidate to the exact source keys and current CPP support.
    pred = {}
    key_arrays = ("sequence_ids", "target_time_utc", "dataset_indices")
    for method, path in banks.items():
        zi = np.load(path, allow_pickle=False)
        for key in key_arrays:
            if not np.array_equal(zi[key], zref[key]):
                raise RuntimeError(f"{method}: source key mismatch for {key}")
        if method != "original_simvp":
            if not np.array_equal(zi["cpp_valid_mask"], zref["cpp_valid_mask"]):
                raise RuntimeError(f"{method}: CPP mask differs from common reference bank")
            if not np.array_equal(zi["cpp_reference_cot"], zref["cpp_reference_cot"], equal_nan=True):
                raise RuntimeError(f"{method}: CPP values differ from common reference bank")
        pred[method] = np.expm1(zi[METHODS[method]].astype(np.float64))
    pred["observed_future_agri_to_R_offline"] = np.expm1(zref["target_agri_r_log1p_cot"].astype(np.float64))
    pred["last_frame_persistence"] = np.expm1(zref["persistence_log1p_cot"].astype(np.float64))
    # Restrict analysis to the station-centred 3x3 field used by the GHI head.
    sl = (slice(None), slice(None), slice(None), slice(6, 9), slice(6, 9))
    ref = ref[sl].astype(np.float64)
    valid = valid[sl]
    pred = {k: v[sl] for k, v in pred.items()}
    if any(not np.isfinite(v).all() for v in pred.values()):
        raise RuntimeError("non-finite COT forecast or comparator values")

    # Compare 15-minute COT changes only where CPP is valid at both endpoints.
    common = valid[:, :, :-1] & valid[:, :, 1:]
    ref_delta = ref[:, :, 1:] - ref[:, :, :-1]
    pred_delta = {k: v[:, :, 1:] - v[:, :, :-1] for k, v in pred.items()}
    pair_dates = np.empty(common.shape, dtype=object)
    for i in range(common.shape[0]):
        for t in range(common.shape[2]):
            pair_dates[i, :, t] = date_key(times[i, t + 1])

    rows: list[dict] = []
    detailed: dict[str, dict] = {}
    rng = np.random.default_rng(SEED)
    for si, station in enumerate(STATIONS):
        mask = common[:, si]
        dtrue = ref_delta[:, si][mask]
        dates = pair_dates[:, si][mask]
        unique_dates = np.unique(dates)
        date_sse_ref = {
            d: float(np.square(dtrue[dates == d]).sum()) for d in unique_dates
        }
        date_n = {d: int(np.sum(dates == d)) for d in unique_dates}
        total_n = int(mask.sum())
        persistence_err = -dtrue
        site_result = {
            "n_pixel_transitions": total_n,
            "target_dates_bjt": int(len(unique_dates)),
            "actual_change_rmse": rmse(dtrue),
            "methods": {},
        }
        for method, dmap in pred_delta.items():
            dpred = dmap[:, si][mask]
            err = dpred - dtrue
            # Date-cluster bootstrap from paired SSE/count sufficient statistics.
            sse_by_date = {d: float(np.square(err[dates == d]).sum()) for d in unique_dates}
            sampled = rng.integers(0, len(unique_dates), size=(N_BOOT, len(unique_dates)))
            boots = np.empty(N_BOOT, dtype=np.float64)
            date_list = list(unique_dates)
            for b, picks in enumerate(sampled):
                n = sum(date_n[date_list[j]] for j in picks)
                sse = sum(sse_by_date[date_list[j]] for j in picks)
                boots[b] = np.sqrt(sse / n)
            d_rmse = rmse(err)
            slope = float(np.dot(dpred, dtrue) / max(np.dot(dtrue, dtrue), 1e-12))
            corr = float(np.corrcoef(dpred, dtrue)[0, 1]) if np.std(dpred) > 0 and np.std(dtrue) > 0 else None
            result = {
                "delta_rmse": d_rmse,
                "delta_mae": float(np.mean(np.abs(err))),
                "delta_bias": float(np.mean(err)),
                "through_origin_slope_predicted_vs_reference_change": slope,
                "change_correlation": corr,
                "date_block_95pct_rmse_interval": [float(x) for x in np.quantile(boots, [0.025, 0.975])],
                "n_pixel_transitions": total_n,
                "target_dates_bjt": int(len(unique_dates)),
            }
            site_result["methods"][method] = result
            rows.append({"station": station, "method": method, **result})

        # Paired date-block bootstrap for each forecast method vs persistence.
        paired = {}
        for method, dmap in pred_delta.items():
            if method == "last_frame_persistence":
                continue
            e_model = dmap[:, si][mask] - dtrue
            e_pers = persistence_err
            diffs = []
            sse_model = {d: float(np.square(e_model[dates == d]).sum()) for d in unique_dates}
            sse_pers = {d: float(np.square(e_pers[dates == d]).sum()) for d in unique_dates}
            for picks in sampled:
                n = sum(date_n[date_list[j]] for j in picks)
                a = np.sqrt(sum(sse_model[date_list[j]] for j in picks) / n)
                b = np.sqrt(sum(sse_pers[date_list[j]] for j in picks) / n)
                diffs.append(a - b)
            observed = rmse(e_model) - rmse(e_pers)
            paired[method + "_minus_persistence_delta_RMSE"] = {
                "difference": observed,
                "date_block_95pct_interval": [float(x) for x in np.quantile(diffs, [0.025, 0.975])],
                "negative_favors_forecast_change": True,
            }
        site_result["paired_comparisons"] = paired
        detailed[station] = site_result

    # Full-cohort temporal response against R(observed future AGRI). This is
    # an offline proxy comparison, not a score against physical COT truth.
    proxy_ref = np.expm1(zref["target_agri_r_log1p_cot"].astype(np.float64))[sl]
    proxy_common = np.isfinite(proxy_ref[:, :, :-1]) & np.isfinite(proxy_ref[:, :, 1:])
    proxy_delta_ref = proxy_ref[:, :, 1:] - proxy_ref[:, :, :-1]
    proxy_results: dict[str, dict] = {}
    proxy_rows: list[dict] = []
    for si, station in enumerate(STATIONS):
        mask = proxy_common[:, si]
        dtrue = proxy_delta_ref[:, si][mask]
        dates = pair_dates[:, si][mask]
        unique_dates = np.unique(dates)
        date_list = list(unique_dates)
        date_n = {d: int(np.sum(dates == d)) for d in unique_dates}
        total_n = int(mask.sum())
        sse_methods: dict[str, dict[str, float]] = {}
        metrics: dict[str, dict] = {}
        for method, dmap in pred_delta.items():
            dpred = dmap[:, si][mask]
            err = dpred - dtrue
            sse = {d: float(np.square(err[dates == d]).sum()) for d in unique_dates}
            sse_methods[method] = sse
            metrics[method] = {
                "delta_rmse": rmse(err),
                "delta_mae": float(np.mean(np.abs(err))),
                "delta_bias": float(np.mean(err)),
                "reference_change_rmse": rmse(dtrue),
                "through_origin_slope_predicted_vs_reference_change": float(np.dot(dpred, dtrue) / max(np.dot(dtrue, dtrue), 1e-12)),
                "change_correlation": float(np.corrcoef(dpred, dtrue)[0, 1]) if np.std(dpred) > 0 and np.std(dtrue) > 0 else None,
                "n_pixel_transitions": total_n,
                "target_dates_bjt": int(len(unique_dates)),
            }
        rng_proxy = np.random.default_rng(SEED + si)
        sampled_proxy = rng_proxy.integers(0, len(unique_dates), size=(N_BOOT, len(unique_dates)))
        for method in ("mse_finetune", "temporal_change_finetune", "spatial_gradient_finetune"):
            diffs = []
            for picks in sampled_proxy:
                n = sum(date_n[date_list[j]] for j in picks)
                a = np.sqrt(sum(sse_methods[method][date_list[j]] for j in picks) / n)
                b = np.sqrt(sum(sse_methods["original_simvp"][date_list[j]] for j in picks) / n)
                diffs.append(a - b)
            metrics[method]["paired_delta_RMSE_minus_original_simvp"] = {
                "difference": metrics[method]["delta_rmse"] - metrics["original_simvp"]["delta_rmse"],
                "date_block_95pct_interval": [float(x) for x in np.quantile(diffs, [0.025, 0.975])],
            }
        persistence_sse = sse_methods["last_frame_persistence"]
        for method in ("original_simvp", "mse_finetune", "temporal_change_finetune", "spatial_gradient_finetune"):
            diffs = []
            for picks in sampled_proxy:
                n = sum(date_n[date_list[j]] for j in picks)
                a = np.sqrt(sum(sse_methods[method][date_list[j]] for j in picks) / n)
                b = np.sqrt(sum(persistence_sse[date_list[j]] for j in picks) / n)
                diffs.append(a - b)
            metrics[method]["paired_delta_RMSE_minus_persistence"] = {
                "difference": metrics[method]["delta_rmse"] - metrics["last_frame_persistence"]["delta_rmse"],
                "date_block_95pct_interval": [float(x) for x in np.quantile(diffs, [0.025, 0.975])],
                "negative_favors_model_trajectory": True,
            }
        proxy_results[station] = {
            "n_pixel_transitions": total_n,
            "target_dates_bjt": int(len(unique_dates)),
            "reference": "R(observed future AGRI), offline proxy response; not independent COT truth",
            "methods": metrics,
        }
        for method, result in metrics.items():
            proxy_rows.append({"station": station, "method": method, **result})

    summary = {
        "state": "PASS_CPP_COMMON_SUPPORT_ADJACENT_LEAD_CHANGE_AUDIT",
        "purpose": "Does the SimVP-to-R forecast preserve observed 15-minute COT evolution, and is the loss upstream in AGRI extrapolation or downstream retrieval response?",
        "bank": str(ref_bank),
        "bank_sha256": sha256(ref_bank),
        "split": "validation only; no test payloads read",
        "support": "CPP analysis: CPP valid at both adjacent forecast leads in station-centred 3x3. Proxy analysis: finite R(observed future AGRI) at both adjacent leads over the full validation cohort.",
        "temporal_definition": "delta at lead t+15 minus lead t for adjacent +15...+240 min target slots; lead pairs are evaluated within each forecast sequence",
        "reference_limit": "CPP is a retrieval reference, not independent physical truth; R(observed future AGRI) is an offline retrieval-response comparator, not truth",
        "overlap_note": "Forecast windows overlap; uncertainty resamples BJT target-date clusters, not individual pixel transitions. Repeated target transitions across initializations remain distinct forecast cases.",
        "caveat": "A strong/weak change score does not isolate cloud-motion error from retrieval error; interpret offline R(observed future AGRI) only as a decomposition aid.",
        "seed": SEED,
        "bootstrap_replicates": N_BOOT,
        "stations": detailed,
        "full_cohort_proxy_response_temporal_change": proxy_results,
    }
    summary["banks"] = {k: {"path": str(v), "sha256": sha256(v)} for k, v in banks.items()}
    summary["reference_source_bank"] = {"path": str(ref_bank), "sha256": sha256(ref_bank)}
    out_json = outdir / "cot_temporal_change_summary.json"
    out_csv = outdir / "cot_temporal_change_metrics.csv"
    out_proxy_csv = outdir / "cot_proxy_temporal_change_metrics.csv"
    out_png = outdir / "cot_temporal_change_scatter.png"
    out_pdf = outdir / "cot_temporal_change_scatter.pdf"
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fields = list(rows[0])
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with out_proxy_csv.open("w", newline="", encoding="utf-8") as f:
        fields = list(dict.fromkeys(key for row in proxy_rows for key in row))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(proxy_rows)

    # Top row: full-cohort model-to-R-response decomposition. Bottom row:
    # independent-but-sparse CPP retrieval reference on common adjacent times.
    fig, axs = plt.subplots(2, 2, figsize=(9.2, 7.0), constrained_layout=True)
    for si, station in enumerate(STATIONS):
        for row, xall, yall, color, reference_label, n_dates in (
            (0, proxy_delta_ref[:, si][proxy_common[:, si]], pred_delta["original_simvp"][:, si][proxy_common[:, si]], "Blues", "R(observed future AGRI) change (offline proxy)", proxy_results[station]["target_dates_bjt"]),
            (1, ref_delta[:, si][common[:, si]], pred_delta["original_simvp"][:, si][common[:, si]], "Oranges", "CPP reference change", detailed[station]["target_dates_bjt"]),
        ):
            ax = axs[row, si]
            lim = float(np.quantile(np.abs(np.concatenate([xall, yall])), 0.995))
            lim = max(lim, 1.0)
            keep = (np.abs(xall) <= lim) & (np.abs(yall) <= lim)
            ax.hexbin(xall[keep], yall[keep], gridsize=48, mincnt=1, bins="log", cmap=color, extent=(-lim, lim, -lim, lim))
            ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.9)
            ax.axhline(0, color="0.4", lw=0.5)
            ax.axvline(0, color="0.4", lw=0.5)
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel=reference_label, ylabel="SimVP forecast→R 15-min COT change", title=f"{station}: {n_dates} BJT dates")
            clipped_pct = 100.0 * (1.0 - np.mean(keep))
            ax.text(0.02, 0.98, f"n={len(xall):,}; clipped={clipped_pct:.2f}%", transform=ax.transAxes, va="top", fontsize=7,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    fig.suptitle("Forecast COT trajectory change versus available references", fontsize=11)
    fig.text(0.5, -0.005, "All panels show adjacent +15-min changes; axes clip only the outer 0.5% tails (fraction reported). Top: full-cohort offline retrieval-response proxy, not truth. Bottom: sparse CPP retrieval reference, not independent physical truth.", ha="center", fontsize=7)
    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)
    print(json.dumps({"summary": str(out_json), "table": str(out_csv), "figure": str(out_pdf), "cpp_source_bank_sha256": summary["reference_source_bank"]["sha256"]}, indent=2))


if __name__ == "__main__":
    main()

