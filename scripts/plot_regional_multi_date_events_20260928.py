"""Plot multi-date validation events with common scaling and paired interval GHI."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/regional_station_cot_20260927"
DATA = RESULTS / "expanded_event_attribution_20260928"
KEYS = ROOT / "audits/regional_error_attribution_20260928/match_keys"
GHI_PRED = RESULTS / "ghi_pilot/forecast_validation_predictions.npz"
LEADS = np.arange(1, 17) * 15
OLD_EVENT_FILES = {
    2311: RESULTS / "sili_trajectories_v2/simvp_better_source.npz",
    2869: RESULTS / "sili_thick_motion/nonzero_motion_cloud_source.npz",
    534: RESULTS / "sili_thick_motion/largest_thick_peak_miss_source.npz",
}
LABELS = {
    "growth": "Growth",
    "decay": "Decay",
    "motion_proxy": "Motion proxy",
    "low_cot_control": "Low-COT control",
    "existing_anchor": "Prior anchor",
}


def paired_ghi(sequence: int) -> tuple[np.ndarray, np.ndarray]:
    pred = np.load(GHI_PRED)
    row_ids = np.load(KEYS / "head_row_ids.npy")
    seq = np.load(KEYS / "sequence.npy")
    station = np.load(KEYS / "station.npy")
    lead = np.load(KEYS / "lead.npy")
    lookup = {int(r): (float(y), float(p)) for r, y, p in
              zip(pred["row_ids"], pred["true_ghi"], pred["pred_ghi"])}
    result = np.full((2, 16), np.nan, np.float64)
    for r in row_ids:
        ri = int(r)
        if seq[ri] == sequence and station[ri] == 0 and ri in lookup:
            result[:, int(lead[ri])] = lookup[ri]
    return result[0], result[1]


def rmse(pred: np.ndarray, true: np.ndarray, valid: np.ndarray) -> tuple[float | None, int]:
    ok = valid & np.isfinite(pred) & np.isfinite(true)
    return (float(np.sqrt(np.mean((pred[ok] - true[ok]) ** 2))) if ok.any() else None,
            int(ok.sum()))


def main() -> None:
    catalog = json.loads((DATA / "event_candidate_catalog.json").read_text(encoding="utf-8"))
    meta = {int(r["sequence"]): r for r in catalog["all_candidates"]}
    selected = {int(r["sequence"]): r for r in catalog["selected"]}
    # Keep the three earlier reviewed cases as anchors; the new deterministic
    # selection adds cases across dates and preserves the prior event examples.
    anchors = [
        (2311, "decay", "2025-08-05T09:15:00", "Legacy anchor; selected in earlier model-error diagnostic"),
        (2869, "motion_proxy", "2025-08-14T11:30:00", "Legacy anchor; selected in earlier motion/thick-cloud diagnostic"),
        (534, "growth", "2025-07-09T12:15:00", "Legacy anchor; selected earlier for largest thick-cloud miss"),
    ]
    cases = []
    for seq, category, init, description in anchors:
        cases.append({"sequence": seq, "selected_category": category, "init_bjt": init,
                      "selection_description": description, "source": OLD_EVENT_FILES[seq]})
    for seq, rec in selected.items():
        if seq == 534 and any(c["sequence"] == seq for c in cases):
            continue
        cases.append({**rec, "source": DATA / f"event_seq{seq:04d}.npz"})
    cases.sort(key=lambda c: (c["init_bjt"], c["sequence"]))

    # Common false-colour channel stretch across all selected dates and leads.
    rgb_samples = []
    for c in cases:
        with np.load(c["source"]) as z:
            rgb = z["rgb"]
            valid_rgb = np.isfinite(rgb).all(axis=-1)
            if valid_rgb.any():
                rgb_samples.append(rgb[valid_rgb])
    pooled = np.concatenate(rgb_samples, axis=0)
    lo = np.percentile(pooled, 2, axis=0)
    hi = np.percentile(pooled, 98, axis=0)
    hi = np.maximum(hi, lo + 1e-8)

    n = len(cases)
    shown = [0, 3, 7, 11, 15]
    rgb_fig, rgb_axes = plt.subplots(n, len(shown), figsize=(14, 2.15*n), layout="constrained")
    full_rgb_fig, full_rgb_axes = plt.subplots(n, 16, figsize=(24, 1.9*n), layout="constrained")
    curve_fig, curve_axes = plt.subplots(n, 2, figsize=(14, 2.25*n), layout="constrained")
    report = {
        "test_used": False, "training_performed": False,
        "cohort": catalog["cohort"], "reference": catalog["reference"],
        "selection_rule": catalog["selection_rule"], "candidate_counts": catalog["candidate_counts"],
        "case_count_including_prior_anchors": n,
        "selection_bias_note": "The three legacy anchors were selected in earlier outcome-aware diagnostics and are shown for continuity only; they are excluded from the new rule-selected sample and must not be used to estimate event prevalence or forecasting success rates.",
        "ghi_label": "Observed 15-minute interval mean; targets +15 to +240 minutes from history end.",
        "rgb": {"channels": "C03/C02/C01 false-colour proxy", "common_percentile_stretch": [2, 98],
                "channel_low": lo.tolist(), "channel_high": hi.tolist()},
        "cases": [],
    }
    csv_rows = []
    colors = {"CPP retrieval reference": "#222222", "R(real future AGRI) · oracle": "#009E73",
              "R(SimVP forecast AGRI)": "#0072B2", "history + fixed translation": "#D55E00"}
    styles = {"CPP retrieval reference": "o-", "R(real future AGRI) · oracle": "^--",
              "R(SimVP forecast AGRI)": "s-", "history + fixed translation": "D-"}
    all_ghi = []

    for i, c in enumerate(cases):
        with np.load(c["source"]) as z:
            fields = {k: np.asarray(z[k]) for k in z.files}
        ref = fields["reference"].astype(np.float32)
        valid = fields["valid"].astype(bool)
        # Show actual future AGRI as false colour, using one scale for all cases.
        rgb = fields["rgb"].astype(np.float32)
        display = np.clip((rgb - lo[None, None, None, :]) /
                          (hi - lo + 1e-8)[None, None, None, :], 0, 1)
        seq, category = int(c["sequence"]), c["selected_category"]
        init = c["init_bjt"].replace("T", " ") + " BJT"
        for ti in range(16):
            ax = full_rgb_axes[i, ti] if n > 1 else full_rgb_axes[ti]
            rgb_valid = np.isfinite(display[ti]).all(axis=-1)
            rgba = np.zeros((*display[ti].shape[:2], 4), dtype=np.float32)
            rgba[..., :3] = np.nan_to_num(display[ti], nan=0.0)
            rgba[..., 3] = rgb_valid.astype(np.float32)
            ax.set_facecolor("#eeeeee")
            ax.imshow(rgba, interpolation="nearest")
            ax.plot(32, 32, marker="+", ms=5, mew=1, color="cyan")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"+{LEADS[ti]}", fontsize=7)
            if ti == 0:
                ax.set_ylabel(f"{LABELS.get(category, category)}\n{init}\nseq {seq}", fontsize=7)
            if float(rgb_valid.mean()) < .9:
                ax.text(.5, .5, "missing", transform=ax.transAxes, ha="center", va="center",
                        fontsize=6, color="#222222", bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
        for col, ti in enumerate(shown):
            ax = rgb_axes[i, col] if n > 1 else rgb_axes[col]
            rgb_valid = np.isfinite(display[ti]).all(axis=-1)
            rgba = np.zeros((*display[ti].shape[:2], 4), dtype=np.float32)
            rgba[..., :3] = np.nan_to_num(display[ti], nan=0.0)
            rgba[..., 3] = rgb_valid.astype(np.float32)
            ax.set_facecolor("#eeeeee")
            ax.imshow(rgba, interpolation="nearest")
            coverage = float(rgb_valid.mean())
            if coverage < .9:
                label = "AGRI missing" if coverage == 0 else f"partial AGRI\n{coverage:.0%} pixels"
                ax.text(.5, .5, label, transform=ax.transAxes, ha="center", va="center",
                        fontsize=8, color="#222222", bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
            ax.plot(32, 32, marker="+", ms=9, mew=1.5, color="cyan")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"+{LEADS[ti]} min")
            if col == 0:
                ax.text(-.12, .5, f"{LABELS.get(category, category)}\n{init}\nseq {seq}",
                        transform=ax.transAxes, ha="right", va="center", fontsize=8)

        cot_ax = curve_axes[i, 0] if n > 1 else curve_axes[0]
        ghi_ax = curve_axes[i, 1] if n > 1 else curve_axes[1]
        station_values = {
            "CPP retrieval reference": np.where(valid[:, 32, 32], ref[:, 32, 32], np.nan),
            "R(real future AGRI) · oracle": fields["oracle"][:, 32, 32],
            "R(SimVP forecast AGRI)": fields["simvp"][:, 32, 32],
            "history + fixed translation": fields["transport"][:, 32, 32],
        }
        for name, y in station_values.items():
            cot_ax.plot(LEADS, y, styles[name], color=colors[name], lw=1.35, ms=3.4,
                        label=name)
        cot_ax.axhline(30, color="#777777", ls=":", lw=.8)
        cot_ax.set_ylim(0, 70)
        cot_ax.grid(alpha=.22)
        cot_ax.set_ylabel("COT", fontsize=8)
        if i == 0:
            cot_ax.legend(fontsize=7, ncol=2, loc="upper left", frameon=False)
        cot_ax.text(.99, .91, f"{LABELS.get(category, category)} · {init}",
                    transform=cot_ax.transAxes, ha="right", va="top", fontsize=8)

        ghi_true, ghi_pred = paired_ghi(seq)
        all_ghi.extend(ghi_true[np.isfinite(ghi_true)].tolist())
        all_ghi.extend(ghi_pred[np.isfinite(ghi_pred)].tolist())
        ghi_ax.plot(LEADS, ghi_true, "o-", color="#222222", lw=1.4, ms=3.5,
                    label="Observed 15-min interval mean")
        ghi_ax.plot(LEADS, ghi_pred, "s--", color="#CC6677", lw=1.25, ms=3.3,
                    label="Saved COT-GHI pilot prediction")
        ghi_ax.grid(alpha=.22)
        ghi_ax.set_ylabel("GHI\n(W m$^{-2}$)", fontsize=8)
        if i == 0:
            ghi_ax.legend(fontsize=7, loc="upper left", frameon=False)

        simvp_rmse, cot_n = rmse(fields["simvp"][:, 32, 32], ref[:, 32, 32], valid[:, 32, 32])
        oracle_rmse, _ = rmse(fields["oracle"][:, 32, 32], ref[:, 32, 32], valid[:, 32, 32])
        ghi_ok = np.isfinite(ghi_true) & np.isfinite(ghi_pred)
        ghi_rmse = float(np.sqrt(np.mean((ghi_pred[ghi_ok] - ghi_true[ghi_ok])**2))) if ghi_ok.any() else None
        metrics = c.get("trajectory_metrics", meta.get(seq, {}).get("trajectory_metrics", {}))
        row = {"category": category, "sequence": seq, "init_bjt": init,
               "valid_CPP_station_leads": cot_n, "CPP_peak_COT": metrics.get("peak_cot"),
               "CPP_peak_lead_min": metrics.get("peak_lead_min"),
               "CPP_trajectory_range_COT": metrics.get("trajectory_range_cot"),
               "history_motion_norm_pixels_per15": metrics.get("motion_norm_pixels_per15"),
               "station_R_forecast_vs_CPP_RMSE_COT": simvp_rmse,
               "station_R_real_AGRI_oracle_vs_CPP_RMSE_COT": oracle_rmse,
               "paired_GHI_leads": int(ghi_ok.sum()), "GHI_pilot_RMSE_Wm2": ghi_rmse,
               "selection_description": c.get("selection_description", "Rule-selected validation event; not random")}
        csv_rows.append(row)
        report["cases"].append({**row,
            "ghi_observed_interval_mean": [float(x) if np.isfinite(x) else None for x in ghi_true],
            "ghi_saved_pilot_prediction": [float(x) if np.isfinite(x) else None for x in ghi_pred],
            "cot_station_series": {k: [float(x) if np.isfinite(x) else None for x in v]
                                   for k, v in station_values.items()},
            "future_CPP_use": "validation reference only; not input to forecast"})

    max_ghi = max(all_ghi) if all_ghi else 1200
    ghi_top = max(1200, int(np.ceil(max_ghi / 200) * 200))
    for i in range(n):
        ghi_ax = curve_axes[i, 1] if n > 1 else curve_axes[1]
        ghi_ax.set_ylim(0, ghi_top)
    for ax in (curve_axes[-1] if n > 1 else curve_axes):
        ax.set_xlabel("Lead from history end (min)")
    for ax in (rgb_axes[-1] if n > 1 else rgb_axes):
        ax.set_xlabel("Actual future AGRI · false colour")

    rgb_fig.suptitle("Sili validation cloud events across dates · actual AGRI C03/C02/C01 proxy\n"
                     "Five snapshots per 4 h sequence; common 2–98% RGB stretch; cyan cross = station pixel")
    full_rgb_fig.suptitle("Full 16-lead actual AGRI progression across validation dates · C03/C02/C01 false colour\n"
                          "One column per 15-min lead; common 2–98% channel stretch; missing source frames are marked")
    curve_fig.suptitle("Multi-date station COT trajectories and paired 15-min GHI intervals\n"
                       "CPP is a retrieval reference; R(real future AGRI) is an offline oracle diagnostic")
    rgb_fig.savefig(DATA / "multi_date_real_agri_progression.png", dpi=170)
    rgb_fig.savefig(DATA / "multi_date_real_agri_progression.pdf")
    full_rgb_fig.savefig(DATA / "multi_date_real_agri_all_16_leads.png", dpi=150)
    curve_fig.savefig(DATA / "multi_date_cot_ghi_trajectories.png", dpi=170)
    curve_fig.savefig(DATA / "multi_date_cot_ghi_trajectories.pdf")
    (DATA / "multi_date_event_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    with (DATA / "multi_date_event_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        w.writeheader(); w.writerows(csv_rows)
    print(json.dumps({"state": "PLOTS_COMPLETE", "n_cases": n,
                      "dates": [c["init_bjt"] for c in cases],
                      "output": str(DATA)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
