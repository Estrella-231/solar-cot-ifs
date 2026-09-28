"""Plot held-out validation cloud events with correctly paired 15-min GHI labels.

All event selection predates this script. RGB is a false-colour proxy made from
real future AGRI C03/C02/C01. CPP is a retrieval reference; oracle R(actual
future AGRI) and future-reference spatial alignment are hindsight diagnostics.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/regional_station_cot_20260927"
KEYS = ROOT / "audits/regional_error_attribution_20260928/match_keys"
OUT = RESULTS / "event_attribution_20260928"
LEADS = np.arange(1, 17) * 15
EVENTS = [
    ("cloud_decay", "2025-08-05 09:15 BJT · dissipating cloud", "sili_trajectories_v2/simvp_better_source.npz", 2311),
    ("moving_thick_cloud", "2025-08-14 11:30 BJT · moving, thickening cloud", "sili_thick_motion/nonzero_motion_cloud_source.npz", 2869),
    ("missed_growth", "2025-07-09 12:15 BJT · missed cloud growth", "sili_thick_motion/largest_thick_peak_miss_source.npz", 534),
]
RGB_STRETCH = {}
for _p in (RESULTS / "sili_trajectories_v2/cases.json",
           RESULTS / "sili_thick_motion/selected_cases.json"):
    for _item in json.loads(_p.read_text(encoding="utf-8")):
        _low = _item.get("rgb_stretch_low", _item.get("rgb_percentile_low"))
        _high = _item.get("rgb_stretch_high", _item.get("rgb_percentile_high"))
        if _low is not None and _high is not None:
            RGB_STRETCH[int(_item["sequence"])] = (np.asarray(_low), np.asarray(_high))


def paired_ghi(sequence: int):
    """Join predictions to the saved head cohort by immutable source row id."""
    pred = np.load(RESULTS / "ghi_pilot/forecast_validation_predictions.npz")
    row_ids = np.load(KEYS / "head_row_ids.npy")
    seq = np.load(KEYS / "sequence.npy")
    station = np.load(KEYS / "station.npy")
    lead = np.load(KEYS / "lead.npy")
    lookup = {int(r): (float(y), float(p)) for r, y, p in
              zip(pred["row_ids"], pred["true_ghi"], pred["pred_ghi"])}
    result = np.full((2, 16), np.nan, dtype=np.float64)
    for r in row_ids:
        r = int(r)
        if seq[r] == sequence and station[r] == 0 and r in lookup:
            result[:, int(lead[r])] = lookup[r]
    return result


def shifted_no_wrap(a: np.ndarray, dr: int, dc: int):
    out = np.full_like(a, np.nan, dtype=np.float32)
    h, w = a.shape
    src_r0, src_r1 = max(0, -dr), min(h, h - dr)
    src_c0, src_c1 = max(0, -dc), min(w, w - dc)
    dst_r0, dst_r1 = src_r0 + dr, src_r1 + dr
    dst_c0, dst_c1 = src_c0 + dc, src_c1 + dc
    out[dst_r0:dst_r1, dst_c0:dst_c1] = a[src_r0:src_r1, src_c0:src_c1]
    return out


def offline_alignment(source: np.ndarray, reference: np.ndarray, valid: np.ndarray, max_shift=16):
    """Optimistic hindsight alignment, scored on one common inner support."""
    candidates = [(dr, dc, shifted_no_wrap(source, dr, dc))
                  for dr in range(-max_shift, max_shift + 1)
                  for dc in range(-max_shift, max_shift + 1)]
    common = valid.copy()
    # Keep exactly the same target pixels for every candidate displacement.
    common[:max_shift, :] = False; common[-max_shift:, :] = False
    common[:, :max_shift] = False; common[:, -max_shift:] = False
    for _, _, shifted in candidates:
        common &= np.isfinite(shifted)
    if common.sum() < 20:
        return {"status": "insufficient_common_support", "n": int(common.sum())}
    base = source[common] - reference[common]
    best = min(candidates, key=lambda item: np.mean((item[2][common] - reference[common]) ** 2))
    residual = best[2][common] - reference[common]
    return {
        "status": "diagnostic_only_future_reference_used",
        "max_integer_shift_pixels": max_shift,
        "n_common_pixels": int(common.sum()),
        "best_shift_row_col": [int(best[0]), int(best[1])],
        "unshifted_rmse_cot": float(np.sqrt(np.mean(base ** 2))),
        "aligned_rmse_cot": float(np.sqrt(np.mean(residual ** 2))),
        "aligned_mae_cot": float(np.mean(np.abs(residual))),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Real AGRI progression: selected frames make the 4-hour sequence readable.
    shown = [0, 3, 7, 11, 15]
    fig, axes = plt.subplots(len(EVENTS), len(shown), figsize=(14, 8), layout="constrained")
    curve_fig, curve_axes = plt.subplots(len(EVENTS), 2, figsize=(13, 10), layout="constrained")
    report = {"test_used": False, "station": "Sili (local crop centre row=163,col=148)",
              "ghi_label": "Observed 15-minute interval mean; lead 1..16 maps to +15..+240 min.",
              "reference": "CPP retrieval product; producer/physical QA unresolved.",
              "selection": "Three already-selected validation examples for event diagnostics, not a representative sample.",
              "events": []}

    for row, (key, title, rel, sequence) in enumerate(EVENTS):
        z = np.load(RESULTS / rel)
        # Saved arrays are in physical COT units; only forecast input features are log1p.
        rgb = z["rgb"]
        lo, hi = RGB_STRETCH[sequence]
        display_rgb = np.clip((rgb - lo[None, None, None, :]) /
                              (hi - lo + 1e-8)[None, None, None, :], 0, 1)
        for col, ti in enumerate(shown):
            ax = axes[row, col]
            img = np.nan_to_num(display_rgb[ti], nan=0.0)
            ax.imshow(img, interpolation="nearest")
            ax.plot(32, 32, marker="+", ms=10, mew=1.8, color="cyan")
            ax.set_title(f"+{LEADS[ti]} min" if row == 0 else "")
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                short = ("08-05 · decay", "08-14 · thick cloud", "07-09 · missed growth")[row]
                ax.text(-0.14, 0.5, short, transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=9)
        cot_ax, ghi_ax = curve_axes[row]
        valid = z["valid"]
        series = {
            "CPP retrieval ref": np.where(valid[:, 32, 32], z["reference"][:, 32, 32], np.nan),
            "R(real future AGRI) · oracle": np.where(valid[:, 32, 32], z["oracle"][:, 32, 32], np.nan),
            "R(SimVP future AGRI)": z["simvp"][:, 32, 32],
            "history + fixed translation": z["transport"][:, 32, 32],
        }
        for name, values in series.items():
            cot_ax.plot(LEADS, values, marker=".", lw=1.5, label=name)
        cot_ax.axhline(30, color="gray", ls=":", lw=1)
        cot_ax.set_ylabel("Station COT")
        cot_ax.grid(alpha=.2)
        cot_ax.set_title(title)
        if row == 0:
            cot_ax.legend(fontsize=7, ncol=2)
        yy = paired_ghi(sequence)
        ghi_ax.plot(LEADS, yy[0], "o-", color="#222222", label="Observed 15-min mean GHI")
        ghi_ax.plot(LEADS, yy[1], "s--", color="#0072B2", label="Existing COT-GHI pilot prediction")
        ghi_ax.set_ylabel("GHI (W m$^{-2}$)")
        ghi_ax.grid(alpha=.2)
        if row == 0:
            ghi_ax.legend(fontsize=7)
        alignment = []
        for ti in range(16):
            if not valid[ti].any() or not np.isfinite(z["transport"][ti]).any():
                alignment.append({"lead_minutes": int(LEADS[ti]), "status": "no_common_reference"})
                continue
            res = offline_alignment(z["transport"][ti], z["reference"][ti], valid[ti])
            alignment.append({"lead_minutes": int(LEADS[ti]), **res})
        # Physical COT maps show whether apparent station errors are spatial or
        # amplitude/evolution errors. All rows use the same 0..70 COT scale.
        map_times = [0, 3, 7, 11, 15]
        fields = [
            ("CPP retrieval ref", z["reference"]),
            ("R(real future AGRI)", z["oracle"]),
            ("R(SimVP future AGRI)", z["simvp"]),
            ("history + fixed translation", z["transport"]),
        ]
        map_fig, map_axes = plt.subplots(4, len(map_times), figsize=(14, 8), layout="constrained")
        im = None
        for ci, ti in enumerate(map_times):
            for ri, (name, cube) in enumerate(fields):
                ax = map_axes[ri, ci]
                panel = np.where(valid[ti], cube[ti], np.nan) if ri == 0 else cube[ti]
                im = ax.imshow(panel, cmap="magma", vmin=0, vmax=70,
                               interpolation="nearest", origin="upper")
                ax.plot(32, 32, marker="+", ms=9, mew=1.5, color="cyan")
                ax.set_xticks([]); ax.set_yticks([])
                if ri == 0:
                    ax.set_title(f"+{LEADS[ti]} min")
                if ci == 0:
                    ax.set_ylabel(name, fontsize=8)
        map_fig.colorbar(im, ax=map_axes, label="COT (CPP/R scale)", shrink=.7, extend="max")
        map_fig.suptitle(title + " · future reference maps (diagnostic)")
        map_fig.savefig(OUT / f"{key}_cot_maps.png", dpi=170)
        map_fig.savefig(OUT / f"{key}_cot_maps.pdf")
        plt.close(map_fig)
        report["events"].append({
            "event": key, "sequence": sequence, "description": title,
            "cot_series_station_pixel": {k: [None if not np.isfinite(v) else float(v) for v in vals]
                                         for k, vals in series.items()},
            "ghi_observed_15min_mean": [None if not np.isfinite(v) else float(v) for v in yy[0]],
            "ghi_existing_pilot_prediction": [None if not np.isfinite(v) else float(v) for v in yy[1]],
            "future_reference_alignment": alignment,
            "alignment_warning": "Uses future CPP to select an integer translation; diagnostic lower bound, not forecast skill.",
        })
    for ax in axes[-1]:
        ax.set_xlabel("Lead time")
    for ax in curve_axes[-1]:
        ax.set_xlabel("Lead from history end (minutes)")
    fig.suptitle("Actual future AGRI progression around Sili · C03/C02/C01 enhanced RGB proxy")
    fig.savefig(OUT / "real_agri_event_progression.png", dpi=180)
    fig.savefig(OUT / "real_agri_event_progression.pdf")
    curve_fig.suptitle("Cloud-event COT and observed 15-minute station GHI · validation diagnostics")
    curve_fig.savefig(OUT / "cot_ghi_event_curves.png", dpi=180)
    curve_fig.savefig(OUT / "cot_ghi_event_curves.pdf")
    (OUT / "event_attribution.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(OUT), "events": [e[0] for e in EVENTS], "test_used": False}))


if __name__ == "__main__":
    main()
