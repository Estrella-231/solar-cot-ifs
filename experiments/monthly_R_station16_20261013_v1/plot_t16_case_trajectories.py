#!/usr/bin/env python3
"""Plot frozen CPP-COT/observed-AGRI validation cases and paired GHI replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


COLORS = {"A": "#0072B2", "AC": "#D55E00", "AP": "#009E73"}
SNAPSHOTS = (0, 4, 8, 12, 16)
ARMS = ("A", "AC", "AP")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def stretch_rgb(raw: np.ndarray, limits: list[list[float]]) -> np.ndarray:
    out = np.empty_like(raw, dtype=np.float32)
    for c in range(3):
        lo, hi = limits[c]
        out[:, c] = np.clip((raw[:, c] - lo) / max(hi - lo, 1e-12), 0, 1)
    return np.moveaxis(out, 1, -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--payload-meta", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(f"output exists; choose a new directory: {a.output}")
    payload_meta = json.loads(a.payload_meta.read_text(encoding="utf-8"))
    if payload_meta.get("test_payloads_read") != 0 or payload_meta.get("GHI_numeric_values_read_after_manifest_freeze") is not True:
        raise ValueError("T16 payload provenance gate failed")
    event = json.loads(a.manifest.read_text(encoding="utf-8"))
    if sha256(a.manifest) != payload_meta["event_manifest_sha256"]:
        raise ValueError("payload and event manifest hash mismatch")
    data = np.load(a.payload, allow_pickle=False)
    rgb = np.asarray(data["false_color_rgb_raw"], dtype=np.float32)
    cot = np.asarray(data["cpp_cot_physical"], dtype=np.float32)
    cmask = np.asarray(data["cpp_quality_mask"], dtype=bool)
    obs = np.asarray(data["observed_ghi_15min_wm2"], dtype=np.float64)
    clear = np.asarray(data["clear_sky_ghi_15min_wm2"], dtype=np.float64)
    pred = np.asarray(data["ghi_head_predictions_3seed"], dtype=np.float64)
    cases = event["selected_cases"]
    if rgb.shape != (len(cases), 17, 3, 16, 16) or cot.shape != (len(cases), 17, 16, 16):
        raise ValueError("unexpected case-array shape")

    # One shared display stretch across all selected validation cases and leads.
    rgb_limits = [[float(np.quantile(rgb[:, :, c], 0.02)), float(np.quantile(rgb[:, :, c], 0.98))] for c in range(3)]
    cot_vmax = 75.0
    leads = np.arange(17, dtype=np.int32) * 15
    a.output.mkdir(parents=True)
    description = [
        "Four preregistered summer validation cases selected from CPP reference COT only.",
        "Each page shows actual AGRI false-colour composites (R=C06, G=C03, B=C02), CPP reference COT on the same station-centred 16x16 grid, and 15-minute interval GHI.",
        "A/AC/AP are contemporaneous validation-head replays using observed AGRI/geometry and, for AC, the frozen R-derived same-time COT input. They are not operational forecasts and do not use predicted future AGRI/COT.",
        "The four cases cover summer only because the audited validation CPP pack contains no complete 17-frame 15-minute reference sequence for DJF, MAM, or SON. No cases were substituted across seasons.",
    ]
    with (a.output / "T16_FIGURE_DESCRIPTION.md").open("w", encoding="utf-8") as f:
        f.write("# T16 validation case figure description\n\n")
        for p in description:
            f.write(f"{p}\n\n")
        f.write("RGB display uses a single linear 2nd-to-98th-percentile stretch per band across the four cases; no spatial interpolation is applied. CPP COT uses a common linear 0–75 scale. GHI head curves show the mean across seeds 42/43/44; shading is the seed minimum-to-maximum range, not a confidence interval. The 0-minute GHI label is the 15-minute interval ending at initialization; subsequent points end at +15 through +240 minutes.\n")

    rgb_stretched = np.asarray([stretch_rgb(rgb[i], rgb_limits) for i in range(len(cases))])
    with (a.output / "case_trajectory_data.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["case_index", "station", "season", "selection_type", "init_time_bjt", "lead_min",
                  "observed_ghi_15min_wm2", "clear_sky_ghi_15min_wm2"] + [f"seed{s}_{arm}_ghi_wm2" for s in (42, 43, 44) for arm in ARMS]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, case in enumerate(cases):
            for k, lead in enumerate(leads):
                row = {"case_index": i + 1, "station": case["station"], "season": case["season"],
                       "selection_type": case["selection_type"], "init_time_bjt": case["init_time_bjt"],
                       "lead_min": int(lead), "observed_ghi_15min_wm2": obs[i, k],
                       "clear_sky_ghi_15min_wm2": clear[i, k]}
                for si, seed in enumerate((42, 43, 44)):
                    for ai, arm in enumerate(ARMS):
                        row[f"seed{seed}_{arm}_ghi_wm2"] = pred[i, k, si, ai]
                w.writerow(row)

    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
                         "axes.spines.top": False, "axes.spines.right": False})
    pdf_path = a.output / "T16_CPP_COT_AGRI_GHI_validation_cases.pdf"
    png_paths = []
    with PdfPages(pdf_path, metadata={"Title": "T16 CPP COT and GHI validation cases",
                                     "Subject": "Validation-only physical case illustrations; same-time head replay, not forecast"}) as pdf:
        for i, case in enumerate(cases):
            fig = plt.figure(figsize=(16.5, 8.6), layout="constrained")
            gs = GridSpec(3, 5, figure=fig, height_ratios=[1.0, 1.0, 1.18])
            station = case["station"].capitalize()
            init = case["init_time_bjt"].replace("T", " ").replace("+08:00", " BJT")
            sel = "maximum COT-change" if case["selection_type"] == "maximum_reference_COT_change" else "deterministic-hash"
            fig.suptitle(f"{station} · {case['season']} · {sel} case · init {init}\n"
                         f"CPP-reference D={case['D_mean_abs_adjacent_cot']:.3f} COT per 15 min; common pixels={case['common_valid_pixels_16x16']}/256",
                         fontsize=14, fontweight="bold")
            for j, k in enumerate(SNAPSHOTS):
                ax = fig.add_subplot(gs[0, j])
                shown = rgb_stretched[i, k]
                ax.imshow(shown, interpolation="nearest", origin="upper")
                ax.plot(8.5, 8.5, marker="*", markersize=11, color="white", markeredgecolor="black", markeredgewidth=0.8)
                ax.set_title(f"+{leads[k]} min", fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel("AGRI false-colour\nR=C06, G=C03, B=C02", fontsize=9)
            cot_image = None
            for j, k in enumerate(SNAPSHOTS):
                ax = fig.add_subplot(gs[1, j])
                frame = np.where(cmask[i, k], cot[i, k], np.nan)
                cot_image = ax.imshow(frame, interpolation="nearest", origin="upper", cmap="magma", vmin=0, vmax=cot_vmax)
                ax.plot(8.5, 8.5, marker="*", markersize=11, color="cyan", markeredgecolor="black", markeredgewidth=0.8)
                ax.set_title(f"+{leads[k]} min", fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel("CPP reference COT\nphysical units; 16×16", fontsize=9)
            fig.colorbar(cot_image, ax=[fig.axes[5], fig.axes[6], fig.axes[7], fig.axes[8], fig.axes[9]],
                         location="right", shrink=0.82, label="COT (common scale 0–75)")

            ax = fig.add_subplot(gs[2, :])
            ax.plot(leads, obs[i], color="#222222", marker="o", markersize=4, linewidth=1.8, label="Observed 15-min mean GHI")
            ax.plot(leads, clear[i], color="#777777", linestyle="--", linewidth=1.4, label="15-min mean clear-sky GHI")
            for ai, arm in enumerate(ARMS):
                v = pred[i, :, :, ai]
                mean = v.mean(axis=1)
                lo, hi = v.min(axis=1), v.max(axis=1)
                marker = {"A": "D", "AC": "s", "AP": "^"}[arm]
                ax.plot(leads, mean, color=COLORS[arm], marker=marker,
                        markersize=3.5, linewidth=1.5, label=f"{arm} head (3-seed mean)")
                ax.fill_between(leads, lo, hi, color=COLORS[arm], alpha=0.10, linewidth=0,
                                label="3-seed min–max range (not CI)" if ai == 0 else None)
            ax.set(xlim=(-5, 245), xticks=np.arange(0, 241, 30), xlabel="Target interval end from initialization (min; each GHI point is a 15-min mean)",
                   ylabel="GHI (W m$^{-2}$)")
            ax.grid(axis="y", color="#d8d8d8", linewidth=0.6)
            ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0))
            ax.text(0.995, 0.03, "Same-time diagnostic replay with observed AGRI and R-COT; not an operational forecast",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#5a3030")
            out_png = a.output / f"case_{i+1}_{case['station']}_{case['init_time_bjt'][:10]}_{sel.replace('-', '_')}.png"
            fig.savefig(out_png, dpi=220, facecolor="white")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
            png_paths.append(out_png.name)

    manifest = {
        "state": "T16_FIGURES_COMPLETE_PARTIAL_SEASONAL_SUPPORT",
        "event_manifest_sha256": sha256(a.manifest),
        "case_payload_sha256": sha256(a.payload),
        "payload_metadata_sha256": sha256(a.payload_meta),
        "plot_script_sha256": sha256(Path(__file__)),
        "rgb_limits_by_band": {b: rgb_limits[k] for k, b in enumerate(("C06", "C03", "C02"))},
        "cot_display_limits_physical": [0.0, cot_vmax],
        "ghi_head_summary": "three-seed mean; translucent min-max seed range, not confidence interval",
        "files": [pdf_path.name, *png_paths, "case_trajectory_data.csv", "T16_FIGURE_DESCRIPTION.md"],
        "test_payloads_read": 0,
    }
    (a.output / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
