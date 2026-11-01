"""Plot four validation false-cloud cases, adjacent frames and same-scene controls."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.backends.backend_pdf import PdfPages
import netCDF4

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "results/cot_staged_diagnostic_20260924/source_cases"
OUT_DIR = CASE_DIR / "forensic_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLP_FILES = {
    # Use ASCII-relative paths because the local NetCDF/HDF5 Windows build
    # rejects these files when opened through this workspace's non-ASCII path.
    ("zhujia", "2025-07-06T08:15:00Z"): Path("results/cot_staged_diagnostic_20260924/source_cases/nc_windows/zhujia_202507060815.nc"),
    ("sili", "2025-07-14T05:15:00Z"): Path("results/cot_staged_diagnostic_20260924/source_cases/nc_windows/sili_202507140515.nc"),
    ("zhujia", "2025-07-06T06:30:00Z"): Path("results/cot_staged_diagnostic_20260924/source_cases/nc_windows/zhujia_202507060630.nc"),
    ("sili", "2025-07-07T05:15:00Z"): Path("results/cot_staged_diagnostic_20260924/source_cases/nc_windows/sili_202507070515.nc"),
}

CLP_CMAP = ListedColormap(["#667788", "#36a6c9", "#e5a23b"])
CLP_CMAP.set_bad("#b8b8b8")
CLP_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], CLP_CMAP.N)


def rgb_composite(raw, limits):
    # Display-only AGRI pseudo-RGB: R=C03, G=C02, B=C01.
    rgb = np.stack([raw[2], raw[1], raw[0]], axis=-1).astype(np.float32)
    for channel, color_index in ((2, 0), (1, 1), (0, 2)):
        lo, hi = limits[str(channel)]
        rgb[..., color_index] = np.clip((rgb[..., color_index] - lo) / (hi - lo), 0, 1)
    return rgb


def load_clp_case(station, timestamp):
    path = CLP_FILES.get((station, timestamp))
    if path is None or not path.is_file():
        return None
    with netCDF4.Dataset(path) as ds:
        return np.ma.filled(ds.variables["CLP"][:], np.nan).astype(np.float32)


def draw_markers(ax, false_rc, control_rc):
    r, c = false_rc
    cr, cc = control_rc
    ax.plot(c, r, marker="x", color="#ff2e2e", markersize=8, markeredgewidth=2.2,
            linestyle="none", zorder=10)
    ax.plot(cc, cr, marker="*", color="white", markeredgecolor="#111111", markersize=11,
            linestyle="none", zorder=11)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xlabel("column")


def main():
    packed = np.load(CASE_DIR / "four_case_temporal_agri_model_inputs.npz")
    original = np.load(CASE_DIR / "four_case_temporal_original_agri_crops.npz")["original_agri"]
    temporal = json.loads(str(packed["manifest_json"].item()))
    limits = json.loads((CASE_DIR / "four_case_temporal_rgb_limits.json").read_text(encoding="utf-8"))
    controls = pd.read_csv(CASE_DIR / "false_cloud_matched_controls_20260924.csv")
    windows = temporal
    # Keep the three-frame ordering from the saved, validation-only manifest.
    by_case = {i: [w for w in windows if w["case_id"] == i] for i in range(1, 5)}
    panel_names = [
        "Actual AGRI pseudo-RGB\nR=C03, G=C02, B=C01",
        "AGRI C13 brightness temperature\n(K)",
        "CPP reference COT\n(0–60; clipped above 60)",
        "CPP cloud phase\n0 clear · 1 water · 2 ice",
        "Model P(cloud)\n(0–1)",
        "Staged model final COT\n(0–60; clipped above 60)",
    ]
    case_pages = []
    timeline_records = []
    with PdfPages(OUT_DIR / "four_case_temporal_forensics_20260924.pdf") as pdf:
        for case_id in range(1, 5):
            frame_indices = [windows.index(w) for w in by_case[case_id]]
            frame_meta = by_case[case_id]
            first = frame_meta[0]
            control = controls.loc[controls.case_id == case_id].iloc[0]
            false_rc = (int(first["pixel_row"]), int(first["pixel_col"]))
            control_rc = (int(control["control_pixel_row"]), int(control["control_pixel_col"]))

            fig = plt.figure(figsize=(14.5, 15.2))
            grid = fig.add_gridspec(6, 3, left=0.12, right=0.81, top=0.90, bottom=0.05,
                                   wspace=0.25, hspace=0.34)
            axes = grid.subplots()
            image_handles = {"thermal": None, "cot": None, "clp": None, "prob": None}
            for col, (idx, meta) in enumerate(zip(frame_indices, frame_meta)):
                raw = original[idx]
                rgb = rgb_composite(raw, limits)
                axes[0, col].imshow(rgb, interpolation="nearest")
                image_handles["thermal"] = axes[1, col].imshow(raw[12], cmap="coolwarm", vmin=200, vmax=305,
                                                                 interpolation="nearest")
                cpp_map = np.where(packed["valid_mask"][idx], packed["cpp_cot"][idx], np.nan)
                image_handles["cot"] = axes[2, col].imshow(cpp_map, cmap="magma", vmin=0, vmax=60,
                                                            interpolation="nearest")
                clp = load_clp_case(meta["station_id"], meta["timestamp_utc"])
                if clp is None:
                    # All zero-COT pixels were audited as CLP=clear. Nonzero
                    # adjacent-frame phase labels are unknown and left masked.
                    cot_map = packed["cpp_cot"][idx]
                    clp = np.full((16, 16), np.nan, dtype=np.float32)
                    clp[(cot_map == 0) & packed["valid_mask"][idx]] = 0
                else:
                    clp[~packed["valid_mask"][idx]] = np.nan
                image_handles["clp"] = axes[3, col].imshow(clp, cmap=CLP_CMAP, norm=CLP_NORM,
                                                            interpolation="nearest")
                image_handles["prob"] = axes[4, col].imshow(packed["cloud_probability"][idx, 0], cmap="viridis",
                                                             vmin=0, vmax=1, interpolation="nearest")
                image_handles["cot"] = axes[5, col].imshow(packed["staged_final_cot"][idx], cmap="magma", vmin=0,
                                                            vmax=60, interpolation="nearest")
                local_time = pd.Timestamp(meta["timestamp_utc"]).strftime("%Y-%m-%d %H:%MZ")
                delta = int(meta["delta_minutes"])
                axes[0, col].set_title(f"{local_time}\nΔt={delta:+d} min", fontsize=10)
                for row in range(6):
                    draw_markers(axes[row, col], false_rc, control_rc)
                central_clp = load_clp_case(meta["station_id"], meta["timestamp_utc"])
                for point_type, (pr, pc) in (("false_pixel", false_rc), ("correct_control", control_rc)):
                    cot_value = float(packed["cpp_cot"][idx, pr, pc])
                    if central_clp is not None:
                        phase = float(central_clp[pr, pc])
                    else:
                        phase = 0.0 if cot_value == 0 else np.nan
                    timeline_records.append({
                        "case_id": case_id,
                        "station_id": meta["station_id"],
                        "point_type": point_type,
                        "timestamp_utc": meta["timestamp_utc"],
                        "timestamp_bjt": meta["timestamp_bjt"],
                        "delta_minutes": int(meta["delta_minutes"]),
                        "pixel_row": pr,
                        "pixel_col": pc,
                        "AGRI_C13_BT_K": float(raw[12, pr, pc]),
                        "cosSOZ_input": float(packed["agri"][idx, 13, pr, pc]),
                        "cpp_COT": cot_value,
                        "cpp_CLP": phase,
                        "cpp_CLP_known": bool(np.isfinite(phase)),
                        "P_cloud": float(packed["cloud_probability"][idx, 0, pr, pc]),
                        "staged_final_COT": float(packed["staged_final_cot"][idx, pr, pc]),
                        "frozen_R_COT": float(packed["frozen_R_cot"][idx, pr, pc]),
                        "cpp_valid": bool(packed["valid_mask"][idx, pr, pc]),
                    })
            row_labels = ["AGRI pseudo-RGB", "AGRI C13 brightness temperature", "CPP reference COT",
                          "CPP cloud phase", "Model P(cloud)", "Staged final COT"]
            for row, label in enumerate(row_labels):
                y = 0.90 - (row + 0.5) * (0.85 / 6)
                fig.text(0.045, y, label, rotation=90, va="center", ha="center", fontsize=9)
            station = first["station_id"].capitalize()
            false_prob = float(packed["cloud_probability"][frame_indices[1], 0, *false_rc])
            false_cpp = float(packed["cpp_cot"][frame_indices[1], *false_rc])
            false_final = float(packed["staged_final_cot"][frame_indices[1], *false_rc])
            control_prob = float(packed["cloud_probability"][frame_indices[1], 0, *control_rc])
            control_final = float(packed["staged_final_cot"][frame_indices[1], *control_rc])
            control_class = str(control["control_kind"])
            control_clp = int(control["control_cpp_CLP"])
            timestamp_bjt = pd.Timestamp(first["case_timestamp_utc"]).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M BJT")
            text = (f"Case {case_id} · {station} · {timestamp_bjt} · validation real-AGRI retrieval\n"
                    f"× false pixel ({false_rc[0]},{false_rc[1]}): CPP COT={false_cpp:.3f}, P(cloud)={false_prob:.3f}, final COT={false_final:.2f} | "
                    f"★ correct {control_class} control ({control_rc[0]},{control_rc[1]}), CLP={control_clp}: "
                    f"P(cloud)={control_prob:.3f}, final COT={control_final:.2f}\n"
                    f"5×5 CPP cloudy-pixel fraction (false/control)={control['false_local_5x5_cloud_fraction']:.2f}/{control['control_local_5x5_cloud_fraction']:.2f}; "
                    "adjacent CLP: cloudy phase unavailable; only COT=0 clear pixels are shown")
            fig.suptitle(text, fontsize=9.5, y=0.985)
            # Dedicated axes keep the physical scales separate and readable.
            def cbar(image, y, label, ticks=None):
                cax = fig.add_axes([0.85, y, 0.018, 0.105])
                cb = fig.colorbar(image, cax=cax, ticks=ticks)
                cb.ax.tick_params(labelsize=7)
                cb.set_label(label, fontsize=8)
            cbar(image_handles["thermal"], 0.765, "C13 (K)")
            cbar(image_handles["cot"], 0.565, "CPP COT", ticks=[0, 20, 40, 60])
            cbar(image_handles["clp"], 0.375, "CPP CLP", ticks=[0, 1, 2])
            cbar(image_handles["prob"], 0.185, "P(cloud)", ticks=[0, 0.5, 1])
            cbar(image_handles["cot"], 0.045, "Final COT", ticks=[0, 20, 40, 60])
            page = OUT_DIR / f"case_{case_id}_temporal_forensics.png"
            fig.savefig(page, dpi=180, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            case_pages.append(page)

    pd.DataFrame(timeline_records).to_csv(OUT_DIR / "four_case_pixel_timelines.csv", index=False)

    # Compact central-time overview for inline review.
    fig, axes = plt.subplots(4, 6, figsize=(18.5, 12.3), layout="constrained")
    titles = ["AGRI pseudo-RGB", "C13 BT (K)", "CPP COT", "CPP CLP", "P(cloud)", "Final COT"]
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontsize=10)
    cot_img = prob_img = clp_img = None
    for case_id in range(1, 5):
        frames = by_case[case_id]
        center_meta = frames[1]
        idx = windows.index(center_meta)
        control = controls.loc[controls.case_id == case_id].iloc[0]
        false_rc = (int(center_meta["pixel_row"]), int(center_meta["pixel_col"]))
        control_rc = (int(control["control_pixel_row"]), int(control["control_pixel_col"]))
        raw = original[idx]
        axes[case_id - 1, 0].imshow(rgb_composite(raw, limits), interpolation="nearest")
        axes[case_id - 1, 1].imshow(raw[12], cmap="coolwarm", vmin=200, vmax=305, interpolation="nearest")
        cpp_map = np.where(packed["valid_mask"][idx], packed["cpp_cot"][idx], np.nan)
        cot_img = axes[case_id - 1, 2].imshow(cpp_map, cmap="magma", vmin=0, vmax=60, interpolation="nearest")
        clp = load_clp_case(center_meta["station_id"], center_meta["timestamp_utc"])
        clp[~packed["valid_mask"][idx]] = np.nan
        clp_img = axes[case_id - 1, 3].imshow(clp, cmap=CLP_CMAP, norm=CLP_NORM, interpolation="nearest")
        prob_img = axes[case_id - 1, 4].imshow(packed["cloud_probability"][idx, 0], cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        axes[case_id - 1, 5].imshow(packed["staged_final_cot"][idx], cmap="magma", vmin=0, vmax=60, interpolation="nearest")
        for ax in axes[case_id - 1]:
            draw_markers(ax, false_rc, control_rc)
        axes[case_id - 1, 0].set_ylabel(f"Case {case_id}\n{center_meta['station_id']}\n{center_meta['timestamp_bjt'][0:16]}", fontsize=9)
    fig.colorbar(cot_img, ax=[*axes[:, 2], *axes[:, 5]], shrink=0.75, label="COT (same 0–60 display scale)")
    fig.colorbar(clp_img, ax=axes[:, 3].tolist(), shrink=0.75, ticks=[0, 1, 2], label="CLP: clear/water/ice")
    fig.colorbar(prob_img, ax=axes[:, 4].tolist(), shrink=0.75, label="P(cloud), 0–1")
    fig.suptitle("False-cloud pixels against same-frame correctly classified controls\nred × = false cloud; white ★ = nearby correct clear/cloud pixel; all panels share row/column coordinates", fontsize=13)
    overview = OUT_DIR / "four_cases_central_frame_overview.png"
    fig.savefig(overview, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("saved:", overview)
    print("saved:", OUT_DIR / "four_case_temporal_forensics_20260924.pdf")
    for page in case_pages:
        print("saved:", page)


if __name__ == "__main__":
    main()
