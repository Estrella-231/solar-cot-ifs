#!/usr/bin/env python3
"""Render frozen T16 observed-AGRI/CPP-COT/R/GHI validation case panels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(p, np.float64) - np.asarray(y, np.float64)) ** 2)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = ap.parse_args()
    repo = args.repo.resolve()
    exp = repo / "experiments/monthly_R_station16_20261013_v1"
    result_dir = exp / "results/t16_partial_cases_v1"
    manifest_path = exp / "results/t16_partial_case_manifest_v1.json"
    input_path = result_dir / "agri_geometry_inputs.npz"
    input_audit_path = result_dir / "agri_geometry_audit.json"
    cot_path = repo / "refine-logs/cot_value_first_20261013/validation_predictions_monthly_R_seed42.npz"
    pack_dir = exp / "results/ghi_head_pack_new_r_v1"
    rows_path = pack_dir / "rows.csv"
    cases = json.loads(manifest_path.read_text(encoding="utf-8"))["selected_cases"]
    input_audit = json.loads(input_audit_path.read_text(encoding="utf-8"))
    if sha256(manifest_path) != input_audit["case_manifest_sha256"]:
        raise RuntimeError("AGRI/geometry extraction is not bound to this frozen case manifest")
    if sha256(input_path) != input_audit["output_npz_sha256"]:
        raise RuntimeError("AGRI/geometry array hash mismatch")
    with np.load(input_path, allow_pickle=False) as z:
        case_ids = z["case_id"].astype(str)
        vis = np.asarray(z["visible_agri_c01_c02_c03"], dtype=np.float32)
        vis_valid = np.asarray(z["visible_valid"], dtype=bool)
        geom = np.asarray(z["geometry_cossoz_cosraa_daymask"], dtype=np.float32)
    if len(case_ids) != len(cases) or vis.shape != (len(cases), 16, 3, 16, 16):
        raise RuntimeError("unexpected extracted AGRI/geometry shape")
    expected_case_ids = [f"{c['station']}_JJA_{c['case_type']}_{c['init_time_BJT'].replace(' ', 'T').replace(':', '')}" for c in cases]
    if case_ids.tolist() != expected_case_ids:
        raise RuntimeError("extracted AGRI case IDs do not match the frozen manifest order")
    if not np.all(vis_valid):
        raise RuntimeError("selected AGRI RGB proxy has invalid pixels; refusing to render missing values as clear sky")

    with np.load(cot_path, allow_pickle=False) as z:
        ref_cot = np.asarray(z["reference_cot"], dtype=np.float32)[:, 0]
        pred_log = np.asarray(z["pred_log1p_cot"], dtype=np.float32)[:, 0]
        cot_valid = np.asarray(z["mask"], dtype=bool)[:, 0]
        cot_sample_indices = np.asarray(z["sample_indices"], dtype=np.int64)
    cot_by_index = {int(i): pos for pos, i in enumerate(cot_sample_indices)}

    with rows_path.open(encoding="utf-8", newline="") as f:
        all_rows = [r for r in csv.DictReader(f) if r["split"] == "val"]
    station_rows: dict[str, list[dict]] = {s: [] for s in ("sili", "zhujia")}
    for row in all_rows:
        station_rows[row["station"].lower()].append(row)
    row_pos = {(st, row["target_time_utc"]): i for st, seq in station_rows.items() for i, row in enumerate(seq)}
    truth = {}
    pred_by_case = []
    for case in cases:
        st = case["station"]
        ys = []
        for ts in case["future_target_times_utc"]:
            if (st, ts) not in row_pos:
                raise RuntimeError(f"selected target missing from GHI validation pack: {(st, ts)}")
            yi = row_pos[(st, ts)]
            ys.append(float(station_rows[st][yi]["ghi"]))
        truth[case["init_time_utc"] + "|" + st] = np.asarray(ys, dtype=np.float64)
        arms = {a: [] for a in ("A", "AC")}
        for seed in (42, 43, 44):
            for arm in arms:
                p = exp / f"results/ghi_heads_new_r_seed{seed}_v1/{st}/{arm}_val_prediction_ghi_wm2.npy"
                arr = np.asarray(np.load(p, allow_pickle=False), dtype=np.float64).reshape(-1)
                if len(arr) != len(station_rows[st]):
                    raise RuntimeError(f"prediction length mismatch: {p}")
                arms[arm].append(arr[[row_pos[(st, ts)] for ts in case["future_target_times_utc"]]])
        pred_by_case.append({k: np.stack(v) for k, v in arms.items()})

    out_dir = repo / "figures/t16_partial_cases_20261013"
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = [out_dir / f"case_{i+1}_{c['station']}_{c['case_type']}.png" for i, c in enumerate(cases)]
    pdf_path = out_dir / "t16_partial_cases.pdf"
    if pdf_path.exists() or any(p.exists() for p in expected):
        raise FileExistsError(f"refusing to overwrite T16 figures in {out_dir}")
    report_rows = []
    lead_idx = [0, 3, 7, 15]

    with PdfPages(pdf_path, metadata={"Title": "T16 partial validation physical cases", "Subject": "Error-blind JJA cases; observed AGRI diagnostic, not forecast input"}) as pdf:
        for case_i, case in enumerate(cases):
            st = case["station"]
            indices = [cot_by_index[int(x)] for x in case["sample_indices"]]
            if len(indices) != 17:
                raise RuntimeError("expected 17 COT records per frozen case")
            fut_idx = indices[1:]
            ref = ref_cot[fut_idx]
            pred = np.expm1(pred_log[fut_idx].astype(np.float64))
            mask = cot_valid[fut_idx]
            y = truth[case["init_time_utc"] + "|" + st]
            arms = pred_by_case[case_i]
            dt_utc = case["future_target_times_utc"]

            fig = plt.figure(figsize=(15.6, 11.0), constrained_layout=True)
            fig.suptitle(
                f"{st.upper()} · {case['season_BJT']} · {case['case_type']} · init {case['init_time_BJT']} BJT\n"
                "Observed future AGRI diagnostic (offline only); CPP reference COT vs frozen R retrieval\n"
                f"Error-blind D={case['D_mean_abs_COT_change']:.3f}; common CPP support="
                f"{case['reference_valid_pixels_common_to_all_17_frames']}/256 pixels",
                fontsize=14, fontweight="bold")
            grid = fig.add_gridspec(5, 4, height_ratios=[1, 1, 1, 0.9, 1.25])
            image_axes = np.empty((4, 4), dtype=object)
            for rr in range(4):
                for cc, idx in enumerate(lead_idx):
                    image_axes[rr, cc] = fig.add_subplot(grid[rr, cc])
            cmap_cot = plt.colormaps["magma"].with_extremes(bad="#bdbdbd")
            cmap_sun = plt.colormaps["cividis"].with_extremes(bad="#bdbdbd")
            rgb_image = None
            ref_image = pred_image = sun_image = None
            for cc, idx in enumerate(lead_idx):
                image_axes[0, cc].set_title(f"+{(idx+1)*15} min", fontsize=10)
                raw = vis[case_i, idx]
                # Fixed physical reflectance stretch [0,1], gamma 1/2.2;
                # input channel order C01,C02,C03 is rendered as RGB C03,C02,C01.
                rgb = np.stack([raw[2], raw[1], raw[0]], axis=-1)
                rgb = np.clip(np.nan_to_num(rgb, nan=0.0), 0.0, 1.0) ** (1.0 / 2.2)
                rgb_image = image_axes[0, cc].imshow(rgb, interpolation="nearest")
                image_axes[0, cc].scatter([8], [8], marker="+", s=70, linewidths=1.5, c="cyan")

                ref_image = image_axes[1, cc].imshow(np.ma.masked_where(~mask[idx], ref[idx]),
                                                      cmap=cmap_cot, norm=Normalize(0, 60), interpolation="nearest")
                image_axes[1, cc].scatter([8], [8], marker="+", s=70, linewidths=1.5, c="cyan")
                pred_image = image_axes[2, cc].imshow(np.ma.masked_where(~mask[idx], pred[idx]),
                                                       cmap=cmap_cot, norm=Normalize(0, 60), interpolation="nearest")
                image_axes[2, cc].scatter([8], [8], marker="+", s=70, linewidths=1.5, c="cyan")

                cossoz = geom[case_i, idx, 0]
                day = geom[case_i, idx, 2] > 0.5
                sza = np.rad2deg(np.arccos(np.clip(cossoz, -1.0, 1.0)))
                sun_image = image_axes[3, cc].imshow(np.ma.masked_where(~day, sza), cmap=cmap_sun,
                                                      norm=Normalize(0, 90), interpolation="nearest")
                image_axes[3, cc].scatter([8], [8], marker="+", s=70, linewidths=1.5, c="white")
                for rr in range(4):
                    ax = image_axes[rr, cc]
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_aspect("equal")
                image_axes[0, cc].set_xlabel("RGB proxy: C03/C02/C01")
            row_titles = ["Observed AGRI RGB proxy", "CPP reference COT", "R retrieved COT", "Solar zenith angle"]
            for rr, title in enumerate(row_titles):
                image_axes[rr, 0].set_ylabel(title, fontsize=10)
            fig.colorbar(ref_image, ax=list(image_axes[1, :]) + list(image_axes[2, :]),
                         shrink=0.82, label="COT (physical units; fixed 0–60 scale)")
            fig.colorbar(sun_image, ax=list(image_axes[3, :]), shrink=0.82, label="Solar zenith (degrees)")

            axg = fig.add_subplot(grid[4, :2])
            axc = fig.add_subplot(grid[4, 2:])
            leads = np.arange(15, 241, 15)
            colors = {"A": "#0072B2", "AC": "#D55E00"}
            for arm in ("A", "AC"):
                for si, seed in enumerate((42, 43, 44)):
                    axg.plot(leads, arms[arm][si], color=colors[arm], lw=1.0, alpha=0.36,
                             linestyle="--" if arm == "A" else "-",
                             label=f"{arm} seed {seed}" if si == 0 else None)
                mean = arms[arm].mean(axis=0)
                axg.plot(leads, mean, color=colors[arm], lw=2.2,
                         linestyle="--" if arm == "A" else "-", label=f"{arm} 3-seed mean")
            axg.plot(leads, y, color="#222222", lw=1.8, marker="o", ms=3.5, label="Observed 15-min GHI")
            rmse_a = float(np.mean([rmse(y, x) for x in arms["A"]]))
            rmse_ac = float(np.mean([rmse(y, x) for x in arms["AC"]]))
            axg.set(xlabel="Lead from init (min)", ylabel="GHI (W m⁻²)", xlim=(10, 245),
                    title=f"Same-time real-AGRI GHI heads · per-seed RMSE mean A={rmse_a:.1f}, AC={rmse_ac:.1f}")
            axg.grid(True, alpha=0.25)
            axg.legend(fontsize=7, ncol=2, frameon=False)

            center_ref = ref[:, 8, 8]
            center_pred = pred[:, 8, 8]
            axc.plot(leads, center_ref, color="#009E73", marker="o", ms=3.5, lw=1.8, label="CPP ref, station pixel")
            axc.plot(leads, center_pred, color="#CC79A7", marker="^", ms=3.5, lw=1.8, linestyle="--", label="R, station pixel")
            axc.set(xlabel="Lead from init (min)", ylabel="COT", xlim=(10, 245), title="Station-pixel COT trajectory")
            axc.grid(True, alpha=0.25)
            axc.legend(fontsize=8, frameon=False)
            png = expected[case_i]
            fig.savefig(png, dpi=220, facecolor="white")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
            report_rows.append({
                "case_id": case_ids[case_i], "station": st, "init_time_BJT": case["init_time_BJT"],
                "case_type": case["case_type"], "D_mean_abs_COT_change": case["D_mean_abs_COT_change"],
                "common_CPP_pixels": case["reference_valid_pixels_common_to_all_17_frames"],
                "case_GHI_truth_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
                "case_A_mean_seed_RMSE_wm2": rmse_a, "case_AC_mean_seed_RMSE_wm2": rmse_ac,
                "case_png": png.name, "case_png_sha256": sha256(png),
                "leads_utc": dt_utc,
            })

    audit = {
        "state": "PASS_PARTIAL_T16_JJA_PHYSICAL_VALIDATION_FIGURES",
        "case_manifest_sha256": sha256(manifest_path),
        "agri_geometry_npz_sha256": sha256(input_path),
        "agri_geometry_audit_sha256": sha256(input_audit_path),
        "reference_cot_npz_sha256": sha256(cot_path),
        "GHI_pack_manifest_sha256": sha256(pack_dir / "pack_manifest.json"),
        "GHI_pack_rows_sha256": sha256(rows_path),
        "pdf_sha256": sha256(pdf_path),
        "cases": report_rows,
        "rendering": {
            "AGRI_rgb_proxy": "fixed reflectance clipping [0,1], gamma 1/2.2; RGB=(C03,C02,C01); false-colour proxy, 16x16, nearest interpolation",
            "COT_maps": "CPP physical COT and expm1(R log1p COT), same 0-60 scale, CPP common-valid mask",
            "solar_geometry": "arccos(cosSOZ), degrees, day_mask, 0-90 scale",
            "GHI_plot": "observed 15-minute target values; each of seeds 42/43/44 plotted thin, three-seed mean thick; per-case RMSE is descriptive",
            "COT_trajectory": "station-center pixel, CPP reference vs frozen R retrieval",
        },
        "test_payload_opened": False,
        "case_selection_used_GHI_values_or_errors": False,
        "interpretation_limit": "Four JJA validation cases only; same-time real-AGRI retrieval diagnostic, not forecast-chain performance or independent truth.",
    }
    audit_path = result_dir / "t16_case_figure_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"state": audit["state"], "pdf": str(pdf_path), "case_count": len(report_rows),
                      "cases": [{k: x[k] for k in ("station", "init_time_BJT", "case_type", "case_A_mean_seed_RMSE_wm2", "case_AC_mean_seed_RMSE_wm2", "case_png")} for x in report_rows],
                      "test_payload_opened": False}, indent=2))


if __name__ == "__main__":
    main()
