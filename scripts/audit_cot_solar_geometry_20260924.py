"""Numerically audit the saved cosSOZ/cosRAA/day-mask model inputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "results/cot_staged_diagnostic_20260924/source_cases"
OUT = CASE_DIR / "solar_geometry_audit_20260924.json"
SIDE_CARS = {
    "zhujia": CASE_DIR / "zhujia_202507060815_sidecar.npz",
    "sili": CASE_DIR / "sili_202507140515_sidecar.npz",
}


def noaa_cos_zenith(timestamp_utc, lat, lon, minute_offset=0.0):
    """NOAA fractional-year approximation; angles are degrees, time is UTC."""
    import pandas as pd
    ts = pd.Timestamp(timestamp_utc) + pd.Timedelta(minutes=float(minute_offset))
    doy = ts.dayofyear
    hour = ts.hour + ts.minute / 60 + ts.second / 3600
    gamma = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    utc_minutes = ts.hour * 60 + ts.minute + ts.second / 60
    solar_minutes = np.mod(utc_minutes + eqtime + 4 * np.asarray(lon), 1440)
    hour_angle = np.deg2rad(solar_minutes / 4 - 180)
    lat_rad = np.deg2rad(lat)
    return np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(hour_angle)


def main():
    pack_dir = CASE_DIR / "pack_readonly"
    rows = json.loads(str(np.load(CASE_DIR / "four_case_temporal_agri_model_inputs.npz")["manifest_json"].item()))
    temporal = np.load(CASE_DIR / "four_case_temporal_agri_model_inputs.npz")
    original_audit = json.loads((CASE_DIR / "original_agri_solar_input_audit_20260924.json").read_text(encoding="utf-8"))
    original_agri = np.load(CASE_DIR / "four_case_temporal_original_agri_crops.npz")["original_agri"]
    norm = json.loads((pack_dir / "norm.json").read_text(encoding="utf-8"))
    mean = np.asarray(norm["mean"], dtype=np.float64)
    std = np.asarray(norm["std"], dtype=np.float64)
    latlon = {}
    for station, path in SIDE_CARS.items():
        d = np.load(path)
        latlon[station] = (d["grid_lat"].astype(np.float64), d["grid_lon"].astype(np.float64))

    all_err = []
    cases = {}
    offsets = np.arange(-10, 10.001, 0.25)
    for cid in range(1, 5):
        frame_meta = [x for x in rows if x["case_id"] == cid]
        frame_results = []
        errors = []
        frame_pairs = []
        for j, meta in enumerate(frame_meta):
            idx = next(i for i, x in enumerate(rows) if x["case_id"] == cid and x["timestamp_utc"] == meta["timestamp_utc"])
            x = temporal["agri"][idx].astype(np.float64)
            cossoz = x[13]
            cosraa = x[14]
            day = x[15]
            lat, lon = latlon[meta["station_id"]]
            independent = noaa_cos_zenith(meta["timestamp_utc"], lat, lon)
            err = cossoz - independent
            errors.append(err)
            all_err.append(err.ravel())
            frame_pairs.append((meta, cossoz.copy(), lat, lon))
            valid = np.isfinite(cossoz) & np.isfinite(independent)
            # Best uniform clock offset for a product-level timing-convention check.
            rmse_by_offset = []
            for dt in offsets:
                pred = noaa_cos_zenith(meta["timestamp_utc"], lat, lon, dt)
                rmse_by_offset.append(float(np.sqrt(np.mean((cossoz[valid] - pred[valid]) ** 2))))
            best = int(np.argmin(rmse_by_offset))
            day_expected = (cossoz >= np.cos(np.deg2rad(85))).astype(np.float32)
            frame_results.append({
                "timestamp_utc": meta["timestamp_utc"],
                "timestamp_bjt": meta["timestamp_bjt"],
                "delta_minutes_from_case": meta["delta_minutes"],
                "cosSOZ_input_range": [float(cossoz.min()), float(cossoz.max())],
                "cosSOZ_noaa_nominal_mae": float(np.mean(np.abs(err[valid]))),
                "cosSOZ_noaa_nominal_rmse": float(np.sqrt(np.mean(err[valid] ** 2)),),
                "cosSOZ_noaa_nominal_max_abs": float(np.max(np.abs(err[valid]))),
                "best_uniform_NOAA_time_offset_min": float(offsets[best]),
                "best_offset_rmse": rmse_by_offset[best],
                "best_offset_rmse_reduction_fraction": float(1 - rmse_by_offset[best] / max(rmse_by_offset[offsets.tolist().index(0.0)], 1e-12)),
                "cosRAA_input_range": [float(cosraa.min()), float(cosraa.max())],
                "day_mask_unique": [float(v) for v in np.unique(day)],
                "day_mask_mismatch_vs_cosSOZ_ge_85deg": int(np.count_nonzero(day != day_expected)),
                "model_normalized_cosSOZ_center": float((cossoz[meta["pixel_row"], meta["pixel_col"]] - mean[13]) / std[13]),
                "model_normalized_cosRAA_center": float((cosraa[meta["pixel_row"], meta["pixel_col"]] - mean[14]) / std[14]),
                "day_mask_center": float(day[meta["pixel_row"], meta["pixel_col"]]),
            })
        stack = np.stack(errors)
        shared_offset_rmse = []
        for dt in offsets:
            squared = []
            for meta, cossoz, lat, lon in frame_pairs:
                nominal = noaa_cos_zenith(meta["timestamp_utc"], lat, lon, dt)
                squared.append((cossoz - nominal) ** 2)
            shared_offset_rmse.append(float(np.sqrt(np.mean(np.stack(squared)))))
        cases[str(cid)] = {
            "station_id": frame_meta[0]["station_id"],
            "false_pixel_rc": [frame_meta[1]["pixel_row"], frame_meta[1]["pixel_col"]],
            "frames": frame_results,
            "case_mean_nominal_rmse": float(np.sqrt(np.mean(stack ** 2))),
            "case_mean_abs_error": float(np.mean(np.abs(stack))),
            "case_best_shared_offset_min": float(offsets[int(np.argmin(shared_offset_rmse))]),
            "case_best_shared_offset_rmse": float(np.min(shared_offset_rmse)),
        }

    all_e = np.concatenate(all_err)
    summary = {
        "scope": "four false-cloud validation cases and their +/-15 minute adjacent real-AGRI validation frames; no test rows accessed",
        "input_contract": {
            "channels": norm["feature_names"],
            "normalization": "training mean/std applied to cosSOZ and cosRAA; day_mask mean=0/std=1, so remains binary",
            "expected_cosSOZ_formula": "cos(deg2rad(SOZ_degrees))",
            "expected_cosRAA_formula": "cos(deg2rad(SOA_degrees - SAA_degrees))",
            "day_mask_source_contract": "original AGRI day_mask based on SOZ < 85 degrees; documented in AGRI spatial payload review",
        },
        "independent_check": "NOAA fractional-year declination/equation-of-time approximation at nominal UTC timestamp and sidecar geolocation; not a high-precision SPA reference and not the original AGRI scan-time angle field",
        "direct_original_agri_reconstruction": original_audit["max_abs_error_over_12_frames"],
        "verified_source_formulae": {
            "cosSOZ": "cos(deg2rad(original AGRI SOZ channel 18))",
            "cosRAA": "cos(deg2rad(original AGRI SOA channel 17 - SAA channel 15))",
            "day_mask": "original AGRI day_mask channel 19 > 0.5",
            "training_normalization": "raw channel normalized by training-set mean/std; day_mask mean=0/std=1",
        },
        "overall_nominal_noaa_minus_input": {
            "mae": float(np.mean(np.abs(all_e))),
            "rmse": float(np.sqrt(np.mean(all_e ** 2))),
            "max_abs": float(np.max(np.abs(all_e))),
        },
        "case_results": cases,
        "interpretation_limit": "The saved solar input channels match the original per-pixel AGRI SOZ/SOA/SAA formula exactly in all 12 selected frames. The fitted NOAA time offset is only a nominal-UTC cross-check; it is not evidence of an input error or a clock correction because the original source angles already match exactly.",
    }
    (CASE_DIR / "solar_geometry_audit_20260924.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Visual comparison on identical 16x16 station grids. NOAA is an
    # independent nominal-UTC approximation, not an authoritative AGRI angle.
    controls = pd.read_csv(CASE_DIR / "false_cloud_matched_controls_20260924.csv")
    fig = plt.figure(figsize=(13.5, 11.6))
    gs = fig.add_gridspec(4, 3, left=0.12, right=0.82, top=0.90, bottom=0.06,
                          wspace=0.25, hspace=0.35)
    axes = gs.subplots()
    handles = {}
    coltitles = ["Saved model input: cosSOZ", "Independent NOAA nominal-UTC cosSOZ", "Input minus NOAA (nominal UTC)"]
    for ax, title in zip(axes[0], coltitles):
        ax.set_title(title, fontsize=10)
    for cid in range(1, 5):
        case_rows = [x for x in rows if x["case_id"] == cid]
        meta = case_rows[1]
        idx = next(i for i, x in enumerate(rows) if x["case_id"] == cid and x["timestamp_utc"] == meta["timestamp_utc"])
        cossoz = np.cos(np.deg2rad(original_agri[idx, 18].astype(np.float64)))
        lat, lon = latlon[meta["station_id"]]
        noaa = noaa_cos_zenith(meta["timestamp_utc"], lat, lon)
        residual = cossoz - noaa
        control = controls.loc[controls.case_id == cid].iloc[0]
        false_rc = (int(meta["pixel_row"]), int(meta["pixel_col"]))
        control_rc = (int(control["control_pixel_row"]), int(control["control_pixel_col"]))
        handles["input"] = axes[cid - 1, 0].imshow(cossoz, cmap="viridis", vmin=0.5, vmax=1.0, interpolation="nearest")
        handles["noaa"] = axes[cid - 1, 1].imshow(noaa, cmap="viridis", vmin=0.5, vmax=1.0, interpolation="nearest")
        handles["resid"] = axes[cid - 1, 2].imshow(residual, cmap="coolwarm", vmin=-0.02, vmax=0.02, interpolation="nearest")
        timestamp_bjt = pd.Timestamp(meta["timestamp_utc"]).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M BJT")
        axes[cid - 1, 0].set_ylabel(f"Case {cid}\n{meta['station_id']}\n{timestamp_bjt}", fontsize=8)
        for ax in axes[cid - 1]:
            r, c = false_rc
            cr, cc = control_rc
            ax.plot(c, r, marker="x", color="#ff2e2e", markersize=8, markeredgewidth=2,
                    linestyle="none", zorder=10)
            ax.plot(cc, cr, marker="*", color="white", markeredgecolor="#111111", markersize=10,
                    linestyle="none", zorder=11)
            ax.set_xticks([0, 5, 10, 15])
            ax.set_yticks([0, 5, 10, 15])
            ax.set_xlabel("column")
    for image, y, label, ticks in [
        (handles["input"], 0.65, "cosSOZ", [0.5, 0.75, 1.0]),
        (handles["resid"], 0.27, "Δ cosSOZ", [-0.02, 0, 0.02]),
    ]:
        cax = fig.add_axes([0.86, y, 0.018, 0.18])
        cb = fig.colorbar(image, cax=cax, ticks=ticks)
        cb.set_label(label, fontsize=9)
        cb.ax.tick_params(labelsize=8)
    fig.suptitle("Solar geometry audit on the four false-cloud validation frames\n"
                 "Saved cosSOZ exactly matches original AGRI SOZ conversion; NOAA is only a nominal-time cross-check. Red × = error; white ★ = correct control.",
                 fontsize=10)
    fig.savefig(CASE_DIR / "forensic_figures/solar_geometry_input_vs_noaa.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary["overall_nominal_noaa_minus_input"], indent=2))
    for cid, item in cases.items():
        print(cid, item["station_id"], "nominal RMSE", item["case_mean_nominal_rmse"], "mean abs", item["case_mean_abs_error"], "best shared offset", item["case_best_shared_offset_min"])


if __name__ == "__main__":
    main()
