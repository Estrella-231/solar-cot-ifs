"""Select matched correctly classified validation controls for four COT cases."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import netCDF4

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "results/cot_staged_diagnostic_20260924/source_cases"
PACK = CASE_DIR / "pack_readonly"
RUN = ROOT / "results/cot_real_agri_staged_hurdle_seed42_20260924"

FALSE_CASES = [
    {"case_id": 1, "station_id": "zhujia", "row_index": 15506, "pixel_row": 0, "pixel_col": 9},
    {"case_id": 2, "station_id": "sili", "row_index": 6655, "pixel_row": 0, "pixel_col": 9},
    {"case_id": 3, "station_id": "zhujia", "row_index": 15499, "pixel_row": 8, "pixel_col": 11},
    {"case_id": 4, "station_id": "sili", "row_index": 6327, "pixel_row": 0, "pixel_col": 6},
]
CLP_FILES = {
    1: Path("results/cot_staged_diagnostic_20260924/source_cases/nc_windows/zhujia_202507060815.nc"),
    2: Path("results/cot_staged_diagnostic_20260924/source_cases/nc_windows/sili_202507140515.nc"),
    3: Path("results/cot_staged_diagnostic_20260924/source_cases/nc_windows/zhujia_202507060630.nc"),
    4: Path("results/cot_staged_diagnostic_20260924/source_cases/nc_windows/sili_202507070515.nc"),
}


def main():
    rows = pd.read_csv(PACK / "rows.csv")
    # Pack targets store COT/100; restore physical CPP COT units for this audit.
    cot = np.load(PACK / "target.npy", mmap_mode="r")[:, 0] * np.float32(100.0)
    mask = np.load(PACK / "mask.npy", mmap_mode="r")[:, 0].astype(bool)
    raw = np.load(PACK / "x_raw.npy", mmap_mode="r")
    stage = np.load(RUN / "stage2_conditional_regressors/best_validation_predictions.npz")
    cls = np.load(RUN / "stage1_classifier/best_validation_probabilities.npz")

    # Model outputs are keyed by the original manifest row id; map each id to
    # both its position in the packed arrays and its output position.
    row_to_pos = {int(row_id): pos for pos, row_id in enumerate(rows["index"].values)}
    output_to_pos = {int(row_id): pos for pos, row_id in enumerate(stage["row_indices"])}
    val = rows[rows["split"] == "validation"]

    selected = []
    for case in FALSE_CASES:
        row_id = case["row_index"]
        pos = row_to_pos[row_id]
        out_pos = output_to_pos[row_id]
        r, c = case["pixel_row"], case["pixel_col"]
        y0, y1 = max(0, r - 2), min(16, r + 3)
        x0, x1 = max(0, c - 2), min(16, c + 3)
        ref_patch = np.asarray(cot[pos, y0:y1, x0:x1])
        ref_mask = np.asarray(mask[pos, y0:y1, x0:x1])
        ref_cf = float((((ref_patch >= 10.0) & ref_mask).sum()) / max(ref_mask.sum(), 1))
        ref_mean = float(ref_patch[ref_mask].mean()) if ref_mask.any() else 0.0
        ref_cos = float(raw[pos, 13, r, c])
        case_time = str(rows.iloc[pos]["timestamp_utc"])

        # A control from the same image gives exactly the same acquisition,
        # atmosphere and solar-time context. Search pixels with a correct
        # CPP phase class and routed-COT behavior.
        pmap = cls["cloud_probability"][out_pos, 0]
        final_map = stage["pred_cot"][out_pos, 0]
        current_cot = np.asarray(cot[pos])
        current_mask = np.asarray(mask[pos])
        with netCDF4.Dataset(CLP_FILES[case["case_id"]]) as ds:
            current_clp = np.ma.filled(ds.variables["CLP"][:], np.nan).astype(np.float32)
        candidates = []
        for rr in range(16):
            for cc in range(16):
                distance = float(np.hypot(rr - r, cc - c))
                if not (0 < distance <= 22):
                    continue
                if not current_mask[rr, cc] or not np.isfinite(current_clp[rr, cc]):
                    continue
                is_clear = current_clp[rr, cc] == 0
                is_cloud = current_clp[rr, cc] in (1, 2)
                correct_clear = is_clear and current_cot[rr, cc] < 0.1 and pmap[rr, cc] < 0.5 and final_map[rr, cc] < 1.0
                correct_cloud = is_cloud and pmap[rr, cc] >= 0.5 and final_map[rr, cc] > 1.0
                if not (correct_clear or correct_cloud):
                    continue
                ay0, ay1 = max(0, rr - 2), min(16, rr + 3)
                ax0, ax1 = max(0, cc - 2), min(16, cc + 3)
                local = current_cot[ay0:ay1, ax0:ax1]
                local_mask = current_mask[ay0:ay1, ax0:ax1]
                local_cf = float(((local >= 10.0) & local_mask).sum() / max(local_mask.sum(), 1))
                local_mean = float(local[local_mask].mean()) if local_mask.any() else 0.0
                cos = float(raw[pos, 13, rr, cc])
                dsun = abs(cos - ref_cos)
                dcf = abs(local_cf - ref_cf)
                score = dsun / 0.015 + dcf / 0.15 + abs(np.log1p(local_mean) - np.log1p(ref_mean))
                candidates.append((score, rr, cc, local_cf, local_mean, dsun,
                                   "clear" if correct_clear else "cloud"))
        if not candidates:
            raise RuntimeError(f"No nearby correctly classified pixel control for case {case['case_id']}")
        # Choose the correct pixel whose local cloud field most closely
        # matches the false case; its true class may be clear or cloudy.
        candidates.sort()
        score, control_r, control_c, control_cf, control_mean, sun_delta, control_kind = candidates[0]
        # Keep the control in the same packed frame and identify its actual
        # pixel coordinate and CPP phase class.
        control_id, control_pos, control_out = row_id, pos, out_pos
        selected.append({
            **case,
            "false_timestamp_utc": case_time,
            "false_timestamp_bjt": str(rows.iloc[pos]["timestamp_bjt"]),
            "false_cosSOZ": ref_cos,
            "false_local_5x5_cloud_fraction": ref_cf,
            "false_local_5x5_mean_COT": ref_mean,
            "false_p_cloud": float(cls["cloud_probability"][out_pos, 0, r, c]),
            "control_row_index": control_id,
            "control_timestamp_utc": str(rows.iloc[control_pos]["timestamp_utc"]),
            "control_timestamp_bjt": str(rows.iloc[control_pos]["timestamp_bjt"]),
            "control_pixel_row": control_r,
            "control_pixel_col": control_c,
            "control_kind": control_kind,
            "control_cosSOZ": float(raw[control_pos, 13, control_r, control_c]),
            "control_local_5x5_cloud_fraction": control_cf,
            "control_local_5x5_mean_COT": control_mean,
            "control_cpp_COT": float(current_cot[control_r, control_c]),
            "control_cpp_CLP": int(current_clp[control_r, control_c]),
            "control_p_cloud": float(pmap[control_r, control_c]),
            "control_predicted_COT": float(final_map[control_r, control_c]),
            "control_frozen_R_COT": float(np.load(RUN / "frozen_R_validation_predictions.npz")["pred_cot"][control_out, 0, control_r, control_c]),
            "match_score": float(score),
            "abs_cosSOZ_difference": float(sun_delta),
        })

    out_csv = CASE_DIR / "false_cloud_matched_controls_20260924.csv"
    pd.DataFrame(selected).to_csv(out_csv, index=False)
    (CASE_DIR / "false_cloud_matched_controls_20260924.json").write_text(
        json.dumps({"selection": "validation only; same frame/station; correctly classified CPP-clear or CPP-water/ice pixel; selected by lowest score over solar angle and 5x5 CPP COT>=10 fraction/mean context; all four selected controls are CPP ice cloud", "cases": selected}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(pd.DataFrame(selected)[["case_id", "false_timestamp_bjt", "false_cosSOZ", "false_local_5x5_cloud_fraction", "false_p_cloud", "control_kind", "control_pixel_row", "control_pixel_col", "control_cpp_CLP", "control_cpp_COT", "control_p_cloud", "control_predicted_COT", "match_score"]].to_string(index=False))
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
