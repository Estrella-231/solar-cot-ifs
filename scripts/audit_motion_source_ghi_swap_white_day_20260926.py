"""Re-score saved paired GHI-head predictions on a stable daylight cohort."""
import argparse
import json
from pathlib import Path

import numpy as np


def metrics(error):
    error = np.asarray(error, dtype=np.float64)
    if not len(error) or not np.isfinite(error).all():
        raise RuntimeError("empty or nonfinite white-day errors")
    return {"n": int(len(error)), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))), "bias": float(error.mean())}


def paired_bootstrap(truth, simvp, motion, days, seed=20260926, reps=2000):
    unique = np.unique(days)
    s0, s1, n = [], [], []
    for day in unique:
        take = days == day
        s0.append(np.square(simvp[take] - truth[take]).sum(dtype=np.float64))
        s1.append(np.square(motion[take] - truth[take]).sum(dtype=np.float64))
        n.append(int(take.sum()))
    s0, s1, n = map(lambda x: np.asarray(x, dtype=np.float64), (s0, s1, n))
    draw = np.random.default_rng(seed).integers(0, len(unique), size=(reps, len(unique)))
    d = np.sqrt(s1[draw].sum(1) / n[draw].sum(1)) - np.sqrt(s0[draw].sum(1) / n[draw].sum(1))
    return {"unit": "BJT initialization date", "dates": int(len(unique)), "repetitions": reps,
            "seed": seed, "delta_motion_minus_simvp_rmse_95pct": np.quantile(d, [0.025, 0.975]).tolist(),
            "probability_motion_rmse_lower": float(np.mean(d < 0))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--prediction-dir", type=Path, required=True)
    p.add_argument("--clear", type=Path, required=True)
    p.add_argument("--sequence", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--clear-min", type=float, default=100.0)
    a = p.parse_args()
    if a.output.exists():
        raise RuntimeError("refuse to overwrite stable-daylight rescore")
    report = json.loads(a.report.read_text(encoding="utf-8"))
    if report.get("state") != "COMPLETE_FIXED_GHI_HEAD_MOTION_SOURCE_COT_SWAP" or report.get("test_used") is not False:
        raise RuntimeError("input-swap predictions incomplete or test-contaminated")
    clear_all = np.load(a.clear, mmap_mode="r", allow_pickle=False)
    seq_all = np.load(a.sequence, mmap_mode="r", allow_pickle=False)
    output = {"state": "COMPLETE_STABLE_DAYLIGHT_RESCORE", "test_used": False,
              "mask": f"clear_sky_GHI >= {a.clear_min:g} W/m2; common for both COT inputs",
              "parent_report": str(a.report), "stations": {}}
    for station in ("Sili", "Zhujia"):
        with np.load(a.prediction_dir / f"{station}_paired_predictions.npz", allow_pickle=False) as z:
            rows = z["pack_row"].astype(np.int64)
            lead = z["lead_minutes"].astype(np.int64)
            days = z["init_day"].astype("U10")
            truth_ghi = z["true_ghi"].astype(np.float64)
            simvp_ghi = z["simvp_ghi"].astype(np.float64)
            motion_ghi = z["motion_source_ghi"].astype(np.float64)
            simvp_kt = z["simvp_kt"].astype(np.float64)
            motion_kt = z["motion_source_kt"].astype(np.float64)
        if rows.max(initial=-1) >= len(clear_all) or rows.max(initial=-1) >= len(seq_all):
            raise RuntimeError("pack-row index exceeds common target arrays")
        clear = np.asarray(clear_all[rows], dtype=np.float64)
        seq = np.asarray(seq_all[rows], dtype=np.int64)
        white = clear >= a.clear_min
        if not white.any():
            raise RuntimeError(f"empty stable daylight set: {station}")
        truth_kt = truth_ghi / clear
        result = {"rows_total": int(len(rows)), "white_day_rows": int(white.sum()),
                  "white_day_fraction": float(white.mean()), "short_15_to_120min": {},
                  "far_180_to_240min": {}, "per_lead": {}}
        for period, period_mask in (("short_15_to_120min", (lead <= 120) & white),
                                    ("far_180_to_240min", (lead >= 180) & white)):
            for name, pred_ghi, pred_kt in (("SimVP", simvp_ghi, simvp_kt),
                                            ("Motion+Source", motion_ghi, motion_kt)):
                result[period][name] = {"ghi_Wm2": metrics((pred_ghi - truth_ghi)[period_mask]),
                                        "kt": metrics((pred_kt - truth_kt)[period_mask])}
        for minute in sorted(np.unique(lead)):
            m = (lead == minute) & white
            result["per_lead"][str(int(minute))] = {}
            for name, pred_ghi, pred_kt in (("SimVP", simvp_ghi, simvp_kt),
                                            ("Motion+Source", motion_ghi, motion_kt)):
                result["per_lead"][str(int(minute))][name] = {
                    "ghi_Wm2": metrics((pred_ghi - truth_ghi)[m]),
                    "kt": metrics((pred_kt - truth_kt)[m])}
        far = (lead >= 180) & white
        result["far_date_bootstrap"] = {
            "GHI_Wm2": paired_bootstrap(truth_ghi[far], simvp_ghi[far], motion_ghi[far], days[far]),
            "kt": paired_bootstrap(truth_kt[far], simvp_kt[far], motion_kt[far], days[far], seed=20260927)}
        # Group by sequence and lead; only compare adjacent leads when both
        # target rows satisfy the identical clear-sky threshold.
        row_by_key = {(int(s), int(l)): i for i, (s, l) in enumerate(zip(seq, lead)) if white[i]}
        changes, far_changes, truth_changes, far_truth_changes = [], [], [], []
        for (s, l), i in row_by_key.items():
            previous = row_by_key.get((s, l - 15))
            if previous is None:
                continue
            dt = truth_kt[i] - truth_kt[previous]
            diff0 = simvp_kt[i] - simvp_kt[previous]
            diff1 = motion_kt[i] - motion_kt[previous]
            changes.append((dt, diff0, diff1, l))
            if l >= 180:
                far_changes.append((dt, diff0, diff1, l))
        for label, values in (("all_adjacent", changes), ("far_current_lead_ge_180", far_changes)):
            arr = np.asarray(values, dtype=np.float64)
            if not len(arr):
                continue
            dtruth, d0, d1 = arr[:, 0], arr[:, 1], arr[:, 2]
            event = np.abs(dtruth) >= .1
            result[label] = {"n": int(len(arr)), "observed_change_event_abs_delta_kt_ge_0.1": int(event.sum()),
                             "SimVP_delta_kt_error": metrics(d0 - dtruth),
                             "Motion+Source_delta_kt_error": metrics(d1 - dtruth),
                             "SimVP_event_delta_kt_error": metrics((d0 - dtruth)[event]),
                             "Motion+Source_event_delta_kt_error": metrics((d1 - dtruth)[event])}
        output["stations"][station] = result
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(a.output.resolve())


if __name__ == "__main__":
    main()
