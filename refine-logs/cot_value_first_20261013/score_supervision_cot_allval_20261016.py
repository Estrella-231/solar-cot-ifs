#!/usr/bin/env python3
"""FP64 common-support scoring for full-validation SimVP supervision banks."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


METHODS = {
    "original_simvp": "baseline",
    "mse_finetune": "mse",
    "temporal_delta": "temporal",
    "spatial_gradient": "spatial",
}
STATIONS = ("sili", "zhujia")
SCALES = {"local_3x3": (slice(7, 10), slice(7, 10)),
          "patch_16x16": (slice(0, 16), slice(0, 16))}
LEADS = tuple(range(15, 241, 15))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stats(pred, truth, mask):
    e = pred[mask] - truth[mask]
    if not e.size:
        return {"n": 0, "rmse": None, "mae": None, "bias": None,
                "pod": None, "far": None, "csi": None, "tp": 0, "fp": 0, "fn": 0}
    tp = int(((pred >= 30.0) & (truth >= 30.0) & mask).sum())
    fp = int(((pred >= 30.0) & (truth < 30.0) & mask).sum())
    fn = int(((pred < 30.0) & (truth >= 30.0) & mask).sum())
    return {"n": int(e.size), "rmse": float(np.sqrt(np.mean(e * e, dtype=np.float64))),
            "mae": float(np.mean(np.abs(e), dtype=np.float64)),
            "bias": float(np.mean(e, dtype=np.float64)), "tp": tp, "fp": fp, "fn": fn,
            "pod": tp / (tp + fn) if tp + fn else None,
            "far": fp / (tp + fp) if tp + fp else None,
            "csi": tp / (tp + fp + fn) if tp + fp + fn else None}


def load_bank(folder: Path):
    profile = json.loads((folder / "profile.json").read_text(encoding="utf-8"))
    if profile.get("split") != "val_only" or profile.get("test_payloads_read") != 0:
        raise RuntimeError(f"invalid split/test receipt in {folder}")
    if profile.get("state") != "COMPLETE_ALL_VALIDATION_S_TO_R_COT_AND_AGRI3X3_BANK":
        raise RuntimeError(f"bank is not complete all-validation output: {folder}")
    bank_path = folder / "forecast_cot_val_bank.npz"
    z = np.load(bank_path, allow_pickle=False)
    need = ("forecast_log1p_cot", "target_agri_r_log1p_cot", "persistence_log1p_cot",
            "cpp_reference_cot", "cpp_valid_mask", "target_time_utc", "dataset_indices", "sequence_ids")
    if any(k not in z.files for k in need):
        raise RuntimeError(f"required array missing in {bank_path}")
    return profile, bank_path, z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True,
                   help="directory containing original_simvp, mse, temporal, spatial bank folders")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)

    loaded = {}
    for name, subdir in METHODS.items():
        folder = a.base / subdir
        loaded[name] = load_bank(folder)
    reference = loaded["original_simvp"][2]
    keys = reference["target_time_utc"].astype(str)
    ids = reference["sequence_ids"].astype(str)
    indices = reference["dataset_indices"].astype(np.int64)
    expected_shape = (len(ids), 2, 16, 16, 16)
    if expected_shape != (3140, 2, 16, 16, 16):
        raise RuntimeError(f"unexpected full validation shape: {expected_shape}")
    if len(set(indices.tolist())) != 3140 or not np.array_equal(indices, np.arange(3140)):
        raise RuntimeError("validation dataset indices are duplicated or out of order")
    for name, (_, _, z) in loaded.items():
        if z["forecast_log1p_cot"].shape != expected_shape:
            raise RuntimeError(f"unexpected forecast shape for {name}")
        for k in ("target_time_utc", "dataset_indices", "sequence_ids", "cpp_valid_mask", "cpp_reference_cot",
                  "target_agri_r_log1p_cot", "persistence_log1p_cot"):
            # equal_nan is unsupported for string and boolean dtypes on some NumPy versions.
            # Keys, indices, IDs and masks are exact; only floating-point reference arrays
            # need NaN-aware equality.
            equal = (np.array_equal(z[k], reference[k], equal_nan=True)
                     if np.issubdtype(z[k].dtype, np.floating)
                     else np.array_equal(z[k], reference[k]))
            if not equal:
                raise RuntimeError(f"common-support mismatch: {name}/{k}")
        if not np.isfinite(z["forecast_log1p_cot"]).all():
            raise RuntimeError(f"non-finite forecast in {name}")

    cpp = reference["cpp_reference_cot"].astype(np.float64)
    mask = reference["cpp_valid_mask"].astype(bool)
    if cpp.shape != expected_shape or int(mask.sum()) == 0 or not np.isfinite(cpp[mask]).all():
        raise RuntimeError("CPP reference support is malformed")
    cpp_dates = set()
    all_dates = set()
    for t in keys.reshape(-1):
        bjt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S") + timedelta(hours=8)
        all_dates.add(bjt.date().isoformat())
    for i in range(3140):
        for li, t in enumerate(keys[i]):
            d = (datetime.strptime(t, "%Y-%m-%d %H:%M:%S") + timedelta(hours=8)).date().isoformat()
            if mask[i, :, li].any():
                cpp_dates.add(d)
            if li and (datetime.strptime(t, "%Y-%m-%d %H:%M:%S") -
                       datetime.strptime(keys[i, li - 1], "%Y-%m-%d %H:%M:%S")).total_seconds() != 900:
                raise RuntimeError(f"noncanonical lead spacing in window {i}")

    rows = []
    aggregate = {name: {scale: {"sse": 0.0, "n": 0, "mae_abs_sum": 0.0, "bias_sum": 0.0,
                                "tp": 0, "fp": 0, "fn": 0} for scale in SCALES}
                 for name in loaded}
    proxy_aggregate = {name: {"sse": 0.0, "n": 0, "mae_abs_sum": 0.0, "bias_sum": 0.0}
                       for name in loaded}
    for name, (_, bank_path, z) in loaded.items():
        pred = np.expm1(z["forecast_log1p_cot"].astype(np.float64))
        target_r = np.expm1(z["target_agri_r_log1p_cot"].astype(np.float64))
        for si, station in enumerate(STATIONS):
            for li, lead in enumerate(LEADS):
                pm = pred[:, si, li]
                tm = target_r[:, si, li]
                proxy_e = pm - tm
                proxy_aggregate[name]["sse"] += float(np.square(proxy_e, dtype=np.float64).sum())
                proxy_aggregate[name]["n"] += int(proxy_e.size)
                proxy_aggregate[name]["mae_abs_sum"] += float(np.abs(proxy_e).sum(dtype=np.float64))
                proxy_aggregate[name]["bias_sum"] += float(proxy_e.sum(dtype=np.float64))
                for scale, (rs, cs) in SCALES.items():
                    m = mask[:, si, li, rs, cs]
                    tr = cpp[:, si, li, rs, cs]
                    pr = pred[:, si, li, rs, cs]
                    s = stats(pr, tr, m)
                    rows.append({"method": name, "station": station, "lead_min": lead,
                                 "scale": scale, **s})
                    e = pr[m] - tr[m]
                    agg = aggregate[name][scale]
                    agg["sse"] += float(np.square(e, dtype=np.float64).sum())
                    agg["n"] += int(e.size)
                    agg["mae_abs_sum"] += float(np.abs(e).sum(dtype=np.float64))
                    agg["bias_sum"] += float(e.sum(dtype=np.float64))
                    agg["tp"] += s["tp"]; agg["fp"] += s["fp"]; agg["fn"] += s["fn"]

    summary = {"state": "PASS_FP64_COMMON_SUPPORT_COT_SCORING",
        "scope": "validation only; CPP is sparse retrieval reference; R(observed future AGRI) is proxy, not physical truth; full-cohort proxy has no independent truth or deployable-input status",
               "test_payloads_read": 0, "windows": 3140, "validation_bjt_dates": len(all_dates),
               "cpp_supported_bjt_dates": len(cpp_dates), "cpp_mask_pixels_full_patch": int(mask.sum()),
               "banks": {}, "aggregate_cpp_metrics": {}, "full_cohort_proxy_response": {}}
    for name, (profile, bank_path, _) in loaded.items():
        summary["banks"][name] = {"bank_sha256": sha(bank_path),
                                  "simvp_checkpoint_sha256": profile["simvp_checkpoint_sha256"],
                                  "r_checkpoint_sha256": profile["r_checkpoint_sha256"],
                                  "windows": profile["windows_completed"],
                                  "test_payloads_read": profile["test_payloads_read"]}
        summary["aggregate_cpp_metrics"][name] = {}
        for scale, v in aggregate[name].items():
            n = v["n"]
            summary["aggregate_cpp_metrics"][name][scale] = {
                "n_pixels": n, "rmse": float(np.sqrt(v["sse"] / n)) if n else None,
                "mae": v["mae_abs_sum"] / n if n else None,
                "bias": v["bias_sum"] / n if n else None,
                "tp": v["tp"], "fp": v["fp"], "fn": v["fn"],
                "pod": v["tp"] / (v["tp"] + v["fn"]) if v["tp"] + v["fn"] else None,
                "far": v["fp"] / (v["tp"] + v["fp"]) if v["tp"] + v["fp"] else None,
                "csi": v["tp"] / (v["tp"] + v["fp"] + v["fn"]) if v["tp"] + v["fp"] + v["fn"] else None}
        pv = proxy_aggregate[name]; n = pv["n"]
        summary["full_cohort_proxy_response"][name] = {
            "n_values": n, "rmse_vs_R_observed_future_AGRI": float(np.sqrt(pv["sse"] / n)),
            "mae_vs_R_observed_future_AGRI": pv["mae_abs_sum"] / n,
            "bias_vs_R_observed_future_AGRI": pv["bias_sum"] / n}

    a.output.mkdir(parents=True)
    with (a.output / "cot_metrics_by_station_lead_scale.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary["source_hashes"] = {"scorer_sha256": sha(Path(__file__))}
    (a.output / "summary_fp64.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
