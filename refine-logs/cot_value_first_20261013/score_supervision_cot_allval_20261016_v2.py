#!/usr/bin/env python3
"""Score R07 banks on a source-reconstructed, station-time keyed CPP reference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

METHODS = {"original_simvp": "baseline", "mse_finetune": "mse",
           "temporal_delta": "temporal", "spatial_gradient": "spatial"}
STATIONS = ("sili", "zhujia")
SCALES = {"local_3x3": (slice(7, 10), slice(7, 10)),
          "patch_16x16": (slice(0, 16), slice(0, 16))}
LEADS = tuple(range(15, 241, 15))
REPS = 2000
SEED = 20261016


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def score(pred, truth, mask):
    e = pred[mask] - truth[mask]
    if not e.size:
        return {"n": 0, "rmse": None, "mae": None, "bias": None,
                "pod": None, "far": None, "csi": None, "tp": 0, "fp": 0, "fn": 0}
    tp = int(((pred >= 30.) & (truth >= 30.) & mask).sum())
    fp = int(((pred >= 30.) & (truth < 30.) & mask).sum())
    fn = int(((pred < 30.) & (truth >= 30.) & mask).sum())
    return {"n": int(e.size), "rmse": float(np.sqrt(np.mean(e * e, dtype=np.float64))),
            "mae": float(np.mean(np.abs(e), dtype=np.float64)), "bias": float(np.mean(e, dtype=np.float64)),
            "tp": tp, "fp": fp, "fn": fn,
            "pod": tp / (tp + fn) if tp + fn else None,
            "far": fp / (tp + fp) if tp + fp else None,
            "csi": tp / (tp + fp + fn) if tp + fp + fn else None}


def reference_key_map(rows_path: Path, predictions_path: Path):
    with rows_path.open(newline="", encoding="utf-8") as f:
        supervised = [r for r in csv.DictReader(f) if r["state"] == "SUPERVISED"]
    with np.load(predictions_path, allow_pickle=False) as z:
        indices = z["sample_indices"].astype(np.int64)
        ref = z["reference_cot"].astype(np.float32)
        mask = z["mask"].astype(bool)
    if ref.ndim != 4 or ref.shape[1] != 1 or mask.shape != ref.shape:
        raise RuntimeError(f"unexpected validation reference shapes: {ref.shape}, {mask.shape}")
    expected_val = {i for i, row in enumerate(supervised) if row["split"] == "val"}
    saved = {int(i): j for j, i in enumerate(indices)}
    if len(saved) != len(indices) or set(saved) != expected_val:
        raise RuntimeError("R predictions do not cover exactly the full-index validation rows")
    ref_keys = {}
    for source_index, row in enumerate(supervised):
        if row["split"] != "val":
            continue
        j = saved[source_index]
        key = (row["station"].lower(), row["target_time_utc"])
        if key in ref_keys:
            raise RuntimeError(f"duplicate station/time CPP key: {key}")
        ref_keys[key] = (ref[j, 0], mask[j, 0])
    return ref_keys, {"rows_sha256": sha(rows_path), "validation_predictions_sha256": sha(predictions_path),
                      "supervised_rows": len(supervised), "validation_samples": len(indices),
                      "reference_keys": len(ref_keys), "key_rule": "(station,target_time_utc), full SUPERVISED index"}


def reindex_reference(target_times, ref_keys):
    n = target_times.shape[0]
    refs = np.full((n, 2, 16, 16, 16), np.nan, dtype=np.float32)
    masks = np.zeros((n, 2, 16, 16, 16), dtype=bool)
    hits = 0
    for wi in range(n):
        for si, station in enumerate(STATIONS):
            for li, raw_time in enumerate(target_times[wi]):
                t = str(raw_time)
                pair = ref_keys.get((station, t))
                if pair is not None:
                    refs[wi, si, li] = pair[0]
                    masks[wi, si, li] = pair[1]
                    hits += int(pair[1].sum())
    return refs, masks, hits


def date_keys(target_times):
    out = np.empty(target_times.shape, dtype="U10")
    for i in range(target_times.shape[0]):
        for j, raw in enumerate(target_times[i]):
            out[i, j] = (datetime.strptime(str(raw), "%Y-%m-%d %H:%M:%S") + timedelta(hours=8)).date().isoformat()
    return out


def bootstrap_pair(metrics_by_method, dates_by_lead, truth, mask, lead_macro, rng, draws):
    """Paired date-block CI for 3x3 macro RMSE and pooled thick-cloud CSI."""
    methods = list(metrics_by_method)
    unique_dates = sorted(set(dates_by_lead.ravel().tolist()))
    # Build station x lead sufficient statistics per date for each method.
    acc = {m: {} for m in methods}
    for method, pred in metrics_by_method.items():
        for scale, (rs, cs) in SCALES.items():
            for si in range(2):
                for li in range(16):
                    per_date = {}
                    for d in unique_dates:
                        # Windows contribute only the target lead whose BJT date is d.
                        wi = np.flatnonzero(dates_by_lead[:, li] == d)
                        if not len(wi):
                            continue
                        mm = mask[wi, si, li, rs, cs]
                        pp = pred[wi, si, li, rs, cs]
                        yy = truth[wi, si, li, rs, cs]
                        e = pp[mm].astype(np.float64) - yy[mm].astype(np.float64)
                        per_date[d] = (float(np.dot(e, e)), int(e.size))
                    acc[method][(scale, si, li)] = per_date

    samples = {(m, scale): [] for m in methods for scale in SCALES}
    csi_samples = {m: [] for m in methods}
    boot_draws = [rng.choice(unique_dates, size=len(unique_dates), replace=True).tolist() for _ in range(draws)]
    for picked in boot_draws:
        for method, pred in metrics_by_method.items():
            for scale, (rs, cs) in SCALES.items():
                rmses = []
                for si in range(2):
                    for li in range(16):
                        sse = 0.0; n = 0
                        for d in picked:
                            item = acc[method][(scale, si, li)].get(d)
                            if item:
                                sse += item[0]; n += item[1]
                        if n:
                            rmses.append(np.sqrt(sse / n))
                samples[(method, scale)].append(float(np.mean(rmses)) if rmses else np.nan)
            tp = fp = fn = 0
            for si in range(2):
                for li in range(16):
                    for d in picked:
                        wi = np.flatnonzero(dates_by_lead[:, li] == d)
                        if not len(wi):
                            continue
                        rs, cs = SCALES["local_3x3"]
                        mm = mask[wi, si, li, rs, cs]
                        pp = pred[wi, si, li, rs, cs]
                        yy = truth[wi, si, li, rs, cs]
                        tp += int(((pp >= 30.) & (yy >= 30.) & mm).sum())
                        fp += int(((pp >= 30.) & (yy < 30.) & mm).sum())
                        fn += int(((pp < 30.) & (yy >= 30.) & mm).sum())
            csi_samples[method].append(tp / (tp + fp + fn) if tp + fp + fn else np.nan)
    # Exact full-support point estimates, using the same pooled 3x3 definition
    # as the bootstrap replicates.
    csi_point = {}
    rs, cs = SCALES["local_3x3"]
    for method, pred in metrics_by_method.items():
        tp = fp = fn = 0
        for si in range(2):
            for li in range(16):
                mm = mask[:, si, li, rs, cs]
                pp = pred[:, si, li, rs, cs]
                yy = truth[:, si, li, rs, cs]
                tp += int(((pp >= 30.) & (yy >= 30.) & mm).sum())
                fp += int(((pp >= 30.) & (yy < 30.) & mm).sum())
                fn += int(((pp < 30.) & (yy >= 30.) & mm).sum())
        csi_point[method] = tp / (tp + fp + fn) if tp + fp + fn else np.nan

    pair_ci = {}
    for candidate in methods:
        if candidate in ("original_simvp", "mse_finetune"):
            continue
        pair_ci[candidate] = {}
        for baseline in ("original_simvp", "mse_finetune"):
            pair_ci[candidate][baseline] = {}
            for scale in SCALES:
                d = np.asarray(samples[(candidate, scale)]) - np.asarray(samples[(baseline, scale)])
                d = d[np.isfinite(d)]
                pair_ci[candidate][baseline][scale + "_macro_rmse_delta"] = {
                    "point": float(np.mean([lead_macro[candidate][scale][k] - lead_macro[baseline][scale][k]
                                             for k in lead_macro[candidate][scale]])),
                    "ci95": [float(np.quantile(d, .025)), float(np.quantile(d, .975))],
                    "bootstrap_reps": int(len(d)), "unit": "BJT target-date block"}
            dc = np.asarray(csi_samples[candidate]) - np.asarray(csi_samples[baseline])
            dc = dc[np.isfinite(dc)]
            pair_ci[candidate][baseline]["local_3x3_pooled_CSI_delta"] = {
                "point": float(csi_point[candidate] - csi_point[baseline]),
                "ci95": [float(np.quantile(dc, .025)), float(np.quantile(dc, .975))],
                "bootstrap_reps": int(len(dc)), "unit": "BJT target-date block"}
    return pair_ci


def bootstrap_pair_vectorized(metrics_by_method, dates_by_lead, truth, mask, rng, draws):
    """Vectorized paired date bootstrap from date-level sufficient statistics."""
    methods = list(metrics_by_method)
    method_ix = {name: i for i, name in enumerate(methods)}
    dates = sorted(set(dates_by_lead.ravel().tolist()))
    date_ix = {date: i for i, date in enumerate(dates)}
    scales = list(SCALES.items())
    m_count, d_count, q_count = len(methods), len(dates), len(scales)
    sse = np.zeros((m_count, d_count, q_count, 2, 16), dtype=np.float64)
    nobs = np.zeros_like(sse, dtype=np.int64)
    event = np.zeros((m_count, d_count, 3), dtype=np.int64)
    idx_by_date_lead = {(date, li): np.flatnonzero(dates_by_lead[:, li] == date)
                        for date in dates for li in range(16)}

    for date in dates:
        di = date_ix[date]
        for li in range(16):
            wi = idx_by_date_lead[(date, li)]
            if wi.size == 0:
                continue
            for si in range(2):
                for qi, (_, (rs, cs)) in enumerate(scales):
                    mm = mask[wi, si, li, rs, cs]
                    yy = truth[wi, si, li, rs, cs]
                    for name, pred in metrics_by_method.items():
                        pp = pred[wi, si, li, rs, cs]
                        err = pp[mm].astype(np.float64) - yy[mm].astype(np.float64)
                        sse[method_ix[name], di, qi, si, li] = np.dot(err, err)
                        nobs[method_ix[name], di, qi, si, li] = err.size

                rs, cs = SCALES["local_3x3"]
                mm = mask[wi, si, li, rs, cs]
                yy = truth[wi, si, li, rs, cs]
                for name, pred in metrics_by_method.items():
                    pp = pred[wi, si, li, rs, cs]
                    mi = method_ix[name]
                    event[mi, di, 0] += int(((pp >= 30.) & (yy >= 30.) & mm).sum())
                    event[mi, di, 1] += int(((pp >= 30.) & (yy < 30.) & mm).sum())
                    event[mi, di, 2] += int(((pp < 30.) & (yy >= 30.) & mm).sum())

    draws = int(draws)
    multiplicity = np.zeros((draws, d_count), dtype=np.int16)
    for ri in range(draws):
        multiplicity[ri] = np.bincount(rng.integers(d_count, size=d_count), minlength=d_count)
    bs_sse = np.einsum("rd,mdqsl->rmqsl", multiplicity, sse, optimize=True)
    bs_n = np.einsum("rd,mdqsl->rmqsl", multiplicity, nobs, optimize=True)
    bs_rmse = np.sqrt(np.divide(bs_sse, bs_n, out=np.full_like(bs_sse, np.nan), where=bs_n > 0))
    bs_valid = np.isfinite(bs_rmse)
    bs_macro = np.divide(np.nansum(bs_rmse, axis=(-2, -1)), bs_valid.sum(axis=(-2, -1)),
                         out=np.full((draws, m_count, q_count), np.nan),
                         where=bs_valid.sum(axis=(-2, -1)) > 0)

    point_rmse = np.sqrt(np.divide(sse.sum(axis=1), nobs.sum(axis=1),
                                   out=np.full((m_count, q_count, 2, 16), np.nan),
                                   where=nobs.sum(axis=1) > 0))
    point_valid = np.isfinite(point_rmse)
    point_macro = np.divide(np.nansum(point_rmse, axis=(-2, -1)), point_valid.sum(axis=(-2, -1)),
                            out=np.full((m_count, q_count), np.nan),
                            where=point_valid.sum(axis=(-2, -1)) > 0)

    bs_event = np.einsum("rd,mdc->rmc", multiplicity, event, optimize=True)
    bs_csi = np.divide(bs_event[:, :, 0], bs_event.sum(axis=2),
                       out=np.full((draws, m_count), np.nan), where=bs_event.sum(axis=2) > 0)
    point_event = event.sum(axis=1)
    point_csi = np.divide(point_event[:, 0], point_event.sum(axis=1),
                          out=np.full(m_count, np.nan), where=point_event.sum(axis=1) > 0)

    out = {}
    for candidate in methods:
        if candidate in ("original_simvp", "mse_finetune"):
            continue
        out[candidate] = {}
        for baseline in ("original_simvp", "mse_finetune"):
            out[candidate][baseline] = {}
            for qi, (scale, _) in enumerate(scales):
                delta = bs_macro[:, method_ix[candidate], qi] - bs_macro[:, method_ix[baseline], qi]
                delta = delta[np.isfinite(delta)]
                out[candidate][baseline][scale + "_macro_rmse_delta"] = {
                    "point": float(point_macro[method_ix[candidate], qi] - point_macro[method_ix[baseline], qi]),
                    "ci95": [float(np.quantile(delta, .025)), float(np.quantile(delta, .975))],
                    "bootstrap_reps": int(len(delta)), "unit": "BJT target-date block"}
            delta_csi = bs_csi[:, method_ix[candidate]] - bs_csi[:, method_ix[baseline]]
            delta_csi = delta_csi[np.isfinite(delta_csi)]
            out[candidate][baseline]["local_3x3_pooled_CSI_delta"] = {
                "point": float(point_csi[method_ix[candidate]] - point_csi[method_ix[baseline]]),
                "ci95": [float(np.quantile(delta_csi, .025)), float(np.quantile(delta_csi, .975))],
                "bootstrap_reps": int(len(delta_csi)), "unit": "BJT target-date block"}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--r-pack-rows", type=Path, required=True)
    p.add_argument("--r-validation-predictions", type=Path, required=True)
    p.add_argument("--bootstrap-reps", type=int, default=REPS)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)

    ref_keys, reference_source = reference_key_map(a.r_pack_rows, a.r_validation_predictions)
    loaded = {}
    for name, subdir in METHODS.items():
        folder = a.base / subdir
        profile = json.loads((folder / "profile.json").read_text(encoding="utf-8"))
        if profile.get("split") != "val_only" or profile.get("test_payloads_read") != 0:
            raise RuntimeError(f"invalid split/test receipt in {folder}")
        if profile.get("state") != "COMPLETE_ALL_VALIDATION_S_TO_R_COT_AND_AGRI3X3_BANK":
            raise RuntimeError(f"incomplete all-validation output in {folder}")
        z = np.load(folder / "forecast_cot_val_bank.npz", allow_pickle=False)
        if any(k not in z.files for k in ("forecast_log1p_cot", "target_agri_r_log1p_cot",
                    "persistence_log1p_cot", "target_time_utc", "dataset_indices", "sequence_ids")):
            raise RuntimeError(f"required model arrays absent in {folder}")
        loaded[name] = (profile, folder / "forecast_cot_val_bank.npz", z)

    baseline = loaded["original_simvp"][2]
    base_keys = baseline["target_time_utc"].astype(str)
    base_ids = baseline["sequence_ids"].astype(str)
    indices = baseline["dataset_indices"].astype(np.int64)
    shape = (len(base_ids), 2, 16, 16, 16)
    if shape != (3140, 2, 16, 16, 16):
        raise RuntimeError(f"unexpected validation forecast shape={shape}")
    if not np.array_equal(indices, np.arange(3140, dtype=np.int64)):
        raise RuntimeError("baseline bank validation indices are not canonical 0..3139")
    for name, (_, _, z) in loaded.items():
        if z["forecast_log1p_cot"].shape != shape or not np.isfinite(z["forecast_log1p_cot"]).all():
            raise RuntimeError(f"bad forecast output shape/finite status: {name}")
        for k in ("target_time_utc", "dataset_indices", "sequence_ids"):
            if not np.array_equal(z[k].astype(str) if k != "dataset_indices" else z[k],
                                  baseline[k].astype(str) if k != "dataset_indices" else baseline[k]):
                raise RuntimeError(f"forecast key mismatch: {name}/{k}")
        if not np.array_equal(z["target_agri_r_log1p_cot"], baseline["target_agri_r_log1p_cot"], equal_nan=True):
            raise RuntimeError(f"proxy-response target bank mismatch: {name}")
        if not np.array_equal(z["persistence_log1p_cot"], baseline["persistence_log1p_cot"], equal_nan=True):
            raise RuntimeError(f"persistence bank mismatch: {name}")

    refs, mask, support_pixels = reindex_reference(base_keys, ref_keys)
    if support_pixels == 0 or not np.isfinite(refs[mask]).all():
        raise RuntimeError("keyed CPP support is empty or nonfinite")
    stored_ref_checks = {}
    for name, (_, _, z) in loaded.items():
        # The historical baseline CPP arrays are known to be misbound. New candidate banks
        # must match the independent keyed reconstruction exactly; all scoring uses refs/mask.
        stored_matches = (np.array_equal(z["cpp_valid_mask"].astype(bool), mask) and
                          np.array_equal(z["cpp_reference_cot"], refs, equal_nan=True))
        stored_ref_checks[name] = stored_matches
        if name != "original_simvp" and not stored_matches:
            raise RuntimeError(f"candidate CPP arrays differ from independent keyed reconstruction: {name}")

    physical = {name: np.expm1(z["forecast_log1p_cot"].astype(np.float64))
                for name, (_, _, z) in loaded.items()}
    proxy = {name: np.expm1(z["target_agri_r_log1p_cot"].astype(np.float64))
             for name, (_, _, z) in loaded.items()}
    dates = date_keys(base_keys)
    all_dates = sorted(set(dates.ravel().tolist()))
    supported_dates = sorted({str((datetime.strptime(str(base_keys[i, li]), "%Y-%m-%d %H:%M:%S") +
                                   timedelta(hours=8)).date())
                              for i in range(shape[0]) for li in range(16) if mask[i, :, li].any()})

    rows = []
    aggregate = {name: {scale: {"sse": 0., "n": 0, "abs": 0., "bias": 0., "tp": 0, "fp": 0, "fn": 0}
                        for scale in SCALES} for name in METHODS}
    proxy_aggregate = {name: {"sse": 0., "n": 0, "abs": 0., "bias": 0.} for name in METHODS}
    lead_macro = {name: {scale: {} for scale in SCALES} for name in METHODS}
    for name in METHODS:
        for si, station in enumerate(STATIONS):
            for li, lead in enumerate(LEADS):
                tr = refs[:, si, li]
                for scale, (rs, cs) in SCALES.items():
                    m = mask[:, si, li, rs, cs]
                    s = score(physical[name][:, si, li, rs, cs], tr[:, rs, cs], m)
                    rows.append({"method": name, "station": station, "lead_min": lead,
                                 "scale": scale, **s})
                    if s["rmse"] is not None:
                        lead_macro[name][scale][f"{station}_{lead}"] = s["rmse"]
                    agg = aggregate[name][scale]
                    pred_slice = physical[name][:, si, li, rs, cs]
                    truth_slice = tr[:, rs, cs]
                    e = pred_slice[m].astype(np.float64) - truth_slice[m].astype(np.float64)
                    agg["sse"] += float(np.dot(e, e)); agg["n"] += int(e.size)
                    agg["abs"] += float(np.abs(e).sum()); agg["bias"] += float(e.sum())
                    agg["tp"] += s["tp"]; agg["fp"] += s["fp"]; agg["fn"] += s["fn"]
                pe = physical[name][:, si, li] - proxy[name][:, si, li]
                proxy_aggregate[name]["sse"] += float(np.square(pe, dtype=np.float64).sum())
                proxy_aggregate[name]["n"] += int(pe.size)
                proxy_aggregate[name]["abs"] += float(np.abs(pe).sum(dtype=np.float64))
                proxy_aggregate[name]["bias"] += float(pe.sum(dtype=np.float64))

    summary = {"state": "PASS_FP64_COMMON_SUPPORT_COT_SCORING_KEYED_CPP_V2",
               "scope": "validation-only; CPP retrieval reference, not independent physical truth; R(observed future AGRI) is a proxy response; no test payloads read",
               "windows": 3140, "validation_bjt_dates": len(all_dates),
               "cpp_supported_bjt_dates": len(supported_dates), "cpp_supported_bjt_date_list": supported_dates,
               "cpp_reference_pixels": support_pixels, "test_payloads_read": 0,
               "cpp_reference_source": reference_source,
               "cpp_storage_check": stored_ref_checks,
               "aggregate_cpp_metrics": {}, "full_cohort_proxy_response": {}, "banks": {},
               "paired_bootstrap": None}
    for name, (profile, bank_path, _) in loaded.items():
        summary["banks"][name] = {"bank_sha256": sha(bank_path),
                                  "simvp_checkpoint_sha256": profile.get("simvp_checkpoint_sha256"),
                                  "r_checkpoint_sha256": profile.get("r_checkpoint_sha256"),
                                  "windows": profile.get("windows_completed"),
                                  "test_payloads_read": profile.get("test_payloads_read")}
        summary["aggregate_cpp_metrics"][name] = {}
        for scale, v in aggregate[name].items():
            n = v["n"]
            summary["aggregate_cpp_metrics"][name][scale] = {
                "n_pixels": n, "rmse_pooled": float(np.sqrt(v["sse"] / n)) if n else None,
                "mae": v["abs"] / n if n else None, "bias": v["bias"] / n if n else None,
                "tp": v["tp"], "fp": v["fp"], "fn": v["fn"],
                "pod": v["tp"] / (v["tp"] + v["fn"]) if v["tp"] + v["fn"] else None,
                "far": v["fp"] / (v["tp"] + v["fp"]) if v["tp"] + v["fp"] else None,
                "csi": v["tp"] / (v["tp"] + v["fp"] + v["fn"]) if v["tp"] + v["fp"] + v["fn"] else None,
                "rmse_station_lead_macro": float(np.mean(list(lead_macro[name][scale].values())))
                if lead_macro[name][scale] else None}
        pv = proxy_aggregate[name]; n = pv["n"]
        summary["full_cohort_proxy_response"][name] = {
            "n_values": n, "rmse_vs_R_observed_future_AGRI": float(np.sqrt(pv["sse"] / n)),
            "mae_vs_R_observed_future_AGRI": pv["abs"] / n, "bias_vs_R_observed_future_AGRI": pv["bias"] / n}

    if a.bootstrap_reps > 0:
        rng = np.random.default_rng(SEED)
        summary["paired_bootstrap"] = {
            "ci_by_candidate_vs_original_and_mse": bootstrap_pair_vectorized(
                physical, dates, refs, mask, rng, a.bootstrap_reps),
            "reps": a.bootstrap_reps, "seed": SEED, "unit": "BJT target-date block",
            "note": "RMSE CI uses equal station x lead macro-averages within each date-block resample; CSI is pooled over common 3x3 CPP support."}

    a.output.mkdir(parents=True)
    with (a.output / "cot_metrics_by_station_lead_scale.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary["source_hashes"] = {"scorer_sha256": sha(Path(__file__))}
    (a.output / "summary_fp64.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
