#!/usr/bin/env python3
"""Validation-only frozen FAC-head COT source substitution for attempt12.

Arms evaluated by the same saved FAC checkpoint:
  forecast_cot: SimVP-predicted AGRI -> frozen corrected R (deployable chain)
  history_persistence: observed history AGRI at init_time -> frozen R
  offline_future_agri: observed future AGRI at target_time -> frozen R (oracle only)

No model is trained. Only validation rows and validation R outputs are scored.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
CENTERS = {"Niushou": (8, 6), "Wangzhai": (9, 8), "Mizhuang": (7, 10)}
HORIZONS = {
    "all": (15, 240),
    "1h": (15, 60),
    "2h": (75, 120),
    "3h": (135, 180),
    "4h": (195, 240),
}
ARMS = ("forecast_cot", "history_persistence", "offline_future_agri")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_model_class(path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("attempt12_train_paired_forecast", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import model definition from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HubeiLocalGHIHead


def metric(y: np.ndarray, p: np.ndarray):
    e = p.astype(np.float64) - y.astype(np.float64)
    return {
        "n": int(e.size),
        "rmse_wm2": float(np.sqrt(np.mean(e * e, dtype=np.float64))),
        "mae_wm2": float(np.mean(np.abs(e), dtype=np.float64)),
        "bias_wm2": float(np.mean(e, dtype=np.float64)),
    }


def cluster_ci(rows_by_station, alt: str, lead_range: tuple[int, int], draws=5000, seed=20261021):
    """Date-cluster CI for equal-station mean RMSE difference (alt - forecast)."""
    lo, hi = lead_range
    date_sets = [{r["date"] for r in rows_by_station[st] if lo <= r["lead_minutes"] <= hi}
                 for st in STATIONS]
    dates = sorted(set.intersection(*date_sets))
    if len(dates) < 2:
        return {"common_target_date_clusters": len(dates), "ci95_delta_rmse_wm2": [None, None]}
    date_index = {d: i for i, d in enumerate(dates)}
    station_stats = {}
    for st in STATIONS:
        items = [r for r in rows_by_station[st] if lo <= r["lead_minutes"] <= hi and r["date"] in date_index]
        sums = {arm: np.zeros(len(dates), dtype=np.float64) for arm in ("forecast_cot", alt)}
        counts = np.zeros(len(dates), dtype=np.int64)
        for r in items:
            j = date_index[r["date"]]
            counts[j] += 1
            for arm in sums:
                err = float(r[arm]) - float(r["truth"])
                sums[arm][j] += err * err
        station_stats[st] = (sums, counts)
    observed_station_delta = []
    for st in STATIONS:
        sums, counts = station_stats[st]
        n = int(counts.sum())
        rmse = {a: float(np.sqrt(sums[a].sum(dtype=np.float64) / n)) for a in sums}
        observed_station_delta.append(rmse[alt] - rmse["forecast_cot"])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(dates), size=(draws, len(dates)))
    delta = np.empty(draws, dtype=np.float64)
    for b, draw in enumerate(samples):
        station_delta = []
        for st in STATIONS:
            sums, counts = station_stats[st]
            n = int(counts[draw].sum())
            if n == 0:
                continue
            rmse = {a: float(np.sqrt(sums[a][draw].sum(dtype=np.float64) / n)) for a in sums}
            station_delta.append(rmse[alt] - rmse["forecast_cot"])
        delta[b] = float(np.mean(station_delta))
    return {
        "common_target_date_clusters": len(dates),
        "observed_equal_station_delta_on_ci_support_wm2": float(np.mean(observed_station_delta)),
        "ci95_delta_rmse_wm2": [float(x) for x in np.quantile(delta, [0.025, 0.975])],
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "delta_definition": "equal-station mean RMSE(alternative COT - forecast COT); negative favors alternative",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--code-dir", type=Path, required=True)
    p.add_argument("--draws", type=int, default=5000)
    a = p.parse_args()
    root, out = a.root, a.output
    pack = root / "GHI_pack_attempt12"
    ghi_runs = root / "GHI_runs_attempt12"
    r_val = root / "R_runs_attempt12" / "full"
    rpack = root / "R_pack_correctutc_attempt2"
    out.mkdir(parents=True, exist_ok=False)

    pack_meta = json.loads((pack / "pack_report.json").read_text())
    if pack_meta.get("test_payloads_read") != 0 or pack_meta.get("buffer_payloads_read") != 0:
        raise ValueError("Attempt12 pack does not certify test/buffer closed")
    rows = read_csv(pack / "rows.csv")
    if {r["split"] for r in rows} - {"train", "val"}:
        raise ValueError("Unexpected split in GHI pack")
    val_by_station = {st: [(i, r) for i, r in enumerate(rows) if r["split"] == "val" and r["station"] == st]
                      for st in STATIONS}
    if any(not val_by_station[st] for st in STATIONS):
        raise ValueError("Missing station validation rows")

    # The full R validation prediction array is joined only through its saved BJT keys.
    r_keys = read_csv(r_val / "val_keys.csv")
    if any(r.get("split") != "val" for r in r_keys):
        raise ValueError("R validation key file contains a non-validation row")
    times = [r["time_bjt"] for r in r_keys]
    if len(times) != len(set(times)):
        raise ValueError("Duplicate R validation time keys")
    time_to_r = {t: i for i, t in enumerate(times)}
    r_pred = np.load(r_val / "val_prediction_cot.npy", mmap_mode="r", allow_pickle=False)
    r_mask = np.load(r_val / "val_valid_mask.npy", mmap_mode="r", allow_pickle=False)
    if r_pred.shape != r_mask.shape or r_pred.shape[0] != len(r_keys) or r_pred.shape[1:] != (1, 16, 16):
        raise ValueError(f"Unexpected R validation shapes: {r_pred.shape}, {r_mask.shape}, keys={len(r_keys)}")

    cot_all = np.load(pack / "cot.npy", mmap_mode="r", allow_pickle=False)
    agri_all = np.load(pack / "agri_raw.npy", mmap_mode="r", allow_pickle=False)
    geo_all = np.load(pack / "geometry.npy", mmap_mode="r", allow_pickle=False)
    y_all = np.load(pack / "ghi_wm2.npy", mmap_mode="r", allow_pickle=False)
    clear_all = np.load(pack / "clear_sky_wm2.npy", mmap_mode="r", allow_pickle=False)
    if agri_all.shape != (len(rows), 13, 3, 3) or geo_all.shape != (len(rows), 2, 3, 3):
        raise ValueError("Unexpected GHI input shapes")
    if cot_all.shape != (len(rows), 1, 3, 3):
        raise ValueError("Unexpected forecast-COT shape")

    norm_path = ghi_runs / "paired_seed42_attempt1" / "normalization_contract.json"
    norm_contract = json.loads(norm_path.read_text())
    if norm_contract.get("fit_split") != "train":
        raise ValueError("Normalization was not fitted on train only")
    norm = norm_contract["stations"]
    trainer = a.code_dir / "train_paired_forecast.py"
    Model = load_model_class(trainer)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)

    predictions_by_station = {}
    key_coverage = {}
    model_hashes = {}
    common_rows = {}
    for st in STATIONS:
        station_pairs = val_by_station[st]
        station_rows = [r for _, r in station_pairs]
        run_dir = ghi_runs / "paired_seed42_attempt1" / st
        bundle = np.load(run_dir / "FAC_validation_predictions_with_keys.npz", allow_pickle=False)
        if len(station_rows) != bundle["pred_ghi_wm2"].shape[0]:
            raise ValueError(f"Validation row count differs for {st}")
        for j, (_, row) in enumerate(station_pairs):
            for key, field in (("station", "station"), ("init_time_bjt", "init_time_bjt"),
                               ("target_time_bjt", "target_time_bjt"), ("lead_minutes", "lead_minutes")):
                if str(bundle[key][j]) != str(row[field]):
                    raise ValueError(f"Saved FAC prediction key mismatch at {st}/{j}/{key}")

        joined = []
        missing_init, missing_target = 0, 0
        y, pred_rows = [], []
        for (global_i, row), j in zip(station_pairs, range(len(station_pairs))):
            init_i = time_to_r.get(row["init_time_bjt"])
            target_i = time_to_r.get(row["target_time_bjt"])
            if init_i is None:
                missing_init += 1
            if target_i is None:
                missing_target += 1
            if init_i is None or target_i is None:
                continue
            rr, cc = CENTERS[st]
            sl = (slice(rr - 1, rr + 2), slice(cc - 1, cc + 2))
            hist = np.asarray(r_pred[init_i, 0, sl[0], sl[1]], dtype=np.float32).copy()
            hist_mask = np.asarray(r_mask[init_i, 0, sl[0], sl[1]], dtype=bool)
            hist[~hist_mask] = 0.0
            future = np.asarray(r_pred[target_i, 0, sl[0], sl[1]], dtype=np.float32).copy()
            future_mask = np.asarray(r_mask[target_i, 0, sl[0], sl[1]], dtype=bool)
            future[~future_mask] = 0.0
            forecast = np.asarray(cot_all[global_i], dtype=np.float32)
            joined.append({"global_i": global_i, "bundle_i": j, "row": row,
                           "history": hist[None], "future": future[None], "forecast": forecast,
                           "history_mask": hist_mask, "future_mask": future_mask})
        key_coverage[st] = {
            "validation_rows": len(station_pairs),
            "common_init_and_target_rows": len(joined),
            "missing_init_time_rows": missing_init,
            "missing_target_time_rows": missing_target,
            "coverage_fraction": len(joined) / len(station_pairs),
        }
        if not joined:
            raise ValueError(f"No common time-key support for {st}")
        common_rows[st] = joined

        ids = np.asarray([x["global_i"] for x in joined], dtype=np.int64)
        agri = np.asarray(agri_all[ids], dtype=np.float32)
        geometry = np.asarray(geo_all[ids], dtype=np.float32)
        target = np.asarray(y_all[ids], dtype=np.float64)
        clear = np.asarray(clear_all[ids], dtype=np.float64)
        if not np.isfinite(agri).all() or not np.isfinite(geometry).all() or not np.isfinite(target).all():
            raise ValueError(f"Nonfinite validation data for {st}")
        z = norm[st]
        agri = (agri - np.asarray(z["agri_mean"], dtype=np.float32)[None, :, None, None]) / \
               np.asarray(z["agri_std"], dtype=np.float32)[None, :, None, None]
        x = np.concatenate((agri, geometry), axis=1).astype(np.float32)
        cots = {
            "forecast_cot": np.stack([x["forecast"] for x in joined]).astype(np.float32),
            "history_persistence": np.stack([x["history"] for x in joined]).astype(np.float32),
            "offline_future_agri": np.stack([x["future"] for x in joined]).astype(np.float32),
        }
        ckpt_path = run_dir / "FAC_best.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ckpt.get("normalization_contract_sha256") != norm_contract.get("sha256"):
            raise ValueError(f"Normalization hash mismatch at {st}")
        if ckpt.get("arm") != "FAC" or ckpt.get("station") != st or int(ckpt.get("seed")) != 42:
            raise ValueError(f"Unexpected checkpoint identity for {st}")
        model = Model("FAC").to(device).eval()
        model.load_state_dict(ckpt["model"], strict=True)
        model_hashes[st] = sha(ckpt_path)
        one_station_preds = {}
        for arm in ARMS:
            c = cots[arm]
            clog = (np.log1p(c) - float(z["cot_log1p_mean"])) / float(z["cot_log1p_std"])
            out_chunks = []
            with torch.inference_mode():
                for start in range(0, len(ids), 1024):
                    xt = torch.from_numpy(x[start:start + 1024])
                    ct = torch.from_numpy(clog[start:start + 1024].astype(np.float32))
                    xt, ct = xt.to(device), ct.to(device)
                    amp = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
                    with amp:
                        kt = model(xt, ct).float().cpu().numpy().astype(np.float64)
                    out_chunks.append(kt * clear[start:start + 1024])
            one_station_preds[arm] = np.concatenate(out_chunks)
        # Reproduce the saved FAC prediction on common keys before interpreting replacements.
        saved = np.asarray(bundle["pred_ghi_wm2"][[x["bundle_i"] for x in joined]], dtype=np.float64)
        reproduction = one_station_preds["forecast_cot"]
        max_abs = float(np.max(np.abs(reproduction - saved)))
        rmse_abs = float(np.sqrt(np.mean((reproduction - saved) ** 2, dtype=np.float64)))
        # CUDA training predictions used FP16 autocast. A CPU-only login-node
        # rerun is FP32 and can differ slightly even with identical weights/inputs.
        # Keep a tight engineering bound and publish both max and RMS deltas.
        parity_tolerance = 3e-1 if device.type == "cpu" else 2e-2
        if max_abs > parity_tolerance:
            raise RuntimeError(f"Frozen FAC baseline reproduction failed for {st}: max_abs={max_abs}; tolerance={parity_tolerance}")
        one_station_preds["baseline_saved"] = saved
        one_station_preds["baseline_reproduction_max_abs_wm2"] = max_abs
        one_station_preds["baseline_reproduction_rmse_abs_wm2"] = rmse_abs
        one_station_preds["truth"] = target
        one_station_preds["rows"] = joined
        predictions_by_station[st] = one_station_preds

    # Each station has its own exact key intersection; station data availability
    # can differ, so do not require artificial cross-station row identity.
    rows_for_ci = {st: [] for st in STATIONS}
    metrics = []
    prediction_rows = []
    for st in STATIONS:
        d = predictions_by_station[st]
        st_rows = d["rows"]
        lead_values = np.asarray([int(x["row"]["lead_minutes"]) for x in st_rows], dtype=np.int16)
        for name, (lo, hi) in HORIZONS.items():
            ids = np.flatnonzero((lead_values >= lo) & (lead_values <= hi))
            if not len(ids):
                continue
            for arm in ARMS:
                m = metric(d["truth"][ids], d[arm][ids])
                metrics.append({"station": st, "horizon": name, "arm": arm, **m,
                                "baseline_reproduction_max_abs_wm2": d["baseline_reproduction_max_abs_wm2"]})
        for i, entry in enumerate(st_rows):
            r = entry["row"]
            obj = {"station": st, "init_time_bjt": r["init_time_bjt"], "target_time_bjt": r["target_time_bjt"],
                   "target_date_bjt": r["target_time_bjt"][:8], "lead_minutes": int(r["lead_minutes"]),
                   "source_role": r["source_role"], "source_sequence_index": int(r["source_sequence_index"]),
                   "truth": float(d["truth"][i]),
                   "forecast_cot_station": float(entry["forecast"][0, 1, 1]),
                   "history_cot_station": float(entry["history"][0, 1, 1]),
                   "offline_future_cot_station": float(entry["future"][0, 1, 1]),
                   "history_cot_valid_station": bool(entry["history_mask"][1, 1]),
                   "offline_future_cot_valid_station": bool(entry["future_mask"][1, 1]),
                   "forecast_cot_3x3_mean": float(np.mean(entry["forecast"], dtype=np.float64)),
                   "history_cot_3x3_mean": float(np.mean(entry["history"], dtype=np.float64)),
                   "offline_future_cot_3x3_valid_mean": float(np.sum(entry["future"][0] * entry["future_mask"]) /
                       max(1, int(np.sum(entry["future_mask"])))),
                   "offline_future_cot_3x3_valid_pixels": int(np.sum(entry["future_mask"]))}
            for arm in ARMS:
                obj[arm] = float(d[arm][i])
            prediction_rows.append(obj)
            rows_for_ci[st].append({"date": r["target_time_bjt"][:8], "lead_minutes": int(r["lead_minutes"]),
                "truth": float(d["truth"][i]), **{arm: float(d[arm][i]) for arm in ARMS}})

    # Bit masks for cluster bootstrap, one for each horizon in HORIZONS order.
    ci = {}
    for h_index, (name, lead_range) in enumerate(HORIZONS.items()):
        lo, hi = lead_range
        base = []
        for s in STATIONS:
            leads = np.asarray([int(x["row"]["lead_minutes"]) for x in predictions_by_station[s]["rows"]])
            selection = (leads >= lo) & (leads <= hi)
            base.append(metric(predictions_by_station[s]["truth"][selection],
                               predictions_by_station[s]["forecast_cot"][selection])["rmse_wm2"])
        base = [x for x in base]
        for arm in ("history_persistence", "offline_future_agri"):
            alt = []
            for s in STATIONS:
                leads = np.asarray([int(x["row"]["lead_minutes"]) for x in predictions_by_station[s]["rows"]])
                selection = (leads >= lo) & (leads <= hi)
                alt.append(metric(predictions_by_station[s]["truth"][selection],
                                  predictions_by_station[s][arm][selection])["rmse_wm2"])
            delta = float(np.mean(np.asarray(alt) - np.asarray(base)))
            ci[f"{name}/{arm}"] = {
                "forecast_equal_station_rmse_wm2": float(np.mean(base)),
                "alternative_equal_station_rmse_wm2": float(np.mean(alt)),
                "delta_rmse_wm2": delta,
                "relative_rmse_reduction_percent": 100.0 * (-delta) / float(np.mean(base)),
                "ci_support_scope": "date-cluster interval restricted to dates present at all three stations; see observed_equal_station_delta_on_ci_support_wm2",
                **cluster_ci(rows_for_ci, arm, lead_range, draws=a.draws,
                             seed=20261021 + h_index * 11 + (1 if arm == "history_persistence" else 2)),
            }

    cot_diagnostic = {}
    for st in STATIONS:
        entries = common_rows[st]
        f = np.stack([x["forecast"][0] for x in entries]).astype(np.float64)
        h = np.stack([x["history"][0] for x in entries]).astype(np.float64)
        target_cot = np.stack([x["future"][0] for x in entries]).astype(np.float64)
        mask = np.stack([x["future_mask"] for x in entries]).astype(bool)
        center = mask[:, 1, 1]
        cdiag = {"target_reference": "offline corrected-R inversion of observed future AGRI; not independent COT truth",
                 "station_pixel_rows": int(center.sum()), "local_3x3_valid_pixel_values": int(mask.sum())}
        for label, source in (("forecast_cot", f), ("history_persistence", h)):
            e = source - target_cot
            ev = e[:, 1, 1][center]
            e3 = e[mask]
            cdiag[label] = {
                "station_pixel_rmse_cot": float(np.sqrt(np.mean(ev * ev, dtype=np.float64))) if ev.size else None,
                "station_pixel_mae_cot": float(np.mean(np.abs(ev), dtype=np.float64)) if ev.size else None,
                "station_pixel_bias_cot": float(np.mean(ev, dtype=np.float64)) if ev.size else None,
                "local_3x3_rmse_cot": float(np.sqrt(np.mean(e3 * e3, dtype=np.float64))) if e3.size else None,
                "local_3x3_mae_cot": float(np.mean(np.abs(e3), dtype=np.float64)) if e3.size else None,
                "local_3x3_bias_cot": float(np.mean(e3, dtype=np.float64)) if e3.size else None,
            }
        cot_diagnostic[st] = cdiag

    with (out / "metrics_by_station_horizon.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        w.writeheader(); w.writerows(metrics)
    with (out / "paired_predictions_common_validation.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prediction_rows[0].keys()))
        w.writeheader(); w.writerows(prediction_rows)
    summary = {
        "state": "VALIDATION_ONLY_FROZEN_FAC_COT_SUBSTITUTION_COMPLETE",
        "interpretation": "fixed-head counterfactual; not a retrained-arm forecast result and not an independent test",
        "sources": {
            "forecast_cot": "corrected-SimVP predicted future AGRI -> frozen corrected R; deployable forecast-chain input",
            "history_persistence": "observed history AGRI at init_time -> same frozen corrected R; persistence diagnostic",
            "offline_future_agri": "observed future AGRI at target_time -> same frozen corrected R; offline oracle input, not deployable",
        },
        "validation_only": True,
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "input_alignment": "station/init_time_bjt/target_time_bjt/lead_minutes exact match; all arms scored on the common intersection",
        "common_support": key_coverage,
        "normalization_fit_split": norm_contract["fit_split"],
        "normalization_contract_sha256": norm_contract["sha256"],
        "pack_report_sha256": sha(pack / "pack_report.json"),
        "ghi_rows_sha256": sha(pack / "rows.csv"),
        "forecast_cot_array_sha256": sha(pack / "cot.npy"),
        "R_validation_keys_sha256": sha(r_val / "val_keys.csv"),
        "R_validation_prediction_sha256": sha(r_val / "val_prediction_cot.npy"),
        "R_validation_mask_sha256": sha(r_val / "val_valid_mask.npy"),
        "trainer_sha256": sha(trainer),
        "inference_device": str(device),
        "inference_mixed_precision": "CUDA autocast FP16, matching trainer" if device.type == "cuda" else "CPU FP32 fallback; baseline parity tolerance still enforced",
        "baseline_reproduction_tolerance_wm2": 3e-1 if device.type == "cpu" else 2e-2,
        "FAC_checkpoint_sha256": model_hashes,
        "baseline_reproduction": {
            st: {"max_abs_wm2": predictions_by_station[st]["baseline_reproduction_max_abs_wm2"],
                 "rmse_abs_wm2": predictions_by_station[st]["baseline_reproduction_rmse_abs_wm2"]}
            for st in STATIONS
        },
        "metrics": metrics,
        "COT_diagnostic_scope": "station-pixel and local 3x3 values from the same corrected R; future AGRI input is observed offline and is not independent truth",
        "COT_error_against_offline_future_R": cot_diagnostic,
        "equal_station_and_date_cluster_bootstrap": ci,
        "bootstrap_draws": a.draws,
        "bootstrap_unit": "target BJT date; equal-station RMSE difference",
        "COT_validity": "R output masked to zero outside saved R val_valid_mask before extracting local 3x3 patches",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("state", "common_support", "baseline_reproduction",
                                                   "COT_error_against_offline_future_R",
                                                   "equal_station_and_date_cluster_bootstrap")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
