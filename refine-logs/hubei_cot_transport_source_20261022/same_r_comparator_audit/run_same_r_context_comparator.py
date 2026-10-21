#!/usr/bin/env python3
"""Validation-only same-frozen-R contextual score on exact common keys."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


SITES = {"Niushou": (8, 6), "Wangzhai": (9, 8), "Mizhuang": (7, 10)}
LOG_DENOM = float(np.log(101.0))
EXPECTED = {
    "r_checkpoint": "8ad4dbd1328325b778308ca4f8a6398dc476043ad41230ed094b9ae4855877c9",
    "r_norm": "11b9014bc07df2d72926ed230ed954b7eea95621869e6479c4aba7f6986bb86a",
    "r_source": "8af2d0e101b492bf2096223fe788b7f3a8a9b5ece2544db45903b95397620e05",
    "simvp_checkpoint": "3db49fce2b83601faaaad4526d43007bb503a42f7fb44613e085c6eda96eac8a",
    "simvp_norm": "40ef5fef39659d861a818e1783d183362be6a726bcdb42d91e4c2bca3cb58e94",
    "simvp_keys": "1f32d09268ac1a5446bb54c717b0cc6e663ca198c3d516d9da821cc94b28f134",
    "simvp_pred": "bb126fa0638f55894f8dbf61d2f800597719db94ce40583409abadf222827483",
    "simvp_geometry": "4dbcf2cf8318283abaa89c9323593d8951a399deeff1281d60f5134c50a6a4f5",
    "forecast_pairs": "1d1f8df9568bbca12b17495f4fd8c37515cad20776f37cf9dde3f445fa3cebe8",
    "bank_rows": "f5e77b9ea7c2922c56306187e7a278605097821adf5811e7859d09438b2beaf9",
    "history_valid": "8972d9eacd66220727d931db2d4704b564009e83e20a6e0e90ee44a9de3002b0",
    "cpp_valid": "873908c4efac930604afd57f84f77b59f3265009bcec57ec7541b05e75c97148",
    "cot_forecast": "d45438e53ad3b50192485b338e81d9c856d2f83f428963bbc1db221ea573e845",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def import_model(source: Path):
    spec = importlib.util.spec_from_file_location("g1s_frozen_r_source", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen R model from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.COTUNet


def make_metrics(rows, names):
    # Match evaluate_station_patch_cot_validation.py: pool all target dates
    # inside each station x lead cell, then macro-average the cells.
    cells = defaultdict(lambda: {n: [0.0, 0.0, 0.0, 0] for n in names})
    for r in rows:
        key = (r["station"], int(r["lead_minutes"]))
        for name in names:
            e = float(r[name]) - float(r["truth"])
            cells[key][name][0] += e * e
            cells[key][name][1] += abs(e)
            cells[key][name][2] += e
            cells[key][name][3] += 1

    cell_values = {}
    for key, values in cells.items():
        cell_values[key] = {n: (np.sqrt(v[0] / v[3]), v[0], v[3], v[1] / v[3], v[2] / v[3])
                            for n, v in values.items() if v[3]}
    overall = {}
    for name in names:
        overall[name] = {
            "equal_station_equal_lead_rmse_cot": float(np.mean([v[name][0] for v in cell_values.values()])),
            "station_center_records": int(sum(v[name][2] for v in cell_values.values())),
            "equal_station_equal_lead_mae_cot": float(np.mean([v[name][3] for v in cell_values.values()])),
            "equal_station_equal_lead_bias_cot": float(np.mean([v[name][4] for v in cell_values.values()])),
        }
    by_station = {}
    for st in SITES:
        station_cells = [v for key, v in cell_values.items() if key[0] == st]
        if station_cells:
            by_station[st] = {name: float(np.mean([v[name][0] for v in station_cells]))
                              for name in names}
    by_lead = {}
    for lead in sorted({int(r["lead_minutes"]) for r in rows}):
        lead_cells = [v for key, v in cell_values.items() if key[1] == lead]
        by_lead[str(lead)] = {name: float(np.mean([v[name][0] for v in lead_cells]))
                              for name in names}
    return overall, by_station, by_lead, cell_values


def bootstrap_delta(rows, alt: str, ref: str, draws=5000, seed=20261022):
    names = [alt, ref]
    cells = defaultdict(lambda: {n: [0.0, 0] for n in names})
    dates = sorted({r["date"] for r in rows})
    for r in rows:
        key = (r["station"], int(r["lead_minutes"]), r["date"])
        for name in names:
            e = float(r[name]) - float(r["truth"])
            cells[key][name][0] += e * e
            cells[key][name][1] += 1
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for b in range(draws):
        draw = rng.choice(dates, size=len(dates), replace=True)
        values = {name: [] for name in names}
        for st in SITES:
            for lead in range(15, 241, 15):
                sumsq = {name: 0.0 for name in names}
                n = 0
                for d in draw:
                    v = cells.get((st, lead, d))
                    if v is None:
                        continue
                    n += v[ref][1]
                    for name in names:
                        sumsq[name] += v[name][0]
                if n:
                    for name in names:
                        values[name].append(float(np.sqrt(sumsq[name] / n)))
        samples[b] = float(np.mean(values[alt]) - np.mean(values[ref]))
    point_delta = []
    for st in SITES:
        for lead in range(15, 241, 15):
            parts = [v for (s, l, _), v in cells.items() if s == st and l == lead]
            if not parts:
                continue
            point_delta.append(
                np.sqrt(sum(v[alt][0] for v in parts) / sum(v[alt][1] for v in parts)) -
                np.sqrt(sum(v[ref][0] for v in parts) / sum(v[ref][1] for v in parts))
            )
    return {
        "alternative_minus_reference_rmse_cot": float(np.mean(point_delta)),
        "ci95_date_block": np.quantile(samples, [0.025, 0.975]).tolist(),
        "target_date_clusters": len(dates),
        "draws": draws,
        "seed": seed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--r-checkpoint", type=Path, required=True)
    p.add_argument("--r-source", type=Path, required=True)
    p.add_argument("--r-norm", type=Path, required=True)
    p.add_argument("--cot-forecast", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, default=256)
    a = p.parse_args()
    base, cache, out = a.base, a.cache, a.output
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True)

    paths = {
        "r_checkpoint": a.r_checkpoint,
        "r_norm": a.r_norm,
        "r_source": a.r_source,
        "simvp_checkpoint": base.parent.parent.parent / "S_runs_attempt12/full/checkpoints/best.ckpt",
        "simvp_norm": base.parent.parent.parent / "prepared_attempt12/norms/full.npz",
        "simvp_keys": base / "keys.csv",
        "simvp_pred": base / "pred_agri_patch.npy",
        "simvp_geometry": base / "future_geometry_patch.npy",
        "forecast_pairs": cache / "forecast_pairs.csv",
        "bank_rows": cache / "bank_rows.csv",
        "history_valid": cache / "history_valid.npy",
        "cpp_valid": cache / "cpp_valid.npy",
        "cot_forecast": a.cot_forecast,
    }
    found_hashes = {k: sha(v) for k, v in paths.items()}
    for k, expected in EXPECTED.items():
        if found_hashes.get(k) != expected:
            raise ValueError(f"Input hash mismatch for {k}: {found_hashes.get(k)} != {expected}")
    # Attempt12 is an MSE-only corrected-geometry baseline, not the attempt13
    # paired MSE arm. Preserve that distinction in the result metadata.
    s_meta = json.loads((base.parent.parent.parent / "S_runs_attempt12/full/run_metadata.json").read_text())
    assert s_meta["best_checkpoint_sha256"] == EXPECTED["simvp_checkpoint"]
    assert s_meta["input_normalization_sha256"] == EXPECTED["simvp_norm"]
    assert s_meta["index_sha256"] == "a8692c0f924b4869840e9c25d88f16c1f869fd88d41452f1db8f9879f29581f8"
    assert s_meta["solar_geometry_corrected"] is True
    assert s_meta["test_payloads_read"] == 0 and s_meta["buffer_payloads_read"] == 0
    r_run = a.r_checkpoint.parent
    r_status = json.loads((r_run / "status.json").read_text())
    r_meta = json.loads((r_run / "run_metadata.json").read_text())
    if r_status.get("state") != "COMPLETE_VALIDATION_ONLY_TEST_CLOSED" or r_meta.get("test_used") is not False:
        raise ValueError("G1S frozen R does not carry the expected validation-only receipt")
    if r_meta.get("model_source_sha256") != found_hashes["r_source"]:
        raise ValueError("G1S R source hash differs from the frozen model code")
    cache_report = json.loads((cache / "cache_report.json").read_text())
    if cache_report.get("test_payloads_read") != 0 or cache_report.get("buffer_payloads_read") != 0:
        raise ValueError("G1S cache report does not certify test/buffer closed")

    r_norm = json.loads(a.r_norm.read_text())
    if r_norm.get("fit_split") != "train only" or r_norm.get("test_used") is not False:
        raise ValueError("Frozen R normalization is not train-only/test-closed")
    expected_channels = ["C01", "C02", "C03", "C04", "C05", "C06", "C09", "C10",
                         "C11", "C12", "C13", "C14", "C15", "cosSOZ", "cosRAA", "day_mask"]
    if r_norm.get("input_channels") != expected_channels:
        raise ValueError(f"Unexpected frozen-R input channel order: {r_norm.get('input_channels')}")
    mean = np.asarray(r_norm["mean"], dtype=np.float32).reshape(1, 16, 1, 1)
    std = np.asarray(r_norm["std"], dtype=np.float32).reshape(1, 16, 1, 1)

    s_keys = csv_rows(base / "keys.csv")
    if any(r["split"] != "val" for r in s_keys):
        raise ValueError("SimVP bank contains non-validation rows")
    s_map = {}
    for si, r in enumerate(s_keys):
        target_times = r["target_times_bjt"].split("|")
        if len(target_times) != 16:
            raise ValueError("SimVP key does not list 16 target frames")
        for ki, target in enumerate(target_times):
            key = (r["init_time_bjt"], target, (ki + 1) * 15)
            if key in s_map:
                raise ValueError(f"Duplicate SimVP key {key}")
            s_map[key] = (si, ki)

    pairs = [r for r in csv_rows(cache / "forecast_pairs.csv") if r["split"] == "val"]
    bank_rows = csv_rows(cache / "bank_rows.csv")
    hvalid = np.load(cache / "history_valid.npy", mmap_mode="r", allow_pickle=False)
    cvalid = np.load(cache / "cpp_valid.npy", mmap_mode="r", allow_pickle=False)
    hcot = np.load(cache / "history_cot_log.npy", mmap_mode="r", allow_pickle=False)
    ctarget = np.load(cache / "cpp_cot.npy", mmap_mode="r", allow_pickle=False)
    if hvalid.shape != cvalid.shape or hcot.shape != ctarget.shape or hvalid.shape != (len(bank_rows), 1, 16, 16):
        raise ValueError("G1S validation cache shape/row mismatch")
    if any(r["split"] not in ("train", "val") for r in bank_rows):
        raise ValueError("G1S bank includes non-train/validation rows")
    cot_z = np.load(a.cot_forecast, allow_pickle=False)
    cot_times = [str(x) for x in cot_z["init_time_bjt"].tolist()]
    if len(cot_times) != len(set(cot_times)):
        raise ValueError("Duplicate G1S COT forecast init keys")
    if cot_z["pred_log_scaled"].shape != (len(cot_times), 16, 1, 16, 16):
        raise ValueError(f"Unexpected G1S COT forecast shape: {cot_z['pred_log_scaled'].shape}")
    cot_map = {t: i for i, t in enumerate(cot_times)}
    bank_time = {r["time_bjt"]: int(r["index"]) for r in bank_rows}
    if len(bank_time) != len(bank_rows):
        raise ValueError("Duplicate G1S COT bank time keys")

    matched = []
    missing_simvp = 0
    for r in pairs:
        init_s, target_s, lead = r["init_time_bjt"], r["target_time_bjt"], int(r["lead_minutes"])
        init_i, target_i = int(r["init_index"]), int(r["target_index"])
        if bank_rows[init_i]["time_bjt"] != init_s or bank_rows[target_i]["time_bjt"] != target_s:
            raise ValueError(f"G1S pair row-index mismatch: {r}")
        init_dt = dt.datetime.strptime(init_s, "%Y%m%d%H%M")
        target_dt = dt.datetime.strptime(target_s, "%Y%m%d%H%M")
        if target_dt - init_dt != dt.timedelta(minutes=lead):
            raise ValueError(f"G1S key/lead mismatch: {r}")
        supported = [site for site, (rr, cc) in SITES.items()
                     if bool(cvalid[target_i, 0, rr, cc]) and bool(hvalid[init_i, 0, rr, cc])]
        if not supported:
            continue
        key = (init_s, target_s, lead)
        if key not in s_map:
            missing_simvp += 1
            continue
        if init_s not in cot_map:
            raise ValueError(f"G1S COT forecast missing init key {init_s}")
        si, ki = s_map[key]
        matched.append((r, init_i, target_i, si, ki, cot_map[init_s], supported))

    if len(matched) != 1824:
        raise ValueError(f"Unexpected scored common key count {len(matched)}; expected 1824")
    if len({r[0]["target_time_bjt"][:8] for r in matched}) != 16:
        raise ValueError("Matched scored support no longer spans the frozen 16 target dates")

    pred_agri = np.load(base / "pred_agri_patch.npy", mmap_mode="r", allow_pickle=False)
    geometry = np.load(base / "future_geometry_patch.npy", mmap_mode="r", allow_pickle=False)
    if pred_agri.shape != (len(s_keys), 16, 13, 16, 16) or geometry.shape != (len(s_keys), 16, 3, 16, 16):
        raise ValueError(f"Unexpected SimVP array shape: {pred_agri.shape}, {geometry.shape}")

    # Run the exact G1S frozen R. Each matched init/lead frame is inferred once.
    unique = {}
    for _, _, _, si, ki, _, _ in matched:
        unique[(si, ki)] = None
    unique_keys = sorted(unique)
    device = torch.device("cuda:0")
    torch.set_num_threads(1)
    Model = import_model(a.r_source)
    model = Model().to(device).eval()
    checkpoint = torch.load(a.r_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    r_outputs = {}
    for start in range(0, len(unique_keys), a.batch):
        batch_keys = unique_keys[start:start + a.batch]
        raw = np.empty((len(batch_keys), 16, 16, 16), dtype=np.float32)
        for j, (si, ki) in enumerate(batch_keys):
            raw[j] = np.concatenate((np.asarray(pred_agri[si, ki], dtype=np.float32),
                                     np.asarray(geometry[si, ki], dtype=np.float32)), axis=0)
        solar_valid = (raw[:, 15:16] > 0.5) & (raw[:, 13:14] > np.cos(np.deg2rad(80.0)))
        x = (raw - mean) / std
        # Match the frozen R training/inference preprocessing exactly: z-score
        # all 16 channels (including day_mask), then replace nonfinite values.
        x[~np.isfinite(x)] = 0.0
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            log_pred = model(torch.from_numpy(x).to(device)).float()
            cot_pred = torch.expm1(log_pred)
            valid_t = torch.from_numpy(solar_valid).to(device)
            cot_pred = cot_pred.masked_fill(~valid_t, 0.0)
        if not torch.isfinite(cot_pred).all() or (cot_pred < 0).any():
            raise ValueError("Frozen R produced nonfinite or negative COT")
        for j, key in enumerate(batch_keys):
            r_outputs[key] = cot_pred[j, 0].detach().cpu().numpy().astype(np.float32)
        print(f"R_INFERRED {min(start + len(batch_keys), len(unique_keys))}/{len(unique_keys)}", flush=True)

    records = []
    for r, init_i, target_i, si, ki, ci, supported in matched:
        lead = int(r["lead_minutes"])
        s_key = (si, ki)
        forecast_map = r_outputs[s_key]
        cot_model_map = np.expm1(np.asarray(cot_z["pred_log_scaled"][ci, lead // 15 - 1, 0], dtype=np.float64) * LOG_DENOM)
        persistence_map = np.expm1(np.asarray(hcot[init_i, 0], dtype=np.float64) * LOG_DENOM)
        for site in supported:
            rr, cc = SITES[site]
            records.append({
                "station": site,
                "init_time_bjt": r["init_time_bjt"],
                "target_time_bjt": r["target_time_bjt"],
                "date": r["target_time_bjt"][:8],
                "lead_minutes": lead,
                "truth": float(ctarget[target_i, 0, rr, cc]),
                "cot_forecast": float(cot_model_map[rr, cc]),
                "persistence": float(persistence_map[rr, cc]),
                "simvp_forecast_agri_to_same_r": float(forecast_map[rr, cc]),
            })
    if len(records) != 5460:
        raise ValueError(f"Unexpected station-center record count {len(records)}")
    names = ["cot_forecast", "persistence", "simvp_forecast_agri_to_same_r"]
    overall, by_station, by_lead, _ = make_metrics(records, names)
    intervals = {
        alt: bootstrap_delta(records, alt, "cot_forecast")
        for alt in ("persistence", "simvp_forecast_agri_to_same_r")
    }
    records_path = out / "paired_validation_records.csv"
    fields = ["station", "init_time_bjt", "target_time_bjt", "date", "lead_minutes", "truth",
              "cot_forecast", "persistence", "simvp_forecast_agri_to_same_r"]
    with records_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    result = {
        "state": "VALIDATION_ONLY_PARTIAL_COMMON_SUPPORT_SAME_R_CONTEXT_DIAGNOSTIC",
        "comparison_status": "contextual_not_matched_simvp_training_arm",
        "scope": "Hubei 16x16 station neighborhood; three station-center pixels; CPP retrieval reference; validation only",
        "support": {
            "all_g1s_validation_pairs": len(pairs),
            "original_g1s_scored_pairs": 3220,
            "matched_init_lead_pairs": len(matched),
            "matched_station_center_records": len(records),
            "target_dates": sorted({r["date"] for r in records}),
            "target_date_count": len({r["date"] for r in records}),
            "lead_counts": {str(lead): sum(int(r["lead_minutes"]) == lead for r in records) for lead in range(15, 241, 15)},
            "simvp_missing_on_original_scored_support": missing_simvp,
            "support_rule": "CPP target valid AND initialization-time historical COT valid; same support for all methods",
        },
        "methods": {
            "cot_forecast": "G1S learned COT forecast from historical actual-AGRI-to-frozen-R COT maps",
            "persistence": "historical COT at init time",
            "simvp_forecast_agri_to_same_r": "attempt12 MSE-selected, corrected-geometry SimVP forecast AGRI passed through the exact G1S frozen R checkpoint",
        },
        "metrics": {"overall_equal_station_equal_lead": overall, "by_station": by_station, "by_lead": by_lead},
        "paired_date_block_95ci_delta_vs_cot_forecast": intervals,
        "attempt12_simvp_model": {
            "checkpoint_sha256": EXPECTED["simvp_checkpoint"],
            "completed_epochs": int(s_meta["completed_epochs"]),
            "selection_metric": "validation AGRI loss",
            "config_sha256": s_meta["config_sha256"],
            "same_trainval_index_and_normalization_as_attempt13": True,
            "not_the_attempt13_mse_control_checkpoint": True,
            "interpretation": "contextual comparator only; different SimVP checkpoint/training schedule, not a paired attempt13 MSE arm",
        },
        "r_lineage": {
            "checkpoint_sha256": EXPECTED["r_checkpoint"],
            "normalization_sha256": EXPECTED["r_norm"],
            "model_source_sha256": EXPECTED["r_source"],
            "normalization_fit": "train only",
            "input_preprocessing": "train-fitted per-channel z-score over all 16 channels including day_mask; normalized nonfinite values set to 0",
            "output_transform": "softplus log1p(COT), converted by expm1",
        },
        "input_sha256": found_hashes,
        "records_sha256": sha(records_path),
        "test_payloads_read": 0,
        "buffer_payloads_read": 0,
        "warning": "This comparison is a validation diagnostic against CPP retrieval, not a physical truth result or independent test. The SimVP comparator is a separate attempt12 MSE-selected checkpoint and covers only the exact intersecting support.",
    }
    (out / "same_r_context_score.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"]["overall_equal_station_equal_lead"], indent=2), flush=True)
    print(json.dumps(result["paired_date_block_95ci_delta_vs_cot_forecast"], indent=2), flush=True)
    print(f"COMPLETE records={len(records)} dates={result['support']['target_date_count']} output={out}", flush=True)


if __name__ == "__main__":
    main()
