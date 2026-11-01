#!/usr/bin/env python3
"""Validation-only counterfactual sensitivity of trained future-COT GHI heads.

Keeps each trained AC/AC50 head, AGRI, geometry, truth, and sample key fixed.
It compares its native future-COT input with (i) future COT zeroed to the
historical-only value and (ii) future COT maps permuted across validation
instances matched by target month, lead, and broad solar-angle regime.
This measures model reliance/alignment, not a deployable forecast score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
HIGH_KT = HERE.parents[1] / "experiments" / "high_kt_loss_20261016"
sys.path.insert(0, str(HIGH_KT))
from train_loss_pair import IntervalHead  # noqa: E402
from train_future_ghi_heads_cot_shrink import load_station  # noqa: E402


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def target_bjt_day(key: dict) -> str:
    t = datetime.strptime(key["target_time_utc"], "%Y-%m-%d %H:%M:%S")
    return (t + timedelta(hours=8)).strftime("%Y-%m-%d")


def lead_group(leads: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(leads), dtype=bool),
        "15-60": leads <= 60,
        "75-120": (leads > 60) & (leads <= 120),
        "135-240": leads > 120,
    }


def shuffled_future(cot: np.ndarray, history_mask: np.ndarray, keys: list[dict],
                    cosz: np.ndarray, seed: int) -> tuple[np.ndarray, dict]:
    """Permute whole future 3x3 COT context fields within comparable regimes."""
    out = np.asarray(cot, dtype=np.float32).copy()
    hm = np.asarray(history_mask, dtype=np.float32).reshape(len(keys), 3)
    leads = np.asarray([int(k["lead_minutes"]) for k in keys], dtype=np.int16)
    month = np.asarray([target_bjt_day(k)[:7] for k in keys])
    solar = np.where(cosz <= 0.0, "night", np.where(cosz < 0.3, "low_sun", "sun"))
    strata: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for i in range(len(keys)):
        strata[(month[i], int(leads[i]), str(solar[i]))].append(i)
    rng = np.random.default_rng(seed)
    moved = 0
    n_singleton = 0
    for ids_list in strata.values():
        ids = np.asarray(ids_list, dtype=np.int64)
        if len(ids) < 2:
            n_singleton += len(ids)
            continue
        perm = rng.permutation(ids)
        moved += int(np.count_nonzero(perm != ids))
        future = 1.0 - hm[ids, :, None, None, None]
        donor = out[perm]
        out[ids] = out[ids] * (1.0 - future) + donor * future
    info = {"strata": len(strata), "singleton_rows": n_singleton,
            "rows_with_different_donor": moved}
    return out, info


def model_cot(cot: np.ndarray, mask: np.ndarray, arm: str, scenario: str) -> np.ndarray:
    c = np.asarray(cot, dtype=np.float32)
    raw_mask = np.asarray(mask, dtype=np.float32)
    if c.ndim != 5 or c.shape[1] != 3:
        raise ValueError(f"expected COT [N,3,1,H,W], got {c.shape}")
    # load_station returns the broadcast-ready [N,3,1,1,1] mask. Unit
    # witnesses may also pass the source [N,3] mask. Do not append singleton
    # axes twice: NumPy would align the extra axes from the left and create a
    # massive accidental [N,N,...] broadcast.
    if raw_mask.shape == (c.shape[0], 3):
        m = raw_mask[:, :, None, None, None]
    elif raw_mask.shape == (c.shape[0], 3, 1, 1, 1):
        m = raw_mask
    else:
        raise ValueError(f"expected history mask [N,3] or [N,3,1,1,1], got {raw_mask.shape}")
    if arm == "AC":
        scale = 1.0
    elif arm == "AC50":
        scale = 0.5
    else:
        raise ValueError(arm)
    if scenario == "native":
        out = c * m + scale * c * (1.0 - m)
        if out.shape != c.shape:
            raise AssertionError(f"COT broadcast changed shape: {c.shape} -> {out.shape}")
        return out
    if scenario == "future_zero":
        out = c * m
        if out.shape != c.shape:
            raise AssertionError(f"COT broadcast changed shape: {c.shape} -> {out.shape}")
        return out
    if scenario == "future_permuted":
        raise ValueError("pass the permuted COT tensor to model_cot for this scenario")
    raise ValueError(scenario)


def predict(model, data, cot: torch.Tensor, batch: int) -> np.ndarray:
    model.eval()
    out = []
    with torch.inference_mode():
        for lo in range(0, len(data["ghi"]), batch):
            hi = min(lo + batch, len(data["ghi"]))
            kt = model(data["agri"][lo:hi], data["geometry"][lo:hi], cot[lo:hi])
            out.append((kt * data["clear"][lo:hi]).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def day_sums(err: np.ndarray, days: np.ndarray, mask: np.ndarray,
             day_index: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray([day_index[d] for d in days], dtype=np.int32)
    use = np.asarray(mask, dtype=bool)
    sse = np.bincount(ids[use], weights=np.square(err[use]), minlength=len(day_index))
    n = np.bincount(ids[use], minlength=len(day_index)).astype(np.float64)
    return sse, n


def aggregate_delta(records, arm: str, scenario: str, group: str,
                    day_weights: np.ndarray | None = None) -> float:
    station_delta = []
    for station in ("sili", "zhujia"):
        seed_delta = []
        for seed in (42, 43, 44):
            f = records[(arm, station, seed, "native", group)]
            c = records[(arm, station, seed, scenario, group)]
            if day_weights is None:
                rf = np.sqrt(f["sse"].sum() / f["n"].sum())
                rc = np.sqrt(c["sse"].sum() / c["n"].sum())
            else:
                rf = np.sqrt(day_weights @ f["sse"] / (day_weights @ f["n"]))
                rc = np.sqrt(day_weights @ c["sse"] / (day_weights @ c["n"]))
            seed_delta.append(float(rc - rf))
        station_delta.append(float(np.mean(seed_delta)))
    return float(np.mean(station_delta))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--run-root", type=Path, required=True,
                    help="completed AC/AC50 head runs, e.g. future_ghi_cot_shrink_v1")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=20261019)
    a = ap.parse_args()
    if a.output.exists() or a.output.with_suffix(".csv").exists():
        raise FileExistsError("refusing to overwrite counterfactual audit output")
    if not torch.cuda.is_available():
        raise RuntimeError("the allocated PBS GPU is required")
    pack_manifest = json.loads((a.pack / "pack_manifest.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((a.run_root / "run_manifest.json").read_text(encoding="utf-8"))
    if pack_manifest.get("state") != "COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK":
        raise ValueError("causal GHI pack incomplete")
    if any(pack_manifest.get(k) is not False for k in
           ("test_payload_opened", "buffer_payload_opened", "future_observed_agri_used_as_input", "cpp_used_as_input")):
        raise ValueError("causal data exclusion manifest gate failed")
    if run_manifest.get("state") != "COMPLETE_FUTURE_GHI_COT_SHRINKAGE_SCREENING_PILOT":
        raise ValueError("AC/AC50 screening pilot is incomplete")
    if run_manifest.get("test_payload_opened") is not False:
        raise ValueError("pilot test exclusion gate failed")

    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    records = {}
    raw_rows = []
    all_days: set[str] = set()
    permutation_summary = {}
    for station in ("sili", "zhujia"):
        data = load_station(a.pack, "val", station, device, pack_manifest)
        keys = data["keys"]
        days = np.asarray([target_bjt_day(k) for k in keys])
        all_days.update(days.tolist())
        leads = np.asarray([int(k["lead_minutes"]) for k in keys], dtype=np.int16)
        cosz = data["geometry"][:, 2, 0].mean(dim=(1, 2)).detach().cpu().numpy()
        cot = data["cot"].detach().cpu().numpy()
        mask = data["history_cot_mask"].detach().cpu().numpy()
        shuffled, shuffle_info = shuffled_future(cot, mask, keys, cosz, 20261019 + (0 if station == "sili" else 100))
        permutation_summary[station] = shuffle_info
        group_masks = lead_group(leads)
        for seed in (42, 43, 44):
            for arm in ("AC", "AC50"):
                ckpt = a.run_root / station / f"seed{seed}" / arm / "best.pt"
                if not ckpt.is_file():
                    raise FileNotFoundError(ckpt)
                state = torch.load(ckpt, map_location=device, weights_only=False)
                if (state.get("station"), int(state.get("seed", -1)), state.get("arm")) != (station, seed, arm):
                    raise ValueError(f"checkpoint identity mismatch: {ckpt}")
                model = IntervalHead().to(device)
                model.load_state_dict(state["model"], strict=True)
                c_native = torch.from_numpy(model_cot(cot, mask, arm, "native")).to(device)
                c_zero = torch.from_numpy(model_cot(cot, mask, arm, "future_zero")).to(device)
                # Shuffling changes only forecast-derived frames; training-time scaling is preserved.
                c_perm_arr = model_cot(shuffled, mask, arm, "native")
                c_perm = torch.from_numpy(c_perm_arr).to(device)
                pred = {
                    "native": predict(model, data, c_native, a.batch_size),
                    "future_zero": predict(model, data, c_zero, a.batch_size),
                    "future_permuted": predict(model, data, c_perm, a.batch_size),
                }
                truth = data["ghi"].detach().cpu().numpy().astype(np.float64)
                for scenario, yhat in pred.items():
                    err = yhat - truth
                    for gname, gmask in group_masks.items():
                        records[(arm, station, seed, scenario, gname)] = {"sse_dates": days,
                                                                          "err": err,
                                                                          "mask": gmask,
                                                                          "rmse": float(np.sqrt(np.mean(err[gmask] ** 2)))}
                        raw_rows.append({"arm": arm, "station": station, "seed": seed,
                                         "scenario": scenario, "lead_group_minutes": gname,
                                         "n": int(gmask.sum()), "rmse_wm2": records[(arm, station, seed, scenario, gname)]["rmse"],
                                         "mean_abs_prediction_change_wm2": float(np.mean(np.abs(yhat - pred["native"])))})
                del model, state, c_native, c_zero, c_perm
                torch.cuda.empty_cache()

    # The same sorted BJT target-date resampling is used for every contrast/group.
    days_sorted = sorted(all_days)
    day_index = {d: i for i, d in enumerate(days_sorted)}
    date_records = {}
    for key, item in records.items():
        arm, station, seed, scenario, gname = key
        sse, n = day_sums(item["err"], item["sse_dates"], item["mask"], day_index)
        date_records[key] = {"sse": sse, "n": n}

    point_ci = {}
    rng = np.random.default_rng(a.bootstrap_seed)
    resamples = rng.integers(0, len(days_sorted), size=(a.bootstrap, len(days_sorted)))
    for arm in ("AC", "AC50"):
        for scenario in ("future_zero", "future_permuted"):
            for gname in ("all", "15-60", "75-120", "135-240"):
                point = aggregate_delta(date_records, arm, scenario, gname)
                draws = np.empty(a.bootstrap, dtype=np.float64)
                for b, sampled in enumerate(resamples):
                    weights = np.bincount(sampled, minlength=len(days_sorted)).astype(np.float64)
                    draws[b] = aggregate_delta(date_records, arm, scenario, gname, weights)
                point_ci[f"{arm}:{scenario}:{gname}"] = {
                    "delta_rmse_replaced_minus_native_wm2": point,
                    "date_block_bootstrap_95pct_ci_wm2": np.quantile(draws, [0.025, 0.975]).tolist(),
                }
    out_csv = a.output.with_suffix(".csv")
    import csv
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0]))
        w.writeheader(); w.writerows(raw_rows)
    report = {
        "state": "COMPLETE_VALIDATION_FUTURE_COT_COUNTERFACTUAL_SENSITIVITY_AUDIT",
        "purpose": "measure whether fixed trained AC/AC50 heads rely on aligned forecast-derived COT; not a deployable score or causal proof",
        "input_source": "frozen SimVP forecast AGRI passed through frozen R; validation pack only",
        "heads": "frozen AC and AC50 heads from the completed three-seed screening pilot",
        "scenarios": {"native": "head-native future predicted COT", "future_zero": "future COT zeroed, history kept", "future_permuted": "future 3x3 fields permuted across validation examples within station x target month x lead x broad solar-angle regime; history kept"},
        "permutation_summary": permutation_summary,
        "target_bjt_date_blocks": len(days_sorted), "bootstrap_replicates": a.bootstrap,
        "paired_date_block_95pct": point_ci,
        "test_payload_opened": False, "buffer_payload_opened": False,
        "future_observed_agri_read": False, "cpp_read": False,
        "pack_manifest_sha256": sha(a.pack / "pack_manifest.json"),
        "run_manifest_sha256": sha(a.run_root / "run_manifest.json"),
        "csv": out_csv.name, "rows": len(raw_rows),
        "interpretation_rule": "positive replacement-minus-native RMSE means destroying/zeroing future-COT alignment hurt this fixed fitted head; near-zero means it has little measurable reliance; neither establishes that COT improves GHI versus a separately trained baseline",
    }
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"state": report["state"], "target_bjt_date_blocks": len(days_sorted),
                      "contrasts": point_ci, "test_payload_opened": False}, indent=2))


if __name__ == "__main__":
    main()
