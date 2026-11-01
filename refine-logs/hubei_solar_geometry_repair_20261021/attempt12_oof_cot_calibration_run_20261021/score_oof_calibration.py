#!/usr/bin/env python3
"""Train-only OOF log1p-COT calibration, then one frozen validation score.

All inputs are attempt12 train/validation artifacts. CPP is only a satellite
retrieval reference. No test or buffer payload is loaded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

LEADS = tuple(range(15, 241, 15))
CENTERS = {"Niushou": (8, 6), "Wangzhai": (9, 8), "Mizhuang": (7, 10)}
METHODS = ("forecast_raw", "forecast_calibrated", "history_persistence")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def stamp(s: str) -> str:
    out = "".join(c for c in str(s) if c.isdigit())
    if len(out) != 12:
        raise ValueError(f"bad BJT timestamp: {s!r}")
    datetime.strptime(out, "%Y%m%d%H%M")
    return out


def fit_date_balanced(groups: dict[str, list[tuple[np.ndarray, np.ndarray]]]) -> dict[str, float | int]:
    """Fit y=a*x+b, clipping slope to [0,1], equal total weight per date."""
    dates = sorted(groups)
    if len(dates) < 3:
        raise ValueError(f"need >=3 train dates, got {len(dates)}")
    moments = []
    for date in dates:
        xs = np.concatenate([p for p, _ in groups[date]])
        ys = np.concatenate([y for _, y in groups[date]])
        if xs.size == 0 or xs.size != ys.size:
            raise ValueError(f"empty or mismatched training date {date}")
        moments.append((xs.mean(dtype=np.float64), ys.mean(dtype=np.float64),
                        np.mean(xs * xs, dtype=np.float64),
                        np.mean(xs * ys, dtype=np.float64), int(xs.size)))
    mux = float(np.mean([m[0] for m in moments], dtype=np.float64))
    muy = float(np.mean([m[1] for m in moments], dtype=np.float64))
    ex2 = float(np.mean([m[2] for m in moments], dtype=np.float64))
    exy = float(np.mean([m[3] for m in moments], dtype=np.float64))
    var = max(ex2 - mux * mux, 1e-12)
    beta_raw = (exy - mux * muy) / var
    beta = float(np.clip(beta_raw, 0.0, 1.0))
    return {"slope": beta, "unclipped_slope": float(beta_raw),
            "intercept": float(muy - beta * mux), "train_dates": len(dates),
            "train_pixels": int(sum(m[4] for m in moments))}


def date_balanced_rmse(groups: dict[str, list[tuple[np.ndarray, np.ndarray]]],
                       a: float = 1.0, b: float = 0.0) -> float:
    vals = []
    for date in sorted(groups):
        es = []
        for x, y in groups[date]:
            es.append((a * x + b - y) ** 2)
        vals.append(float(np.mean(np.concatenate(es), dtype=np.float64)))
    return float(np.sqrt(np.mean(vals, dtype=np.float64)))


def build_train_groups(root: Path):
    base = root
    rp = base / "prepared_attempt12" / "R_packs" / "full"
    fold_csv = base / "preparation_attempt1" / "metadata" / "train_date_folds.csv"
    fold_by_date = {stamp(r["date_BJT"] + "0000")[:8]: int(r["heldout_fold"])
                    for r in read_csv(fold_csv)}
    rows = read_csv(rp / "rows.csv")
    refs = np.load(rp / "target.npy", mmap_mode="r", allow_pickle=False)
    masks = np.load(rp / "mask.npy", mmap_mode="r", allow_pickle=False)
    if refs.ndim != 4 or refs.shape[1:] != (1, 16, 16) or masks.shape != refs.shape:
        raise ValueError(f"unexpected train target/mask shapes {refs.shape}/{masks.shape}")
    if len(rows) != refs.shape[0]:
        raise ValueError("train R-pack row count differs from arrays")
    target_index: dict[tuple[str, int], int] = {}
    for r in rows:
        if r["split"] != "train":
            continue
        time = stamp(r["time_bjt"])
        fold = fold_by_date.get(time[:8])
        if fold is None:
            raise ValueError(f"train target date has no fold: {time}")
        key = (time, fold)
        if key in target_index:
            raise ValueError(f"duplicate train CPP time/fold key {key}")
        target_index[key] = int(r["index"])
    groups_by_fold: dict[int, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {
        k: defaultdict(list) for k in range(5)}
    source_meta = {}
    for fold in range(5):
        cot_path = base / "COT_banks_attempt12" / f"fold_{fold}.npy"
        key_path = base / "S_banks_attempt12" / f"fold_{fold}" / "val" / "keys.csv"
        bank = np.load(cot_path, mmap_mode="r", allow_pickle=False)
        seqs = read_csv(key_path)
        if bank.shape != (len(seqs), 16, 1, 16, 16):
            raise ValueError(f"fold {fold} bank/key shape mismatch {bank.shape} / {len(seqs)}")
        for si, seq in enumerate(seqs):
            if seq["split"] != "val":
                raise ValueError("OOF bank key role is not val");
            init = stamp(seq["init_time_bjt"])
            targets = [stamp(x) for x in seq["target_times_bjt"].split("|")]
            if len(targets) != 16:
                raise ValueError(f"expected 16 leads for {init}")
            init_dt = datetime.strptime(init, "%Y%m%d%H%M")
            for li, lead in enumerate(LEADS):
                target = targets[li]
                if datetime.strptime(target, "%Y%m%d%H%M") != init_dt + timedelta(minutes=lead):
                    raise ValueError(f"lead alignment error {init} -> {target} at {lead}")
                target_fold = fold_by_date.get(target[:8])
                if target_fold != fold:
                    raise ValueError(f"target date/fold mismatch {target} bank fold {fold}")
                ri = target_index.get((target, fold))
                if ri is None:
                    continue
                mask = np.asarray(masks[ri, 0], dtype=bool)
                if not mask.any():
                    continue
                pred = np.asarray(bank[si, li, 0], dtype=np.float64)
                truth = np.asarray(refs[ri, 0], dtype=np.float64) * 100.0
                finite = mask & np.isfinite(pred) & np.isfinite(truth)
                if not finite.any():
                    continue
                pz = np.log1p(np.clip(pred[finite], 0.0, 100.0))
                yz = np.log1p(np.clip(truth[finite], 0.0, 100.0))
                groups_by_fold[fold][target[:8]].append((pz, yz))
        source_meta[str(fold)] = {"cot_sha256": sha(cot_path), "keys_sha256": sha(key_path),
                                  "sequences": len(seqs), "usable_date_blocks": len(groups_by_fold[fold])}
    hashes = {"rows_sha256": sha(rp / "rows.csv"),
        "target_sha256": sha(rp / "target.npy"), "mask_sha256": sha(rp / "mask.npy"),
        "train_date_folds_sha256": sha(fold_csv),
        "cot_bank_manifest_sha256": sha(base / "COT_banks_attempt12" / "source_manifest.json")}
    return groups_by_fold, source_meta, hashes, refs.shape


def bootstrap_delta(date_stats: dict[str, dict[str, tuple[float, int]]], left: str, right: str,
                    draws: int, seed: int) -> dict[str, object]:
    dates = sorted(date_stats)
    if len(dates) < 2:
        return {"n_dates": len(dates), "delta_rmse": None, "ci95": None}
    sse_l = np.asarray([date_stats[d][left][0] for d in dates], dtype=np.float64)
    sse_r = np.asarray([date_stats[d][right][0] for d in dates], dtype=np.float64)
    count = np.asarray([date_stats[d][left][1] for d in dates], dtype=np.int64)
    if not np.array_equal(count, [date_stats[d][right][1] for d in dates]):
        raise ValueError("paired methods have different mask support")
    point = float(np.sqrt(sse_l.sum() / count.sum()) - np.sqrt(sse_r.sum() / count.sum()))
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(dates), size=(draws, len(dates)))
    denom = count[ids].sum(axis=1)
    vals = np.sqrt(sse_l[ids].sum(axis=1) / denom) - np.sqrt(sse_r[ids].sum(axis=1) / denom)
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return {"n_dates": len(dates), "delta_rmse": point,
            "ci95": [float(lo), float(hi)], "draws": draws, "block": "BJT target date"}


def validation_score(root: Path, a: float, b: float, draws: int = 5000, seed: int = 20261021):
    base = root
    bank = np.load(base / "COT_banks_attempt12" / "validation.npy", mmap_mode="r", allow_pickle=False)
    skeys = read_csv(base / "S_banks_attempt12" / "full" / "val" / "keys.csv")
    rv = base / "R_runs_attempt12" / "full"
    rkeys = read_csv(rv / "val_keys.csv")
    rindex = {stamp(r["time_bjt"]): i for i, r in enumerate(rkeys)}
    if len(rindex) != len(rkeys) or bank.shape != (len(skeys), 16, 1, 16, 16):
        raise ValueError("validation bank/key mismatch or duplicate R key")
    hist = np.load(rv / "val_prediction_cot.npy", mmap_mode="r", allow_pickle=False)
    cpp = np.load(rv / "val_reference_cpp_cot.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(rv / "val_valid_mask.npy", mmap_mode="r", allow_pickle=False)
    if hist.shape != cpp.shape or cpp.shape != valid.shape or cpp.shape != (len(rkeys), 1, 16, 16):
        raise ValueError(f"validation R array mismatch {hist.shape}/{cpp.shape}/{valid.shape}")
    if valid.dtype not in (np.dtype(bool), np.dtype(np.uint8), np.dtype(np.float32)):
        raise ValueError(f"unexpected mask dtype {valid.dtype}")
    if valid.dtype != np.dtype(bool) and not np.isin(np.unique(valid), [0, 1]).all():
        raise ValueError("validation mask is not binary")
    centers = {k: (v[0], v[1]) for k, v in CENTERS.items()}
    scopes = [("all_patch", slice(0, 16), slice(0, 16))]
    for station, (y, x) in centers.items():
        scopes.extend([(f"{station}_pixel", slice(y, y+1), slice(x, x+1)),
                       (f"{station}_3x3", slice(y-1, y+2), slice(x-1, x+2))])
    accum = defaultdict(lambda: {"sse": defaultdict(float), "sae": defaultdict(float),
                                 "bias": defaultdict(float), "n": defaultdict(int),
                                 "dates": defaultdict(lambda: {m: [0.0, 0] for m in METHODS})})
    common_keys = set()
    nearzero = defaultdict(lambda: [0, 0])
    for si, seq in enumerate(skeys):
        init = stamp(seq["init_time_bjt"])
        targets = [stamp(x) for x in seq["target_times_bjt"].split("|")]
        if len(targets) != 16:
            raise ValueError("validation sequence has incorrect lead count")
        init_dt = datetime.strptime(init, "%Y%m%d%H%M")
        for li, lead in enumerate(LEADS):
            target = targets[li]
            if datetime.strptime(target, "%Y%m%d%H%M") != init_dt + timedelta(minutes=lead):
                raise ValueError(f"validation lead misaligned for {init}/{target}")
            ti = rindex.get(target); hi = rindex.get(init)
            if ti is None or hi is None:
                continue
            pred = np.asarray(bank[si, li, 0], dtype=np.float64)
            histp = np.asarray(hist[hi, 0], dtype=np.float64)
            ref = np.asarray(cpp[ti, 0], dtype=np.float64)
            mask = np.asarray(valid[ti, 0], dtype=bool).copy()
            mask &= np.isfinite(pred) & np.isfinite(histp) & np.isfinite(ref)
            if not mask.any():
                continue
            common_keys.add((init, target, lead))
            cal = np.clip(np.expm1(a * np.log1p(np.clip(pred, 0, 100)) + b), 0, 100)
            vals = {"forecast_raw": pred, "forecast_calibrated": cal,
                    "history_persistence": histp}
            date = target[:8]
            for name, ys, xs in scopes:
                m = mask[ys, xs]
                if not m.any():
                    continue
                cell = accum[(lead, name)]
                for method, field in vals.items():
                    err = np.asarray(field[ys, xs][m] - ref[ys, xs][m], dtype=np.float64)
                    ss = float(np.sum(err * err, dtype=np.float64))
                    sa = float(np.sum(np.abs(err), dtype=np.float64))
                    sg = float(np.sum(err, dtype=np.float64))
                    n = int(err.size)
                    cell["sse"][method] += ss; cell["sae"][method] += sa
                    cell["bias"][method] += sg; cell["n"][method] += n
                    cell["dates"][date][method][0] += ss
                    cell["dates"][date][method][1] += n
                # Fixed near-zero false-alarm readout on all patch only.
                if name == "all_patch":
                    nz = mask & (ref <= 0.5)
                    for method, field in vals.items():
                        nearzero[method][0] += int(np.sum(nz & (field > 1.0)))
                        nearzero[method][1] += int(np.sum(nz))
    if len(common_keys) != 1164:
        raise ValueError(f"expected 1,164 exact common keys, got {len(common_keys)}")
    rows = []
    for (lead, scope), cell in sorted(accum.items()):
        date_stats = cell["dates"]
        for method in METHODS:
            n = cell["n"][method]
            rows.append({"lead_minutes": lead, "scope": scope, "method": method,
                         "n_pixels": n, "rmse_cot": float(np.sqrt(cell["sse"][method]/n)),
                         "mae_cot": float(cell["sae"][method]/n),
                         "bias_cot_pred_minus_cpp": float(cell["bias"][method]/n),
                         "delta_vs_raw_ci": json.dumps(bootstrap_delta(date_stats, method, "forecast_raw", draws, seed+lead)),
                         "delta_vs_persistence_ci": json.dumps(bootstrap_delta(date_stats, method, "history_persistence", draws, seed+lead+1))})
    for name, ys, xs in scopes:
        # pooled scope over all leads
        cell = defaultdict(lambda: [0.0, 0])
        for (lead, scope), c in accum.items():
            if scope != name:
                continue
            for d, block in c["dates"].items():
                for method in METHODS:
                    cell[(method, d)][0] += block[method][0]
                    cell[(method, d)][1] += block[method][1]
        date_stats = defaultdict(dict)
        for (method, d), pair in cell.items():
            date_stats[d][method] = pair
        for method in METHODS:
            sse = sum(x[method][0] for x in date_stats.values())
            count = sum(x[method][1] for x in date_stats.values())
            rows.append({"lead_minutes": "all", "scope": name, "method": method,
                         "n_pixels": int(count), "rmse_cot": float(np.sqrt(sse/count)),
                         "mae_cot": None, "bias_cot_pred_minus_cpp": None,
                         "delta_vs_raw_ci": json.dumps(bootstrap_delta(date_stats, method, "forecast_raw", draws, seed+10)),
                         "delta_vs_persistence_ci": json.dumps(bootstrap_delta(date_stats, method, "history_persistence", draws, seed+11))})
    nz = {m: {"false_alarm_pixels": int(nearzero[m][0]),
              "reference_nearzero_pixels": int(nearzero[m][1]),
              "false_alarm_fraction": (nearzero[m][0]/nearzero[m][1]
                                       if nearzero[m][1] else None)} for m in METHODS}
    return rows, {"common_init_target_lead_keys": len(common_keys), "target_dates": len({k[1][:8] for k in common_keys}),
                  "nearzero_cpp_le_0p5_forecast_gt_1": nz,
                  "r_val_keys_sha256": sha(rv / "val_keys.csv"),
                  "validation_bank_sha256": sha(base / "COT_banks_attempt12" / "validation.npy"),
                  "validation_cpp_sha256": sha(rv / "val_reference_cpp_cot.npy"),
                  "validation_mask_sha256": sha(rv / "val_valid_mask.npy"),
                  "history_cot_sha256": sha(rv / "val_prediction_cot.npy"),
                  "simvp_val_keys_sha256": sha(base / "S_banks_attempt12" / "full" / "val" / "keys.csv")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True, help="local cache of attempt12 source files")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    groups_by_fold, fold_sources, train_hashes, train_shape = build_train_groups(args.root)
    crossfit = []
    for heldout in range(5):
        train = defaultdict(list); test = groups_by_fold[heldout]
        for fold, groups in groups_by_fold.items():
            if fold != heldout:
                for date, vals in groups.items(): train[date].extend(vals)
        coef = fit_date_balanced(train)
        raw = date_balanced_rmse(test)
        cal = date_balanced_rmse(test, coef["slope"], coef["intercept"])
        crossfit.append({"heldout_fold": heldout, "train_fit": coef,
                         "heldout_date_balanced_rmse_log1p_raw": raw,
                         "heldout_date_balanced_rmse_log1p_calibrated": cal,
                         "delta_calibrated_minus_raw": cal - raw,
                         "heldout_dates": len(test)})
    all_train = defaultdict(list)
    for groups in groups_by_fold.values():
        for date, vals in groups.items(): all_train[date].extend(vals)
    coef = fit_date_balanced(all_train)
    val_rows, val_meta = validation_score(args.root, coef["slope"], coef["intercept"])
    args.output.mkdir(parents=True)
    with (args.output / "validation_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(val_rows[0])); w.writeheader(); w.writerows(val_rows)
    receipt = {"state":"COMPLETE_TRAIN_ONLY_OOF_COT_CALIBRATION_VALIDATION_DIAGNOSTIC",
               "claim_scope":"attempt12 Hubei validation against CPP retrieval reference, not independent COT truth/test",
               "test_payloads_read":0,"buffer_payloads_read":0,
               "fit":"single global date-balanced affine map in log1p space across all leads/pixels; slope clipped [0,1]",
               "parameters":coef,"train_array_shapes":list(train_shape),"train_hashes":train_hashes,
               "fold_source_manifest":fold_sources,
               "crossfit_by_date_fold":crossfit,"validation":val_meta,
               "validation_metrics_csv_sha256":sha(args.output / "validation_metrics.csv")}
    (args.output / "calibration_receipt.json").write_text(json.dumps(receipt,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"state":receipt["state"],"parameters":coef,
                      "crossfit_delta": [x["delta_calibrated_minus_raw"] for x in crossfit],
                      "validation_common_keys":val_meta["common_init_target_lead_keys"],
                      "validation_metrics":str(args.output / "validation_metrics.csv")},indent=2))


if __name__ == "__main__":
    main()
