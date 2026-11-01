"""Exploratory, H-only regime decomposition of P-XC vs no-COT GHI OOF errors."""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "results/direct_cot_dualpath_hcv_v1_local4060"
BANK = ROOT / "data/cot_traj_hcv_inputs_v1/bank"
OUT = RUN / "score/regime_diagnostic"
STATIONS = ("sili", "zhujia")
SEEDS = (42, 43, 44)
ARMS = ("none", "pxc")
REPS = 5000
RNG_SEED = 20261021


def key_id(k):
    return (k["init_time_utc"], k["target_time_utc"], int(k["lead_minutes"]))


def bjt_date(k):
    return (datetime.fromisoformat(k["target_time_utc"]) + timedelta(hours=8)).date()


def block_id(k):
    d = bjt_date(k)
    return f"{d.year:04d}-{d.month:02d}-B{(d.day - 1) // 2:02d}"


def feature_maps(station):
    folder = BANK / station / "train"
    keys = [json.loads(x) for x in (folder / "keys.jsonl").read_text(encoding="utf-8").splitlines()]
    kmap = {key_id(k): i for i, k in enumerate(keys)}
    if len(kmap) != len(keys):
        raise ValueError(f"duplicate row-bank keys: {station}")
    cot = np.load(folder / "p_xc.npy", mmap_mode="r")[:, 0, 1, 1].astype(np.float64)
    geom = np.load(folder / "geometry_raw.npy", mmap_mode="r")[:, 0, 1, 1].astype(np.float64)
    return kmap, cot, geom


def stratum_values(cot, cos_soz):
    out = {}
    bins = (("COT < 1", 0.0, 1.0), ("1 <= COT < 10", 1.0, 10.0),
            ("10 <= COT < 30", 10.0, 30.0), ("COT >= 30", 30.0, np.inf))
    for name, lo, hi in bins:
        out[f"predicted_center_{name}"] = (cot >= lo) & (cot < hi)
    soz = np.degrees(np.arccos(np.clip(cos_soz, -1, 1)))
    out["SOZ < 80 deg"] = soz < 80.0
    out["80 <= SOZ < 90 deg"] = (soz >= 80.0) & (soz < 90.0)
    return out


def collect():
    features = {st: feature_maps(st) for st in STATIONS}
    records = {name: defaultdict(lambda: [0.0, 0.0, 0]) for st in STATIONS for name in []}
    strata_names = None
    # records[stratum][station,seed,block] = [SSE_PXC, SSE_none, row_count]
    records = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0]))
    for st in STATIONS:
        rowmap, cot, cos_soz = features[st]
        for fold in range(5):
            loaded = {}
            refkeys = None
            for arm in ARMS:
                for seed in SEEDS:
                    run = RUN / st / f"fold{fold}" / arm / f"seed{seed}"
                    keys = [json.loads(x) for x in (run / "oof_keys.jsonl").read_text(encoding="utf-8").splitlines()]
                    with np.load(run / "oof_predictions.npz", allow_pickle=False) as z:
                        pred = z["prediction_ghi"].astype(np.float64)
                        truth = z["truth_ghi"].astype(np.float64)
                    if len(keys) != len(pred) or len(pred) != len(truth):
                        raise ValueError(f"row/pred mismatch: {st}/{fold}/{arm}/{seed}")
                    if refkeys is None:
                        refkeys = keys
                    elif keys != refkeys:
                        raise ValueError(f"paired key mismatch: {st}/{fold}/{arm}/{seed}")
                    loaded[(arm, seed)] = (pred, truth)
            for i, k in enumerate(refkeys):
                ki = key_id(k)
                if ki not in rowmap:
                    raise ValueError(f"OOF key absent from H bank: {st}/{ki}")
                ri = rowmap[ki]
                strats = stratum_values(np.asarray([cot[ri]]), np.asarray([cos_soz[ri]]))
                if strata_names is None:
                    strata_names = list(strats)
                b = block_id(k)
                for seed in SEEDS:
                    p, y = loaded[("pxc", seed)]
                    n, yn = loaded[("none", seed)]
                    if not np.array_equal(y, yn):
                        raise ValueError("truth mismatch")
                    for name, mask in strats.items():
                        if not bool(mask[0]):
                            continue
                        rec = records[name][(st, seed, b)]
                        rec[0] += float((p[i] - y[i]) ** 2)
                        rec[1] += float((n[i] - y[i]) ** 2)
                        rec[2] += 1
    return records, strata_names


def calc_delta(d, blocks, weights):
    vals = []
    for st in STATIONS:
        for seed in SEEDS:
            sse = [0.0, 0.0]
            count = 0
            for b, w in zip(blocks, weights):
                a = d.get((st, seed, b))
                if a and w:
                    sse[0] += a[0] * w
                    sse[1] += a[1] * w
                    count += a[2] * w
            if count == 0:
                return np.nan
            vals.append(np.sqrt(sse[0] / count) - np.sqrt(sse[1] / count))
    return float(np.mean(vals))


def summarize(records, strata_names):
    rng = np.random.default_rng(RNG_SEED)
    results = {}
    for name in strata_names:
        d = records[name]
        blocks = sorted({key[2] for key in d})
        if not blocks:
            continue
        full = np.ones(len(blocks), dtype=np.int64)
        point = calc_delta(d, blocks, full)
        draws = np.empty(REPS, dtype=np.float64)
        for j in range(REPS):
            draws[j] = calc_delta(d, blocks, np.bincount(rng.integers(0, len(blocks), len(blocks)), minlength=len(blocks)))
        # Counts per station and seed make sparse strata visible.
        counts = {}
        for st in STATIONS:
            counts[st] = {}
            for seed in SEEDS:
                counts[st][str(seed)] = int(sum(v[2] for (s, se, _), v in d.items() if s == st and se == seed))
        results[name] = {
            "delta_rmse_pxc_minus_none_wm2": point,
            "ci95_low": float(np.nanquantile(draws, .025)),
            "ci95_high": float(np.nanquantile(draws, .975)),
            "two_day_blocks": len(blocks),
            "rows_station_seed": counts,
            "interpretation": "exploratory H-only post-hoc stratification; positive means P-XC worsens",
        }
    return results


def main():
    metrics_path = RUN / "score/dualpath_hcv_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("state") != "COMPLETE_COT_DUALPATH_H_CV_SCORE":
        raise RuntimeError("complete H-only score required")
    for k in ("test_payload_opened", "buffer_payload_opened", "future_observed_agri_used_as_input", "cpp_used_as_input"):
        if metrics.get(k) is not False:
            raise RuntimeError(f"forbidden source flag {k}")
    records, strata_names = collect()
    results = summarize(records, strata_names)
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "state": "COMPLETE_EXPLORATORY_HCV_REGIME_DIAGNOSTIC",
        "source_run_manifest_sha256": metrics["run_manifest_sha256"],
        "comparison": "P-XC forecast COT GHI head minus no-COT GHI head; same H OOF keys, common truth, paired seeds",
        "conditioning_variables": "P-XC predicted station-center COT bins (0-1, 1-10, 10-30, >=30) and target solar-zenith bands (<80, 80-90 degrees)",
        "bootstrap": {"unit": "fixed calendar two-day BJT blocks", "reps": REPS, "seed": RNG_SEED},
        "results": results,
        "test_payload_opened": False,
        "buffer_payload_opened": False,
        "future_observed_agri_used_as_input": False,
        "cpp_used_as_input": False,
        "claim_boundary": "post-hoc subgroup diagnostics only; not a confirmatory result and not a model-selection gate",
    }
    (OUT / "regime_metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    names = list(results)
    x = np.arange(len(names))
    vals = np.asarray([results[n]["delta_rmse_pxc_minus_none_wm2"] for n in names])
    lo = np.asarray([results[n]["ci95_low"] for n in names])
    hi = np.asarray([results[n]["ci95_high"] for n in names])
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    colors = ["#9b2226" if v > 0 else "#0a9396" for v in vals]
    ax.bar(x, vals, color=colors, alpha=.85)
    ax.errorbar(x, vals, yerr=np.vstack((vals-lo, hi-vals)), fmt="none", ecolor="#222222", capsize=4, lw=1.3)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylabel("Paired ΔRMSE (W m⁻²), P-XC − no COT\npositive = COT head is worse")
    ax.set_title("Exploratory H-only COT utility by forecast regime")
    ax.grid(axis="y", alpha=.2)
    for i, n in enumerate(names):
        ax.text(i, hi[i] + .15, f"n={sum(results[n]['rows_station_seed'][s]['42'] for s in STATIONS):,}", ha="center", va="bottom", fontsize=8)
    fig.savefig(OUT / "regime_diagnostic.png", dpi=180)
    fig.savefig(OUT / "regime_diagnostic.pdf")
    print(json.dumps({"state": result["state"], "strata": results}, indent=2), flush=True)


if __name__ == "__main__":
    main()
