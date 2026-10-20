#!/usr/bin/env python3
"""Independent FP64 rescoring of the Hubei four-arm same-time GHI experiment.

The script reads only saved validation predictions, the audited train/validation
GHI pack, and the correct-UTC CPP train/validation supervision sidecar. It never
opens test or buffer payloads. Date-block bootstrap uses the frozen consecutive
two-day validation blocks and keeps all station/seed/arm comparisons paired.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_remote_attempt2"
PRED_ROOT = RESULTS
PACK = RESULTS / "ghi_fourarm_pack_attempt5"
CPP_DIR = RESULTS / "cpp_attempt2"
OUT = ROOT / "results_fourarm_attempt2"
OUT.mkdir(parents=True, exist_ok=True)

STATIONS = ["Niushou", "Wangzhai", "Mizhuang"]
ARMS = ["A", "AC", "AM", "ACM"]
SEEDS = [42, 43, 44]
# Pairs are (reference, candidate); positive relative gain means candidate is better.
CONTRASTS = [("A", "AC"), ("A", "AM"), ("A", "ACM"), ("AC", "ACM"), ("AM", "ACM")]
COT_BINS = [
    ("CPP_COT_[0,0.1)", 0.0, 0.1),
    ("CPP_COT_[0.1,5)", 0.1, 5.0),
    ("CPP_COT_[5,30)", 5.0, 30.0),
    ("CPP_COT_[30,100]", 30.0, 100.0 + 1e-12),
    ("CPP_COT_>100", 100.0 + 1e-12, np.inf),
]
N_BOOT = 20_000
BOOT_SEED = 20261020


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    e = p - y
    if w is None:
        mse = float(np.mean(e * e, dtype=np.float64))
        mae = float(np.mean(np.abs(e), dtype=np.float64))
        bias = float(np.mean(e, dtype=np.float64))
    else:
        w = np.asarray(w, dtype=np.float64)
        den = float(np.sum(w, dtype=np.float64))
        if den <= 0:
            return {"rmse": float("nan"), "mae": float("nan"), "bias": float("nan")}
        mse = float(np.sum(w * e * e, dtype=np.float64) / den)
        mae = float(np.sum(w * np.abs(e), dtype=np.float64) / den)
        bias = float(np.sum(w * e, dtype=np.float64) / den)
    return {"rmse": float(np.sqrt(mse)), "mae": mae, "bias": bias}


def read_prediction(seed: int, station: str, arm: str) -> tuple[dict, dict, str]:
    p = PRED_ROOT / f"ghi_fourarm_seed{seed}_attempt2" / station / f"{arm}_validation_predictions.npz"
    jpath = p.with_name(f"{arm}_validation.json")
    if not p.is_file() or not jpath.is_file():
        raise FileNotFoundError(p if not p.exists() else jpath)
    meta = json.loads(jpath.read_text(encoding="utf-8"))
    digest = sha256(p)
    if meta.get("prediction_sha256") != digest:
        raise AssertionError(f"prediction SHA mismatch: {p}")
    ckpt = p.with_name(f"{arm}_best.pt")
    if not ckpt.is_file() or sha256(ckpt) != meta.get("checkpoint_sha256"):
        raise AssertionError(f"checkpoint SHA mismatch or missing checkpoint: {ckpt}")
    with np.load(p, allow_pickle=False) as z:
        a = {k: z[k].copy() for k in z.files}
    n = len(a["target_time_bjt"])
    for key in ["station", "split", "pred_ghi_wm2", "truth_ghi_wm2", "clear_sky_wm2"]:
        if len(a[key]) != n:
            raise AssertionError(f"length mismatch {p}: {key}")
    if a["pred_ghi_wm2"].dtype != np.float64 or a["truth_ghi_wm2"].dtype != np.float64:
        raise AssertionError(f"pred/truth must be float64 in saved output: {p}")
    if set(a["split"].astype(str)) != {"val"}:
        raise AssertionError(f"non-validation split in prediction: {p}")
    if set(a["station"].astype(str)) != {station}:
        raise AssertionError(f"station mismatch in prediction: {p}")
    if int(meta.get("test_payloads_read", -1)) != 0:
        raise AssertionError(f"test gate failed in metadata: {p}")
    if meta.get("checkpoint_reload_parity") != "PASS" or float(meta.get("checkpoint_reload_max_abs_delta_wm2", np.inf)) != 0.0:
        raise AssertionError(f"checkpoint reload parity failed: {p}")
    if not all(np.all(np.isfinite(a[k])) for k in ["pred_ghi_wm2", "truth_ghi_wm2", "clear_sky_wm2"]):
        raise AssertionError(f"nonfinite saved prediction values: {p}")
    keys = list(zip(a["station"].astype(str), a["target_time_bjt"].astype(str)))
    if len(set(keys)) != n:
        raise AssertionError(f"duplicate station/time keys: {p}")
    order = np.argsort(np.asarray([f"{s}/{t}" for s, t in keys], dtype=str))
    a = {k: v[order] for k, v in a.items()}
    keyarr = np.asarray([keys[i][0] + "/" + keys[i][1] for i in order], dtype=str)
    return a, meta, digest


def main() -> None:
    rows = pd.read_csv(PACK / "rows.csv", dtype={"time_bjt": str, "station": str, "split": str})
    if not bool(rows["time_bjt"].str.fullmatch(r"\d{12}").all()):
        raise AssertionError("rows.csv time_bjt must be canonical 12-digit BJT")
    rows_val = rows.loc[rows["split"].eq("val")].copy()
    if rows_val.duplicated(["station", "time_bjt"]).any():
        raise AssertionError("duplicate station/time key in validation pack")
    pack_keys = {(r.station, r.time_bjt): (float(r.ghi_wm2), float(r.clear_sky_wm2)) for r in rows_val.itertuples()}
    allpred: dict[tuple[int, str, str], dict] = {}
    metadata: dict[str, dict] = {}
    file_hashes: dict[str, str] = {}
    key_reference: dict[str, list[str]] = {}
    max_truth_pack_abs = 0.0
    max_clear_pack_abs = 0.0
    for seed in SEEDS:
        seed_root = PRED_ROOT / f"ghi_fourarm_seed{seed}_attempt2"
        epoch_path = seed_root / "all_metrics.jsonl"
        epoch_rows = [json.loads(line) for line in epoch_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(epoch_rows) != len(STATIONS) * len(ARMS) * 30:
            raise AssertionError(f"wrong epoch record count for seed {seed}: {len(epoch_rows)}")
        for station in STATIONS:
            for arm in ARMS:
                got = sorted(int(r["epoch"]) for r in epoch_rows if r.get("station") == station and r.get("arm") == arm and int(r.get("seed", -1)) == seed)
                if got != list(range(1, 31)):
                    raise AssertionError(f"missing or duplicate epochs for {seed}/{station}/{arm}: {got}")
        for station in STATIONS:
            station_keys = None
            station_truth = None
            station_clear = None
            station_init_hashes = set()
            for arm in ARMS:
                a, meta, digest = read_prediction(seed, station, arm)
                keys = [s + "/" + t for s, t in zip(a["station"].astype(str), a["target_time_bjt"].astype(str))]
                if station_keys is None:
                    station_keys = keys
                    station_truth = a["truth_ghi_wm2"].astype(np.float64)
                    station_clear = a["clear_sky_wm2"].astype(np.float64)
                else:
                    if keys != station_keys:
                        raise AssertionError(f"prediction keys not paired across arms: {seed}/{station}/{arm}")
                    if not np.array_equal(a["truth_ghi_wm2"], station_truth):
                        raise AssertionError(f"truth differs across arms: {seed}/{station}/{arm}")
                    if not np.array_equal(a["clear_sky_wm2"], station_clear):
                        raise AssertionError(f"clear-sky differs across arms: {seed}/{station}/{arm}")
                station_init_hashes.add(meta["initialization_sha256"])
                for i, t in enumerate(a["target_time_bjt"].astype(str)):
                    k = (station, t)
                    if k not in pack_keys:
                        raise AssertionError(f"prediction key absent from audited pack: {seed}/{station}/{t}")
                    ghi, cs = pack_keys[k]
                    max_truth_pack_abs = max(max_truth_pack_abs, abs(float(a["truth_ghi_wm2"][i]) - ghi))
                    max_clear_pack_abs = max(max_clear_pack_abs, abs(float(a["clear_sky_wm2"][i]) - cs))
                allpred[(seed, station, arm)] = a
                name = f"seed{seed}/{station}/{arm}"
                metadata[name] = meta
                file_hashes[name] = digest
            if len(station_init_hashes) != 1:
                raise AssertionError(f"initialization was not paired across arms: {seed}/{station}")
            key_reference[f"seed{seed}/{station}"] = station_keys or []

    # Compare support across arms/seeds within each station, then against the
    # complete validation pack without incorrectly equating different stations.
    station_reference_keys: dict[str, list[str]] = {}
    for station in STATIONS:
        reference = key_reference[f"seed{SEEDS[0]}/{station}"]
        for seed in SEEDS:
            label = f"seed{seed}/{station}"
            if key_reference[label] != reference:
                raise AssertionError(f"seed validation key order/support differs: {label}")
        station_reference_keys[station] = reference
        pack_keys_for_station = sorted(f"{s}/{t}" for s, t in pack_keys if s == station)
        if sorted(reference) != pack_keys_for_station:
            missing = sorted(set(pack_keys_for_station) - set(reference))[:10]
            extra = sorted(set(reference) - set(pack_keys_for_station))[:10]
            raise AssertionError(f"validation prediction support != pack for {station}: missing={missing}; extra={extra}")
    n_reference_keys = sum(map(len, station_reference_keys.values()))
    if max_truth_pack_abs > 1e-4 or max_clear_pack_abs > 1e-4:
        raise AssertionError(f"saved truth/clear-sky differs from audited pack: {max_truth_pack_abs=}, {max_clear_pack_abs=}")

    # Add corrected-UTC CPP reference at the station pixel; these values are used
    # only to define diagnostic cloud strata, never to select the full GHI score.
    with np.load(CPP_DIR / "cpp_supervision_trainval.npz", allow_pickle=False) as z:
        cpp_time = z["time_bjt"].astype(str)
        cpp_split = z["split"].astype(str)
        cpp_cot = z["cot"].astype(np.float64)
        cpp_cot_valid = z["cot_valid_mask"].astype(bool)
        cpp_clp = z["clp"].astype(np.int64)
        cpp_clp_valid = z["clp_valid_mask"].astype(bool)
        cpp_station_names = z["station_names"].astype(str).tolist()
        cpp_centers = z["station_local_centers"].astype(int)
    cpp_index = {t: i for i, t in enumerate(cpp_time) if cpp_split[i] == "val"}
    if len(cpp_index) != int(np.sum(cpp_split == "val")):
        raise AssertionError("duplicate CPP validation time keys")
    cpp_station_index = {s: i for i, s in enumerate(cpp_station_names)}
    if set(cpp_station_index) != set(STATIONS):
        raise AssertionError(f"unexpected CPP station map: {cpp_station_names}")
    row_records: list[dict] = []
    for seed in SEEDS:
        for station in STATIONS:
            for arm in ARMS:
                a = allpred[(seed, station, arm)]
                si = cpp_station_index[station]
                rr, cc = map(int, cpp_centers[si])
                for i, t in enumerate(a["target_time_bjt"].astype(str)):
                    cidx = cpp_index.get(t)
                    cval = np.nan
                    clpval = -1
                    if cidx is not None and cpp_cot_valid[cidx, rr, cc]:
                        cval = float(cpp_cot[cidx, rr, cc])
                    if cidx is not None and cpp_clp_valid[cidx, rr, cc]:
                        clpval = int(cpp_clp[cidx, rr, cc])
                    row_records.append({
                        "seed": seed, "station": station, "arm": arm,
                        "time_bjt": t, "date_bjt": t[:8], "split": "val",
                        "pred": float(a["pred_ghi_wm2"][i]),
                        "truth": float(a["truth_ghi_wm2"][i]),
                        "clear_sky": float(a["clear_sky_wm2"][i]),
                        "cpp_cot_center": cval, "cpp_clp_center": clpval,
                    })
    long = pd.DataFrame.from_records(row_records)
    if len(long) != n_reference_keys * len(SEEDS) * len(ARMS):
        raise AssertionError("long table row count mismatch")
    # Each unique validation date belongs to a consecutive two-day block.
    dates = sorted(rows_val["time_bjt"].str[:8].unique())
    date_blocks: dict[str, int] = {}
    if len(dates) % 2:
        raise AssertionError(f"expected paired two-day validation dates, got {len(dates)}")
    for block_idx in range(0, len(dates), 2):
        d0, d1 = dates[block_idx:block_idx + 2]
        if (pd.Timestamp(d1) - pd.Timestamp(d0)).days != 1:
            raise AssertionError(f"validation dates do not form consecutive two-day blocks: {d0},{d1}")
        date_blocks[d0] = block_idx // 2
        date_blocks[d1] = block_idx // 2
    long["date_block"] = long["date_bjt"].map(date_blocks)
    if long["date_block"].isna().any():
        raise AssertionError("unmapped validation date block")

    # CPP reference coverage by month, keeping full GHI validation support visible.
    unique_rows = long.drop_duplicates(["station", "time_bjt"]).copy()
    unique_rows["month_bjt"] = unique_rows["time_bjt"].str[:6]
    unique_cot = unique_rows["cpp_cot_center"].to_numpy(dtype=np.float64)
    unique_cot_valid = np.isfinite(unique_cot)
    monthly_cov_rows = []
    for month, mg in unique_rows.groupby("month_bjt", sort=True):
        mc = mg["cpp_cot_center"].to_numpy(dtype=np.float64)
        mclp = mg["cpp_clp_center"].to_numpy(dtype=np.int64)
        row = {
            "month_bjt": month,
            "ghi_validation_station_time_n": int(len(mg)),
            "ghi_validation_unique_time_n": int(mg["time_bjt"].nunique()),
            "cpp_cot_center_valid_n": int(np.isfinite(mc).sum()),
            "cpp_clp_center_valid_n": int((mclp >= 0).sum()),
            "n_validation_days": int(mg["date_bjt"].nunique()),
            "n_two_day_blocks": int(mg["date_block"].nunique()),
        }
        for name, lo, hi in COT_BINS:
            row[name + "_n"] = int(np.sum(np.isfinite(mc) & (mc >= lo) & (mc < hi)))
        monthly_cov_rows.append(row)
    monthly_cov = pd.DataFrame(monthly_cov_rows)
    monthly_cov.to_csv(OUT / "cpp_reference_coverage_by_month.csv", index=False)

    # Aggregate metrics by station and seed, with a fixed validation support.
    metric_rows = []
    for (seed, station, arm), g in long.groupby(["seed", "station", "arm"], sort=True):
        m = metrics(g["truth"].to_numpy(), g["pred"].to_numpy())
        metric_rows.append({"seed": seed, "station": station, "arm": arm, "n": len(g), **m})
    per_seed = pd.DataFrame(metric_rows)
    per_seed.to_csv(OUT / "metrics_by_station_seed_arm.csv", index=False, float_format="%.10g")

    station_rows = []
    for (station, arm), g in per_seed.groupby(["station", "arm"], sort=True):
        station_rows.append({
            "station": station, "arm": arm,
            "n_per_seed": int(g["n"].min()),
            "rmse_mean_seeds": float(g["rmse"].mean()),
            "rmse_sd_seeds": float(g["rmse"].std(ddof=1)),
            "mae_mean_seeds": float(g["mae"].mean()),
            "bias_mean_seeds": float(g["bias"].mean()),
        })
    station_summary = pd.DataFrame(station_rows)
    station_summary.to_csv(OUT / "metrics_by_station_arm.csv", index=False, float_format="%.10g")

    aggregate = []
    for arm, g in per_seed.groupby("arm", sort=False):
        # Equal weighting across the three station RMSEs and the three seed replicates.
        station_seed_rmse = g["rmse"].to_numpy(dtype=np.float64)
        row_g = long.loc[long["arm"].eq(arm)]
        row_err = row_g["pred"].to_numpy(dtype=np.float64) - row_g["truth"].to_numpy(dtype=np.float64)
        aggregate.append({
            "arm": arm,
            "equal_station_seed_rmse": float(np.mean(station_seed_rmse, dtype=np.float64)),
            "sd_station_seed_rmse": float(np.std(station_seed_rmse, ddof=1)),
            "equal_station_seed_mae": float(np.mean(g["mae"].to_numpy(dtype=np.float64))),
            "equal_station_seed_bias": float(np.mean(g["bias"].to_numpy(dtype=np.float64))),
            "pooled_row_rmse": float(np.sqrt(np.mean(row_err * row_err, dtype=np.float64))),
        })
    aggregate_df = pd.DataFrame(aggregate)
    a_rmse = float(aggregate_df.loc[aggregate_df.arm.eq("A"), "equal_station_seed_rmse"].iloc[0])
    aggregate_df["gain_vs_A_percent"] = (a_rmse - aggregate_df["equal_station_seed_rmse"]) / a_rmse * 100.0
    aggregate_df.to_csv(OUT / "metrics_overall_arm.csv", index=False, float_format="%.10g")

    # Prepare keyed arrays for block bootstrap; resampling the same two-day
    # validation clusters keeps every arm and seed strictly paired.
    block_ids = np.array(sorted(date_blocks.values()), dtype=int)
    block_ids = np.unique(block_ids)
    rng = np.random.default_rng(BOOT_SEED)

    def bootstrap_contrast(mask: np.ndarray, ref: str, cand: str, draw_counts: np.ndarray,
                           eligible_blocks: list[int], block_sse: dict, block_n: dict) -> dict:
        if len(eligible_blocks) < 2:
            return {"ref": ref, "candidate": cand, "n_blocks": len(eligible_blocks), "n_boot": 0,
                    "delta_rmse_ci95": [None, None], "gain_percent_ci95": [None, None],
                    "delta_rmse_point": None, "gain_percent_point": None}

        def replicate_scores(arm: str) -> np.ndarray:
            cell_scores = []
            for seed in SEEDS:
                for station in STATIONS:
                    key = (seed, station, arm)
                    sse = block_sse[key]
                    ns = block_n[key]
                    den = draw_counts @ ns
                    num = draw_counts @ sse
                    rmse = np.full(len(draw_counts), np.nan, dtype=np.float64)
                    ok = den > 0
                    rmse[ok] = np.sqrt(num[ok] / den[ok])
                    cell_scores.append(rmse)
            mat = np.vstack(cell_scores).T
            return np.nanmean(mat, axis=1)

        ref_scores = replicate_scores(ref)
        cand_scores = replicate_scores(cand)
        ok = np.isfinite(ref_scores) & np.isfinite(cand_scores) & (ref_scores != 0)
        arr = np.column_stack((cand_scores[ok] - ref_scores[ok],
                               100.0 * (ref_scores[ok] - cand_scores[ok]) / ref_scores[ok]))
        # Point estimate on the original fixed support.
        point: dict[str, list[float]] = {ref: [], cand: []}
        for seed in SEEDS:
            for station in STATIONS:
                for arm in [ref, cand]:
                    g = long.loc[mask & long["seed"].eq(seed) & long["station"].eq(station) & long["arm"].eq(arm)]
                    if len(g):
                        point[arm].append(metrics(g["truth"].to_numpy(), g["pred"].to_numpy())["rmse"])
        rp = float(np.mean(point[ref], dtype=np.float64))
        cp = float(np.mean(point[cand], dtype=np.float64))
        return {
            "ref": ref, "candidate": cand, "n_blocks": len(eligible_blocks), "n_boot": int(len(arr)),
            "delta_rmse_point": cp - rp,
            "gain_percent_point": 100.0 * (rp - cp) / rp if rp else None,
            "delta_rmse_ci95": np.percentile(arr[:, 0], [2.5, 97.5]).tolist() if len(arr) else [None, None],
            "gain_percent_ci95": np.percentile(arr[:, 1], [2.5, 97.5]).tolist() if len(arr) else [None, None],
        }

    # Support strata. CPP values are retrieval-reference labels; they define
    # diagnostic bins only and never filter the full-support GHI headline.
    stratum_masks: dict[str, np.ndarray] = {"all_validation": np.ones(len(long), dtype=bool)}
    cpp_valid = np.isfinite(long["cpp_cot_center"].to_numpy(dtype=np.float64))
    cvals = long["cpp_cot_center"].to_numpy(dtype=np.float64)
    for name, lo, hi in COT_BINS:
        stratum_masks[name] = cpp_valid & (cvals >= lo) & (cvals < hi)
    clp = long["cpp_clp_center"].to_numpy(dtype=np.int64)
    clp_valid_bool = clp >= 0
    stratum_masks["CPP_CLP_clear"] = clp_valid_bool & (clp == 0)
    stratum_masks["CPP_CLP_water"] = clp_valid_bool & (clp == 1)
    stratum_masks["CPP_CLP_ice"] = clp_valid_bool & (clp == 2)

    stratified_rows = []
    stratified_station_rows = []
    bootstrap_results = []
    for sname, mask in stratum_masks.items():
        g = long.loc[mask]
        unique_samples = g[["station", "time_bjt"]].drop_duplicates()
        blocks_here = sorted(g["date_block"].unique().astype(int).tolist())
        for arm in ARMS:
            ga = g.loc[g["arm"].eq(arm)]
            stats = []
            for station in STATIONS:
                for seed in SEEDS:
                    one = ga.loc[ga["station"].eq(station) & ga["seed"].eq(seed)]
                    if len(one):
                        stats.append(metrics(one["truth"].to_numpy(), one["pred"].to_numpy()))
            if stats:
                stratified_rows.append({
                    "stratum": sname, "arm": arm,
                    "unique_station_time_n": int(len(unique_samples)),
                    "station_seed_cells_present": int(len(stats)),
                    "stations_present": int(ga["station"].nunique()),
                    "date_blocks": int(len(blocks_here)),
                    "rmse_equal_station_seed": float(np.mean([m["rmse"] for m in stats], dtype=np.float64)),
                    "mae_equal_station_seed": float(np.mean([m["mae"] for m in stats], dtype=np.float64)),
                    "bias_equal_station_seed": float(np.mean([m["bias"] for m in stats], dtype=np.float64)),
                })
            for station in STATIONS:
                station_seed_stats = []
                station_n = None
                for seed in SEEDS:
                    one = ga.loc[ga["station"].eq(station) & ga["seed"].eq(seed)]
                    if len(one):
                        station_n = len(one)
                        station_seed_stats.append(metrics(one["truth"].to_numpy(), one["pred"].to_numpy()))
                if station_seed_stats:
                    stratified_station_rows.append({
                        "stratum": sname, "station": station, "arm": arm,
                        "n_per_seed": int(station_n), "seeds_present": int(len(station_seed_stats)),
                        "rmse_mean_seeds": float(np.mean([m["rmse"] for m in station_seed_stats], dtype=np.float64)),
                        "rmse_sd_seeds": float(np.std([m["rmse"] for m in station_seed_stats], ddof=1)) if len(station_seed_stats) > 1 else 0.0,
                        "mae_mean_seeds": float(np.mean([m["mae"] for m in station_seed_stats], dtype=np.float64)),
                        "bias_mean_seeds": float(np.mean([m["bias"] for m in station_seed_stats], dtype=np.float64)),
                    })
        # Block-sufficient-statistic bootstrap: fast, exact for RMSE under
        # resampled date-block multiplicities, and paired across all arms.
        eligible_blocks = blocks_here
        draws = rng.choice(eligible_blocks, size=(N_BOOT, len(eligible_blocks)), replace=True) if len(eligible_blocks) >= 2 else np.empty((0, 0), dtype=int)
        draw_counts = np.stack([(draws == b).sum(axis=1) for b in eligible_blocks], axis=1) if len(eligible_blocks) >= 2 else np.empty((0, 0), dtype=np.int64)
        gs = g.copy()
        gs["err2"] = np.square(gs["pred"].to_numpy(dtype=np.float64) - gs["truth"].to_numpy(dtype=np.float64))
        agg = gs.groupby(["seed", "station", "arm", "date_block"], sort=False).agg(sse=("err2", "sum"), n=("err2", "size"))
        block_sse = {}
        block_n = {}
        for seed in SEEDS:
            for station in STATIONS:
                for arm in ARMS:
                    try:
                        one = agg.loc[(seed, station, arm)]
                        block_sse[(seed, station, arm)] = np.asarray([float(one.loc[b, "sse"]) if b in one.index else 0.0 for b in eligible_blocks], dtype=np.float64)
                        block_n[(seed, station, arm)] = np.asarray([float(one.loc[b, "n"]) if b in one.index else 0.0 for b in eligible_blocks], dtype=np.float64)
                    except KeyError:
                        block_sse[(seed, station, arm)] = np.zeros(len(eligible_blocks), dtype=np.float64)
                        block_n[(seed, station, arm)] = np.zeros(len(eligible_blocks), dtype=np.float64)
        for ref, cand in CONTRASTS:
            result = bootstrap_contrast(mask, ref, cand, draw_counts, eligible_blocks, block_sse, block_n)
            result["stratum"] = sname
            result["unique_station_time_n"] = int(len(unique_samples))
            bootstrap_results.append(result)
    strata_df = pd.DataFrame(stratified_rows)
    strata_df.to_csv(OUT / "metrics_by_cpp_cot_clp_stratum.csv", index=False, float_format="%.10g")
    strata_station_df = pd.DataFrame(stratified_station_rows)
    strata_station_df.to_csv(OUT / "metrics_by_cpp_stratum_station_arm.csv", index=False, float_format="%.10g")
    boot_df = pd.DataFrame(bootstrap_results)
    boot_df.to_json(OUT / "paired_two_day_block_bootstrap.json", orient="records", force_ascii=False, indent=2)

    # Seed-by-station paired deltas, ensuring any large outlier can be seen.
    deltas = []
    for seed in SEEDS:
        for station in STATIONS:
            sub = per_seed.loc[per_seed.seed.eq(seed) & per_seed.station.eq(station)].set_index("arm")
            base = float(sub.loc["A", "rmse"])
            for ref, cand in CONTRASTS:
                if ref != "A":
                    continue
                cand_rmse = float(sub.loc[cand, "rmse"])
                deltas.append({"seed": seed, "station": station, "ref": ref, "candidate": cand,
                               "ref_rmse": base, "candidate_rmse": cand_rmse,
                               "delta_candidate_minus_ref": cand_rmse - base,
                               "gain_percent": 100.0 * (base - cand_rmse) / base})
    pd.DataFrame(deltas).to_csv(OUT / "paired_seed_station_deltas.csv", index=False, float_format="%.10g")

    audit = {
        "status": "INDEPENDENT_SAVED_PREDICTION_RESCORING_PASS",
        "experiment": "Hubei 3-station same-time GHI reconstruction four-arm paired experiment",
        "input_mode": "single-frame station-centered 3x3 AGRI + solar geometry, optional predicted COT/p_cloud",
        "stations": STATIONS, "seeds": SEEDS, "arms": ARMS,
        "n_prediction_files": len(file_hashes),
        "all_36_checkpoint_sha256_verified_against_saved_metadata": True,
        "all_3_seed_logs_have_30_epochs_for_each_station_arm": True,
        "n_validation_station_time_keys": n_reference_keys,
        "n_station_seed_arm_prediction_rows": len(long),
        "validation_rows_by_station": {s: int(rows_val.station.eq(s).sum()) for s in STATIONS},
        "all_36_files_sha256_verified_against_saved_metadata": True,
        "checkpoint_reload_parity_all_36": all(m.get("checkpoint_reload_parity") == "PASS" and float(m.get("checkpoint_reload_max_abs_delta_wm2")) == 0.0 for m in metadata.values()),
        "all_36_trainval_split_only": True,
        "all_test_payloads_read": 0,
        "pack_rows_exact_key_match": True,
        "maximum_truth_vs_pack_abs_delta_wm2": max_truth_pack_abs,
        "maximum_clear_sky_vs_pack_abs_delta_wm2": max_clear_pack_abs,
        "paired_initialization_per_station_seed": True,
        "validation_dates_bjt": dates,
        "two_day_blocks": [[dates[i], dates[i+1]] for i in range(0, len(dates), 2)],
        "cpp_reference_support_unique_time_keys": len(cpp_index),
        "cpp_reference_coverage_by_validation_month": monthly_cov.to_dict(orient="records"),
        "cpp_reference_is_retrieval_reference_not_independent_truth": True,
        "checkpoint_selected_on_same_validation_set": True,
        "cpp_thickness_bin_definition": ["[0,0.1)", "[0.1,5)", "[5,30)", "[30,100]", ">100"],
        "cpp_stratification_uses_station_center_pixel_only": True,
        "bootstrap": {"unit": "consecutive two-day BJT validation date block", "replicates": N_BOOT, "seed": BOOT_SEED},
        "prediction_sha256": file_hashes,
        "prediction_metadata": {k: {"best_epoch": v["best_epoch"], "initialization_sha256": v["initialization_sha256"],
                                     "checkpoint_sha256": v["checkpoint_sha256"], "prediction_sha256": v["prediction_sha256"],
                                     "checkpoint_reload_parity": v["checkpoint_reload_parity"],
                                     "checkpoint_reload_max_abs_delta_wm2": v["checkpoint_reload_max_abs_delta_wm2"]}
                                for k, v in metadata.items()},
        "source_pack_report_sha256": sha256(PACK / "pack_report.json"),
        "source_pack_sha_manifest_sha256": sha256(PACK / "SHA256.json"),
        "cpp_trainval_sidecar_sha256": sha256(CPP_DIR / "cpp_supervision_trainval.npz"),
        "cpp_rows_correct_utc_sha256": sha256(CPP_DIR / "rows_correct_utc.csv"),
    }
    (OUT / "independent_rescore_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    # Human-readable summary table with confidence intervals for planned contrasts.
    lines = [
        "# 湖北三站云物理输入四臂GHI结果（保存预测独立复算）",
        "",
        "范围：湖北牛首、王寨、米庄；共同验证集；单帧站点3×3同期GHI重建；三种子；总体支持不由CPP覆盖筛选。",
        "CPP站点中心COT/CLP仅用于分层诊断，是卫星检索参考，不是独立真值。训练集和验证集以外载荷未读取。",
        "",
        "## Overall validation metrics",
        "",
        "| Arm | Equal-station × seed RMSE (W/m²) | Across-cell SD | MAE | Bias | RMSE gain vs A |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in aggregate_df.itertuples():
        lines.append(f"| {r.arm} | {r.equal_station_seed_rmse:.3f} | {r.sd_station_seed_rmse:.3f} | {r.equal_station_seed_mae:.3f} | {r.equal_station_seed_bias:+.3f} | {r.gain_vs_A_percent:+.2f}% |")
    lines += [
        "",
        "RMSE汇总先逐站、逐seed计算RMSE，再对三站和三seed等权平均；括号中的SD是9个站点×seed单元的描述性标准差，不是抽样标准误。保存的行级预测为float64，指标由本脚本重新计算。",
        "",
        "## Paired two-day date-block bootstrap",
        "",
        "正的gain表示候选组RMSE低于参照组。区间基于10个连续两日验证日期块、20,000次配对重抽样。",
        "",
        "| Stratum | Candidate − reference | RMSE delta (W/m²), 95% CI | Relative gain, 95% CI | Blocks |",
        "|---|---|---:|---:|---:|",
    ]
    for r in boot_df.itertuples():
        if r.n_boot == 0 or r.delta_rmse_point is None:
            continue
        dl, du = r.delta_rmse_ci95
        gl, gu = r.gain_percent_ci95
        lines.append(f"| {r.stratum} | {r.candidate} − {r.ref} | {r.delta_rmse_point:+.3f} [{dl:+.3f}, {du:+.3f}] | {r.gain_percent_point:+.2f}% [{gl:+.2f}, {gu:+.2f}] | {r.n_blocks} |")
    lines += [
        "",
        "## CPP COT/CLP strata",
        "",
        "| Stratum | Arm | N unique station-time | Blocks | Equal-station × seed RMSE (W/m²) | MAE | Bias |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in strata_df.itertuples():
        lines.append(f"| {r.stratum} | {r.arm} | {r.unique_station_time_n} | {r.date_blocks} | {r.rmse_equal_station_seed:.3f} | {r.mae_equal_station_seed:.3f} | {r.bias_equal_station_seed:+.3f} |")
    lines += [
        "",
        "## CPP reference coverage by validation month",
        "",
        "Counts are unique station-time records. Months with no valid CPP center pixel remain in the table as zero-coverage months; they remain included in the all-validation GHI score.",
        "",
        "| BJT month | GHI val station-time N | Unique target times | CPP COT center valid N | CPP CLP center valid N | Val days | Two-day blocks |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in monthly_cov.itertuples():
        lines.append(f"| {r.month_bjt} | {r.ghi_validation_station_time_n} | {r.ghi_validation_unique_time_n} | {r.cpp_cot_center_valid_n} | {r.cpp_clp_center_valid_n} | {r.n_validation_days} | {r.n_two_day_blocks} |")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "This experiment measures same-time GHI reconstruction from the input AGRI frame. It does not measure future GHI forecasting, and it does not establish that an auxiliary field is accurate in an independent sense. Per-COT strata are restricted to the CPP reference-valid station-center support; the all-validation result retains every common GHI validation row.",
        "",
        "The saved checkpoint for each station/arm/seed was selected using this same validation set. The date-block interval is conditional on those selected checkpoints; it is not an independent-test interval and does not include model-selection uncertainty.",
        "",
        "Test and buffer payloads read: 0.",
    ]
    # Compact station-specific COT-layer comparisons, averaged over seeds.
    station_section = [
        "",
        "## Station-specific CPP COT strata",
        "",
        "| CPP station-center COT stratum | Station | Arm | N/seed | RMSE (W/m²), mean over seeds | Seed SD |",
        "|---|---|---|---:|---:|---:|",
    ] + [
        f"| {r.stratum} | {r.station} | {r.arm} | {r.n_per_seed} | {r.rmse_mean_seeds:.3f} | {r.rmse_sd_seeds:.3f} |"
        for r in strata_station_df.itertuples()
        if r.stratum.startswith("CPP_COT_")
    ]
    boundary_idx = lines.index("## Interpretation boundary")
    lines[boundary_idx:boundary_idx] = station_section
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "out_dir": str(OUT),
        "n_val_station_time_keys": n_reference_keys,
        "rows_per_station": {s: int(rows_val.station.eq(s).sum()) for s in STATIONS},
        "validation_dates": dates,
        "n_two_day_blocks": len(block_ids),
        "overall": aggregate_df.to_dict(orient="records"),
        "main_contrasts": boot_df.loc[boot_df["stratum"].eq("all_validation")].to_dict(orient="records"),
        "cpp_strata_n": strata_df.loc[strata_df.arm.eq("A"), ["stratum", "unique_station_time_n", "date_blocks", "stations_present"]].to_dict(orient="records"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
