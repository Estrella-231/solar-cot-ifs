#!/usr/bin/env python3
"""Paired, date-block validation summary for Hubei FA/FAC forecast-GHI heads.

Reads only the three saved validation prediction bundles and their independent
audit receipts. It does not open train tensors, test payloads, or buffer data.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SEEDS = (42, 43, 44)
STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
ARMS = ("FA", "FAC")
EXPECTED_N = {"Niushou": 2316, "Wangzhai": 2112, "Mizhuang": 2295}
LEADS = tuple(range(15, 241, 15))


def load_bundle(path: Path):
    with np.load(path, allow_pickle=False) as z:
        station = z["station"].astype(str)
        init = z["init_time_bjt"].astype(str)
        target = z["target_time_bjt"].astype(str)
        lead = z["lead_minutes"].astype(np.int32)
        truth = np.asarray(z["truth_ghi_wm2"], dtype=np.float64)
        pred = np.asarray(z["pred_ghi_wm2"], dtype=np.float64)
        clear = np.asarray(z["clear_sky_ghi_wm2"], dtype=np.float64)
    keys = list(zip(station.tolist(), init.tolist(), target.tolist(), lead.tolist()))
    if len(keys) != len(set(keys)):
        raise AssertionError(f"duplicate keys: {path}")
    if not np.isin(lead, LEADS).all():
        raise AssertionError(f"invalid leads: {path}")
    for i, t, l in zip(init, target, lead):
        if dt.datetime.strptime(t, "%Y%m%d%H%M") - dt.datetime.strptime(i, "%Y%m%d%H%M") != dt.timedelta(minutes=int(l)):
            raise AssertionError(f"init/target/lead mismatch: {path}")
    if not (np.isfinite(truth).all() and np.isfinite(pred).all() and np.isfinite(clear).all()):
        raise AssertionError(f"nonfinite values: {path}")
    days = np.asarray([t[:8] for t in target])
    return {"keys": keys, "station": station, "target": target, "lead": lead,
            "truth": truth, "pred": pred, "clear": clear, "days": days}


def metrics(truth: np.ndarray, pred: np.ndarray, weights: np.ndarray | None = None):
    e = pred - truth
    if weights is None:
        return float(np.sqrt(np.mean(e * e))), float(np.mean(np.abs(e)))
    den = float(weights.sum())
    if den <= 0:
        return float("nan"), float("nan")
    return float(np.sqrt(np.sum(weights * e * e) / den)), float(np.sum(weights * np.abs(e)) / den)


def date_bootstrap(data, dates, n_boot: int, seed: int, block_days: int):
    """Common target-date resampling weights, shared by stations/seeds/arms."""
    rng = np.random.default_rng(seed)
    date_to_i = {d: i for i, d in enumerate(dates)}
    if block_days == 1:
        candidate_blocks = [[i] for i in range(len(dates))]
    elif block_days == 2:
        candidate_blocks = []
        for i, d in enumerate(dates):
            next_day = (dt.datetime.strptime(d, "%Y%m%d").date() + dt.timedelta(days=1)).strftime("%Y%m%d")
            if next_day in date_to_i:
                candidate_blocks.append([i, date_to_i[next_day]])
        if not candidate_blocks:
            raise AssertionError("no consecutive two-day validation blocks available")
    else:
        raise ValueError("only 1-day and 2-day blocks are supported")
    row_day_i = {}
    for s in SEEDS:
        for st in STATIONS:
            for arm in ARMS:
                d = data[s][st][arm]
                row_day_i[(s, st, arm)] = np.asarray([date_to_i[x] for x in d["days"]], dtype=np.int32)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        counts = np.zeros(len(dates), dtype=np.float64)
        n_draws = len(dates) if block_days == 1 else int(np.ceil(len(dates) / 2))
        block_choices = rng.integers(0, len(candidate_blocks), size=n_draws)
        for choice in block_choices:
            counts[candidate_blocks[int(choice)]] += 1.0
        by_station = []
        for st in STATIONS:
            seed_diffs = []
            for s in SEEDS:
                rmse_by_arm = {}
                for arm in ARMS:
                    d = data[s][st][arm]
                    w = counts[row_day_i[(s, st, arm)]]
                    rmse_by_arm[arm] = metrics(d["truth"], d["pred"], w)[0]
                seed_diffs.append(rmse_by_arm["FAC"] - rmse_by_arm["FA"])
            by_station.append(float(np.mean(seed_diffs)))
        deltas[b] = float(np.mean(by_station))
    return {
        "n_boot": n_boot,
        "seed": seed,
        "block_calendar_days": block_days,
        "percentile_95_ci_fac_minus_fa_wm2": [float(np.quantile(deltas, .025)), float(np.quantile(deltas, .975))],
        "bootstrap_mean_fac_minus_fa_wm2": float(np.mean(deltas)),
        "probability_fac_rmse_lower_in_bootstrap": float(np.mean(deltas < 0)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audit-root", type=Path, required=True,
                   help="parent directory containing seed audit directories")
    p.add_argument("--audit-dir-template", default="forecast_ghi_seed{seed}_20261017")
    p.add_argument("--run-dir-template", default="forecast_FA_FAC_seed{seed}")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    a = p.parse_args()

    data = {}
    hashes = {}
    provenance = {}
    for seed in SEEDS:
        sd = a.audit_root / a.audit_dir_template.format(seed=seed)
        audit = json.loads((sd / f"seed{seed}_validation_audit.json").read_text(encoding="utf-8"))
        if audit["state"] != "HUBEI_FORECAST_GHI_SEED_VALIDATION_AUDIT_PASS_TEST_CLOSED" or audit["test_payloads_read"] != 0:
            raise AssertionError(f"seed {seed} audit not passed/test closed")
        hashes[str(seed)] = audit["run_metadata_sha256"]
        run = sd / a.run_dir_template.format(seed=seed)
        run_metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
        if run_metadata.get("state") != "HUBEI_GHI_HEADS_SEED_COMPLETE" or run_metadata.get("seed") != seed:
            raise AssertionError(f"run metadata incomplete or seed mismatch: {seed}")
        if run_metadata.get("test_payloads_read") != 0:
            raise AssertionError(f"test/buffer gate not closed: {seed}")
        if run_metadata.get("pack_sha256") != audit.get("pack_report_sha256"):
            raise AssertionError(f"audit and run pack hashes differ: {seed}")
        provenance[str(seed)] = {
            "trainer_sha256": run_metadata.get("trainer_sha256"),
            "normalization_contract_sha256": run_metadata.get("normalization_contract_sha256"),
            "pack_sha256": run_metadata.get("pack_sha256"),
            "run_metadata_sha256": audit["run_metadata_sha256"],
        }
        data[seed] = {}
        for st in STATIONS:
            data[seed][st] = {}
            for arm in ARMS:
                bundle = run / st / f"{arm}_validation_predictions_with_keys.npz"
                data[seed][st][arm] = load_bundle(bundle)
                d = data[seed][st][arm]
                if len(d["keys"]) != EXPECTED_N[st]:
                    raise AssertionError(f"unexpected N: seed={seed} station={st} arm={arm}")
            fa, fac = data[seed][st]["FA"], data[seed][st]["FAC"]
            if (fa["keys"] != fac["keys"] or not np.array_equal(fa["truth"], fac["truth"])
                    or not np.array_equal(fa["clear"], fac["clear"])):
                raise AssertionError(f"FA/FAC support/truth/clear-sky differ: {seed}/{st}")

    # Verify common validation support and truth across independent seeds.
    for st in STATIONS:
        ref = data[SEEDS[0]][st]["FA"]
        for seed in SEEDS[1:]:
            cur = data[seed][st]["FA"]
            if (ref["keys"] != cur["keys"] or not np.array_equal(ref["truth"], cur["truth"])
                    or not np.array_equal(ref["clear"], cur["clear"])):
                raise AssertionError(f"seed support/truth/clear-sky differ: {seed}/{st}")

    rows = []
    lead_rows = []
    for st in STATIONS:
        for seed in SEEDS:
            rec = {"station": st, "seed": seed}
            for arm in ARMS:
                d = data[seed][st][arm]
                rmse, mae = metrics(d["truth"], d["pred"])
                rec[f"{arm}_rmse_wm2"] = rmse
                rec[f"{arm}_mae_wm2"] = mae
                for lead in LEADS:
                    m = d["lead"] == lead
                    if m.any():
                        lr, lm = metrics(d["truth"][m], d["pred"][m])
                    else:
                        lr, lm = float("nan"), float("nan")
                    lead_rows.append({"station": st, "seed": seed, "arm": arm,
                                      "lead_minutes": lead, "n": int(m.sum()),
                                      "rmse_wm2": lr, "mae_wm2": lm})
            rec["FAC_minus_FA_rmse_wm2"] = rec["FAC_rmse_wm2"] - rec["FA_rmse_wm2"]
            rec["FAC_minus_FA_mae_wm2"] = rec["FAC_mae_wm2"] - rec["FA_mae_wm2"]
            rows.append(rec)

    per_lead_summary = []
    for lead in LEADS:
        lead_rec = {"lead_minutes": lead}
        arm_mean = {}
        for arm in ARMS:
            vals = [x["rmse_wm2"] for x in lead_rows if x["lead_minutes"] == lead and x["arm"] == arm]
            arm_mean[arm] = float(np.mean(vals))
            lead_rec[f"{arm}_rmse_station_seed_equal_mean_wm2"] = arm_mean[arm]
        lead_rec["FAC_minus_FA_rmse_wm2"] = arm_mean["FAC"] - arm_mean["FA"]
        per_seed_lead_delta = []
        for seed in SEEDS:
            fa_values = [x["rmse_wm2"] for x in lead_rows if x["lead_minutes"] == lead and x["arm"] == "FA" and x["seed"] == seed]
            fac_values = [x["rmse_wm2"] for x in lead_rows if x["lead_minutes"] == lead and x["arm"] == "FAC" and x["seed"] == seed]
            per_seed_lead_delta.append(float(np.mean(fac_values) - np.mean(fa_values)))
        lead_rec["seeds_fac_lower_rmse"] = int(np.sum(np.asarray(per_seed_lead_delta) < 0))
        lead_rec["per_seed_fac_minus_fa_wm2"] = per_seed_lead_delta
        per_lead_summary.append(lead_rec)

    # Main descriptive estimand: average station RMSE equally within each seed,
    # then average the three seeds equally. Negative FAC-FA favors COT.
    per_seed = {}
    for seed in SEEDS:
        by_arm = {}
        for arm in ARMS:
            by_arm[arm] = float(np.mean([next(x for x in rows if x["seed"] == seed and x["station"] == st)[f"{arm}_rmse_wm2"] for st in STATIONS]))
        per_seed[str(seed)] = {"station_equal_fa_rmse_wm2": by_arm["FA"],
                               "station_equal_fac_rmse_wm2": by_arm["FAC"],
                               "fac_minus_fa_rmse_wm2": by_arm["FAC"] - by_arm["FA"]}
    seed_deltas = np.asarray([per_seed[str(s)]["fac_minus_fa_rmse_wm2"] for s in SEEDS])
    site_summary = {}
    for st in STATIONS:
        site_summary[st] = {}
        for arm in ARMS:
            vals = np.asarray([next(x for x in rows if x["seed"] == s and x["station"] == st)[f"{arm}_rmse_wm2"] for s in SEEDS])
            site_summary[st][arm] = {"rmse_mean_wm2": float(vals.mean()), "rmse_sd_across_seeds_wm2": float(vals.std(ddof=1))}
        diffs = np.asarray([next(x for x in rows if x["seed"] == s and x["station"] == st)["FAC_minus_FA_rmse_wm2"] for s in SEEDS])
        site_summary[st]["FAC_minus_FA"] = {"mean_wm2": float(diffs.mean()), "sd_across_seeds_wm2": float(diffs.std(ddof=1)), "seeds_improved": int(np.sum(diffs < 0))}

    dates = sorted({d for st in STATIONS for arm in ARMS for s in SEEDS for d in data[s][st][arm]["days"].tolist()})
    ci_1d = date_bootstrap(data, dates, a.bootstrap, 20261017, 1)
    ci_2d = date_bootstrap(data, dates, a.bootstrap, 20261018, 2)
    overall_delta = float(seed_deltas.mean())
    baseline = float(np.mean([per_seed[str(s)]["station_equal_fa_rmse_wm2"] for s in SEEDS]))
    pack_hashes = {v["pack_sha256"] for v in provenance.values()}
    trainer_hashes = {v["trainer_sha256"] for v in provenance.values()}
    norm_hashes = {v["normalization_contract_sha256"] for v in provenance.values()}
    if len(pack_hashes) != 1 or len(trainer_hashes) != 1 or len(norm_hashes) != 1:
        raise AssertionError("three seeds do not share one pack, trainer, and normalization contract")
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = {
        "state": "HUBEI_FORECAST_GHI_THREE_SEED_VALIDATION_SUMMARY_TEST_CLOSED",
        "scope": "validation only; model selection evidence, not independent test or paper-confirmatory result",
        "pack_sha256": next(iter(pack_hashes)),
        "seed_run_metadata_sha256": hashes,
        "seed_provenance": provenance,
        "analysis_script_sha256": script_sha,
        "analysis_input_templates": {"audit_dir": a.audit_dir_template, "run_dir": a.run_dir_template},
        "test_payloads_read": 0,
        "support": {st: {"n_per_seed_per_arm": EXPECTED_N[st]} for st in STATIONS},
        "target_dates": dates,
        "n_target_dates": len(dates),
        "primary_estimand": "equal mean of station RMSEs within each seed, then equal mean across three seeds; paired validation support",
        "seed_table": per_seed,
        "station_table": site_summary,
        "main_result": {
            "fa_station_equal_mean_rmse_wm2": baseline,
            "fac_station_equal_mean_rmse_wm2": baseline + overall_delta,
            "fac_minus_fa_rmse_wm2": overall_delta,
            "relative_rmse_reduction_percent_positive_favors_fac": float(-100.0 * overall_delta / baseline),
            "seeds_with_negative_delta": int(np.sum(seed_deltas < 0)),
            "date_block_bootstrap_1day": ci_1d,
            "moving_2day_block_sensitivity": ci_2d,
        },
        "raw_station_seed_table": rows,
        "per_lead_table": lead_rows,
        "per_lead_equal_station_seed_summary": per_lead_summary,
        "key_findings": [
            f"Across the three seeds the station-equal FAC-minus-FA RMSE difference is {overall_delta:.3f} W/m2; negative values favor FAC.",
            f"FAC reduces RMSE in {int(np.sum(seed_deltas < 0))}/3 seed-level paired aggregates; all three seeds use the same validation truth/support.",
            "These are validation results used for model selection; keep the independent test closed until the complete chain, analysis code, and figures are frozen.",
        ],
        "next_step": "Freeze this validation summary and analysis code, complete forecast-chain provenance/case diagnostics, then perform one-time held-out test evaluation under the registered split; do not retune from test.",
    }
    a.output_dir.mkdir(parents=True, exist_ok=True)
    (a.output_dir / "three_seed_validation_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, content in (("three_seed_station_seed_metrics.csv", rows), ("three_seed_per_lead_metrics.csv", lead_rows)):
        with (a.output_dir / name).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(content[0].keys()))
            w.writeheader()
            w.writerows(content)
    print(json.dumps(report["main_result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
