#!/usr/bin/env python3
"""Independent FP64 audit and paired validation bootstrap for Hubei A/AC/AP/AL.

Reads only the frozen train/validation pack and saved validation predictions.
It never opens or decodes test/buffer payloads.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
STATIONS = ("Niushou", "Wangzhai", "Mizhuang")
SEEDS = (42, 43, 44)
ARMS = ("A", "AC", "AP", "AL")
N_BOOT = 2000
BOOT_SEED = 20261013


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    e = p - y
    sst = np.sum((y - np.mean(y)) ** 2, dtype=np.float64)
    return {
        "n": int(y.size),
        "rmse_wm2": float(np.sqrt(np.mean(e * e, dtype=np.float64))),
        "mae_wm2": float(np.mean(np.abs(e), dtype=np.float64)),
        "bias_wm2": float(np.mean(e, dtype=np.float64)),
        "r2": float(1.0 - np.sum(e * e, dtype=np.float64) / sst) if sst > 0 else float("nan"),
    }


def load_predictions(pack: pd.DataFrame, baseline_root: Path | None = None) -> tuple[dict, dict, dict]:
    """Load keyed predictions and align every arm to the common pack rows.

    The original baseline_runs directory contains an earlier checkpoint set.
    The current same-time comparison must use the normreload runs named by the
    2026-10-17 cold-start receipt; accepting positional NPYs here would allow a
    reordered prediction vector to be scored against unrelated timestamps.
    """
    experiment_root = ROOT.parent
    if baseline_root is None:
        baseline_root = experiment_root / "audit_receipts" / "same_time_normreload_runs_20261017"
    coldstart_path = experiment_root / "audit_receipts" / "coldstart_normreload_20261017_v3" / "same_time_coldstart_audit.json"
    coldstart = json.loads(coldstart_path.read_text(encoding="utf-8"))
    if coldstart.get("state") != "HUBEI_CHECKPOINT_COLDSTART_NORMRELOAD_AUDIT_PASS_TEST_CLOSED":
        raise AssertionError("normreload cold-start audit is not passed")
    if coldstart.get("test_payloads_read") != 0 or coldstart.get("buffer_payloads_read") != 0:
        raise AssertionError("normreload cold-start audit did not keep test/buffer closed")
    pack_report_path = ROOT / "pack" / "pack_report.json"
    feature_report_path = ROOT / "AL_FEATURE_REPORT.json"
    if sha256(pack_report_path) != coldstart.get("pack_report_sha256"):
        raise AssertionError("local AL pack differs from current normreload cold-start pack; refusing cross-pack comparison")
    pack_report = json.loads(pack_report_path.read_text(encoding="utf-8"))
    expected_split_sha256 = pack_report.get("split_sha256")
    feature_report = json.loads(feature_report_path.read_text(encoding="utf-8"))
    if feature_report.get("pack_report_sha256") != coldstart.get("pack_report_sha256"):
        raise AssertionError("AL feature report does not bind the current pack")
    pack_rows = pack.reset_index(drop=True)
    pack_val = pack_rows.loc[pack_rows["split"].eq("val")].copy()
    if set(pack["split"].unique()) - {"train", "val"}:
        raise AssertionError("pack includes split other than train/val")
    preds: dict[tuple[int, str, str], np.ndarray] = {}
    truths: dict[tuple[str, str], np.ndarray] = {}
    times: dict[tuple[str, str], np.ndarray] = {}

    for station in STATIONS:
        sv = pack_val.loc[pack_val.station.eq(station)].reset_index(drop=True)
        if sv.empty:
            raise AssertionError(f"no validation rows for {station}")
        if sv.time_bjt.astype(str).duplicated().any():
            raise AssertionError(f"duplicate station/time in pack for {station}")
        tkey = (station, "truth")
        truths[tkey] = sv.ghi_wm2.to_numpy(np.float64)
        times[(station, "dates")] = sv.time_bjt.astype(str).str.slice(0, 8).to_numpy()
        pack_times = sv.time_bjt.astype(str).to_numpy()
        expected_keys = list(zip([station] * len(pack_times), pack_times))

        for seed in SEEDS:
            run = baseline_root / f"same_time_seed{seed}_normreload_v1"
            run_meta_path = run / "run_metadata.json"
            run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
            provenance = coldstart["run_provenance"][str(seed)]
            if sha256(run_meta_path) != provenance["run_metadata_sha256"]:
                raise AssertionError(f"run metadata hash differs from cold-start receipt: seed{seed}")
            if run_meta.get("pack_sha256") != coldstart.get("pack_report_sha256"):
                raise AssertionError(f"run pack differs from cold-start audited pack: seed{seed}")
            if run_meta.get("split_sha256") != expected_split_sha256:
                raise AssertionError(f"baseline split differs from AL pack: seed{seed}")
            if run_meta.get("split_sha256") != pack_rows.attrs.get("split_sha256", run_meta.get("split_sha256")):
                # The full split hash is checked below against the AL pack report;
                # this branch only rejects a missing or malformed run value.
                if not run_meta.get("split_sha256"):
                    raise AssertionError(f"missing split hash: seed{seed}")
            base = run / station
            for arm in ("A", "AC", "AP"):
                bundle_path = base / f"{arm}_validation_predictions_with_keys.npz"
                bundle_sha = sha256(bundle_path)
                receipt_rows = [r for r in coldstart["results"] if r["seed"] == seed and r["station"] == station and r["arm"] == arm]
                if len(receipt_rows) != 1 or bundle_sha != receipt_rows[0]["prediction_bundle_sha256"]:
                    raise AssertionError(f"keyed {arm} bundle not bound by cold-start receipt: {station}/seed{seed}")
                with np.load(bundle_path, allow_pickle=False) as z:
                    keys = list(zip(z["station"].astype(str).tolist(), z["target_time_bjt"].astype(str).tolist()))
                    p = np.asarray(z["pred_ghi_wm2"], dtype=np.float64).reshape(-1)
                    y = np.asarray(z["truth_ghi_wm2"], dtype=np.float64).reshape(-1)
                    clear = np.asarray(z["clear_sky_ghi_wm2"], dtype=np.float64).reshape(-1)
                if keys != expected_keys:
                    raise AssertionError(f"{arm} keyed validation rows differ from pack order: {station}/seed{seed}")
                if (not np.isfinite(p).all() or not np.array_equal(y, truths[tkey])
                        or not np.array_equal(clear, sv.clear_sky_wm2.to_numpy(np.float64))):
                    raise AssertionError(f"{arm} keyed prediction/truth/clear-sky mismatch: {station}/seed{seed}")
                pred_path = base / f"{arm}_val_prediction_ghi_wm2.npy"
                if not np.array_equal(p, np.asarray(np.load(pred_path, allow_pickle=False), dtype=np.float64).reshape(-1)):
                    raise AssertionError(f"{arm} keyed bundle differs from legacy prediction vector: {station}/seed{seed}")
                meta = json.loads((base / f"{arm}_validation.json").read_text(encoding="utf-8"))
                if int(meta["val_n"]) != len(sv) or int(meta["seed"]) != seed:
                    raise AssertionError(f"{arm} metadata count/seed mismatch: {station}/{seed}")
                m = metrics(truths[tkey], p)
                if abs(m["rmse_wm2"] - float(meta["rmse_wm2"])) > 0.03:
                    raise AssertionError(f"{arm} RMSE does not reproduce metadata: {station}/{seed}")
                preds[(seed, station, arm)] = p

            al_dir = ROOT / "runs_maskfix_v2" / f"seed{seed}" / station
            z = np.load(al_dir / "AL_validation_predictions_with_keys.npz", allow_pickle=False)
            if not np.all(z["station"].astype(str) == station):
                raise AssertionError(f"AL station-key mismatch: {station}/{seed}")
            al_times = z["target_time_bjt"].astype(str)
            if not np.array_equal(al_times, pack_times):
                raise AssertionError(f"AL target key order mismatch with pack: {station}/{seed}")
            # AL NPZ stores float32 labels; the common pack remains the FP64
            # recomputation reference. This tolerance covers float32 roundoff only.
            if not np.allclose(z["truth_ghi_wm2"].astype(np.float64), truths[tkey], rtol=0, atol=1e-4):
                raise AssertionError(f"AL truth mismatch with common pack: {station}/{seed}")
            p_al = z["pred_ghi_wm2"].astype(np.float64).reshape(-1)
            if not np.isfinite(p_al).all() or len(p_al) != len(sv):
                raise AssertionError(f"bad AL prediction length/finite for {station}/{seed}")
            saved = np.asarray(np.load(al_dir / "AL_val_prediction_ghi_wm2.npy", allow_pickle=False), dtype=np.float64).reshape(-1)
            if not np.allclose(saved, p_al, rtol=0, atol=1e-5):
                raise AssertionError(f"AL paired prediction files differ: {station}/{seed}")
            meta = json.loads((al_dir / "AL_validation.json").read_text(encoding="utf-8"))
            if int(meta["val_n"]) != len(sv) or int(meta["seed"]) != seed:
                raise AssertionError(f"AL metadata count/seed mismatch: {station}/{seed}")
            if meta.get("station") != station or meta.get("arm") != "AL" or meta.get("test_payloads_read") != 0:
                raise AssertionError(f"AL metadata station/arm/test boundary mismatch: {station}/{seed}")
            if sha256(al_dir / "AL_validation_predictions_with_keys.npz") != meta.get("bundle_sha256"):
                raise AssertionError(f"AL bundle hash differs from station metadata: {station}/{seed}")
            if sha256(al_dir / "AL_best.pt") != meta.get("checkpoint_sha256"):
                raise AssertionError(f"AL checkpoint hash differs from station metadata: {station}/{seed}")
            if sha256(al_dir / "AL_val_prediction_ghi_wm2.npy") != meta.get("prediction_sha256"):
                raise AssertionError(f"AL prediction file hash differs from metadata: {station}/{seed}")
            if sha256(al_dir / "AL_validation_predictions_with_keys.npz") != meta.get("bundle_sha256"):
                raise AssertionError(f"AL keyed bundle hash differs from metadata: {station}/{seed}")
            m = metrics(truths[tkey], p_al)
            if abs(m["rmse_wm2"] - float(meta["rmse_wm2"])) > 0.03:
                raise AssertionError(f"AL RMSE does not reproduce metadata: {station}/{seed}")
            preds[(seed, station, "AL")] = p_al

            # All 4 interventions share the same exact truth vector and timestamp sequence.
            for arm in ARMS:
                if len(preds[(seed, station, arm)]) != len(pack_times):
                    raise AssertionError(f"support length differs: {arm}/{station}/{seed}")
    return preds, truths, times


def point_summary(preds: dict, truths: dict) -> tuple[dict, dict]:
    cells = {}
    for arm in ARMS:
        per_seed_station = {}
        for seed in SEEDS:
            for station in STATIONS:
                m = metrics(truths[(station, "truth")], preds[(seed, station, arm)])
                per_seed_station[f"{seed}/{station}"] = m
        cells[arm] = per_seed_station
    means = {}
    for arm in ARMS:
        means[arm] = {
            key: float(np.mean([cells[arm][f"{s}/{st}"][key] for s in SEEDS for st in STATIONS]))
            for key in ("rmse_wm2", "mae_wm2", "bias_wm2", "r2")
        }
    deltas = {f"{a}_minus_{b}": means[a]["rmse_wm2"] - means[b]["rmse_wm2"]
              for a, b in (("AC", "A"), ("AC", "AP"), ("AL", "A"), ("AL", "AP"), ("AC", "AL"))}
    return {"cells": cells, "mean_station_equal_seed_equal": means, "rmse_deltas_wm2": deltas}, cells


def bootstrap(preds: dict, truths: dict, dates_by_station: dict, block_pairs: list[list[str]]) -> dict:
    # For each station/seed/arm/date retain sufficient statistics so date resampling
    # recomputes the specified within-cell RMSE before station/seed equal weighting.
    sq = {}
    count = {}
    date_set = sorted({d for ds in dates_by_station.values() for d in ds})
    for seed in SEEDS:
        for station in STATIONS:
            dates = dates_by_station[station]
            y = truths[(station, "truth")]
            for arm in ARMS:
                e2 = (preds[(seed, station, arm)] - y) ** 2
                for d in date_set:
                    mask = dates == d
                    sq[(seed, station, arm, d)] = float(np.sum(e2[mask], dtype=np.float64))
                    count[(station, d)] = int(np.sum(mask))

    # Match the frozen validation date set and the 11 contiguous two-day val blocks.
    if sorted(date_set) != sorted(d for pair in block_pairs for d in pair):
        raise AssertionError("frozen validation date blocks do not match prediction dates")

    def estimate(weight: dict[str, int]) -> dict[str, float]:
        out = {}
        for arm in ARMS:
            vals = []
            for seed in SEEDS:
                for station in STATIONS:
                    ss = sum(weight.get(d, 0) * sq[(seed, station, arm, d)] for d in date_set)
                    nn = sum(weight.get(d, 0) * count[(station, d)] for d in date_set)
                    if nn <= 0:
                        raise AssertionError("bootstrap sample has no support")
                    vals.append(np.sqrt(ss / nn))
            out[arm] = float(np.mean(vals))
        return out

    rng = np.random.default_rng(BOOT_SEED)
    date_draws = np.empty((N_BOOT, len(ARMS)), dtype=np.float64)
    block_draws = np.empty_like(date_draws)
    for i in range(N_BOOT):
        drawn_dates = rng.choice(date_set, size=len(date_set), replace=True)
        weights = {d: int(np.sum(drawn_dates == d)) for d in date_set}
        est = estimate(weights)
        date_draws[i] = [est[a] for a in ARMS]

        chosen = rng.integers(0, len(block_pairs), size=len(block_pairs))
        w2 = {d: 0 for d in date_set}
        for j in chosen:
            for d in block_pairs[int(j)]:
                w2[d] += 1
        est2 = estimate(w2)
        block_draws[i] = [est2[a] for a in ARMS]

    deltas = (("AC", "A"), ("AC", "AP"), ("AL", "A"), ("AL", "AP"), ("AC", "AL"))
    result = {}
    for mode, draws in (("single_date", date_draws), ("contiguous_2day_blocks", block_draws)):
        result[mode] = {}
        for a, b in deltas:
            delta = draws[:, ARMS.index(a)] - draws[:, ARMS.index(b)]
            result[mode][f"{a}_minus_{b}"] = {
                "point_wm2": float(np.mean([metrics(truths[(st, "truth")], preds[(s, st, a)])["rmse_wm2"] - metrics(truths[(st, "truth")], preds[(s, st, b)])["rmse_wm2"] for s in SEEDS for st in STATIONS])),
                "ci95_percentile_wm2": [float(x) for x in np.quantile(delta, [0.025, 0.975])],
                "fraction_replicates_below_zero": float(np.mean(delta < 0)),
            }
        result[mode]["endpoint_rmse_ci95_wm2"] = {
            arm: [float(x) for x in np.quantile(draws[:, ARMS.index(arm)], [0.025, 0.975])]
            for arm in ARMS
        }
        result[mode]["bootstrap_replicates"] = N_BOOT
        result[mode]["seed"] = BOOT_SEED
    return result


def main() -> None:
    pack_dir = ROOT / "pack"
    report = json.loads((pack_dir / "pack_report.json").read_text(encoding="utf-8"))
    feature_report = json.loads((ROOT / "AL_FEATURE_REPORT.json").read_text(encoding="utf-8"))
    coldstart_path = ROOT.parent / "audit_receipts" / "coldstart_normreload_20261017_v3" / "same_time_coldstart_audit.json"
    coldstart = json.loads(coldstart_path.read_text(encoding="utf-8"))
    if sha256(pack_dir / "pack_report.json") != coldstart.get("pack_report_sha256"):
        raise AssertionError("local pack report is stale relative to current baseline receipt")
    if feature_report.get("pack_report_sha256") != coldstart.get("pack_report_sha256"):
        raise AssertionError("local AL feature report is not for the current normreload pack")
    report_split_sha256 = report.get("split_sha256")
    if report["test_and_buffer_values_read"] != 0 or feature_report["test_payloads_read"] != 0:
        raise AssertionError("test/buffer gate is not closed")
    if feature_report["ghi_labels_read"] is not False or feature_report["cpp_payloads_read"] != 0:
        raise AssertionError("feature bank crossed its declared boundary")
    rows = pd.read_csv(pack_dir / "rows.csv", dtype={"time_bjt": str})
    if not set(rows["split"].unique()).issubset({"train", "val"}):
        raise AssertionError("pack contains non-train/validation rows")
    preds, truths, _ = load_predictions(rows)
    point, cells = point_summary(preds, truths)
    val_rows = rows.loc[rows.split.eq("val")]
    val_dates = sorted(val_rows.time_bjt.astype(str).str.slice(0, 8).unique())
    dates_by_station = {
        st: val_rows.loc[val_rows.station.eq(st), "time_bjt"].astype(str).str.slice(0, 8).to_numpy()
        for st in STATIONS
    }
    assignment = pd.read_csv(ROOT.parent / "frozen_split_20261015_v1" / "day_assignment.csv", dtype={"date_BJT": str})
    vd = assignment.loc[assignment.split.eq("val"), "date_BJT"].str.replace("-", "", regex=False).tolist()
    block_pairs = [vd[i:i + 2] for i in range(0, len(vd), 2)]
    if any(len(p) != 2 or pd.Timestamp(p[1]) - pd.Timestamp(p[0]) != pd.Timedelta(days=1) for p in block_pairs):
        raise AssertionError("validation dates are not frozen consecutive two-day pairs")
    if sorted(val_dates) != sorted(d for p in block_pairs for d in p):
        raise AssertionError("pack dates differ from frozen validation date mapping")
    boot = bootstrap(preds, truths, dates_by_station, block_pairs)

    metadata = {}
    for seed in SEEDS:
        meta = json.loads((ROOT / "runs_maskfix_v2" / f"seed{seed}" / "run_metadata.json").read_text(encoding="utf-8"))
        if meta["state"] != "HUBEI_GHI_HEAD_AL_SEED_COMPLETE" or meta["test_payloads_read"] != 0:
            raise AssertionError(f"AL run incomplete or test gate violated: seed {seed}")
        if meta["split_sha256"] != report["split_sha256"]:
            raise AssertionError(f"AL split hash mismatch: seed {seed}")
        metadata[str(seed)] = {"state": meta["state"], "gpu": meta["gpu"], "elapsed_s": meta["elapsed_s"],
                               "model_sha256": meta["model_sha256"], "parameter_count": meta["parameter_count"],
                               "test_payloads_read": meta["test_payloads_read"]}

    out = {
        "state": "HUBEI_AL_INDEPENDENT_VALIDATION_RECOMPUTE_COMPLETE_TEST_CLOSED",
        "scope": "validation only; same-time station GHI estimate; not an independent test or forecast result",
        "data_contract": {
            "split_sha256": report["split_sha256"],
            "pack_report_sha256": sha256(pack_dir / "pack_report.json"),
            "feature_report_sha256": sha256(ROOT / "AL_FEATURE_REPORT.json"),
            "stations": list(STATIONS), "seeds": list(SEEDS), "arms": list(ARMS),
            "validation_n": {st: int(np.sum(val_rows.station.eq(st))) for st in STATIONS},
            "validation_dates_n": len(val_dates), "validation_dates": val_dates,
            "test_buffer_values_read": 0,
        },
        "runs": metadata,
        "fp64_recomputed_metrics": point,
        "shared_date_bootstrap": boot,
        "claim_boundary": [
            "All metrics are on the fixed validation split used for model selection; they are not independent confirmation.",
            "The AL arm adds frozen R dec1 hidden features, not COT physical values; AL is a control for whether R local representations can explain the AC increment.",
            "No future lead-time/forecast claim follows from this same-time estimate experiment.",
        ],
    }
    out_path = ROOT / "AL_RESULT_AUDIT_20261017.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# 湖北 AL 结果独立复算（验证集）",
        "",
        "范围：固定验证集上的同期站点 GHI 估计；不代表独立测试或未来预报。test/buffer 有效载荷读取数为 0。",
        "",
        "| 组别 | RMSE (W/m²) | MAE (W/m²) | Bias (W/m²) | 平均站点/种子 R² |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        m = point["mean_station_equal_seed_equal"][arm]
        lines.append(f"| {arm} | {m['rmse_wm2']:.3f} | {m['mae_wm2']:.3f} | {m['bias_wm2']:.3f} | {m['r2']:.4f} |")
    lines += ["", "| 对照差（前者−后者，RMSE；负值有利前者） | 点差 W/m² | 单日 bootstrap 95% CI | 连续两日块 95% CI |", "|---|---:|---:|---:|"]
    for key in ("AC_minus_A", "AC_minus_AP", "AL_minus_A", "AL_minus_AP", "AC_minus_AL"):
        d = point["rmse_deltas_wm2"][key]
        c1 = boot["single_date"][key]["ci95_percentile_wm2"]
        c2 = boot["contiguous_2day_blocks"][key]["ci95_percentile_wm2"]
        lines.append(f"| {key.replace('_minus_', ' − ')} | {d:.3f} | [{c1[0]:.3f}, {c1[1]:.3f}] | [{c2[0]:.3f}, {c2[1]:.3f}] |")
    lines += [
        "",
        "解释边界：A/AC/AP/AL 的结果都来自模型选择用的 validation 日期，bootstrap 只量化该验证支持上的日期抽样变化，不把 validation 变成 test。AL 输入是冻结 R 的局地中间特征，不是 COT 数值，因此只能作为 R 表征控制。",
        "",
        "逐站逐种子 FP64 复算、键对齐、文件与 split 哈希详见同目录 `AL_RESULT_AUDIT_20261017.json`。",
    ]
    (ROOT / "AL_RESULT_AUDIT_20261017.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"state": out["state"], "means": point["mean_station_equal_seed_equal"],
                      "deltas": point["rmse_deltas_wm2"],
                      "single_day_ci": {k: v["ci95_percentile_wm2"] for k, v in boot["single_date"].items() if k.endswith("minus_A") or k.endswith("minus_AP")},
                      "two_day_ci": {k: v["ci95_percentile_wm2"] for k, v in boot["contiguous_2day_blocks"].items() if k.endswith("minus_A") or k.endswith("minus_AP")},
                      "output": str(out_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
