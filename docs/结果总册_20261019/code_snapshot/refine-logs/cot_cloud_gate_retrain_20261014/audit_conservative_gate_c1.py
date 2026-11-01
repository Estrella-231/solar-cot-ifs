#!/usr/bin/env python3
"""Independent FP64 validation recomputation and BJT-date paired bootstrap."""
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PRED_PATH = ROOT / "remote_result_c1_attempt3" / "validation_predictions.npz"
RESULT_PATH = ROOT / "remote_result_c1_attempt3" / "result.json"
SOURCE_PATH = ROOT / "remote_result_c1_attempt3" / "source_audit.json"
OUT_JSON = ROOT / "C1_INDEPENDENT_AUDIT_20261015.json"
OUT_MD = ROOT / "C1_INDEPENDENT_AUDIT_20261015.md"
N_BOOT = 2000
BOOT_SEED = 20261015
CLOUD_COT = 0.1
THICK_COT = 30.0
STATIONS = ("sili", "zhujia")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scores(pred, reference, clp, valid):
    p, y, label = pred[valid], reference[valid], clp[valid]
    e = p.astype(np.float64) - y.astype(np.float64)
    target_cloud, pred_cloud = label > 0, p >= CLOUD_COT
    target_thick, pred_thick = y >= THICK_COT, p >= THICK_COT

    def confusion(pr, gt):
        tp = int(np.count_nonzero(pr & gt))
        fp = int(np.count_nonzero(pr & ~gt))
        fn = int(np.count_nonzero(~pr & gt))
        tn = int(np.count_nonzero(~pr & ~gt))
        return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "POD": tp / (tp + fn) if tp + fn else None,
                "FAR": fp / (tp + fp) if tp + fp else None,
                "CSI": tp / (tp + fp + fn) if tp + fp + fn else None}

    return {"valid_pixels": int(valid.sum()),
            "COT_RMSE": float(np.sqrt(np.mean(e * e))),
            "SSE": float(np.sum(e * e, dtype=np.float64)),
            "cloud": confusion(pred_cloud, target_cloud),
            "thick": confusion(pred_thick, target_thick)}


def daily_sufficient(pred, reference, clp, valid):
    p, y, label = pred[valid], reference[valid], clp[valid]
    e = p.astype(np.float64) - y.astype(np.float64)
    c_gt, c_pr = label > 0, p >= CLOUD_COT
    t_gt, t_pr = y >= THICK_COT, p >= THICK_COT
    return np.asarray([
        float(valid.sum()), float(np.sum(e * e, dtype=np.float64)),
        float(np.count_nonzero(c_pr & c_gt)), float(np.count_nonzero(c_pr & ~c_gt)),
        float(np.count_nonzero(~c_pr & c_gt)),
        float(np.count_nonzero(t_pr & t_gt)), float(np.count_nonzero(t_pr & ~t_gt)),
        float(np.count_nonzero(~t_pr & t_gt)),
    ], dtype=np.float64)


def metric_from_suff(s):
    n, sse, ctp, cfp, cfn, ttp, tfp, tfn = s
    return {"COT_RMSE": np.sqrt(sse / n) if n else np.nan,
            "cloud_POD": ctp / (ctp + cfn) if ctp + cfn else np.nan,
            "cloud_FAR": cfp / (ctp + cfp) if ctp + cfp else np.nan,
            "thick_POD": ttp / (ttp + tfn) if ttp + tfn else np.nan,
            "thick_CSI": ttp / (ttp + tfp + tfn) if ttp + tfp + tfn else np.nan}


def main():
    with np.load(PRED_PATH, allow_pickle=False) as z:
        prob = z["prob_cloud"].astype(np.float64)
        old_prob = z["old_B1_prob_cloud"].astype(np.float64)
        raw = z["raw_cot"].astype(np.float64)
        ref = z["reference_cot"].astype(np.float64)
        clp = z["clp"].astype(np.uint8)
        valid = z["valid"].astype(bool)
        station = z["station"].astype(str)
        utc = z["target_time_utc"].astype(str)
        sample_indices = z["sample_indices"].astype(np.int64)
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    if result["state"] != "COMPLETE_C1_FIXED_TRAINING_AND_VALIDATION" or result["test_payloads_read"] != 0:
        raise ValueError("run completion/test-closure marker")
    if source["R_frozen"] is not True or source["R_validation_source"] != "checkpoint-bound saved validation predictions":
        raise ValueError("frozen R provenance")
    if not (prob.shape == old_prob.shape == raw.shape == ref.shape == clp.shape == valid.shape):
        raise ValueError("prediction array shapes disagree")
    if prob.ndim != 3 or len(station) != len(prob) or len(utc) != len(prob) or len(sample_indices) != len(prob):
        raise ValueError("validation key/array dimensions")
    if len(set(zip(station.tolist(), utc.tolist()))) != len(prob):
        raise ValueError("duplicate station/target keys")
    if not np.isin(station, STATIONS).all() or not np.isin(clp[valid], (0, 1, 2)).all():
        raise ValueError("station or CLP support")
    if not all(np.isfinite(a[valid]).all() for a in (prob, old_prob, raw, ref)):
        raise ValueError("nonfinite values on common support")
    if np.any(raw[valid] < 0) or np.any(ref[valid] < 0) or np.any((prob[valid] < 0) | (prob[valid] > 1)):
        raise ValueError("physical range violation")

    threshold = float(result["threshold"])
    arms = {
        "raw": raw,
        "old_B1_0p5": np.where(old_prob < 0.5, 0.0, raw),
        "full_last_0p5": np.where(prob < 0.5, 0.0, raw),
        "full_last_conservative": np.where(prob < threshold, 0.0, raw),
    }
    rows_by_station = {s: station == s for s in STATIONS}
    point = {}
    daily = {}
    for arm_name, prediction in arms.items():
        by_station = {}
        for s in STATIONS:
            sel = rows_by_station[s]
            by_station[s] = scores(prediction[sel], ref[sel], clp[sel], valid[sel])
        point[arm_name] = {"by_station": by_station,
                           "station_equal_mean": {
                               "COT_RMSE": float(np.mean([by_station[s]["COT_RMSE"] for s in STATIONS])),
                               "cloud_FAR": float(np.mean([by_station[s]["cloud"]["FAR"] for s in STATIONS])),
                           }}

    bjt_dates = []
    for t in utc:
        bjt_dates.append((datetime.strptime(t, "%Y-%m-%d %H:%M:%S") + timedelta(hours=8)).date().isoformat())
    bjt_dates = np.asarray(bjt_dates)
    unique_dates = np.unique(bjt_dates)
    day_position = {d: i for i, d in enumerate(unique_dates)}
    n_dates, n_samples = len(unique_dates), len(prob)
    if n_dates != 46:
        raise ValueError(f"unexpected validation date support: {n_dates}")
    day_counts = np.zeros(n_dates, dtype=np.int64)
    for i, d in enumerate(bjt_dates):
        day_counts[day_position[d]] += 1

    for arm_name, prediction in arms.items():
        daily[arm_name] = {}
        for s in STATIONS:
            sel = rows_by_station[s]
            stats = np.zeros((n_dates, 8), dtype=np.float64)
            for d, pos in day_position.items():
                idx = sel & (bjt_dates == d)
                stats[pos] = daily_sufficient(prediction[idx], ref[idx], clp[idx], valid[idx])
            daily[arm_name][s] = stats

    direct = {}
    for arm_name, comparison_name in (("old_B1_0p5", "old_B1_0p5"),
                                      ("full_last_0p5", "full_last_0p5"),
                                      ("full_last_conservative", "full_last_conservative")):
        stored = result["comparisons"][comparison_name]
        direct[comparison_name] = {}
        for s in STATIONS:
            vals = point[arm_name]["by_station"][s]
            got = stored["by_station"][s]
            direct[comparison_name][s] = {
                "cot_rmse_abs_difference": float(abs(vals["COT_RMSE"] - got["gated"]["COT_RMSE"])),
                "cloud_pod_abs_difference": float(abs(vals["cloud"]["POD"] - got["gated"]["cloud"]["POD"])),
                "cloud_far_abs_difference": float(abs(vals["cloud"]["FAR"] - got["gated"]["cloud"]["FAR"])),
                "thick_pod_abs_difference": float(abs(vals["thick"]["POD"] - got["gated"]["thick"]["POD"])),
                "thick_csi_abs_difference": float(abs(vals["thick"]["CSI"] - got["gated"]["thick"]["CSI"])),
            }
            if max(direct[comparison_name][s].values()) > 1e-10:
                raise ValueError("independent recomputation differs from saved result: " + comparison_name + "/" + s)

    rng = np.random.default_rng(BOOT_SEED)
    boot_values = {name: {} for name in arms if name != "raw"}
    for b in range(N_BOOT):
        draw = np.bincount(rng.integers(0, n_dates, size=n_dates), minlength=n_dates).astype(np.float64)
        boot_metric = {}
        for arm_name in arms:
            by_station = {}
            for s in STATIONS:
                aggregate = draw @ daily[arm_name][s]
                by_station[s] = metric_from_suff(aggregate)
            boot_metric[arm_name] = by_station
        raw_equal_rmse = np.mean([boot_metric["raw"][s]["COT_RMSE"] for s in STATIONS])
        for arm_name in boot_values:
            arm_equal_rmse = np.mean([boot_metric[arm_name][s]["COT_RMSE"] for s in STATIONS])
            for s in STATIONS:
                rec = boot_values[arm_name].setdefault(s, {m: [] for m in (
                    "delta_COT_RMSE", "delta_cloud_POD", "delta_cloud_FAR",
                    "delta_thick_POD", "delta_thick_CSI")})
                rec["delta_COT_RMSE"].append(boot_metric[arm_name][s]["COT_RMSE"] - boot_metric["raw"][s]["COT_RMSE"])
                for metric, output in (("cloud_POD", "delta_cloud_POD"), ("cloud_FAR", "delta_cloud_FAR"),
                                       ("thick_POD", "delta_thick_POD"), ("thick_CSI", "delta_thick_CSI")):
                    rec[output].append(boot_metric[arm_name][s][metric] - boot_metric["raw"][s][metric])
            boot_values[arm_name].setdefault("station_equal_mean", {"delta_COT_RMSE": []})
            boot_values[arm_name]["station_equal_mean"]["delta_COT_RMSE"].append(arm_equal_rmse - raw_equal_rmse)

    intervals = {}
    for arm_name, groups in boot_values.items():
        intervals[arm_name] = {}
        for group, metrics in groups.items():
            intervals[arm_name][group] = {}
            for metric, vals in metrics.items():
                lo, hi = np.quantile(np.asarray(vals, np.float64), [0.025, 0.975])
                intervals[arm_name][group][metric] = [float(lo), float(hi)]

    selected = "full_last_conservative"
    conservative = result["comparisons"][selected]
    guards = {}
    for s in STATIONS:
        raw_m = point["raw"]["by_station"][s]
        gate_m = point[selected]["by_station"][s]
        guards[s] = {
            "cloud_FAR_improved": gate_m["cloud"]["FAR"] < raw_m["cloud"]["FAR"],
            "cloud_POD_drop_le_1pp": gate_m["cloud"]["POD"] - raw_m["cloud"]["POD"] >= -0.01 - 1e-12,
            "thick_POD_drop_le_1pp": gate_m["thick"]["POD"] - raw_m["thick"]["POD"] >= -0.01 - 1e-12,
            "thick_CSI_drop_le_0p01": gate_m["thick"]["CSI"] - raw_m["thick"]["CSI"] >= -0.01 - 1e-12,
        }
        guards[s]["all_protection_guards_pass"] = all(guards[s].values())
    point_delta = conservative["station_equal_mean"]["delta_COT_RMSE"]
    bootstrap_ci = intervals[selected]["station_equal_mean"]["delta_COT_RMSE"]

    # Verify all immutable files listed in the training result manifest.
    output_dir = PRED_PATH.parent
    hash_checks = {}
    for name, expected in result["output_hashes"].items():
        path = output_dir / name
        actual = sha(path)
        hash_checks[name] = {"expected": expected, "actual": actual, "match": actual == expected}
        if actual != expected:
            raise ValueError("training output hash mismatch: " + name)

    report = {
        "state": "PASS_INDEPENDENT_FP64_AND_DATE_BOOTSTRAP",
        "split": "exploratory validation already viewed in B1; not an independent test",
        "samples": int(n_samples), "valid_pixels": int(valid.sum()),
        "bjt_target_dates": int(n_dates), "bootstrap_replicates": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "threshold_frozen_from_train_oof": threshold,
        "test_payloads_read": 0,
        "point_metrics": point,
        "date_cluster_bootstrap_95pct_CI_for_deltas": intervals,
        "independent_recompute_max_abs_diffs_vs_training_json": direct,
        "conservative_gate_point_guards": guards,
        "conservative_gate_station_equal_delta_COT_RMSE": float(point_delta),
        "conservative_gate_station_equal_delta_COT_RMSE_95pct_CI": bootstrap_ci,
        "conservative_gate_RMSE_CI_excludes_zero": bool(bootstrap_ci[1] < 0 or bootstrap_ci[0] > 0),
        "training_output_sha256_checks": hash_checks,
    }
    OUT_JSON.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = [
        "# C1 独立复算与日期簇不确定性",
        "",
        "状态：独立 FP64 指标复算与训练输出哈希检查通过；本结果仍是探索性验证，不是独立 test。",
        "",
        f"固定阈值：`{threshold:.8f}`（仅由训练日期 OOF 冻结）；有效验证像元 {int(valid.sum()):,}，BJT 目标日期 {n_dates}，配对日期 bootstrap {N_BOOT} 次。",
        "",
        "| 对照 | 站点等权 COT RMSE | 相对原 R ΔRMSE | 四里云 POD Δ | 竺家云 POD Δ | 四里云 FAR Δ | 竺家云云 FAR Δ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    raw_equal = point["raw"]["station_equal_mean"]["COT_RMSE"]
    for arm_name, label in (("old_B1_0p5", "旧 B1，p=0.5"), ("full_last_0p5", "新分类器，p=0.5"),
                            ("full_last_conservative", "新分类器，OOF 保守阈值")):
        v = point[arm_name]
        lines.append("| {} | {:.6f} | {:+.6f} | {:+.3f} pp | {:+.3f} pp | {:+.3f} pp | {:+.3f} pp |".format(
            label, v["station_equal_mean"]["COT_RMSE"], v["station_equal_mean"]["COT_RMSE"] - raw_equal,
            100 * (v["by_station"]["sili"]["cloud"]["POD"] - point["raw"]["by_station"]["sili"]["cloud"]["POD"]),
            100 * (v["by_station"]["zhujia"]["cloud"]["POD"] - point["raw"]["by_station"]["zhujia"]["cloud"]["POD"]),
            100 * (v["by_station"]["sili"]["cloud"]["FAR"] - point["raw"]["by_station"]["sili"]["cloud"]["FAR"]),
            100 * (v["by_station"]["zhujia"]["cloud"]["FAR"] - point["raw"]["by_station"]["zhujia"]["cloud"]["FAR"])))
    lines += ["", "| 候选 | Δ站点等权 COT RMSE 的 BJT 日期配对 95% CI |", "|---|---:|"]
    for arm_name, label in (("old_B1_0p5", "旧 B1，p=0.5"), ("full_last_0p5", "新分类器，p=0.5"),
                            ("full_last_conservative", "新分类器，OOF 保守阈值")):
        ci = intervals[arm_name]["station_equal_mean"]["delta_COT_RMSE"]
        lines.append(f"| {label} | [{ci[0]:+.6f}, {ci[1]:+.6f}] |")
    lines += ["", "保守阈值的逐站 FAR、云 POD 和厚云保护条件："]
    for s in STATIONS:
        g = guards[s]
        lines.append(f"- {s}: cloud FAR improved={g['cloud_FAR_improved']}; cloud POD drop ≤1pp={g['cloud_POD_drop_le_1pp']}; thick POD drop ≤1pp={g['thick_POD_drop_le_1pp']}; thick CSI drop ≤0.01={g['thick_CSI_drop_le_0p01']}.")
    lines += ["", f"阈值点估计 ΔRMSE={point_delta:+.9f} W/m²；日期簇 95% CI 为 [{bootstrap_ci[0]:+.6f}, {bootstrap_ci[1]:+.6f}] W/m²。",
              "", "该验证集曾用于 B1 筛选，因此即使区间排除零，也只能作为探索性证据；不能据此声称独立泛化或论文主结果。", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "state", "samples", "valid_pixels", "bjt_target_dates", "bootstrap_replicates",
        "conservative_gate_station_equal_delta_COT_RMSE",
        "conservative_gate_station_equal_delta_COT_RMSE_95pct_CI",
        "conservative_gate_RMSE_CI_excludes_zero", "conservative_gate_point_guards")}, indent=2))


if __name__ == "__main__":
    main()
