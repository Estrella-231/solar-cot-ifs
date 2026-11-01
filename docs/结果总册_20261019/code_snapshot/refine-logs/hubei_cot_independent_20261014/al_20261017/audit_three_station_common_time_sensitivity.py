#!/usr/bin/env python3
"""Paired validation sensitivity restricted to timestamps observed at all stations.

This is a secondary support sensitivity for the same-time GHI head experiment.
It reads only the frozen train/validation pack and saved validation predictions;
test and buffer payloads remain closed.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from audit_al_results import ARMS, BOOT_SEED, N_BOOT, SEEDS, STATIONS, load_predictions


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    pack_dir = ROOT / "pack"
    pack_report = json.loads((pack_dir / "pack_report.json").read_text(encoding="utf-8"))
    feature_report = json.loads((ROOT / "AL_FEATURE_REPORT.json").read_text(encoding="utf-8"))
    if pack_report["test_and_buffer_values_read"] != 0 or feature_report["test_payloads_read"] != 0:
        raise AssertionError("test/buffer gate is not closed")
    rows = pd.read_csv(pack_dir / "rows.csv", dtype={"time_bjt": str})
    preds, truths, _ = load_predictions(rows)
    val = rows.loc[rows["split"].eq("val")].copy()
    val["time_bjt"] = val["time_bjt"].astype(str)
    per_station = {
        st: set(val.loc[val["station"].eq(st), "time_bjt"].tolist())
        for st in STATIONS
    }
    common_times = sorted(set.intersection(*(per_station[st] for st in STATIONS)))
    if not common_times:
        raise AssertionError("no common target timestamps across all stations")
    common_dates = sorted({t[:8] for t in common_times})
    assignment = pd.read_csv(ROOT.parent / "frozen_split_20261015_v1" / "day_assignment.csv", dtype={"date_BJT": str})
    val_dates = assignment.loc[assignment["split"].eq("val"), "date_BJT"].str.replace("-", "", regex=False).tolist()
    blocks = [val_dates[i : i + 2] for i in range(0, len(val_dates), 2)]
    if any(len(b) != 2 or pd.Timestamp(b[1]) - pd.Timestamp(b[0]) != pd.Timedelta(days=1) for b in blocks):
        raise AssertionError("frozen validation dates are not consecutive two-day blocks")
    if not set(common_dates).issubset(set(val_dates)):
        raise AssertionError("common target timestamp is outside frozen validation dates")

    input_hashes = {
        "pack_rows_csv": sha256(pack_dir / "rows.csv"),
        "pack_report_json": sha256(pack_dir / "pack_report.json"),
        "al_feature_report_json": sha256(ROOT / "AL_FEATURE_REPORT.json"),
        "frozen_date_assignment_csv": sha256(ROOT.parent / "frozen_split_20261015_v1" / "day_assignment.csv"),
        "predictions": {},
    }
    for seed in SEEDS:
        for station in STATIONS:
            for arm in ("A", "AC", "AP"):
                path = ROOT / "baseline_runs" / f"seed{seed}" / station / f"{arm}_val_prediction_ghi_wm2.npy"
                input_hashes["predictions"][f"{seed}/{station}/{arm}"] = sha256(path)
            path = ROOT / "runs_maskfix_v2" / f"seed{seed}" / station / "AL_validation_predictions_with_keys.npz"
            input_hashes["predictions"][f"{seed}/{station}/AL"] = sha256(path)

    # For each station and date, retain paired squared-error sums/counts on the
    # exact all-station common-time support. The same date/block draws are used
    # for every arm, station, and seed.
    sum_sq: dict[tuple[int, str, str, str], float] = {}
    for station in STATIONS:
        sv = val.loc[val["station"].eq(station)].reset_index(drop=True)
        times = sv["time_bjt"].to_numpy()
        keep = np.isin(times, common_times)
        if int(keep.sum()) != len(common_times):
            raise AssertionError(f"station {station} did not retain the full common-time set")
        y = sv["ghi_wm2"].to_numpy(np.float64)
        for seed in SEEDS:
            for arm in ARMS:
                err2 = (preds[(seed, station, arm)][keep] - y[keep]) ** 2
                kept_dates = times[keep].astype(str)
                for date in val_dates:
                    sum_sq[(seed, station, arm, date)] = float(
                        np.sum(err2[np.char.startswith(kept_dates, date)], dtype=np.float64)
                    )

    # The common support has identical target timestamps at every station.
    n_by_date = {d: sum(1 for t in common_times if t.startswith(d)) for d in val_dates}
    supported_dates = [d for d in val_dates if n_by_date[d] > 0]
    supported_blocks = [[d for d in b if n_by_date[d] > 0] for b in blocks]
    supported_blocks = [b for b in supported_blocks if b]
    if not supported_dates or not supported_blocks:
        raise AssertionError("common timestamps do not intersect the frozen validation blocks")

    def estimate(date_weights: dict[str, int]) -> dict[str, float]:
        result: dict[str, float] = {}
        for arm in ARMS:
            cell_rmse = []
            for seed in SEEDS:
                for station in STATIONS:
                    sse = sum(date_weights.get(d, 0) * sum_sq[(seed, station, arm, d)] for d in supported_dates)
                    n = sum(date_weights.get(d, 0) * n_by_date[d] for d in supported_dates)
                    if n <= 0:
                        raise AssertionError("bootstrap replicate has no common support")
                    cell_rmse.append(float(np.sqrt(sse / n)))
            result[arm] = float(np.mean(cell_rmse))
        return result

    point_weights = {d: 1 for d in supported_dates}
    point = estimate(point_weights)
    deltas = (("AC", "A"), ("AC", "AP"), ("AL", "A"), ("AL", "AP"), ("AC", "AL"))
    rng = np.random.default_rng(BOOT_SEED)
    draws = {f"{a}_minus_{b}": np.empty(N_BOOT, dtype=np.float64) for a, b in deltas}
    for i in range(N_BOOT):
        selected = rng.integers(0, len(supported_blocks), size=len(supported_blocks))
        weights = {d: 0 for d in supported_dates}
        for j in selected:
            for d in supported_blocks[int(j)]:
                weights[d] += 1
        boot = estimate(weights)
        for a, b in deltas:
            draws[f"{a}_minus_{b}"][i] = boot[a] - boot[b]

    delta_point = {f"{a}_minus_{b}": point[a] - point[b] for a, b in deltas}
    out = {
        "state": "HUBEI_THREE_STATION_COMMON_TIME_VALIDATION_SENSITIVITY_COMPLETE_TEST_CLOSED",
        "scope": "same-time GHI estimate; validation-only support sensitivity, not independent test or forecast evidence",
        "test_buffer_values_read": 0,
        "split_sha256": pack_report["split_sha256"],
        "input_sha256": input_hashes,
        "common_target_timestamps_n": len(common_times),
        "common_validation_dates_n": len(common_dates),
        "common_validation_dates": common_dates,
        "per_station_validation_n": {st: len(per_station[st]) for st in STATIONS},
        "common_support_n_by_station": {st: len(common_times) for st in STATIONS},
        "point_mean_station_equal_seed_equal_rmse_wm2": point,
        "paired_rmse_differences_wm2": delta_point,
        "contiguous_two_day_block_bootstrap": {
            key: {
                "ci95_percentile_wm2": [float(x) for x in np.quantile(values, [0.025, 0.975])],
                "fraction_below_zero": float(np.mean(values < 0)),
            }
            for key, values in draws.items()
        },
        "bootstrap_replicates": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "block_sampling_note": "Resampled only frozen consecutive two-day validation blocks containing at least one common target timestamp; empty-support days contribute no rows.",
    }
    out_path = ROOT / "THREE_STATION_COMMON_TIME_SENSITIVITY_20261017.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# 三站共同目标时刻敏感性（validation only）",
        "",
        "本分析只在三站同时存在完整标签的目标时刻上，重算同期GHI估计结果；不是独立测试，也不是未来预报。test/buffer有效载荷读取数为0。",
        "",
        f"- 三站共同目标时刻：{len(common_times)}；涉及验证日期：{len(common_dates)}。",
        f"- 每个站各自完整验证数：{', '.join(f'{st} {len(per_station[st])}' for st in STATIONS)}。",
        "- 指标为站点等权、种子等权RMSE；区间按冻结连续两日日期对进行共享配对bootstrap，只保留共同支持中有样本的日期。",
        "",
        "| 方法 | 共同支持 RMSE (W/m²) |",
        "|---|---:|",
    ]
    for arm in ARMS:
        lines.append(f"| {arm} | {point[arm]:.3f} |")
    lines += ["", "| 配对差（前者−后者，RMSE） | 点差 (W/m²) | 连续两日块95% CI (W/m²) |", "|---|---:|---:|"]
    for a, b in deltas:
        key = f"{a}_minus_{b}"
        ci = out["contiguous_two_day_block_bootstrap"][key]["ci95_percentile_wm2"]
        lines.append(f"| {a} − {b} | {delta_point[key]:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |")
    lines += [
        "",
        "解释边界：这是模型选择用validation队列上的共同时间支持敏感性；不能取代独立test。A/AC/AP/AL均由原验证预测及共同pack真值复算。",
        "",
        f"机器可读报告：`{out_path.name}`。",
    ]
    md_path = ROOT / "THREE_STATION_COMMON_TIME_SENSITIVITY_20261017.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_path), "n_common_times": len(common_times), "dates": common_dates,
                      "point": point, "deltas": delta_point,
                      "two_day_ci": {k: v["ci95_percentile_wm2"] for k, v in out["contiguous_two_day_block_bootstrap"].items()}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
