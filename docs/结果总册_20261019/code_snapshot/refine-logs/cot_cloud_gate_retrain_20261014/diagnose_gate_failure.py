#!/usr/bin/env python3
"""Decompose paired validation changes caused by the fixed 0.5 hard COT gate."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
cp = np.load(ROOT / "remote_result_retry1/validation_predictions.npz")
rp = np.load(ROOT / "R_validation_predictions_seed42.npz")
clp_pack = np.load(ROOT / "clp_trainval.npz")
idx = rp["sample_indices"].astype(int)
assert np.array_equal(idx, cp["sample_indices"])
assert np.array_equal(clp_pack["sample_indices"][idx], idx)
assert np.array_equal(clp_pack["station"][idx], cp["station"])
assert np.array_equal(clp_pack["target_time_utc"][idx], cp["target_time_utc"])

v = rp["mask"][:, 0].astype(bool)
assert np.array_equal(v, cp["valid"])
clp = clp_pack["clp"][idx, 0]
assert np.all(np.isin(clp[v], (0, 1, 2)))
cloud_ref = clp >= 1
p = cp["prob_cloud"].astype(np.float64)
pred_cloud = p >= 0.5
raw = np.maximum(np.expm1(rp["pred_log1p_cot"][:, 0].astype(np.float64)), 0)
y = rp["reference_cot"][:, 0].astype(np.float64)
gate = raw * pred_cloud
stations = cp["station"].astype(str)
rows, cols = np.indices((16, 16))
center3 = ((rows >= 7) & (rows <= 9) & (cols >= 7) & (cols <= 9))[None]

cot_bins = [(0, .1, "[0,0.1)"), (.1, 1, "[0.1,1)"), (1, 5, "[1,5)"),
            (5, 15, "[5,15)"), (15, 30, "[15,30)"), (30, np.inf, "[30,+inf)")]

def summarize(sel):
    m = v & sel
    n = int(m.sum())
    if n == 0:
        return {"n": 0}
    sse0 = float(np.square(raw[m] - y[m]).sum())
    sse1 = float(np.square(gate[m] - y[m]).sum())
    missed = m & cloud_ref & ~pred_cloud
    cleared = m & ~pred_cloud
    return {
        "n": n,
        "raw_rmse": float(np.sqrt(sse0 / n)),
        "gate_rmse": float(np.sqrt(sse1 / n)),
        "delta_sse_gate_minus_raw": sse1 - sse0,
        "delta_rmse": float(np.sqrt(sse1 / n) - np.sqrt(sse0 / n)),
        "gate_zero_fraction": float(cleared.sum() / n),
        "cloud_reference_fraction": float((m & cloud_ref).sum() / n),
        "true_cloud_missed_pixels": int(missed.sum()),
        "missed_cloud_fraction_of_reference_cloud": float(missed.sum() / max((m & cloud_ref).sum(), 1)),
        "missed_cloud_ref_cot_mean": float(y[missed].mean()) if missed.any() else None,
        "missed_cloud_raw_cot_mean": float(raw[missed].mean()) if missed.any() else None,
    }

result = {"threshold": .5, "valid_pixels": int(v.sum()), "test_payloads_read": 0, "by_station": {}}
for s in ("sili", "zhujia"):
    sm = stations == s
    sm3 = sm[:, None, None]
    st = {"all": summarize(sm3)}
    for c in (0, 1, 2):
        st[f"CLP_{c}"] = summarize(sm3 & (clp == c))
    st["by_reference_COT_bin"] = {name: summarize(sm3 & (y >= lo) & (y < hi)) for lo, hi, name in cot_bins}
    st["station_center_3x3"] = summarize(sm3 & center3)
    st["outside_center_3x3"] = summarize(sm3 & ~center3)
    # Pixel-location diagnostics; rows/cols are patch coordinates, not lat/lon.
    map_delta_sse = np.zeros((16, 16), dtype=np.float64)
    map_missed = np.zeros((16, 16), dtype=np.int64)
    map_ref_cloud = np.zeros((16, 16), dtype=np.int64)
    for r in range(16):
        for c in range(16):
            q = sm & v[:, r, c]
            map_delta_sse[r, c] = np.square(gate[:, r, c][q] - y[:, r, c][q]).sum() - np.square(raw[:, r, c][q] - y[:, r, c][q]).sum()
            map_missed[r, c] = np.count_nonzero(q & cloud_ref[:, r, c] & ~pred_cloud[:, r, c])
            map_ref_cloud[r, c] = np.count_nonzero(q & cloud_ref[:, r, c])
    st["spatial_patch_diagnostics"] = {
        "delta_sse_by_pixel": map_delta_sse.tolist(),
        "missed_cloud_count_by_pixel": map_missed.tolist(),
        "reference_cloud_count_by_pixel": map_ref_cloud.tolist(),
        "center_3x3_missed_cloud": int(map_missed[7:10, 7:10].sum()),
        "center_3x3_reference_cloud": int(map_ref_cloud[7:10, 7:10].sum()),
    }
    result["by_station"][s] = st

# Contribution split by cloud truth and whether classifier cleared the R estimate.
for s in ("sili", "zhujia"):
    sm = stations == s
    sm3 = sm[:, None, None]
    per = {}
    for label, cond in (("CPP_clear", ~cloud_ref), ("CPP_cloud", cloud_ref)):
        m = v & sm3 & cond
        d = np.square(gate[m] - y[m]) - np.square(raw[m] - y[m])
        per[label] = {
            "n": int(m.sum()), "delta_sse": float(d.sum()),
            "clear_decisions": int((m & ~pred_cloud).sum()),
            "clear_decision_delta_sse": float((np.square(gate[m & ~pred_cloud] - y[m & ~pred_cloud]) - np.square(raw[m & ~pred_cloud] - y[m & ~pred_cloud])).sum()),
            "cloud_decision_delta_sse": float((np.square(gate[m & pred_cloud] - y[m & pred_cloud]) - np.square(raw[m & pred_cloud] - y[m & pred_cloud])).sum()),
        }
    result["by_station"][s]["error_change_by_CPP_cloud_truth"] = per

(ROOT / "gate_failure_decomposition.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

lines = ["# 固定阈值 COT 门控退化原因诊断", "", "仅使用冻结验证集保存的分类概率、R COT、CPP CLP 和共同有效掩膜；未训练模型、未读取测试集。",
         "", "## 站点与CPP云类分解", "", "| 站点/分组 | 像元数 | 原R RMSE | 门控RMSE | ΔRMSE | 门控清零比例 | CPP云漏检数 | 漏检云参考COT均值 | ΔSSE |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for s in ("sili", "zhujia"):
    for group in ("all", "CLP_0", "CLP_1", "CLP_2"):
        a = result["by_station"][s][group]
        lines.append(f"| {s} / {group} | {a['n']} | {a.get('raw_rmse',float('nan')):.4f} | {a.get('gate_rmse',float('nan')):.4f} | {a.get('delta_rmse',float('nan')):+.4f} | {a.get('gate_zero_fraction',float('nan')):.3%} | {a.get('true_cloud_missed_pixels',0)} | {a.get('missed_cloud_ref_cot_mean',float('nan')) if a.get('missed_cloud_ref_cot_mean') is not None else float('nan'):.3f} | {a.get('delta_sse_gate_minus_raw',0):+.1f} |")
lines += ["", "## 按参考COT范围的门控变化", "", "| 站点 | CPP参考COT范围 | N | 原R RMSE | 门控RMSE | ΔRMSE | 漏检有云像元 | ΔSSE |", "|---|---|---:|---:|---:|---:|---:|---:|"]
for s in ("sili", "zhujia"):
    for b, a in result["by_station"][s]["by_reference_COT_bin"].items():
        lines.append(f"| {s} | {b} | {a['n']} | {a.get('raw_rmse',float('nan')):.4f} | {a.get('gate_rmse',float('nan')):.4f} | {a.get('delta_rmse',float('nan')):+.4f} | {a.get('true_cloud_missed_pixels',0)} | {a.get('delta_sse_gate_minus_raw',0):+.1f} |")
(ROOT / "gate_failure_decomposition.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("WROTE gate_failure_decomposition.json and .md")
