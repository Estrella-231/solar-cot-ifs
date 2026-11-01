from pathlib import Path
import csv, json

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "cot_forecast_case_values.csv"
CASE = ROOT / "case_selection_and_metrics.json"
OUT_JSON = ROOT / "arrival_diagnostic.json"
OUT_MD = ROOT / "COT_FORECAST_ARRIVAL_DIAGNOSTIC_20261014.md"
THRESHOLD = 30.0
with CASE.open(encoding="utf-8") as f:
    case = json.load(f)
series = {}
support = {}
with SRC.open(newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        key = (row["station"], row["series"])
        lead = int(row["lead_min"])
        series.setdefault(key, {})[lead] = float(row["mean_cot"])
        support.setdefault(key, {})[lead] = int(row["n_common_3x3_pixels"])
methods = ["CPP reference", "S→R forecast", "Persistence-R", "Future observed AGRI→R (offline)"]
rows = []
for station in ("Sili", "Zhujia"):
    ref = series[(station, "CPP reference")]
    leads = sorted(ref)
    for method in methods:
        vals = series[(station, method)]
        if sorted(vals) != leads:
            raise ValueError(f"lead mismatch: {station} {method}")
        if any(support[(station, method)][l] != support[(station, "CPP reference")][l] for l in leads):
            raise ValueError(f"common-support mismatch: {station} {method}")
        peak_lead = max(leads, key=lambda l: vals[l])
        crossing = next((l for l in leads if vals[l] >= THRESHOLD), None)
        ref_peak = max(ref.values())
        rows.append({
            "station": station, "series": method,
            "n_common_pixels_per_lead": support[(station, method)][leads[0]],
            "first_3x3_mean_cot_ge_30_lead_min": crossing,
            "threshold_reached": crossing is not None,
            "peak_3x3_mean_cot": vals[peak_lead], "peak_lead_min": peak_lead,
            "peak_difference_from_CPP_peak": vals[peak_lead] - ref_peak,
            "reference_peak_cot": ref_peak,
            "reference_first_threshold_lead_min": next((l for l in leads if ref[l] >= THRESHOLD), None),
            "lead_values": [vals[l] for l in leads],
        })
result = {
    "state": "COMPLETE_SINGLE_PREDECLARED_VALIDATION_CASE_DIAGNOSTIC",
    "split": case["split"], "test_payloads_read": 0,
    "sequence_id": case["selected_sequence_id"], "target_date_bjt": case["selected_target_date_bjt"],
    "selection_rule": case["selection_rule"], "bank_sha256": case["bank_sha256"],
    "metric": "first 15-min lead where mean physical COT over the fixed CPP-common 3x3 pixels reaches 30; no interpolation",
    "threshold_cot": THRESHOLD, "leads_min": sorted(series[("Sili", "CPP reference")]),
    "results": rows,
    "limitations": [
        "One mechanically selected validation case; descriptive only and no confidence interval.",
        "CPP is a retrieval reference, not independent truth.",
        "Future observed AGRI→R is an offline reference, not deployable forecast input.",
        "No first crossing within +15..+240 min is reported as not reached in horizon, not as a numeric arrival-time error.",
        "The 3x3 neighborhood mean can hide sub-neighborhood cloud motion; this diagnostic is not cloud-object tracking."
    ]
}
OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
labels = {"CPP reference":"CPP参考", "S→R forecast":"SimVP→R", "Persistence-R":"持续性→R", "Future observed AGRI→R (offline)":"真实未来AGRI→R（离线参考）"}
lines = [
    "# COT外推到达时间与峰值案例诊断",
    "",
    f"案例：{result['sequence_id']}，目标日 {result['target_date_bjt']} BJT。按预先规则选择该日：{result['selection_rule']}。",
    "",
    "在两站固定3×3邻域、每个lead的CPP共同有效像元上，计算物理COT空间均值。到达定义为首个均值达到固定厚云阈值COT≥30的15分钟lead，不做时效插值。",
    "",
    "| 站点 | 序列 | CPP首达时效 | 本方法首达时效 | 本方法峰值及lead | 相对CPP峰值差 | 支持像元/lead |",
    "|---|---|---:|---:|---:|---:|---:|"
]
by_station = {s: {r['series']: r for r in rows if r['station']==s} for s in ('Sili','Zhujia')}
for station in ('Sili','Zhujia'):
    ref = by_station[station]['CPP reference']
    ref_cross = ref['reference_first_threshold_lead_min']
    for method in methods[1:]:
        x=by_station[station][method]
        fmt=lambda v: f"+{v} min" if v is not None else "240 min内未达到"
        lines.append(f"| {station} | {labels[method]} | {fmt(ref_cross)} | {fmt(x['first_3x3_mean_cot_ge_30_lead_min'])} | {x['peak_3x3_mean_cot']:.2f} @ +{x['peak_lead_min']} min | {x['peak_difference_from_CPP_peak']:+.2f} | {x['n_common_pixels_per_lead']} |")
lines += [
    "",
    "## 结果解释",
    "",
    "四里CPP参考在+135 min首次达到厚云阈值，并在+240 min达到3×3均值峰值53.72；SimVP→R和持续性在预报窗内均未达到阈值，峰值分别只有29.61和28.34。竺家CPP参考在+30 min已达到阈值、+195 min峰值44.03；SimVP→R与持续性在整段时效都没有达到阈值，3×3均值峰值分别为0.47和1.84。该固定案例显示预测链没有复现参考厚云信号及其到达过程。",
    "",
    "离线真实未来AGRI→R在四里+15 min越过阈值但随后没有追上CPP参考峰值；竺家整段未到阈值。它只能帮助区分外推输入误差与反演链差异，不能作为可部署预测成绩。该图/表是单一、机械选出的验证案例，不能估计总体到达误差；CPP也不是独立物理真值。",
    "",
    "来源为既有`cot_forecast_case_values.csv`与`case_selection_and_metrics.json`；脚本逐站逐lead校验方法和CPP有效像元数相同。完整机器结果见`arrival_diagnostic.json`。test载荷读取为0。",
]
OUT_MD.write_text("\n".join(lines)+"\n", encoding="utf-8")
print(json.dumps({"json":str(OUT_JSON),"markdown":str(OUT_MD),"rows":len(rows),"test_payloads_read":0},ensure_ascii=False))
