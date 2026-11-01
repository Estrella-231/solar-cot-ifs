# 湖北三站 forecast-GHI FA/FAC 三种子验证结果

> 历史版本说明：本文记录的是旧的未保存完整train-only归一化合同的结果。当前正式validation摘要以[归一化重载复现报告](FORECAST_GHI_NORMRELOAD_THREE_SEED_VALIDATION_20261017.md)为准；旧指标不得与新run混算。

日期：2026-10-17。状态：**validation 汇总已完成；独立 test/buffer 仍关闭**。

## 口径

- 比较对象：FA（预测 AGRI）与 FAC（同一预测 AGRI + 预测 COT），使用同一五折 OOF forecast 输入包及固定 validation 预测输入。
- 三个独立训练 seed：42/43/44；各站各臂均训练30 epoch，batch=1536。每 seed 保存的验证预测已由独立 FP64 脚本复算；站内 FA/FAC 逐样本键、真值一致，三个 seed 的验证支持也一致。
- 支持：Niushou 每臂每 seed n=2,316；Wangzhai n=2,112；Mizhuang n=2,295。验证目标覆盖22个 BJT 日期、lead +15…+240 min。指标单位沿用原工程 W/m² 数值合同；站点字段仪器校准资料尚未独立验真。
- 主汇总：先对三站 RMSE 等权，再对三个 seed 等权；区间按共同目标日期对全部站点、两臂和三个 seed 配对重抽样。该结果仍属于用于模型选择的 validation，不是独立确认性结论。

## 原始验证指标

RMSE / MAE，单位 W/m²；差值为 FAC−FA，负数表示 FAC 较低。

| 站点 | Seed | FA RMSE | FAC RMSE | Δ RMSE | FA MAE | FAC MAE | Δ MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 牛首 | 42 | 74.32 | 75.08 | +0.76 | 49.04 | 49.40 | +0.36 |
| 牛首 | 43 | 76.10 | 78.32 | +2.22 | 49.09 | 50.46 | +1.36 |
| 牛首 | 44 | 79.05 | 78.41 | −0.63 | 52.10 | 51.70 | −0.41 |
| 王寨 | 42 | 73.88 | 73.93 | +0.04 | 46.87 | 46.89 | +0.03 |
| 王寨 | 43 | 71.42 | 74.11 | +2.70 | 45.47 | 47.34 | +1.87 |
| 王寨 | 44 | 71.81 | 74.35 | +2.55 | 44.97 | 47.32 | +2.36 |
| 米庄 | 42 | 84.76 | 84.84 | +0.08 | 54.08 | 54.38 | +0.30 |
| 米庄 | 43 | 80.96 | 82.14 | +1.18 | 51.08 | 52.28 | +1.20 |
| 米庄 | 44 | 82.95 | 86.04 | +3.09 | 52.36 | 54.95 | +2.58 |

## 汇总和 lead 检查

- 三站等权、三 seed 等权的 FA RMSE 为 **77.25**，FAC 为 **78.58 W/m²**；FAC−FA = **+1.33 W/m²**，相对 RMSE 不是下降，而是高 **1.72%**。
- 按各 seed 三站等权，FAC−FA RMSE：seed42 **+0.29**、seed43 **+2.03**、seed44 **+1.67 W/m²**；**0/3 个 seed 整体改善**。
- 日期配对 bootstrap：单日块95%区间 **[−0.42, 3.82] W/m²**；连续两日块敏感性区间 **[−0.30, 3.82] W/m²**。两个区间均跨0，当前 validation 不能确认 COT 有收益；点估计方向偏向 FA。
- 16个 lead 的站点×seed等权 RMSE 中，FAC 点估计在全部 lead 均高于 FA（差值约 **+0.87 至 +2.00 W/m²**）；个别 lead 仅有一个 seed 的站点等权差值为负，平均方向没有反转。

## 解释边界与下一步

目前这套预测头没有显示“加入预测COT提高未来站点GHI精度”。因此不能把 COT 增益写入论文，也不能因为这个 validation 结果而重调阈值或筛选天气/lead。完成预定 forecast 链路来源、案例/分层审计，并冻结代码、权重和指标脚本后，再按冻结 split 对 test 做一次性独立评价；test 结果不能用于调参。若最终 test 同样不支持收益，应把负结果如实报告，并结合 COT 误差/时间对齐证据解释，不靠新增随机 seed 追求正方向。

原始机器结果和复算收据：

- `audit_receipts/forecast_ghi_three_seed_validation_20261017/three_seed_validation_summary.json`
- `audit_receipts/forecast_ghi_three_seed_validation_20261017/three_seed_station_seed_metrics.csv`
- `audit_receipts/forecast_ghi_three_seed_validation_20261017/three_seed_per_lead_metrics.csv`
- `audit_receipts/forecast_ghi_seed42_20261017/seed42_validation_audit.json`
- `audit_receipts/forecast_ghi_seed43_20261017/seed43_validation_audit.json`
- `audit_receipts/forecast_ghi_seed44_20261017/seed44_validation_audit.json`

分析入口：`../../scripts/analyze_hubei_forecast_ghi_three_seed_validation_20261017.py`。训练均绑定 forecast pack SHA-256 `9421c0d5e19094454e5b94b98de797ac90248d3e5b56fdef6dacda339df359dd`；三个 seed 的独立审计均记录 `test_payloads_read=0`。
