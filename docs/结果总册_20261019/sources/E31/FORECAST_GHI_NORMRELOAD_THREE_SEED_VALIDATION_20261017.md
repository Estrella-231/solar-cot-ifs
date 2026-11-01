# 湖北 forecast-GHI：归一化重载后三种子验证结果

日期：2026-10-17。该结果取代旧的未将train-only归一化数值合同完整写入checkpoint的FA/FAC三种子结果；范围仍是validation选模，不是独立test。

## 对照和复核

- FA：预测AGRI + 历史太阳几何；FAC：同一预测AGRI + 预测COT + 相同几何。
- 种子42/43/44，均使用相同OOF输入包（SHA `9421c0d5e19094454e5b94b98de797ac90248d3e5b56fdef6dacda339df359dd`）、同一训练器及train-only normalization合同。
- 三站各臂/seed验证支持完全配对，共22个目标日期、lead +15至+240 min；站点样本量为牛首2316、王寨2112、米庄2295（每seed每臂）。保存预测及checkpoint重载复算通过，test/buffer载荷读取0。

## 汇总结果

三站等权、seed等权 RMSE：FA **77.150**，FAC **78.155 W/m²**；FAC−FA = **+1.005 W/m²**，即FAC高1.30%。单日日期簇bootstrap 10,000次95%区间为 **[−0.722, +3.780] W/m²**；连续两日块敏感性为 **[−0.681, +3.813] W/m²**，均跨0。FAC只在1/3个seed级汇总中RMSE较低。

各站三seed平均RMSE（W/m²）：

| 站点 | FA | FAC | FAC−FA |
|---|---:|---:|---:|
| 牛首 | 76.49 | 77.27 | +0.78 |
| 王寨 | 72.11 | 73.85 | +1.74 |
| 米庄 | 82.85 | 83.35 | +0.49 |

三站等权FAC−FA在16个lead上均为正，范围 **+0.45至+1.83 W/m²**。这只是描述性逐时效validation表现，并非逐lead独立显著性检验。

![FA 与 FAC 逐时效验证 RMSE 及配对差](audit_receipts/forecast_ghi_three_seed_normreload_20261017/figures/validation_per_lead_FA_FAC.png)

图中阴影表示三个训练seed的范围，不是置信区间；右图正值表示加入预测COT后RMSE较大。

## 结论边界与下一步

当前validation不能确认预测COT改善未来GHI预报；点估计反而偏向FA，但日期块区间跨0。不得用该validation作论文独立确认性结论或据此重调样本。下一步冻结预测输入链、分层/案例、分析代码和图表后，才可重新申请独立test-freeze审计。

机器收据：`audit_receipts/forecast_ghi_three_seed_normreload_20261017/three_seed_validation_summary.json`；逐lead数据：同目录`three_seed_per_lead_metrics.csv`。
