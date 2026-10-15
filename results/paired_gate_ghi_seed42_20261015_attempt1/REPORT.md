# 保守晴空门控 COT 的配对 GHI 头结果（seed 42）

日期：2026-10-15。实验范围：湖南四里、竺家；这是当前验证集上的探索性结果，不是独立测试结论。test 数据未读取。

## 对照设计与完整性

比较原始 R-COT（raw）与 train-OOF 选出的保守晴空门控 R-COT（gate）。门控阈值 `t_clear=0.2935079336` 只由 train OOF 确定。两臂使用相同的 LocalGHIHead AC 结构、seed 42、每站相同初始化权重、epoch 样本顺序、30 轮训练和 batch 2000；只有 COT bank 与各自 train-only COT 归一化改变。

raw/gate pack 审计通过：54,937 条样本键一致（50,458 train、4,479 val）；AGRI、太阳几何、标签及行顺序一致；test payload 读取数为 0。训练更新数核对通过：四里每臂 390 次、竺家每臂 420 次。

## 验证结果

正的 RMSE 差表示 gate 更差。

| 站点 | raw RMSE | gate RMSE | Δ RMSE | raw MAE | gate MAE | Δ MAE | gate bias − raw bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| 四里（n=2,128） | 132.639 | 132.716 | +0.077 | 73.454 | 73.527 | +0.072 | +0.098 |
| 竺家（n=2,351） | 106.325 | 106.902 | +0.577 | 72.907 | 73.431 | +0.524 | +0.550 |
| 站点等权 RMSE | 119.482 | 119.809 | **+0.327** | — | — | — | — |

单位均为 W m⁻²。两站 RMSE、MAE 点估计都略有退化；bias 向上移动、绝对偏差变小，但不足以抵消平方误差和绝对误差的增加。

按目标日期分层、站内对 raw/gate 成对重采样 2,000 次，站点等权 ΔRMSE 的 95% 区间为 **[−0.043, +0.751] W m⁻²**。四里和竺家的验证支持分别覆盖 44、50 个目标日期，其中共同日期 42 天。连续两日块敏感性区间为 **[−0.051, +0.528] W m⁻²**。区间均跨零，因此总体 GHI 差异尚不能判定为稳定退化或改善；当前点估计偏向 raw。

## 决策

按预注册停止规则，seed 42 有任一站退化就不自动扩展 seed 43/44。两站点估计均退化，故停止这轮门控候选扩种；保留结果为负向/收益未证实的验证证据，不调阈值、不打开 test，也不将它外推成未来 forecast AGRI 的性能结论。

下一步应先归档本轮模型与预测、完成 CPP 云顶高来源与质量/时空对齐审计，再决定是否提出一个新的单因素物理假设。不能仅因 bias 改善就把门控称为有效。

## 可复核文件

- `result.json`：训练预算、两站两臂指标、更新数与权重哈希。
- `pair_training_contract.json`：训练合同和 pack/头源码哈希。
- `paired_pack_audit.json`：两组输入/标签/键一致性门禁。
- `paired_date_bootstrap.json`：独立指标复算和日期簇区间。
- `sili/paired_validation_predictions.npz`、`zhujia/paired_validation_predictions.npz`：验证时刻、真值及两臂逐样本预测。
- `threshold_frozen.json`：由训练 OOF 冻结的门控阈值。
- `../../refine-logs/cot_cloud_gate_retrain_20261014/analyze_paired_ghi_gate_seed42.py`：独立复算脚本。
