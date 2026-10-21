# 历史 COT 外推、持平基线与变化/物理监督审计（2026-10-21）

## 结论

“预测 COT 过平/过弱，因而输给历史 COT 持平”并非未探索的问题。仓库已有直接的 forecast-vs-persistence 对照，也试过时间变化损失、空间梯度损失和 COT 辅助监督。现有证据在不同区域/验证合同下不一致，因此不能把它概括成“模型普遍输给持平”；更不能把 COT 标签辅助损失称为严格的云物理定律约束。

## 已有直接比较：未来 COT 预测 vs 历史持平

湖北 attempt12 使用历史 AGRI → 修正几何 SimVP → 冻结 R 得到未来 COT；基线把初始化时的 R(COT) 持平到未来。CPP 仅定义共同检索支持。四个一小时分组的像元加权 RMSE：

| 预报时段 | SimVP→R 未来 COT | 历史 COT 持平 | 未来预测减持平 |
|---|---:|---:|---:|
| 1 h | 10.255 | 8.946 | +1.309 |
| 2 h | 9.787 | 9.238 | +0.549 |
| 3 h | 10.194 | 9.617 | +0.577 |
| 4 h | 12.063 | 9.086 | +2.977 |

在这份湖北验证诊断里，持平基线四个时段都更低，4 h 差距最大；CPP 是检索参考而非独立真值。

## 既有减少“过平”的训练尝试

湖南 2026-10-17 全量验证的固定 3×3 局地支持上，原 SimVP、等预算 AGRI-MSE、时间变化损失、空间梯度损失、历史 COT 持平的 COT RMSE 分别为 10.664、10.750、10.756、10.759、20.487。时间变化/梯度项没有降低总体 RMSE；持平基线在这份湖南局地评估上明显更差。时间变化审计显示其代理变化误差改善较小且站点不一致，不是稳定的空间 COT 技能提升。

同一湖南旧划分下，AGRI+冻结 R-COT 辅助与 CPP-only COT 辅助的 RMSE 为 10.741、10.744，均未超过原 SimVP 的 10.664。CPP-only 厚云 CSI 小幅提高，但近零 COT 虚警显著增加，未通过既定门槛。

## 什么还没有做

上述辅助损失是让冻结 R(预测 AGRI) 靠近 CPP COT 参考的监督目标。它不是云水/光学厚度守恒、云团输运方程、辐射传输闭合等显式物理定律。因而“缺少物理约束”作为严格机制解释尚未被直接验证；已有结果只测试了特定的 COT/变化代理监督。

## 当前湖北配对实验定位

当前湖北候选是修正太阳几何、湖北自有 train/validation 和冻结 R 下，将 AGRI-only MSE 与 AGRI+冻结 R-COT 辅助监督做相同初始化、相同训练预算的地区复核。它复核的是既有训练思路能否在湖北成立，不是首次提出该方法，也不是新的物理定律约束。attempt12 的历史输运比较是已有证据；本轮不应重复它。

## 下游 GHI 头对 COT 输入的响应

Attempt12 还做过固定 GHI 头替换 COT 输入的验证诊断：AGRI、太阳几何、GHI 标签、头部权重和预处理均固定，只比较 SimVP→R 预测 COT、历史 COT 持平、以及离线未来 AGRI→R COT。三站只覆盖约18%的完整验证行，且日期区间较窄；在该受限共同支持上，离线未来 AGRI→R 输入对应等站 RMSE 117.75 W m⁻²，低于预测 COT 输入的122.58 W m⁻²；两者 RMSE 差（离线未来输入减预测 COT 输入）的95%日期簇区间为−10.63至−1.39 W m⁻²。离线未来图在预报起报时不可得，不能作为可部署成绩或独立 COT 真值。

该替换诊断说明固定 GHI 头能响应 COT 输入，和“GHI 头完全忽略 COT”不一致；它与“预测 COT 链质量限制了部分下游收益”相容，但受限样本、离线未来图和 R 的共同代理性质不足以证明唯一瓶颈。一个保存的验证云团案例也显示时效错位：牛首 +120 min 预测 COT/离线未来 AGRI→R COT 为17.48/39.79，+240 min 则为47.90/24.51。该例提示增厚/消散时机值得检查，但不能代表总体。

因此当前 attempt13 先检验 COT 辅助监督能否改善未来 AGRI→R COT；只有在共同 CPP 支持上同时优于原模型与持平基线，才值得将资源转向轨迹/运动演变建模。若 COT 端改善而 GHI 仍无收益，再另行检查下游头；不能由离线替换提前宣称预报收益。

## 作业记录更正

PBS `215355.tc6000` 已被提前取消，原因是把 Lightning 进度条的 282 误读为训练步数超合同。实际上它由 256 个训练 batch + 26 个 validation batch 构成；合同仍为每轮 256 个 optimizer updates。该中断是执行者误判，不是模型失败，日志与目录保留。相同配对合同已在新目录 `hubei_cot_auxiliary_retrain_attempt13_retry4` 重新提交为 PBS `215356.tc6000`。2026-10-21 12:26 CST 实时状态为 R，node22 单卡 A800；尚未产生训练结果。test/buffer 保持关闭。

## 依据

- `refine-logs/hubei_solar_geometry_repair_20261021/ATTEMPT12_RESULTS_20261021.md`
- `refine-logs/cot_value_first_20261013/EXPERIMENT_TRACKER.md`，R07 full-validation 与 temporal-change audit 记录
- `refine-logs/cot_aware_simvp_20261017/EXPERIMENT_PLAN.md`、`EXPERIMENT_TRACKER.md`
- `refine-logs/hubei_solar_geometry_repair_20261021/hubei_cot_auxiliary_retrain_attempt13/EXPERIMENT_PLAN.md`
## Lead-wise bias check: not uniformly too weak

对已保存的 attempt12 `forecast_cot_metrics.csv` 再按全16×16共同 CPP 支持复核，`bias_cot` 定义为预测 COT − CPP 参考 COT。偏差随时效变号：+75/+90/+105 min 分别为 −0.55/−1.49/−1.83；+195/+210/+225/+240 min 则为 +2.38/+2.87/+3.45/+4.43。+240 min 的预测链 RMSE 为14.26，历史持平为11.43。多数所列短时效仅覆盖4个目标日期，+240 min覆盖14日；CPP仍只是检索参考。

因此“预测COT在所有时效都幅值偏弱”与现有验证统计不符；长时效反而有明显正偏。RMSE输给持平可能涉及时效依赖的增厚/消散幅值误差、位置/到达时间误差或反演误差，当前均值偏差表不能区分这些机制；“过平”还需用同键场标准差、时间增量及事件位置/到达误差专门检验。此项复核是验证诊断，不是独立真值结论。来源：`receipts/attempt12/forecast_cot_metrics.csv` 与 `code_pipeline_attempt12/evaluate_forecast_cot.py`。
