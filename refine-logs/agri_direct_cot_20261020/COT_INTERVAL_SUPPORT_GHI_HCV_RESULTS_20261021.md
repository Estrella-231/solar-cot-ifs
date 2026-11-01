# 15 分钟区间平均预测 COT 的 GHI 筛选结果

日期：2026-10-21。状态：完成，未通过预设收益门。结果仅来自外层训练集合 H 的内部日期块交叉验证，不是独立验证或测试结论。

## 问题与对照

15 分钟 GHI 标签对应右闭区间 `(target_time−15 min, target_time]` 的地面辐照度均值，而此前 GHI 头只使用区间末端的预测 COT。这里仅改变 COT 的时间支持：比较无 COT、末端时刻中心 3×3 COT，以及该 15 分钟区间两端 COT 的算术平均。第一段区间的左端使用起报时已有的历史反演 COT；其后使用前一时效的预测 COT。区间平均是端点梯形近似的简化形式，不是辐射传输积分。

三组都使用同一 AGRI 预测、太阳几何、站点 GHI 样本、3×3 COT 支持、GHI 头结构、日期折、训练预算和配对初始化。COT 两组按同一几何规则处理特征支持；所有组保留相同 GHI 行。测试集、缓冲日期、未来真实 AGRI 和 CPP/COT 均未作为输入。

## 结果

主指标为候选减参照的 GHI RMSE，正数代表候选更差。误差区间由 112 个固定北京时间两日块进行 5,000 次配对 bootstrap。

| 对比 | 全时效 +15～+240 min ΔRMSE，W m⁻²（95%区间） | 短时效 +15～+60 min ΔRMSE，W m⁻²（95%区间） |
|---|---:|---:|
| 区间平均 COT − 末端 COT | +0.062 [−0.107, +0.233] | −0.009 [−0.230, +0.215] |
| 区间平均 COT − 无 COT | +0.444 [−1.791, +2.729] | −1.484 [−3.373, +0.308] |

相对末端 COT，四里全时效点差为 −0.040 W m⁻²，竺家为 +0.164 W m⁻²；三个配对种子中只有一个全时效改善。两站、种子和删去单个两日块后的估计均未形成稳健一致的增益。所有主要置信区间均跨零，预设七项条件未通过。

![15分钟区间平均预测COT的H内部GHI筛选图](results/direct_cot_interval_support_hcv_v1_local4060/score/analysis/interval_support_hcv_diagnostic.png)

## 解释与边界

把 COT 从区间末端改成 15 分钟端点平均，并未解决预测 COT 加入 GHI 头后的增益问题。短时效相对无 COT 的点估计较好，但区间仍跨零，不能据此称为收益。该候选按原定合同关闭；不在同一 H 数据上搜索其它平均权重、阈值或子群。

此前“过去真实 AGRI 和历史反演 COT→未来 COT”的 P-XC 模型已完成固定 12 轮训练，但其有限 CPP 参考评分和 GHI 结果仍不能证明下游收益。当前 H/V 清单没有未使用的新 GHI 日期；test/buffer 数值载荷继续关闭。下一步应先只读核验是否已有新的、来源及标签质量可绑定且从未参与开发的地面观测日期；若不存在，就暂不能做独立确认，也不能用反复 H-CV 选出的点估计代替它。

## 复现收据

- 90 个头：两站 × 5 折 × 3 种子 × 3 组；每头 30 轮、batch 512；每站 H 键逐样本 OOF 一次（四里 72,345、竺家 79,417）。
- 运行设备：本地 NVIDIA GeForce RTX 4060 Laptop GPU；未提交 PBS。
- 训练来源：冻结 H-only `direct_cot_forecast_row_bank_v1`、H-only `direct_cot_hv_inference_v1`、U-only `R_U` 和湖南 `grid_static`；中心 3×3 的预测 COT 与 rowbank 逐行最大绝对差为 0。
- `run_manifest.json` SHA-256：`4193685798022fc40fdb228441d9e757cf185e156873a3cc2916b78c3f5e7625`。
- 配对评分 JSON SHA-256：`d828cc058851851923095e16866cd26fcc6562ce2a0c5ba2c5d868641afdfeeed`。
- Figure PNG SHA-256：`ba7e0c0695f687250d0d5c5e5c7b8bfd5d202edc22a2990650c24ec23e7ca6f93`。
- 来源收据标记 test/buffer、未来观测 AGRI、CPP 输入均为 `false`；未修改论文正文。

机器可读评分、逐时效/绝对 RMSE 表、PDF 图和训练模型位于 `results/direct_cot_interval_support_hcv_v1_local4060/`。
