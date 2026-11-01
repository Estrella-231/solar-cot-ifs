# 本轮交付索引

2026-10-20。范围：云掩膜历史核查、文献与方法选择、分类器几何/输入对齐审计、四组配对训练、FP64评分和B2分层分析。

| 文件 | 内容 |
|---|---|
| `EXPERIMENT_PLAN_20261020.md` | 本次完整计划日期版 |
| `EXPERIMENT_PLAN.md` | 相同计划的固定入口 |
| `EXPERIMENT_TRACKER.md` | 已完成准备与未执行训练分开登记 |
| `LITERATURE_METHOD_DECISION_20261020.md` | 原文接口、读取层级、语义与方法选择 |
| `HISTORY_AUDIT_20261020.md` | 旧实验去重、单位/标签区分及当前资产 |
| `rescore_old_gate_dayclean.py` | 旧硬门控保存预测的FP64重评分脚本 |
| `OLD_GATE_DAYCLEAN_20261020.json` | 旧固定权重剔除已确认无效观测日后的结果、区间与源摘要 |
| `ASSET_AVAILABILITY_20261020.json` | 远端stat/metadata及完整键覆盖结果，包含SSH结束255说明 |
| `REVIEW_NOTES_20261020.md` | 两名只读审阅发现及已修正内容 |
| `remote_preflight_audit.py` / `REMOTE_PREFLIGHT_AUDIT_20261020.json` | 站点-帧键和split精确核验、分类器旧太阳几何与修正几何差值 |
| `train_four_arm.py` | 四组17槽配对训练器，含行键复查、无效站日排除、共享初始化及梯度见证 |
| `run_four_arm.pbs` | node22 单卡 PBS 运行脚本 |
| `score_four_arm.py` | 保存预测的FP64重算、主配对差和两日块置信区间 |
| `analyze_four_arm_strata.py` | CPP COT厚度与CLP类别分层，含完整日期块bootstrap |
| `analyze_four_arm_high_ghi.py` | GHI≥800 W/m²子集的RMSE/MAE/MBE/SSE诊断 |
| `plot_four_arm_strata.py` | 生成CPP COT分层及CLP分层配对差图（PNG/PDF） |
| `B2_RESULTS_20261020.md` | 四组总评、区间、分层、高GHI结果和证据边界 |
| `EXPERIMENT_TRACKER.md` | 记录 PBS `215069.tc6000`、geometry/alignment审计和最终结果 |

主`refine-logs/EXPERIMENT_PLAN.md`和`EXPERIMENT_TRACKER.md`已登记本研究线。PBS `215069.tc6000` 在浙师大 node22/A800 GPU4 完成24/24配对训练。没有改论文、重训R或分类器、创建自动化或读取test载荷。

代码SHA-256：`train_four_arm.py` `cb98e42ff67f2746cc615f3bbb199990489bfb5ca84918d5e88d6ae4f9526a66`；`run_four_arm.pbs` `a4600272f94b04c3ec7609c9cb8988e7ecd25fea409a2c7e14ed39f63fae8695`；`score_four_arm.py` `945b6cfe0a744fc903c2e3f6c242bd8fc076f7e8ebd2ad8edc8dc9a70551a409`；`analyze_four_arm_strata.py` `4feaad005dbfbaade35c9aeaa954304b1a0783f9be991a7bfdca5544f63f0eecd`；`analyze_four_arm_high_ghi.py` `1dc0c333a258b9ad827afb84eca71ffe196e6f7dae60e63dc9d66a36354f3911`；`plot_four_arm_strata.py` `e8fd41810117d81e7eb2778cc60956f58925f68067cc7652017d21fba8ae830c`.
