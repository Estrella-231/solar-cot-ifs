# Experiment Tracker — COT-aware SimVP

更新时间：2026-10-17 11:51 BJT。此次不制作图表，不改写当前验证结果，也未启动PBS训练。

| 阶段 | 状态 | 证据 / 下一步 |
|---|---|---|
| 单因素假设与配对合同 | READY | 对照是已完成的等预算MSE微调；候选只增加冻结R的COT辅助损失。见 `EXPERIMENT_PLAN.md`。 |
| COT损失核心函数 | PASS_CPU_WITNESS | `cot_auxiliary.py` / `test_cot_auxiliary.py`；swc CPU合成见证确认CPP掩膜、R回退标签、SimVP输入梯度有限非零，R参数梯度为空。 |
| train/val日期与序列键 | PASS_REMOTE_AUDIT | 35,320 train、3,140 val；抽查并验证序列键`init_time_bjt`与SimVP行的`BJT_start+105min`绑定，所有目标帧索引按UTC键逐帧核对。 |
| 固定训练行配对 | PASS_REMOTE_AUDIT | 复用既有MSE `fixed_training_rows.json`，32,000行、seed42、batch8、4,000步；SHA=`db8511fc2e8ac426711047216b285af3de8069603777d129d1ae67814c21a364`。 |
| R/CPP sidecar来源 | PASS_REMOTE_AUDIT | 两个包按当前frame-key SHA `75fbd1c1…`绑定；CPP形状、几何/有效掩膜及R输出共55,733帧；四个部分无test载荷。 |
| 新训练入口与实数GPU profile | TODO | 尚未完成训练入口集成；完成后先在单卡做真实数据profile、峰值显存/吞吐及checkpoint重载见证，再决定启动。 |
| PBS资源 | CHECKED_NOT_SUBMITTED | 2026-10-17 03:30 CST队列：214221(Q)、214514(R)，未结束2个；214513已完成。node21 GPU快照中多卡有其他用户负载，当前作业只分配1卡；无训练提交。 |
| 预测COT独立复算 | BLOCKED_ON_TRAIN | 固定同一CPP共同支持，比较原SimVP、MSE微调与COT辅助候选；逐lead/站点宏平均，报告厚云CSI、近零虚警和AGRI护栏。 |
| 论文双语更新 | WAIT_FOR_RESULT | 只有训练候选通过预设门槛才更新英文LaTeX与中文Markdown；未出现正结果前不增加主张。 |

## 操作边界

test保持关闭。COT损失筛选不能替代匹配预测输入训练的GHI头，也不能据此声称GHI预报获益。不得覆盖冻结SimVP、MSE对照、原SimVP→R主结果或已有论文PDF。
