# Experiment Tracker — COT-aware SimVP

更新时间：2026-10-17 11:50 BJT。论文引言、讨论和结论已完成一轮双语收口；未引入新实验结果。

| 阶段 | 状态 | 证据 / 下一步 |
|---|---|---|
| 单因素假设与配对合同 | READY | AGRI-MSE与AGRI-MSE+COT辅助损失；相同seed42、训练行、batch8、4,000步和初始化，COT辅助仅用训练数据确定权重。|
| 论文双语写作 | PASS_COMPILE | 引言明确三项研究问题；讨论分开同期估计、预测COT和未测的预测输入GHI技能；结论同步收束。LaTeX内置编译成功。|
| 核心损失与冻结R反传 | PASS_REMOTE_WITNESS | `cot_auxiliary.py`：R权重梯度为0，SimVP预测输入梯度有限非零；CPP掩膜和R回退目标通过CPU合成见证。真实月度R CPU反传见证通过。|
| train/val日期与序列键 | PASS_REMOTE_AUDIT | 35,320 train、3,140 val；16个未来帧逐项按UTC帧键核验，初始化与样本时间索引吻合；test载荷关闭。|
| 固定训练行和对照权重 | PASS_REMOTE_AUDIT | 复用MSE对照固定32,000行、seed42、batch8、4,000步；step-4000权重存在，并在脚本启动前核验步骤。|
| 同掩膜COT比较 | IMPLEMENTED_AND_TESTED_SYNTHETIC | 候选、原始SimVP和等预算MSE step-4000均保存逐样本预测银行。独立FP64审计器按同一CPP支持重算patch与中心3×3宏平均，配对10,000次BJT目标日期bootstrap，并检查厚云CSI、近零虚警和AGRI护栏；合成统计见证通过。|
| PBS资源与运行 | GATED_QUEUE_LIMIT | 2026-10-17 11:50全账户有3个未结束任务：`214221`排队、`214517`与`214521`运行；达到上限，未提交训练作业。候选此前`214518`（校验清单目录）、`214519`（模块目录）、`214520`（Path参数/模型加载）均在更新前退出；修复后已通过远端代码散列和真实月度数据合同CPU见证，最新独立审计器仍待同步服务器。等待队列低于3后重新qsub；A800单卡实数profile门槛仍有效。|
| 论文结果更新 | WAIT_FOR_AUDIT | 训练结果须经独立复算、同掩膜对照和AGRI/厚云护栏检查后，才同步写入正文；test保持关闭。|

## 执行边界

该实验最多支持预测COT筛选结论，不能单独证明GHI预报收益。若通过预设门槛，再考虑seed43/44；未通过则记录为负结果，不按验证集调权重。不得覆盖冻结SimVP、MSE对照、原SimVP→R主结果或已有论文PDF。
