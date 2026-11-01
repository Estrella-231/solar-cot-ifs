# SimVP 监督改进执行清单

日期：2026-10-03；计划版本：v1.1。隔离训练器/PBS脚本已写，尚未提交任务、监控或读取test。

| ID | 阶段 | 目的/候选 | 数据 | 关键产物/指标 | 优先级 | 状态 |
|---|---|---|---|---|---|---|---|
| R00 | M0 | 来源、损失实现、掩膜/梯度见证 | 新train/val | hash、loss witnesses、test_reads=0 | 必须 | TODO |
| R01 | M0 | train-only λ标定、单卡profile | 固定32 train batches | λ、显存、真实吞吐、共同microbatch | 必须 | TODO |
| R02 | M0 | 冻结样本、事件规则、损失贡献 | 新train/完整val | sample hashes、loss mass、R日期审计 | 必须 | TODO |
| R03 | M1 | S0固定初始模型 | 完整val | 四季/通道/lead、真实事件锚点 | 必须 | TODO |
| R04 | M1 | A_mse seed42，4000步 | 新train/val | 固定step0/2000/4000输出 | 必须 | TODO |
| R05 | M1 | B_temporal seed42，4000步 | 同A | Lt、物理AGRI、峰值/时序 | 必须 | TODO |
| R06 | M1 | C_spatial seed42，4000步 | 同A | Ls、物理AGRI、虚警/边缘 | 必须 | TODO |
| R07 | M2 | 合规R下COT及真实事件配对 | 冻结val | 峰值MAE/POD/FAR/CSI/到达、块bootstrap | 必须 | WAIT_R_AUDIT |
| R08 | M2 | 依据冻结筛选线决定去留 | 同一支持 | decision.json、真实图、负结果 | 必须 | TODO |
| R09 | M3 | A/优胜项各seed43/44，共4次 | 同合同 | 微调随机性；非独立预训练seed | 条件 | NOT_RELEASED |
| R10 | M4 | 分层采样单因素 | 新train/自然val | 仅在采样证据充分后定义 | 可选 | NOT_RELEASED |
| R11 | M4 | COT辅助监督/B+C组合 | 新train/val | 各自独立协议 | 可选 | NOT_RELEASED |
| R12 | M4 | 配对重训GHI头与闭环 | 新train/val | 相同15min区间标签GHI | 条件 | NOT_RELEASED |

执行顺序：R00→R01/R02→R03→R04/R05/R06→R07/R08；R07可等待新R而不阻断纯AGRI微调，但不能跳过COT门禁宣称成功。拟申请单节点3卡并行执行A/B/C，每臂一张卡，batch8；通过profile后才开始4000步正式更新。所有提交仍受实时全账户PBS上限约束。

参考：[完整计划](EXPERIMENT_PLAN.md)。每次实施更新实际代码版本、服务器目录、PBS、时间、硬件、完成标志及失败原因；不能把TODO自动改成已开始。
