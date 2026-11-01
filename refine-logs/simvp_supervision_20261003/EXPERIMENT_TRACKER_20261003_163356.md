# SimVP 监督改进执行清单

日期：2026-10-03；计划版本：v1.2。V1作业211897.tc6000在GPU前置校准中OOM，确认原因是无须保留整张SimVP反向图；V2修复校准梯度路径并通过哈希/语法/CPU损失见证。PBS 211898.tc6000于16:20 BJT在node23三张A100运行；GPU前置校准PASS、test_payloads_read=0，16:23:47 A/B/C启动step-0完整验证及profile。截至16:34，三臂进程均活跃并在跑验证，三卡显存各约7.5GiB，采样利用率会随计算批次波动；profile/step0文件尚未落盘。Luna只读监控中。

| ID | 阶段 | 目的/候选 | 数据 | 关键产物/指标 | 优先级 | 状态 |
|---|---|---|---|---|---|---|---|
| R00 | M0 | 来源、损失实现、掩膜/梯度见证 | 新train/val | hash、loss witnesses、test_reads=0 | 必须 | PASS_CPU_HASH_SYNTAX |
| R01 | M0 | train-only λ标定、3卡并行profile | 固定32 train batches | λ、显存、真实吞吐、共同microbatch | 必须 | PREFLIGHT_PASS_PROFILE_RUNNING |
| R02 | M0 | 冻结样本、事件规则、损失贡献 | 新train/完整val | sample hashes、loss mass、R日期审计 | 必须 | SAMPLE_ORDER_FROZEN_LOSS_MASS_PENDING |
| R03 | M1 | S0固定初始模型 | 完整val | 四季/通道/lead、真实事件锚点 | 必须 | STEP0_VALIDATION_RUNNING |
| R04 | M1 | A_mse seed42，4000步 | 新train/val | 固定step0/2000/4000输出 | 必须 | TODO |
| R05 | M1 | B_temporal seed42，4000步 | 同A | Lt、物理AGRI、峰值/时序 | 必须 | TODO |
| R06 | M1 | C_spatial seed42，4000步 | 同A | Ls、物理AGRI、虚警/边缘 | 必须 | TODO |
| R07 | M2 | 合规R下COT及真实事件配对 | 冻结val | 峰值MAE/POD/FAR/CSI/到达、块bootstrap | 必须 | WAIT_R_AUDIT |
| R08 | M2 | 依据冻结筛选线决定去留 | 同一支持 | decision.json、真实图、负结果 | 必须 | TODO |
| R09 | M3 | A/优胜项各seed43/44，共4次 | 同合同 | 微调随机性；非独立预训练seed | 条件 | NOT_RELEASED |
| R10 | M4 | 分层采样单因素 | 新train/自然val | 仅在采样证据充分后定义 | 可选 | NOT_RELEASED |
| R11 | M4 | COT辅助监督/B+C组合 | 新train/val | 各自独立协议 | 可选 | NOT_RELEASED |
| R12 | M4 | 配对重训GHI头与闭环 | 新train/val | 相同15min区间标签GHI | 条件 | NOT_RELEASED |

执行顺序：R00→R01/R02→R03→R04/R05/R06→R07/R08；R07可等待新R而不阻断纯AGRI微调，但不能跳过COT门禁宣称成功。V1作业211897已失败退出，保留原目录和日志；修正版V2作业211898于16:20 BJT在node23运行，每臂一张A100、batch8。train-only标定λt=0.0448684、λs=0.0611386；16:24快照三臂在各自GPU初始化验证，step0结果和profile尚未落盘。提交前队列为1个未结束作业（211884），本次后共2/3；后续实时状态以PBS为准。

参考：[完整计划](EXPERIMENT_PLAN.md)。每次实施更新实际代码版本、服务器目录、PBS、时间、硬件、完成标志及失败原因；不能把TODO自动改成已开始。
