# COT 运动与演变实验跟踪表

日期：2026-09-24。所有条目为设计状态；未提交PBS、未训练、未打开test。

| Run ID | 阶段 | 目的/系统 | Split | 主检查 | 优先级 | 状态 | 前置条件 |
|---|---|---|---|---|---|---|---|
| D000 | M0 | geometry24全链路、输入方向/时刻见证、来源锁定 | train/val | 24几何与16预测逐帧对应；无test | MUST | PLANNED | 选定修复版S训练完成 |
| D001 | M0 | halo覆盖、原生patch与拼接R一致性、单卡profile | train | 位移源覆盖/拼接误差/吞吐 | MUST | PLANNED | D000 |
| D010 | M1 | A0/AH/AI/O_diag匹配来源训练H | train/val | 同键GHI RMSE/MAE/bias，理想参考潜力 | MUST | PLANNED | D000/D001，新bank |
| D011 | M1 | P0持续与AP纯搬运 | train/val | COT空间/轨迹/越界，零训练参照 | MUST | PLANNED | D000/D001 |
| D020 | M2 | E_phys与其H；AP的H | train/val | AD/AP相对AH/AI的GHI与COT误差 | MUST | CONDITIONAL | M1有信号、覆盖足够 |
| D030 | M3 | 同起点E_phys_cont/E_task，固定H反传 | train/val | 梯度见证、相同步数、COT漂移 | MUST | CONDITIONAL | D020通过 |
| D031 | M3 | 固定两种E、从相同初始化重训新H | train/val | AT相对AD/AH/AI的配对增益 | MUST | CONDITIONAL | D030 |
| D040 | M4 | 删除搬运与同容量自由场控制 | train/val | 机制与物理语义归因 | CONDITIONAL | NOT_STARTED | D031通过 |
| D041 | M4 | seed42/43/44配对复核 | train/val | 原始点估计、日期配对CI、seed差异 | CONDITIONAL | NOT_STARTED | 配置/超参数已冻结 |
| D042 | M4 | 图像底图AFNO复核/去边缘项/加ramp项 | train/val | 每次仅一因素，次要结果 | OPTIONAL | NOT_STARTED | 核心结论成立且有明确诊断需求 |
| D050 | final | 独立时间test与双语论文结果 | test | 预登记冻结协议 | CONDITIONAL | LOCKED | 来源/实现/多seed/审计要求满足 |

预算：GPU小时及各阶段墙钟上限待D001实测填入；未创建作业或持续监控。工程门槛为提案中的预登记候选值，不是已观察到的结果。
