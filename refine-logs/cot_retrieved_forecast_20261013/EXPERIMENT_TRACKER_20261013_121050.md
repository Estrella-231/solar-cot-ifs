# 反演COT序列预测实验 tracker

日期：2026-10-13。独立候选分支，计划已交付；尚未启动新数据生产或训练，不覆盖主任务T01–T32状态。

| ID | 实验块 | 工作/控制因素 | 状态 | 下一动作/依赖 |
|---|---|---|---|---|
| RC00 | 方案 | P-R与P-Hybrid、支持域、预算和判据 | PLAN_READY | 当前版本为建议合同，运行前绑定实测数据/profile |
| RC01 | B0 | 合规新月度区域R及CPP、真实图像16例核对 | BLOCKED_NEW_R | 站点tile CPP单时刻工程 smoke 已成功；旧R与新test日期重叠34/52天，不能用于新test标签；新split R与全train/val CPP/COT bank尚未完成 |
| RC02 | B0 | train/val唯一帧R-COT bank及质量掩膜 | BLOCKED_NEW_R | 唯一帧时间键manifest已生成（train 51,236、val 4,497）；需日期安全新R后才生成R-COT，test不读取 |
| RC03 | B0 | 24帧序列与共同监督支持 | MANIFEST_READY_BANK_BLOCKED | 8→16 manifest 35,320 train、3,140 val已由冻结清单生成并通过时间/分割校验；待R/CPP bank及逐帧有效性 |
| RC04 | B1 | RetrievedCOTForecaster_v1及因果/mask/时间轴检查 | PASS_SYNTHETIC_ENGINEERING_SMOKE | FD-201 RTX 4090上合成单批前向/反向/更新/checkpoint reload通过；不能替代真实数据或时间轴验证；记录见 `smoke_results/smoke_gpu0_fd201_20261013.json` |
| RC05 | B1 | 小样本拟合、真实端到端单卡profile | BLOCKED_ON_NEW_R_AND_BANK | 必须先有按新月度train训练、日期安全的R及train/val帧bank；本次GPU smoke不是profile。提交时重新查PBS/GPU；历史快照不能代表当前资源 |
| RC06 | B1 | 持久性、纯输运、SimVP→R同样本评价 | PLANNED | 新R/S来源、完整16lead预测 |
| RC07 | B1 | P-R seed42，最多2048train窗口、12epoch | PLANNED | RC05；与RC08同初始化/预算/支持域 |
| RC08 | B1 | P-Hybrid seed42，仅替换有效位置监督数值 | PLANNED | RC05；与RC07配对，合计8 GPU小时评审点 |
| RC09 | B3 | 固定事件真实RGB/CPP/R/预测连续图 | PLANNED | RC02冻结事件，RC06–08后绘图；标明日夜/无效支持 |
| RC10 | B2 | 两臂全量seed42/43/44共6次 | CONDITIONAL | 小试方向支持；随机重启，合计48 GPU小时评审点 |
| RC11 | B2 | 独立FP64复算、配对日期CI、覆盖与误差归因 | PLANNED | 每一批保存预测后执行；不依赖其他窗口审批 |
| RC12 | B2 | 仅一项归因：物理幅值loss/历史控制/CPP-only | CONDITIONAL | 必须有RC09/RC11证据，另名固定预算 |
| RC13 | B4 | 同GHI头：无COT/SimVP-COT/P-COT | CONDITIONAL | 主任务同期价值与本候选COT均支持；最多18次小头另计 |
| RC14 | 交付 | 分析/决策/仓库同步 | PLAN_DELIVERED_RESULTS_PENDING | 完成实验即分析并提交相关工件；当前无新实验结果 |

状态更新：模型代码已通过合成工程 smoke；真实帧bank、日期安全的新split R和预测训练尚未完成。train/val唯一帧及8→16序列元数据清单已生成，test payload访问为0，元数据校验通过。旧R日期对照显示新test 52天中34天落入旧R训练/验证日期，不能用旧R产物形成独立test教师标签。CPP单时刻tile smoke仅验证官方tile读取与融合链路，不证明反演精度。下一步依赖新split R及真实train/val唯一帧COT bank；资源快照过期，训练前必须重新查询。
