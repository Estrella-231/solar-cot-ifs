# 反演COT序列预测实验 tracker

日期：2026-10-13。独立候选分支，计划已交付；尚未启动新数据生产或训练，不覆盖主任务T01–T32状态。

| ID | 实验块 | 工作/控制因素 | 状态 | 下一动作/依赖 |
|---|---|---|---|---|
| RC00 | 方案 | P-R与P-Hybrid、支持域、预算和判据 | PLAN_READY | 当前版本为建议合同，运行前绑定实测数据/profile |
| RC01 | B0 | 合规新月度区域R及CPP、真实图像16例核对 | PARTIAL | 总控确认原始AGRI/辅助文件存在且一个train时刻的站点tile CPP实跑成功；全train/val CPP/COT bank未生产，运行时再核服务器 |
| RC02 | B0 | train/val唯一帧R-COT bank及质量掩膜 | PLANNED | RC01；按时刻连接旧资产，不套旧序列顺序 |
| RC03 | B0 | 24帧序列与共同监督支持 | PLANNED | RC02；新test不读取 |
| RC04 | B1 | RetrievedCOTForecaster_v1及因果/mask/时间轴检查 | CODE_IMPLEMENTED_NOT_SMOKED | `scripts/retrieved_cot_forecaster_20261013.py`；模型前向/反向witness待可用Python/Torch运行环境 |
| RC05 | B1 | 小样本拟合、真实端到端单卡profile | BLOCKED_ON_BANK_AND_FREE_GPU | RC03/RC04；FD-201队列查询为空但6张4090均有显存/他人进程；不能视作空卡 |
| RC06 | B1 | 持久性、纯输运、SimVP→R同样本评价 | PLANNED | 新R/S来源、完整16lead预测 |
| RC07 | B1 | P-R seed42，最多2048train窗口、12epoch | PLANNED | RC05；与RC08同初始化/预算/支持域 |
| RC08 | B1 | P-Hybrid seed42，仅替换有效位置监督数值 | PLANNED | RC05；与RC07配对，合计8 GPU小时评审点 |
| RC09 | B3 | 固定事件真实RGB/CPP/R/预测连续图 | PLANNED | RC02冻结事件，RC06–08后绘图；标明日夜/无效支持 |
| RC10 | B2 | 两臂全量seed42/43/44共6次 | CONDITIONAL | 小试方向支持；随机重启，合计48 GPU小时评审点 |
| RC11 | B2 | 独立FP64复算、配对日期CI、覆盖与误差归因 | PLANNED | 每一批保存预测后执行；不依赖其他窗口审批 |
| RC12 | B2 | 仅一项归因：物理幅值loss/历史控制/CPP-only | CONDITIONAL | 必须有RC09/RC11证据，另名固定预算 |
| RC13 | B4 | 同GHI头：无COT/SimVP-COT/P-COT | CONDITIONAL | 主任务同期价值与本候选COT均支持；最多18次小头另计 |
| RC14 | 交付 | 分析/决策/仓库同步 | PLAN_DELIVERED_RESULTS_PENDING | 完成实验即分析并提交相关工件；当前无新实验结果 |

状态：模型代码已写入但未执行烟测。2026-10-13 11:47 BJT对FD-201只读复查显示PBS账户队列为空；6张4090中GPU0–2各19,393/24,564 MiB，GPU3–5各18,709/24,564 MiB，利用率46%–100%，均有其他负载/占用。CPP bank与COT预测训练均未开始；test载荷未读取。下一步先本地代码接口审阅，再待CPP/合规R bank和可用单卡完成profile。
