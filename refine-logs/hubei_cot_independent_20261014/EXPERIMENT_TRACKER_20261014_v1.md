# 湖北独立COT实验执行表

2026-10-14。PLAN_READY：只完成方案与范围修订；无本计划PBS任务、无新训练或test读取。
权威方案：[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。

| ID | 阶段 | 任务 | 数量 | 状态 | 下一证据 |
|---|---|---|---:|---|---|
| HB00 | 数据 | 原始实测/CPP/grid及覆盖 | 三站 | PARTIAL | 学院AGRI已定位；原共享盘访问与实测时间语义待核验 |
| HB01 | 数据 | 湖北日期划分、标签/归一化 | 1套 | TODO | train/val/test/buffer清单，test仅元数据 |
| HB02 | R | 旧来源审计或新R训练、COT bank | 最多先1个R | TODO | 来源、重载、COT和GHI键连接 |
| HB03 | 适配 | 去两站硬编码、三站见证与profile | 3臂 | TODO | 有限梯度、参数量、共同batch及端到端耗时 |
| HB04 | 对照 | seed42 A/AC/AP | 9头 | TODO | 同键预测、重载及FP64复算 |
| HB05 | 对照 | seed43/44 A/AC/AP | 18头 | TODO | 三seed完整结果，不按方向取消 |
| HB06 | 评价 | 三站/省等权、共享日期CI、天气/案例 | 1包 | TODO | 主要终点及容量比较；保留负结果 |
| HB07 | 归因 | 条件AL同R隐特征 | 9头 | CONDITIONAL | AC获益或需要物理语义归因时触发 |
| HB08 | 预报准备 | 新划分S核验、预测AGRI/COT bank | 1套 | LATER | COT外推与输入可用性证据 |
| HB09 | 预报 | FA/FAC三站三seed；条件FAP | 18(+9)头 | LATER | 先完成同期结论和B3实施附件 |

批次执行不等于PBS任务个数；全账户未结束任务≤3。湖南作业/结果独立保留。

