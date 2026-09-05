# 第二篇实验跟踪表

日期：2026-09-05；本次未提交服务器作业。TODO不是已开跑，门禁待核不是观测失败。

| ID | 阶段 | 任务 | 依赖 | 状态 | 产物 |
|---|---|---|---|---|---|
| P200a | 审计 | 湖南SimVP训练区域/split/权重/通道 | 现存资产查证 | TODO | asset_manifest.json |
| P200b | 审计 | COT/CPP合同、缺测、可用通道与区域 | 无 | TODO | cot_contract.json |
| P200c | 审计 | IFS周期、发布时刻、step和SSRD单位 | 无 | TODO | ifs_availability.parquet |
| P200d | 审计 | 共同cohort、标签、网格与先前test暴露 | M0核验 | TODO | cohort_manifest与audit.json |
| P201a | smoke | 32序列端到端及8条过拟合 | P200a/b/d | WAITING_PREREQUISITES | smoke_audit.json |
| P201b | profile | 共享cache/单卡batch/吞吐/显存 | P201a | WAITING_PREREQUISITES | profile.json |
| P202a | pilot | A/AC seed42 train/val | P201b | WAITING_PREREQUISITES | validation_predictions |
| P202b | pilot | AN/ACN seed42 train/val | P202a+P200c | WAITING_PREREQUISITES | validation_predictions |
| P203 | 决定 | wide与同CPP隐特征对照/域偏移 | P202 | WAITING_PREREQUISITES | go_no_go.md |
| P204 | 正式 | 核心四格3种子及必要对照 | P203通过+协议冻结 | WAITING_PREREQUISITES | checkpoints/config/norm/hash |
| P205 | 评价 | 同cohort test、CI、COT误差传播 | P204冻结 | WAITING_PREREQUISITES | predictions/metrics/figures |
| P206 | 写作 | result-to-claim、双语结果与结论 | P205 | WAITING_RESULTS | claim_audit及paper |

后续每个组每个seed拆分一行，登记服务器绝对run_dir、job_id、启动/结束时间、实际配置、输入cache哈希、退出码、验证选模规则。不得用本机报告快照更新“服务器正在运行”。
