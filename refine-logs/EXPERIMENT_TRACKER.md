# 第二篇实验跟踪表

## 2026-09-10 来源核查与后续授权

用户已授权来源核验后推进空间表示最小对照。最新证据见docs/SOURCE_FOLLOWUP_20260910.md：FD-107登录恢复，CPP生产目录仍700、writer绑定未闭合；新IFS根chensr/ifs_hres_china/raw_nc含2024–2026目录，2025抽样header有100时刻，历史下载请求包含湖南长步长，但时间坐标读取超时、历史发布合同未放行。空间表示准备见docs/COT_SPATIAL_REPRESENTATION_NEXT_20260910.md，可复用已有cot_log1p场，不重建或重跑S/R。本轮未提交训练、未打开test，监控保持关闭。

## 2026-09-10 按需核查：208030完成，种子稳定性未通过

服务器日志出现GHI_MULTISEED_TRAINING_AND_AUDIT_COMPLETE；audits/head_ghi_multiseed_20260910.json为PASS_ALL_12_FIXED_LR_PREDICTIONS_AND_CPU_RELOADS。8新增训练与12固定LR候选同99849验证行/真值/NumPy指标/CPU重载均验收通过，test_used=false。qstat记录已清除，Unknown Job Id不是失败，不重复提交；未取得PBS最终exit_status。

AC−A的GHI RMSE（W/m²，负值有利COT）：original seed42/43/44为−1.010415/−1.080577/+1.090858，均值−0.333378，样本标准差1.233923，2/3改善；QC1为−0.506901/+2.702890/+0.740733，均值+0.978907，样本标准差1.618096，1/3改善。仅新增43/44：original平均+0.005140，QC1平均+1.721812。原始三seed平均A149.959893、AC149.626515；QC1平均A149.132355、AC150.111262。

按冻结协议，新增种子方向不一致，当前四维COT统计拼接配置未通过种子稳定性判据。不能延续seed42的0.67%/0.34%作为稳定提升；不能由此推导COT普遍无效。分站亦不稳定，QC1四里三个种子均退化。保持所有候选与负结果，不自动加seed、调参、改变表示、打开test或恢复小时监控。若用户后续授权研究新表示，须另立单因素和参数匹配对照协议。

## 2026-09-10 最新授权：多种子验证

PBS208030已提交（单A100，提交前账户空队列），复用seed42、新增43/44，original/QC1各A/AC，共8新增训练；固定学习率不逐seed选取。协议docs/HEAD_GHI_MULTISEED_PROTOCOL_20260910.md，执行与验收入口docs/HEAD_GHI_MULTISEED_RUN_20260910.md。单seed208020已完成；本轮完成须以全部训练及12候选独立验收为准。用户未要求持续监控，小时任务继续关闭；无自动后续扩展或test。

## 2026-09-10 按需检查：208020训练及验收完成

服务器日志已出现GHI_LOSS_TRAINING_AND_AUDIT_COMPLETE；head_ghi_paired_20260910.json为PASS_ALL_8_PREDICTION_AUDIT，head_ghi_reload_20260910.json为PASS_4_SELECTED_CPU_CHECKPOINT_WITNESSES，均无test。qstat记录已清除，Unknown Job Id不作为失败或重投依据；未取得PBS最终exit_status，不宣称已核验退出码。启动到最终见证约6分半，非已核验PBS计费墙钟。

相同99849验证行、两站等权GHI RMSE（W/m²）：original旧A149.817451/AC154.578853，新A150.016921/AC149.006507；QC1旧A149.905825/AC151.248653，新A149.795619/AC149.288718。新目标下COT相对A改善0.673534%/0.338395%，差中差−5.771817/−1.849729 W/m²。新损失两cohort两档同LR的AC−A均为负，但收益并非两站一致：original四里+1.833593、竺家−3.854423；QC1四里+1.477040、竺家−2.490842 W/m²。original新A较旧A略差0.199470，不能称所有模型均改善。

这是单seed且验证选模的初步正向证据，支持目标函数选择会影响当前COT增益，不证明唯一机理、普遍有效或正式显著性。下一步建议冻结本配置做成对种子稳定性与分站/分时效检查，不自动启动；test/IFS/CPP源绑定限制保留，小时监控仍关闭。

## 2026-09-10 用户授权GHI损失对照

用户明确要求按推荐推进。独立协议见docs/HEAD_GHI_LOSS_PROTOCOL_20260910.md，执行记录见docs/HEAD_GHI_LOSS_RUN_20260910.md。PBS208020已提交并确认R/1 A100，原始与QC1各A/AC两档LR共8次小头训练，唯一改变损失，完成状态以训练及独立验收为准。旧配置关闭仍保留为历史结论，不阻止本次已授权的独立损失诊断。小时监控保持关闭，不自动扩展后续实验。

最新：固定 64 目标的 COT 配对诊断已完成，58 匹配目标/872 配对/152 缺 lead 原样保留；同 mask COT RMSE 4.293596→10.131402，32 组均为外推输入更差。独立 NumPy 重算 635 项指标差 0；无明确实现错误，旧 cache warning 保留，不启动正式扩展。见 COT_PAIR_DIAGNOSTIC_RESULTS_20260908.md 与 COT_PAIR_RESULT_TO_CLAIM_20260908.md。以下早期“下一步配对”等文字为历史记录。

本轮后续：207626数值见证完成（1 A100/50s/C/exit0），CPU/GPU归一化相同；后端均关闭时R最大差4.053e-6，原设置对旧cache有一分片max0.001033306未过0.001，保留差异警告。没有具体通道/norm/pack错误，不重训。下一步64个预定val目标的真实/外推输入同CPP掩膜诊断，先metadata/空间几何门禁。详见R_CACHE_NUMERIC_RESULTS_20260908.md。

更新：2026-09-08自动检查。PBS207616正常完成，单A100墙钟7分05秒，两套×A/AC×两档LR共8次探索pilot全部保留。全量预测SHA/相同99849验证行/真值及NumPy重算通过，4个选定模型独立CPU抽样重载通过。original A149.817451/AC154.578853，QC1 A149.905825/AC151.248653 W/m²，加入COT后分别差3.1781%/0.8958%；尚不支持本配置COT增益，进入P202无收益诊断，不放行正式扩展。QC1为疑似平直段敏感性，保留537605train；original538053train；全部源值、大kt、损失和冻结S/R保留。test未使用，IFS仍受门禁。详见HEAD_PAIRED_PILOT_RESULTS_20260908.md。每小时heartbeat cot继续启用并已切换诊断优先级。历史段落保留。

| ID | 阶段 | 任务 | 依赖 | 状态 | 产物 |
|---|---|---|---|---|---|
| P200a | 审计 | 湖南SimVP训练区域/split/权重/通道 | 现存资产查证 | FROZEN_VALIDATION_RELOAD_PASS_207522 | configs/s_frozen_hunan_seed42.json |
| P200b | 审计 | COT/CPP合同、缺测、可用通道与区域 | 无 | CPP_REPAIR_ACCEPTED_TEACHER_BINDING_UNVERIFIED | cpp_aligned_20260905_v2/acceptance.json |
| P200b-R | 上游训练 | 修正CPP下的湖南R | train/val完整pack、GPU门禁 | COMPLETE_VERIFIED_FROZEN_207372 | configs/r_frozen_repaired_seed42.json |
| P200c | 审计 | IFS周期、发布时刻、step和SSRD单位 | 无 | BLOCKED_RELEASE_MISSING_STEPS1TO4_ONLY | audits/ifs_station_source_20260907.json |
| P200d | 审计 | 共同cohort、标签、网格与先前test暴露 | M0核验 | QC_SENSITIVITY_DISPOSITION_FROZEN_ORIGINAL_PAIR_REQUIRED | audits/head_qc_view_20260908.json |
| P201a | smoke | 32序列端到端及8条过拟合 | P200a/b/d | FEATURE_PIPELINE_AND_A_AC_OVERFIT_PASS_207615 | audits/head_overfit8_20260908.json |
| P201b | profile | 共享cache/单卡batch/吞吐/显存 | P201a | HEAD_PROFILE_COMPLETE_BATCH2048 | runs/head_A_AC_seed42_20260908_v1/profile.json |
| P202a | pilot | A/AC seed42 train/val | P201b+source QA | CLOSED_THIS_CONFIGURATION_8_RUNS_NO_COT_GAIN_DIAGNOSTICS_COMPLETE | audits/head_paired_pilot_20260908.json; audits/head_paired_reload_20260908.json |
| P202a-R | 诊断 | 冻结R的6分片计算后端复现 | A/AC负结果+特征检查 | COMPLETE_207626_BACKEND_DEPENDENT_DIFFERENCES_WARNING_RETAINED | audits/r_cache_numeric_witness_20260908.json |
| P202a-pair | 诊断 | 固定目标 real/forecast AGRI 的 CPP 参考误差 | 数值见证+坐标/几何绑定 | COMPLETE_872_PAIRS_635_METRICS_EXACT_NO_IMPLEMENTATION_ERROR | audits/cot_pair_diag_20260908_v2/report.json; audits/cot_pair_results_audit_20260908.json |
| P202a-spatial | 用户追加诊断 | COT/AGRI实际像元、区域与源生成地理绑定 | 用户最新明确要求 | FIXED12_PAYLOAD_PASS_PRODUCER_BINDING_UNVERIFIED | docs/COT_SPATIAL_REAUDIT_RESULTS_20260908.md |
| P202a-cause | 用户追加原因诊断 | 全8候选误差/目标函数/epoch分解；固定模型实际train残差 | 已存结果及冻结best | COMPLETE_208011_FOUR_FORWARD_AND_INDEPENDENT_LOSS_SUMMARY | docs/HEAD_LOSS_MASS_RESULTS_20260910.md |
| P202b | pilot | AN/ACN seed42 train/val | P202a+P200c | WAITING_PREREQUISITES | validation_predictions |
| P203 | 决定 | wide与同CPP隐特征对照/域偏移 | P202 | CLOSED_THIS_CONFIGURATION_NO_IMPLEMENTATION_ERROR | docs/HEAD_PAIRED_RESULT_TO_CLAIM_20260908.md |
| P204 | 正式 | 核心四格3种子及必要对照 | P203通过+协议冻结 | WAITING_PREREQUISITES | checkpoints/config/norm/hash |
| P205 | 评价 | 同cohort test、CI、COT误差传播 | P204冻结 | WAITING_PREREQUISITES | predictions/metrics/figures |
| P206 | 写作 | result-to-claim、双语结果与结论 | P205 | WAITING_RESULTS | claim_audit及paper |

后续每个组每个seed拆分一行，登记服务器绝对run_dir、job_id、启动/结束时间、实际配置、输入cache哈希、退出码、验证选模规则。不得用本机报告快照更新“服务器正在运行”。

## 2026-09-07 推进记录

最新：R正式训练已完成。PBS207372，node21/GPU0，单卡profile选batch1024；8样本eval Huber从0.19621降至0.000288，通过后重新随机初始化。正式完成46轮（0–45），最佳第38轮（index37）；val Huber0.019415，COT MAE2.032317/RMSE4.246485。按保存预测重算和6样本CPU权重重载检查均通过，test未用。总PBS墙钟2min，正式循环计时15.66s（不含打包、传输、环境启动和profile）。权重与norm登记到r_frozen_repaired_seed42.json。S尚未完成选模，P201/P202保持等待，不将R检索误差当作GHI收益。

CPP修复成功46,036，源缺失84,904，剩余读取错误0；COT候选21,140（train11,978/val6,536/test2,626）。新R的数据包只读取train/val，逐条核验原始和sidecar哈希、网格、身份及掩膜，训练期归一化。旧配置和权重不复用。共享盘小文件读取仍在进行，不能称R已经训练。

已实现并部署方案规定的base16 U-Net/非负log1p(COT)/masked Huber，CPU两项测试及独立agent文档复跑通过。自动衔接程序在本机后台运行，等待数据完成后传输到学院、逐文件校验、PBS单卡profile、8样本过拟合，门禁通过才随机重置正式训练。学院根目录 `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907`；作业号将写入该目录 `submitted_job.json`。

GHI基础清单972,126行；按明确的15min区间晴空积分加入kt后972,028行（98行区间晴空值为0，保留排除原因）。无CPP质量筛选、无kt上截断、未计算测试指标，GHI回算最大误差1.14e-13 W/m²。FD-107输出 `/home/Data_Pool_3/wangyc/irradiance_paper/solar_cot_ifs_20260905/data/ghi_label_cohort_20260907_v1`。这仍不是IFS共同交集或已就绪的外推特征数据。

IFS完整缓存索引23,360条、仅step1–4；按实际source_path抽查2024/2025/2026三份NC均只有4个time，SSRD为J m**-2，没有可审计release。AN/ACN继续等待补全来源；不影响R以及后续A/AC。详见 `docs/PROGRESS_20260907.md`。

## 2026-09-05 P200执行更新
本轮已实际部署CPU审计和重划脚本。M0的8文件hash通过；COT候选重划12630/6954/2878，两次服务器执行与本地一致。旧COT训练/选模时段与论文val/test重叠，旧SimVP候选为湖北迁移权重，不能直接复用。IFS抽查未保留release信息，站点缓存只覆盖step1–4。当前未启动GPU训练或完整pilot，原因是上游科学合同尚未通过，不是等待用户重复批准。
新增产物见docs/P200_AUDIT_20260905.md、data/cot_candidate_m0_split_v1和scripts。下一任务是CPP质量来源/样本检查与合规湖南SimVP准备；IFS分支独立推进。测试代码8项通过，CPU复跑通过，不表示训练环境就绪。

## 2026-09-05 CPP质量来源更新
源到缓存18例核验通过，但旧cpp_valid_mask仅是finite掩膜，源文件没有独立QA字段。COT源Range为0–100，旧训练85阈值是实现过滤。旧CPP空间裁剪全部偏移一像元；18份独立修正样例坐标误差均为0。全量重建与新候选覆盖统计尚未完成，旧22,462条清单仅保留历史候选意义。源NC生成批次与实际checkpoint的关联仍未核实。详见docs/CPP_QUALITY_PROVENANCE_20260905.md及configs/cot_contract.json。新增6项测试，总计14项通过；未启动训练。

## 2026-09-05 CPP修复执行
用户已授权修复。已实现逐文件坐标匹配、独立sidecar、原文件hash关联、写后读回、完整inventory和断点续跑；配套读取器替换CPP并保留原AGRI/GHI。服务器启动时间22:52:05 BJT，小批量18例全部ok后自动进入全量，两进程。当前状态必须查询/tmp/cpp_repair_20260905.log及data/cpp_aligned_20260905_v2/status.json，不提前宣称完成。执行说明：docs/CPP_REPAIR_RUNBOOK_20260905.md。本地17项测试通过，服务器3项集成测试通过。

实际进展：v2于22:56:36 BJT通过18/18小批量门禁并启动全量，进程3798830。18份原文件hash均与修复前审计一致，坐标误差全为0；其中12份符合当前参考监督掩膜。全量尚未验收。首轮SQLite共享盘失败已改为JSONL，失败目录保留。
