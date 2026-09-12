# 每小时自动推进交接

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

## 2026-09-10 最新：诊断完成，用户要求停止持续监控

用户授权续跑207702后，新PBS208011已通过Gold并完成4个冻结模型前向，4×99849验证预测与既有结果逐值一致，独立NumPy实际loss汇总通过。见HEAD_LOSS_MASS_RESULTS_20260910.md。原低Gcs<20约5.7%训练行占实际kt损失98.87%～99.06%，QC1为95.46%～95.64%；属于固定checkpoint实际残差证据，尚不是AC退化的唯一因果证明。

用户明确“你不用持续监控”，cot每小时自动检查已删除。停止轮询与自动提交，等待用户按需检查或推进。以下旧“Gold等待恢复”“仍无输出”“每小时ACTIVE”仅是历史记录，不能再按旧文字重复运行207702/208011或创建新自动任务。

## 用户新增并发限制

同时最多3个未结束PBS任务，按账户跨项目统计，保守计入Q/R/H/W/T/E等所有非完成状态。每次qsub前实时核查qstat -u slfu，已有3个等待；串行提交并逐次重查，不能各论文分别提交3个，也不终止他人任务。GPU最多8卡限制仍保留。最新查询qstat -u slfu成功返回空队列；这不能倒推207702失败时的全队列状态，也不能证明Gold已恢复。207702是一个PBS任务，内部拟串行检查4个模型，不是同时提交4个任务。

## 最新用户要求：检查COT负收益原因（优先于仅等待producer的旧记录）

已完成 `docs/COT_FAILURE_CAUSES_20260908.md` 的8候选/全epoch/相同99849验证行分解：kt目标在GHI误差上隐含1/Gcs²权重；original低/中Gcs改善而Gcs≥600 RMSE+7.559，QC1并非相同分布；主要净退化在竺家。四维统计压缩为实现事实，既有R外推域退化仍保留。不能把零预测参考损失占比称为实际训练残差占比，不能声称唯一因果原因。

已准备四个既有best checkpoint全train/val只前向及独立实际loss汇总。PBS207702在prologue因 `Gold is abnormal, please contact administrator!` 失败（C/exit−2、约1秒），Python未启动，audits/head_loss_mass_20260908_v1不存在；失败日志已封存。下一步需明确计费/调度恢复证据后，确认原作业结束/输出状态，给重试另名PBS日志，再运行同一只前向协议。不要绕过PBS、反复盲投、修改权重或把提交失败当模型失败。前向成功后运行scripts/summarize_head_loss_mass_20260908.py并核对保存数组。模型/loss对照仅为建议，尚未冻结/执行；没有授权变化不扩大超参和test。producer新路径问题仍开放，但不再把它当作所有原因诊断的唯一前提。

## 最新用户授权：主动复查区域与实际像元（优先于下方旧停止记录）

用户最新怀疑 COT 区域等有误，要求修改小时任务并检查。自动任务已更新，当前状态 FIXED12_PAYLOAD_PASS_PRODUCER_BINDING_UNVERIFIED。先读 docs/COT_SPATIAL_REAUDIT_RESULTS_20260908.md 和 CPP_PRODUCER_GEOMETRY_REVIEW_20260908.md：冻结12行的源CPP→sidecar→R target/mask、full AGRI→Combined→R输入已逐值检查并独立复核通过；未发现本批下游湖南区域、站点互换、转置/翻转或一像元偏移错误。原HDF固定两站×C02/C13共4片经DN/LUT还原与full/Combined完全一致，另一程序仅读两张LUT后独立重建也通过。这些是有界抽样，不能宣称全量或源产品绝对地理注册已获认证。

剩余具体缺口是实际 Result/CPP NC writer、生成批次、checkpoint、模型输出到经纬度网格的注册/拼接以及真实AGRI输入时刻。相邻训练代码不写基础NC，NC缺少生产绑定字段；有界目录调查已结束，不能靠正确LAT/LON或文件名排除生产时已错配。已向用户询问生成脚本/批次日志的服务器路径；检查是否有该问题的新回复或明确新来源即可，没有线索不重复全树搜索、12行payload、HDF或旧配对诊断。若获得具体生产记录，先只读核验绑定，预先固定复现协议后使用既有12行检查写NC前的中间张量；不得按误差寻找最优位置。只有发现可复现错误并独立复核，才另名修复、判断影响资产并公平复跑原预算。当前没有因空间证据需要启动的训练，原科学负结果及test/IFS/样本公平规则保留。

自动任务 `cot`，heartbeat 每小时，ACTIVE，绑定本任务；不要重复创建。用户已授权继续已批准步骤。先读本清单、项目 AGENTS.md / IRRADIANCE_ERROR_PREVENTION.md、最新 STATUS 和 tracker，再核查状态。没有新证据或可行动变化时保持安静，不为每小时产生输出而重复训练、诊断或扫描。

## 最近巡检与主机时间

本轮定时巡检（调度 UTC 2026-09-08T08:13:55Z）：PBS 活动队列为空，既有完成日志/结果与本地登记 SHA 一致，无新可行动实验或 IFS 来源。补齐了学院服务器原先缺失的既有 ifs_station_source_20260907.json 报告副本，复制后 SHA a735934f31784c0dc9c6a68496cb5a90a65e11617723e22dce0af165fea3c98f 一致，没有重读源 NC 或改变任何数据/指标。

主机时钟检查：参考 UTC 08:16:11、本地 BJT 16:16:11，服务器回报 UTC 07:58:27，约慢 17分45秒。后续巡检必须分别记录调度/本地和服务器时间，优先用 hash/完成标记判断变更，不能直接用跨主机 mtime 判定新旧或卡死。没有修改系统时钟，也没有据此调整预报数据 UTC/BJT 或 target/init/lead。详情 audits/hourly_live_check_20260908T0813Z.json；研究结论与停止条件不变。

## 最新状态（覆盖历史报告中的待执行文字）

2026-09-08 15:39 BJT 实时 qstat -u slfu 无活动作业。本篇 S/R、train/val cache、head pack、8 次 A/AC pilot、后端数值见证和固定目标 COT 配对诊断全部已完成。207626 已从 qstat -f 的活动/保留记录中清除；先前 C/exit0/50s 快照与持久日志的完成标记均保留，不能因 Unknown Job Id 重新提交。CPU 配对诊断日志最终为 COMPLETE_CPU_CPP_REFERENCE_DIAGNOSTIC_WITH_CACHE_WARNING。

1. **GHI pilot 结果保留。** 207616 共 original/QC1 × A/AC × 两档 LR 的 8 次 seed42 探索训练完成。相同 99849 验证行，original A149.817451/AC154.578853，QC1 A149.905825/AC151.248653 W/m²，COT 后分别更差 3.1781%/0.8958%。预测键/真值/SHA/NumPy 全量重算和 4 个选择模型的固定 8 行 CPU 见证通过。未证明 COT 增益。
2. **数值 warning 保留。** 207626 完成，六原分片 130 序列 GPU 重放；CPU/GPU normalization 完全相同。benchmark/cuDNN TF32 同时关闭时 GPU/CPU 最大 log1p 差 4.053e-6；原设置对旧缓存仍有一分片 0.001033306 > 原门 0.001。没有放宽阈值，没有隔离 TF32 单因素，尚不能称旧缓存数值完全复现，也未定位具体模型/特征实现错误。
3. **同目标 COT 诊断完成。** 事前固定 64 个 R validation 目标，58 匹配目标/872 配对，152 缺 lead 和 6 完全缺配对原样保留。58 目标原文件/sidecar/S grid 坐标相同，872 配对三几何 669696 元素差 0。两支相同冻结 R、CPU FP32、batch32、norm/参考/mask，仅换 AGRI。耗时 18.607 秒，无 GPU/训练/GHI/test。相同 219310 像元贡献的 COT RMSE 真实输入 4.293596、外推输入 10.131402；两站全部 32 组均外推误差更高。独立保存 NPZ 审计 635 项指标差 0，原参考/mask/重复 real/全部配对键通过；不是第二次模型前向。
4. **诊断版本不重复。** 几何 v1 仅 UTC Z/+00:00 表示检查失败；诊断 v1 仅 checkpoint 相对 pack 路径解析失败，均在相应数值读取前停止且原失败保留。geometry v2 和 cot_pair_diag v2 是完整结果，不再运行 v1。诊断 v2 仅解析修复，同科学协议；数据时刻未修改。PowerShell JSON 的 DateTime 本地化显示也无需修数据。
5. **解释边界。** CPP 为参考产品、teacher binding 仍未闭合；R 曾在 validation 选模，同 target 跨 lead 重复计权且覆盖不同。不能将 58 unique-real RMSE 与 872 pair-pooled forecast 混比；不能把输入替换误差退化直接当作 GHI 负收益因果证明、普遍 COT 无效或正式 test 结论。

## 当前执行决定与触发条件

- 阅读 `COT_PAIR_RESULT_TO_CLAIM_20260908.md` 的最新二次语义审查；same-family/provisional 不能伪装正式接受。独立数值通过不自动批准科学主张。
- 按原 P202 止损规则，无明确实现错误时关闭本配置的正向扩大训练：不跑 P203 wide/AL，不扩 P204 多种子，不打开 P205 test，不为了获得正结果改架构、loss、kt 截断、日期/天气筛选或超参预算。保留全部 original/QC1、S/R 和旧 cache。当前没有待提交的 GPU 任务。
- 只有新证据定位到可复现的具体实现错误，才记录证据、独立复核、另名修复并按原同预算 original/QC1 A/AC 两档 LR 成对复跑；不得同时引入新方法。未解释完的后端差异自身不等于已证实错误，也不是反复重跑的授权理由。
- IFS 分支独立受 release/native-step/单位合同约束。已有 source 仅 step1–4 且无 release。`IFS_SOURCE_FOLLOWUP_20260908.md` 的有界本地线索复查没有新湖南历史库存/发布时间证据；旧湖北范围的 schedule 合同不适用此湖南范围。无新明确 manifest/来源/历史发布依据时，不重复原三 NC/全 index 审计，不重复搜全树。
- 若出现具体新 IFS 来源，先限定日期/文件/字段只读核实：区分实际 release、保守发布上界和 valid-time 离线匹配，核实历史适用性、同 cycle 所需 steps、SSRD 前一步和单位。通过并冻结合同后仍需独立设计/审查同 cohort 四格对照，不能自动宣称 COT×IFS 互补。不得跨缺口插值或伪造 release。
- 文稿只收窄到证据实际支持的范围。pilot 不直接写成正式正文最终结论；如改论文，英文 LaTeX 与中文 Markdown 同步，工程路径/哈希/修复历史留在工程文档或补充材料。
- 在以上条件均未变化时，结束该次小时检查且不通知；保留 heartbeat 活跃。重大完成、失败、新阶段或确实需要用户决策时才通知。不要自动扩大到 diffusion、第一篇或新的研究路线。

## 已完成资产与权威入口

服务器根：`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907`。
Python：`/public/home/slfu/miniconda3/envs/swc/bin/python`。

- S207380、R207372 已冻结：configs/s_frozen_hunan_seed42.json 与 r_frozen_repaired_seed42.json。
- 207522 train/val cache 全 1332 分片及 head pack 已验收；head pack train538053/val99849。QC1 train537605/val99849，448 条仅 train 疑似平直敏感性，不是已确诊设备故障或 BSRN 通用剔除标准，其他大 kt 保留。BSRN 主来源边界见 QC_PRIMARY_SOURCES_20260908.md。
- 207615 profile/8序列过拟合完成，实测 head batch2048；207616 两原目录 runs/head_A_AC_seed42_20260908_v1 和 runs/head_A_AC_qc1_seed42_20260908_v1。
- GHI 证据：audits/head_paired_pilot_20260908.json、head_paired_reload_20260908.json；docs/HEAD_PAIRED_PILOT_RESULTS_20260908.md。
- 数值证据：audits/r_cache_numeric_witness_20260908.json；docs/R_CACHE_NUMERIC_RESULTS_20260908.md 与 R_CACHE_NUMERIC_REVIEW_20260908.md。禁止重复 qsub207626。
- COT 诊断：audits/cot_pair_diag_20260908_v2/report.json、同目录 cot_pair_predictions_20260908_v1.npz；audits/cot_pair_results_audit_20260908.json。
- COT 完整报告/全时效图：docs/COT_PAIR_DIAGNOSTIC_RESULTS_20260908.md、COT_PAIR_RESULTS_REVIEW_20260908.md；figures/cot_pair_diag_20260908_v2/。
- 协议 SHA74a94abda6c1b7054b26baf586bf3b7e2e94fe62c71efc2ac5a0f3ee3998ccf5，不改原协议。没有来源/代码新变化不重复计算。

需要使用 GPU 的已批准新动作仍先优化一张 A100，batch 按实测吞吐与显存余量；最多 8 卡、给他人留资源，绝不终止他人作业。当前无 GPU 动作需求。
