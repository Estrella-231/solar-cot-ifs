> Current status (2026-11-02): this is a retained historical planning/tracking document. Use [the repository index](../docs/REPOSITORY_INDEX.md) and [current model route](../docs/GHI_HEAD_ROUTE_20261102.md); old pending jobs and mandatory isolation steps below are not current instructions.

## 2026-10-22 湖北历史COT外推来源对照：attempt15部分完成

为检验训练期R-COT输入与外推时R-COT输入来源不一致是否会压低未来COT技能，已从原train日期划出14个未参与训练/模型选择的日期作内部留出；前后各清除1日，成对缓存和独立CPU审计通过。两组仅训练历史COT的来源不同，使用相同CPP检索参考目标、几何、样本、模型、种子、损失和更新预算。PBS `215542.tc6000` 在node22使用A800 GPU1/3/4/5完成seed42/43四组训练（每组10轮、1,350更新）；进程只在写训练后收据时因变量作用域错误退出，已有checkpoint、训练日志和预测已保留并核验。seed44首次补跑`215543.tc6000`在训练前因服务器目录写错而5秒失败、未产生更新；路径修复后`215544.tc6000`正排队请求node22两张卡并行完成剩余两组，固定每组global batch 32、1,350更新。COT评分尚未运行，无结果可判断；官方月度验证/test/buffer均未读取。详见`hubei_cot_transport_source_20261022/oof_source_cv_attempt15/EXPERIMENT_PLAN.md`与项目tracker。

## 2026-10-20 云掩膜方法核查与规划

已核查晴空加权、hurdle、oracle清零、学习硬门控及旧GHI重训；按用户确认的无效观测日复算旧硬门控，等站RMSE原始COT 102.881→硬门控103.215 W/m²，未见收益。判云分类器的太阳几何/键对齐已核验；54,895站点-帧键匹配，冻结分类器沿用旧太阳几何，本轮不冒称修正几何判云。PBS `215069.tc6000` 已在浙师大 node22/A800 GPU4完成24次四组配对训练。等站等seed RMSE为AGRI 89.485、+COT 88.264、+判云 88.186、联合 87.795 W/m²。主比较联合−COT为−0.469 W/m²（0.531%），日期块95%CI [−1.007,+0.047]跨0，且四里小幅退化；按冻结判据C1**未证实**，二值B3不启动。厚云分层出现探索性下降，但CPP/CLP仅覆盖1,265/4,226行，厚云169行/6个共同支持块且无多重比较校正；高GHI子集CI也跨0。评分与图见[结果报告](cloud_probability_input_20261020/B2_RESULTS_20261020.md)及`cloud_probability_input_20261020/results/four_arm_paired_v1_attempt1/figures/`。真实连续云事件图待整理；test未读取。

## 2026-10-18 湖北 test 链门禁修复与复核

真实 test 仍为 **NO-GO**，真实 test/buffer 载荷保持未读取。本轮修复并经新鲜独立只读复核确认：合成评估器的路径隔离、一次性账本反复制、严格时间戳解析和完整两日bootstrap。相关单测29/29通过。随后补齐FA/FAC/FAP三站三seed共27个已训练head checkpoint哈希，独立复核确认旧18项不变；但这仅是头部权重身份清单。full-train SimVP/R、真实真值builder、缺日AGRI、建包/推理前共享ledger、六月14日旧回放接触裁决和冻结统计配置仍未闭合。完整结论见 [湖北 test 链复核 v4](hubei_cot_independent_20261014/audit_receipts/test_chain_audit_20261018_v4/EXPERIMENT_AUDIT.md) 和 [测试门禁进度](hubei_cot_independent_20261014/PROGRESS_20261018_TEST_GATES.md)。未重复训练，未启动真实test评价。

## 2026-10-13 详细计划：COT反演主实验与并行推进

最新详细合同：[COT价值优先计划](cot_value_first_20261013/EXPERIMENT_PLAN.md)；[32项任务表](cot_value_first_20261013/EXPERIMENT_TRACKER.md)。2026-10-13晚按Yi × Sun评价框架新增validation同期COT反演复算，具体结果见`cot_value_first_20261013/COT_RETRIEVAL_VALIDATION_RESULTS_20261013.md`；此为反演参考一致性，不是未来预报结论。无新增训练、未读取test。已实时核实新月度SimVP来源合同（211815训练，12 epochs，seed42，月度split、test载荷0，checkpoint SHA在远端training_complete.json）；未来16lead的S→R预测bank仍未找到，下一步生成/验收validation冻结推理bank。此前“PLAN_ONLY”是本日早先状态，不代表本次进展。
# 第二篇实验跟踪表

## 2026-10-02 按月留块重训

合同 `hunan_monthly_block_2d_buffer1d_seed42_v1`：train35320 / val3140 / test3202，26个完整月份；2025-09稀疏月仅训练。独立全窗口、帧分离、缓冲、四季支持审计及远端源码/PBS/hash预检通过。CPU缓存与train-only归一化PBS `211814.tc6000`，单卡profile→12轮随机初始化SimVP依赖PBS `211815.tc6000`，提交时Q/H。当前新test只核对元数据，R/H按新划分重建待后续，旧模型不能用作新test成绩。执行路径和停止条件见 `../docs/MONTHLY_BLOCK_SPLIT_20261002.md`。

19:43:20 BJT准备211814已在node23启动R，转GPU队列并暂请求1卡，但归一化由六个CPU worker执行。19:44～19:45部署运行校验全OK，已处理至少3000/51236训练唯一帧、约15GiB本地缓存、无日志错误；211815仍H afterok依赖。尚未开始CUDA profile或SimVP训练，不报告新模型性能。

## 2026-09-24 候选方案登记（未执行）

GHI指导的COT运动与演变方案及D000–D050跟踪项见 [独立tracker](cot_dynamics_20260924/EXPERIMENT_TRACKER.md)。B0已启动，数据合同与CPU诊断有实测结果；GPU pilot因全账户PBS上限待空位，尚无E或GHI头训练，也未创建持续自动监控。先验收geometry24全链路，再做理想未来COT诊断与固定搬运，最后有条件增加小型演变网络和GHI梯度。

## 2026-09-18 用户决定：移除 COT 全局平均类统计特征

用户明确：区域均值、p90 等"做平均的" COT 统计特征全部从计划移除，不再作为未来实验的输入或对照组。`EXPERIMENT_PLAN.md` 已同步修订：删除 AC_stat_sp 组与四统计槽、B1/QC1/预算对应收缩；旧四统计量 AC 三 seed 结果仅作历史负结果锚点只读保留，不重训不扩展。`cot_gated_station_v3` 读出中的四统计拼接槽一并移除，该代码在 HPC 上尚未修改，B0 重跑前须改代码并重做 CPU witness。历史协议、结果文档（含反事实、多种子、weather 分层）保持原样不改写。

## 2026-09-17 当前执行入口：空间COT方向

论文、proposal和实验计划已修订；下方记录保留为历史，旧头的完成状态不表示新空间头完成。

| 阶段 | 状态 | 下一步与完成证据 |
|---|---|---|
| 双语正文/方案 | 已修订，结果待定 | 英文与中文同章节、同公式；新方案未继承旧审查分数 |
| B0 来源与实现 | PARTIAL PASS | 209381完成侧车绑定；209384 smoke通过。CPP历史生产链仍未闭合，不能进入新头pilot |
| B1 表示对照 | COMPLETE EXPLORATORY / STOP EXPANSION | PBS209422完成；AC_map 148.6723、A_sp 148.8904、AC_stat_sp 149.8294 W/m²，未达到1% pilot门槛，暂不自动扩展 |
| B2 归因 | NOT_STARTED | 容量、同监督latent、非中心空间置换；B1通过后展开 |
| B3 IFS增量 | BLOCKED | release/cycle/step/单位闭合，共同cohort四格对照 |
| B4 稳健性与测试 | NOT_STARTED | QC1预定敏感性、误差传播；协议冻结前不打开test |

2026-09-17 B0实际记录：PBS 209380 首次打包因夜间无标签缓存元组被误判为键不一致而停止，未产生GPU smoke或训练；修复为跳过无标签元组且仍要求每条标签恰好一次后，PBS 209381 写出新的 head_spatial_cot_trainval_20260917_v2。它验收637902行、1332个shard、训练像元全局log1p COT mean=2.0386260760/std=1.5273211328，test_used=false。PBS 209384 显式绑定v2完成GPU smoke，exit_status=0、batch2048、14,425参数、peak 0.1828575GiB/39.5861GiB，test_used=false。两次作业均已完成；未提交pilot、三seed、IFS或test，未恢复监控。

当前方案详见 EXPERIMENT_PLAN.md；任何状态推进必须附实际证据。

2026-09-17 用户明确要求“先不管，用现有的cot模型先做实验”。据此提交PBS 209422.tc6000，路径为 runs/spatial_b1_exploratory_seed42_20260917_v1；它固定S/R、base pack、空间sidecar、原始cohort、GHI损失、batch2048和LR0.0003，顺序运行A_sp/AC_stat_sp/AC_map各10epoch。运行合同明确CPP历史生产链为UNRESOLVED、test_used=false；结果只作探索性空间表示筛选。不得从本次单seed pilot推导稳定提升、物理因果或IFS协同。

PBS209422已完成，三组均保存checkpoint和validation predictions。完整原始数值及停止判断见docs/SPATIAL_B1_PILOT_RESULTS_20260917.md；本次不自动提交seed43/44、QC1、IFS或test。

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
## 2026-09-17 空间 COT B1/B2 判定

用户授权先用现有 COT 模型做实验。B0 sidecar 与 GPU smoke 已通过；B1 seed42 `A_sp/AC_stat_sp/AC_map` 验证 pilot 完成。随后在冻结 `AC_map` 检查点上完成无训练激活/归一化/空间置换诊断：COT 置零使等站 RMSE 148.6723→152.2370 W/m²，但非中心置换仅使其变为148.9558 W/m²。B2 seed42 控制 `AC_shuffle/A_proxy` 已完成（PBS209426，服务器 `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/runs/spatial_b2_controls_seed42_20260917_v1`）：148.6884/148.9985 W/m²，均与 AC_map 近似。结论：当前头使用 COT 幅值但未显示空间排列收益；空间图路线不进入 seed43/44、IFS 或 test，下一步仅允许定义幅值表示或先闭合 CPP provenance。首个控制脚本缺少 GPU 资源声明导致209425无CUDA，已修正并重提；不计为科学结果。
## 2026-09-18 站点感知对齐融合

服务器实时复核`configs/s_frozen_hunan_seed42.json`和`audits/station_grid_contract_20260908_v1.json`：Sili `[155:171,140:156]`、Zhujia `[65:81,142:158]`，两站最近网格均为patch `[8,8]`。用户授权后实现`station_aware_spatial_v2`：13 AGRI和1 COT逐像元14通道卷积，融合中心像元、中心5×5和全patch特征。B0 PBS209540通过，COT首层梯度和预测扰动均非零；全pack profile选batch16384（9.99GiB，888416 rows/s）。B1 PBS209543完成、test/IFS均未读：A_station 151.1860、AC_station 151.2517 W/m²，四里+0.7752、竺家−0.6437；但batch16384令10epoch更新数比原B1 batch2048合同少八倍，只保留作吞吐筛选，不能判定COT退化。batch2048重跑PBS209558在Python启动前因Gold异常退出（exit−2、资源时间0），无有效新结果；调度恢复前不扩展多seed/IFS/test。详见docs/STATION_AWARE_SPATIAL_V2_20260918.md。

## 2026-10-21 13:16 CST：attempt13 retry4 中途验证刷新

- 13:16:55 CST 实时 `qstat -u slfu` 显示未结束作业2/3：PBS `215356.tc6000` 为 `R`、node22、单GPU；评分作业 `215359.tc6000` 为 `H`，依赖 `afterok:215356.tc6000`。
- attempt13 MSE-control 正在 epoch 3 的241/282 batch（约85%）。完整 epoch 2 的物理验证 `val_cot_rmse=15.24987`，为当前中途 best；不是最终技能结果。COT-aux 臂尚未启动，评分未启动；未见失败/完成标记。
- 证据：远端 `.../hubei_cot_auxiliary_retrain_attempt13_retry4/runs/mse_control/training.log` 及 `checkpoints/`；完整路径见该 attempt 的 `EXPERIMENT_PLAN.md`。本次只读，test/buffer读取0，未操作PBS作业。

## 2026-10-22 OOF-history attempt15 score closure

六个COT外推fit已完成，并使用保存预测做FP64评分。训练时按4张A800（seed42/43四fit并发）后接2张A800（seed44两fit并发），每fit保持global batch32、world size1、1350更新。内部分组留出COT RMSE：OOF历史来源10.751、full-R历史来源11.332，点差−0.581（−5.13%），但日期块95%区间[−3.497,+1.990]跨零；所有种子区间也跨零，按预定门槛停止该cohort继续训练。厚云CPP参考分层点估计改善而薄/中等分层变差；CPP是检索参考，不是独立真值。此诊断没有GHI评估，官方月度验证/test/buffer读取均为0。详情与复现证据：`solar-cot-ifs/refine-logs/hubei_cot_transport_source_20261022/oof_source_cv_attempt15/RESULT_REPORT_20261022.md`；下一步仍是恢复全区域CPP原始源及质量/时间/网格溯源，不以此结果放行GHI训练。

同日只读源路径复查发现 `/public/home/slfu/ttzhou/swc/Data_Pool_3` 新可见，但无CPP/COT命名文件、仅有一个2023-01-23 CLP文件，预期Hubei GHI目录缺失；FD-201/FD-107仍在远端命令前断开。详见 attempt15 的 `source_route_recheck_20261022.txt`。这不是原始源恢复，不打开产品载荷，也不解除 full-region CPP 门禁。
