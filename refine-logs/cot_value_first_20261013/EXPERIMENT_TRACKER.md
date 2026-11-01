# COT价值优先任务表

## 本轮更新：同期反演主线写作与厚度分层（2026-10-14）

按用户最新要求，先完成已确定Method并整理结果占位。T43已同步扩写paper/main.tex及main_zh.md为8节、10个公式，核对当前16×16 R、3×3逐像元双分支H、归一化、标签、损失与配对训练；工程来源另存paper/METHOD_EVIDENCE_20261014.md。独立只读协作复核无实质性方法/翻译错误；机械比较10个公式一致，引用/标签无悬空，图1文件存在。编译器仍在初始化阶段报平台标准目录缺失，整稿PDF未验证。

T44完成已有保存验证输出的COT厚度分析：原4,479行主结果与共享52日期CI复现；有效中心CPP子集1,284行/46日，厚云30≤COT≤100共173行/25日，A→AC RMSE为74.929→58.105 W m⁻²（−22.45%），AC−A区间[−21.862,−12.359]。四里/竺家各三个seed均改善。较低厚度组降幅1.48%/1.70%/1.70%，CI均跨0；分层仅为描述性验证分析，不构成单调趋势或预报收益。表与结论已同步进中英正文。

当前写作主线为同期总体收益→厚度依赖与站点差异→湖北独立重复→预报应用。湖北结果和匹配训练forecast-GHI结果仍待完成，不提前声明预报提升，不把全部收益衰减归因于外推；本轮无训练/PBS/test读取。已有B4停止状态未因写作改变。

| ID | 工作 | 产物 | 状态 |
|---|---|---|---|
| T43 | 按实际代码扩写同期方法、预报应用及中英结果占位 | paper/main.tex、main_zh.md、METHOD_EVIDENCE_20261014.md、PAPER_PLAN.md | PASS_SOURCE_AND_BILINGUAL_REVIEW；整稿编译BLOCKED_PLATFORM |
| T44 | 已存验证预测按中心CPP/R-COT阈值分层 | results/cot_thickness_story_20261014/ | PASS_VALIDATION_DIAGNOSTIC；原主结果复现；厚云组改善，其他参考层CI跨0 |

以下“最新状态”及过程段保留为此前阶段记录；当前写作状态以上段及T43/T44为准。

最新状态（2026-10-14 BJT）：T36全3,140窗bank、T37固定头全验证归因、M0/M1晴空语义/门控诊断及T18 R-AL表征控制均已完成并通过保存预测的FP64复算。T38已用T18同一4,479行保存预测精确复现预注册共享52日期bootstrap与站点分别抽样敏感性区间；CI差异由抽样设计解释，主分析口径不变。T39将冻结验证结果与主张边界同步写入中英文稿；T40新增脚本驱动的双面板结果图，修正初审可读性意见后通过复审。T32补充更新工程交接，明确当前模型链与验证证据，并标注2026-10-03主体为历史快照。T41实时复核发现旧Motion+Source COT bank绑定旧R且R来源NC生成绑定未核实，不并入当前新月度共同支持。T42再次调用内置编译器仍报平台标准目录缺失；没有LaTeX诊断可用于修源，需等待编译环境变化后再试。预测COT/AGRI输入替换会增加冻结AC头误差，但不构成forecast-head成绩；B3不支持启动B4。T18显示AL与AC的验证差值未分辨，现有COT增益不能唯一归因于显式COT物理语义。test仍关闭。

历史进展：T34已完成PBS 213860：271窗/两站/16 lead的站点中心3×3预测AGRI实值已保存，形状[271,2,16,13,3,3]；dataset index、sequence ID及每lead target_time与现有COT bank逐项一致（PASS），test payload读取0。T35完成PBS 213861固定AC头2×2离线归因，上游CPP支持bank中的7,005实例、8个BJT日期：只替换AGRI的两站等权ΔRMSE +11.73 [2.27,17.54]；真实AGRI下只替换COT +5.14 [−3.98,11.91]；预测AGRI下COT替换 +10.14 [−3.18,18.28]；联合替换FF−TT +21.88 [1.09,34.55] W m⁻²（95%配对日期块CI）。这是冻结真实输入AC头的oracle/分布偏移诊断，不是已训练forecast-head结果。T33完成PBS 213859固定seed42 AC头离线归因：真实目标AGRI/几何不变，仅替换同期R-COT为预测AGRI→冻结R的COT。7,005 forecast实例、529个唯一站点目标键、8个BJT目标日；窗口来自上游CPP支持bank，非完整自然验证集。两站等权delta-RMSE=+5.14 W m⁻²，按目标日期块95% CI [−3.98,11.91]跨0；四里+4.73、竺家+5.55。长lead差值扩大（+240 min分别+21.36/+12.08）。这是oracle-AGRI COT-only归因，不是端到端预报成绩。独立FP64复算及预测实例见`refine-logs/cot_value_first_20261013/results/forecast_cot_fixed_head_oracle_agri_20261014_v1/`。T19/T20冻结SimVP→R CPP支持验证bank完成（271窗、550,400共同有效像元、8个完整双站16lead窗口；test关闭）。FP64主口径3×3逐lead站点等权后lead等权：S→R RMSE25.538、持续性28.252，差−2.714，1日日期块95% CI [−6.476,0.493]跨0；厚云COT≥30 CSI 0.082 vs 0.189，训练参考尺度11×11 SSIM 0.2356 vs 0.2468，均未优于持续性。T22完成文献与M03图件，并补完整共同支持到达诊断：8窗全来自2025-08-21且相互重叠；四里SimVP→R/持续性各命中1/8，命中比CPP早105 min；竺家均0/8，CPP两站均8/8有厚云参考事件。该结果仅描述性、无CI；旧输运产物不符合新月度双站键，不并入。依据T17，GHI预报头仍gated，不启动B4训练；GHI预报增益未评估，test保持关闭。

| ID | 阶段 | 执行者 | 前置任务 | 工作 | 验收产物 | 资源 | 状态 |
|---|---|---|---|---|---|---|---|
|T01|B0|当前会话|—|既有211902/212019及R/S/H来源审计|asset_inventory.json|CPU|PARTIAL（旧R、历史COT bank、月度SimVP及CPP资产已按生产记录远端核验；211902精确作业映射仍未确认）|
|T02|B0|当前会话|—|实时队列、可用GPU/磁盘/路径核验|resource_snapshot.json|CPU/GPU|PASS_FOR_ZJNU_SINGLE_GPU（zjnu-hpc node22 8×A800-80GB物理空闲；PBS已接受单卡任务213623；FD-107/201本轮未参与）|
|T03|B0|当前会话|—|既有站点工作簿按新日期清单抽取标签，连接Ineichen晴空值并核对唯一target/掩膜|data_contract.json|CPU|PASS（train/val标签及clear-sky已落盘；最终主掩膜train 50,458、val 4,479）|
|T04|B0|当前会话|T01,T03|复用已训练R及CPP资产，核对新日期归属和覆盖|retrieval_contract.json|CPU/GPU|PASS_FOR_CPP_SUPPORTED_TRAIN_VAL（新月度R seed42训练及FP64验证完成；pack/转移审计通过；13个训练时刻CPP NetCDF/HDF读取失败，且CPP仅支持部分候选）|
|T05|B0|当前会话|T02,T03|节点本地盘缓存、checksum与读取吞吐|cache_manifest.json|CPU/IO|PASS_SINGLE_LOAD_IN_MEMORY（R训练已从已审核pack一次加载至内存并保持训练期间常驻；不逐epoch重读共享文件）|
|T06|B0|当前会话|T04,T05|按唯一train/val目标时刻复用冻结R生成观测AGRI的站点16×16 log1p(COT) bank|bank_manifest.json|GPU|PASS（55,733/55,733帧；train 51,236/val 4,497；source errors=0；R checkpoint SHA绑定通过；test未打开）|
|T07|B0|当前会话|T01,T03|3×3 A/AC/AP架构、参数及真实数据可用性witness|head_contract.json|CPU/GPU|PASS；54,937条共同train/val pack，独立键/形状/有限值/哈希审计通过，test未读取。|
|T08|B0|当前会话|T05,T06,T07|单卡吞吐、batch与30epoch预算冻结|profile.json|GPU|PASS；node22单张A800 profile选择batch=2000，至少12 updates/epoch，峰值分配0.189 GiB；已记录小头算力轻的限制。|
|T09|B1|当前会话|T08|四里R-A/AC/AP seed42|3 runs|GPU|PASS；PBS 213829，30 epoch；输出及独立FP64核验见 ghi_heads_new_r_seed42_v1。|
|T10|B1|当前会话|T08|竺家R-A/AC/AP seed42|3 runs|GPU|PASS；PBS 213829，30 epoch；输出及独立FP64核验见 ghi_heads_new_r_seed42_v1。|
|T11|B1|当前会话|T09,T10|六臂重载与逐样本FP64验收|pilot_audit.json|CPU/GPU|PASS_PROVISIONAL；保存预测FP64独立复算与checkpoint CPU reload均完成；CPU/A800最大逐点差0.452 W m-2，RMSE差≤0.0046 W m-2，作为跨设备浮点差记录。单种子未作主结论。|
|T12|B1|当前会话|T11|两站R-A/AC/AP seed43|6 runs|GPU|PASS；PBS 213832/213833，逐样本FP64复算完成；AC−A=-5.684、AP−A=-0.104 W m-2站点等权RMSE。|
|T13|B1|当前会话|T11|两站R-A/AC/AP seed44|6 runs|GPU|PASS；PBS 213838 Sili、213839 Zhujia均exit_status=0；各A/AC/AP 30 epoch；新test未读取。|
|T14|B2|当前会话|T12,T13|18臂主终点、配对CI与分层|experiments/monthly_R_station16_20261013_v1/results/GHI_HEAD_THREE_SEED_RESULTS_20261013.md；GHI_WEATHER_STRATIFICATION_20261013.md|CPU|PASS_VALIDATION_ONLY；FP64与日期块bootstrap通过。总体AC−A=-3.740、AC−AP=-3.582 W m⁻²。季节描述分层均为负；同键CPP/CLP云况诊断中，cloudy子集840行、AC−A=-3.730 [−7.813,−0.289]、AC−AP=-3.559 [−7.831,−0.119]，clear/mixed CI跨0。该分层仅1291/4479行有CLP键，主GHI cohort保持不变；CLP非独立真值。|
|T15|B2|当前会话|T11|固定头COT扰动与输入依赖诊断|experiments/monthly_R_station16_20261013_v1/results/COT_INPUT_DEPENDENCE_VALIDATION_20261013.md|CPU|PASS_VALIDATION_DIAGNOSTIC；物理零/中心保留外围置零/保值邻域置换/同站同月错配均已完成；置换敏感度很小，错配与置零会明显退化；结果仅为冻结头输入依赖诊断，不作因果证明。|
|T16|B2|当前会话|T03,T06,T14|预先选16例并输出真实图像/GHI案例|experiments/monthly_R_station16_20261013_v1/results/t16_partial_case_manifest_v1.json；refine-logs/cot_value_first_20261013/results/t16_cpp_event_selection_v1/T16_EVENT_MANIFEST_20261013_v2.json；experiments/monthly_R_station16_20261013_v1/results/t16_cpp_case_figures_v2/|CPU|PARTIAL_CPP_REFERENCE_CASES；仅JJA有连续17帧CPP参考序列，四里/竺家各2例。缺口是CPP历史参考的15分钟时间覆盖，不是AGRI或新R-COT bank全时段缺失；T16 selector只需17帧公共有效像元>0，并未要求256/256。全覆盖口径与方案见`results/T16_SEASONAL_COVERAGE_AUDIT_20261014.md`。v2图件是同期观测回放，test关闭。|
|T17|B2|当前会话|T14,T15,T16|决定应用验证/错误修复/收缩结论|refine-logs/cot_value_first_20261013/results/phase_decision_T17_20261013.json|CPU|PASS_ADVANCE_TO_B3_WITH_BOUNDED_CLAIM；验证集AC−A=-3.740、AC−AP=-3.582 W m⁻²，三seed同向且日期块CI低于0；仅支持进入物理COT外推验证，不支持test泛化、GHI预报增益或空间纹理归因。|
|T18|B2条件|当前会话|T14|R-AL同监督隐特征两站三seed|6 runs + audit|GPU/CPU|PASS_VALIDATION_ONLY_REPRESENTATION_CONTROL; PBS 213869 exit 0, six 30-epoch runs complete; common-key FP64 audit and independent local recompute pass. AL−A/AL−AP improve; AL−AC CI crosses 0. See T18_RAL_CONTRACT_20261014.md and results/T18_RAL_RESULTS_20261014.md.|
|T19|B3|current|T01,T04,T05|Frozen-S source, train/val prediction and geometry validation|refine-logs/cot_value_first_20261013/results/forecast_contract.json|CPU/GPU|PASS_CPP_SUPPORTED_VALIDATION_DIAGNOSTIC; PBS213858 completed 271 CPP-supported validation windows, source/array hash and shape checks pass, test_payloads_read=0. No existing transport bank found. See `results/forecast_cot_val_bank_20261014_v1/` and the FP64 result report.|
|T20|B3|当前会话|T19|预测COT与历史COT train/val bank|forecast_cot_manifest.json|GPU|PARTIAL_VAL_ONLY; CPP-supported validation bank generated for 271 windows, not train; train inference is not required for frozen-model validation scoring.|
|T21|B3|current|T20|16lead COT station/local/patch common-support evaluation|cot_lead_metrics.json；MOTION_SOURCE_REUSE_AUDIT_20261014.md|CPU|PARTIAL_SIMVP_PERSISTENCE_OFFLINE_R; FP64 1x1/3x3/16x16 lead metrics and thick-cloud/arrival diagnostics saved. Live audit confirms the located Motion+Source bank is old-contract and bound to old R (`cot_repaired_seed42_v1`) with NC-generation binding unverified; excluded from current monthly comparison. CPP support is sparse and not independent truth.|
|T22|B3|当前会话|T21|文献可比性、覆盖、厚云与到达诊断|results/cot_forecast_case_20261014_v1/COT_FORECAST_ARRIVAL_COHORT_DIAGNOSTIC_20261014.md|CPU|PASS_BOUNDED_VALIDATION_DIAGNOSTIC; 完整共同支持窗口8个，但全来自2025-08-21且相互重叠。固定3×3 COT≥30首达：四里SimVP→R/Persistence各命中1/8（唯一命中较CPP早105min），竺家均0/8；CPP厚云窗口均8/8。离线目标AGRI→R也未普遍恢复峰值。仅描述性、无CI，不能视为8个独立事件；CPP是检索参考。SSIM和厚云CSI不优于持续性，T17的GHI预报头仍gated；test关闭。|
|T23|B4|当前会话|T17,T20|F-A/H/F/HF结构、因果梯度与预算见证|forecast_head_contract.json|GPU|NOT_APPLICABLE；B3未达到forecast-GHI扩展门槛|
|T24|B4|当前会话|T23|四里F四臂seed42|4 runs|GPU|NOT_APPLICABLE；随T23关闭|
|T25|B4|当前会话|T23|竺家F四臂seed42|4 runs|GPU|NOT_APPLICABLE；随T23关闭|
|T26|B4|当前会话|T24,T25|八臂配对键/预测/重载验收|forecast_pilot_audit.json|CPU/GPU|NOT_APPLICABLE；随T23关闭|
|T27|B4|当前会话|T26|两站F四臂seed43/44|16 runs|GPU|NOT_APPLICABLE；随T23关闭|
|T28|B4条件|当前会话|T27,T14|C1成立而C2弱时F-O六臂及替换诊断|6 runs + offline audit|GPU|NOT_APPLICABLE；无forecast-head训练，不满足条件分支前提|
|T29|B4|当前会话|T27,T21|预报增量、COT误差传播与公共cohort|forecast_results.json|CPU|NOT_APPLICABLE；预测AGRI/COT仍仅有冻结同期头离线替换诊断|
|T30|验收|当前会话|T14,T17,T22,T29|冻结当前验证结论与test许可边界|results/analysis_freeze_validation_only_20261014.json|CPU|PROVISIONAL_NARROW_CLAIM_REVIEWED；同源Codex复核，不是跨模型/完整完整性接受；未授权test|
|T31|验收|当前会话|T30|新test一次正式评价、复算与结果保全|test_receipt.json|CPU/GPU|CLOSED；须独立明确放行|
|T32|交付|当前会话|T31|中英稿、科学图表、工程交接与结论范围|paper + handoff|CPU|PARTIAL；中英稿主张与冻结结果同步（T39）；Fig.1已生成并复审（T40）；`docs/END_TO_END_HANDOFF_20261003.md` 已补当前合同/结果并标出旧快照；整稿编译因内置编译器平台目录不可用而未验证，test关闭。|
|T33|B3 attribution|当前会话|T17,T20,T22|固定AC头真实目标AGRI/几何不变，比较同期R-COT与SimVP→R预测COT|forecast_cot_fixed_head_oracle_agri_20261014_v1/FORECAST_COT_FIXED_HEAD_ORACLE_AGRI_DIAGNOSTIC.md|GPU inference + CPU FP64|PASS_BOUNDED_DIAGNOSTIC；CPP支持bank的7,005行/8 BJT目标日；站点等权ΔRMSE +5.14 W m⁻²，日期块CI [−3.98,11.93]跨0；oracle-AGRI且窗重叠，不是部署成绩。|
|T34|B3 attribution|当前会话|T20,T33|保存271个冻结SimVP验证窗的两站站点中心3×3预测AGRI实值，并与现有COT bank键一致性核验|forecast_agri3x3_bank_20261014_v1/forecast_cot_val_bank.npz|GPU inference|PASS；PBS 213860；[271,2,16,13,3,3]；序列/索引/时刻键精确匹配，test读取0。|
|T35|B3 attribution|当前会话|T33,T34|冻结AC头2×2离线对照：真实/预测AGRI × 同期R-COT/预测R-COT|fixed_head_2x2_agri_cot_attribution_20261014_v1/FIXED_HEAD_2X2_AGRI_COT_ATTRIBUTION_20261014.md|GPU inference + CPU FP64|PASS_BOUNDED_DIAGNOSTIC；7,005行/8 BJT日；AGRI替换+11.73 [2.27,17.54]，COT单因素CI跨0，联合替换+21.88 [1.09,34.55] W m⁻²；固定头 oracle/分布偏移压力测，不是训练后的forecast-GHI成绩。|

条件项T18/T28由阶段判定决定，未启动不算失败；若启动，T30必须纳入其审计结果。T17关闭预报扩展时T23–T29标NOT_APPLICABLE并记录理由，不无限等待，也不伪造预报结果；T30/T31只针对已有证据支持且符合门禁的同期反演范围。R重建如必要由T04拆出独立子任务/预算，不悄悄算作已完成。

复旦承接第4个及以上并发GPU任务；PBS账户≤3未结束作业。当前会话直接准备、提交和评价，不再需要交接、请求单或签署收据。

## 2026-10-13：阶段决策与SimVP外推源审计

- T17判定通过：在新月度split验证集上，AC三seed均优于A/AP，AC−A=-3.740 W m⁻²，日期块95%CI [-6.180,-1.597]；进入B3物理COT外推评价，但不宣称test泛化、空间纹理学习或预报GHI收益。T15显示邻域排列敏感性很低，T16真实案例仅覆盖JJA，均作为限制保留。判定文件：`results/phase_decision_T17_20261013.json`。
- T18 R-AL条件已触发：AC收益与R感受野/CPP监督混杂并存；冻结特征层/投影和输入bank合同后再训练，不改变已完成A/AC/AP结果。
- T19已现场只读核验月度SimVP `best.pt` SHA-256 `f255da84...11c50ec4`、12轮完成记录、monthly split、normalization及模型/数据代码哈希，预测几何为8帧历史+16帧逐lead目标时刻。详细候选合同：`results/forecast_contract.json`。尚未产生预测bank，也未做新的PBS提交。
- 本轮现场查询看到账户作业 `213850.tc6000` 为运行态；因此没有提交训练/推理作业。先本地完成推理代码和一批回放见证，再提交必要的单卡bank任务。
运行时记录host、job/pid、GPU、命令/config、数据及模型来源、时间、输出及结果。缺实际数据才记阻塞，缺scope receipt或窗口回复不算。旧windows目录保留已有产物。执行规则见EXPERIMENT_PLAN.md第10节。

## 2026-10-13 11:57 BJT：使用学院服务器空闲 A800

- `zjnu-hpc` (`tc6000`) 可访问；`qstat -u slfu` 提交前为空。GPU2 (`node22`) 备注 `gv maint`，但本机 `nvidia-smi` 实际列出8张 NVIDIA A800-SXM4-80GB，显存各约2 MiB、利用率0%；PBS调度接受了 node22 的单卡请求。
- 已提交并启动 `213623.tc6000`，状态 `R`，node22，PBS申请 `ppn=8:gpus=1`，请求8小时。启动日志确认Gold预留成功，实际只暴露分配的GPU 0；作业启动时该卡约2 MiB显存/0%利用率。
- 作业任务：用官方旧合同冻结R `cot_repaired_seed42_v1`（SHA-256 `0cbdb932f49945259d5f1704316edaf6481be7cfdb54692236f829988925799c`）和月度train-only SimVP归一化，对冻结新清单的唯一train/val真实AGRI目标帧生成四里/竺家站点16×16 `log1p(COT)` bank。候选表SHA-256 `b85c2841e33cc92eda848acb183f926d49564b6c131d6418730e6b53978e2807`；模型实现SHA与checkpoint绑定均通过。此为旧R对新月度真实输入的下游复用，不是新月度R训练或COT精度结论；无test行/载荷读取。
- 代码已通过本地与服务器Python语法编译，首个train帧及对应AGRI/几何文件存在。输出目录 `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/cot_value_first_20261013/results/real_agri_cot_bank_rseed42_monthly_v1/`。本记录只说明作业已运行，完整bank与覆盖状态待作业收据。
- 2026-10-13 12:00 BJT进度核验：作业仍为`R`，已处理1,408/55,733个唯一train/val目标帧，缺失输入0。node22 GPU0约571 MiB/80 GiB、利用率0%；其余7卡仍各约2 MiB/0%。当前任务受小文件读取吞吐限制，暂不并开更多卡以免增加共享存储竞争。

## 2026-10-13 12:35 BJT：开始新月度 R 重建

- 按用户要求，独立训练新的 `monthly_station16_COTUNet_seed42_v1`，不再把旧 `cot_repaired_seed42_v1` 的下游推理银行误称为新split R。新模型从随机初始化开始，采用新月度 train/val 日期键、AGRI 13+几何3输入、站点16×16 patch、train-only归一化及有限安全的 `log1p(COT)` masked Huber。
- 数据来源门禁通过：candidate manifest SHA-256 `b85c2841e33cc92eda848acb183f926d49564b6c131d6418730e6b53978e2807`，CPP inventory SHA-256 `0a303ec6965be360689f3891e96b5e4085ad3e6550b2db6504a8b2ac35565dd2`；输入文件只允许train/val。PBS node22不能访问CPP源，完整打包改由FD-107读取原始CPP与月度train/val AGRI。
- 更新后的pack脚本通过本地语法检查及FD-107真实源smoke：一个train时刻、一个val时刻，共4个站点样本；AGRI/CPP最大网格坐标误差0.0°；有效零COT保留；读取错误0；test载荷读取0。完整pack已在FD-107单CPU进程启动，PID `179560`，输出 `.../monthly_R_station16_20261013_v1/data/full_trainval_pack_v1`，日志 `logs/full_trainval_pack_v1.log`。该步骤I/O为主，不占GPU。
- 新训练模型拓扑已对照历史 `COTUNet16` 固定为两级下采样及 base width 32；训练脚本语法通过，已同步到PBS项目脚本目录且SHA-256核验一致。PBS全账户最新队列只有 `213623.tc6000`（旧R诊断银行，仍在运行）；本轮尚未qsub。
- node22两次1-epoch训练smoke均完成；第二次已修复只读memmap warning，成功保存/重载checkpoint与验证预测。独立CPU FP64审计通过，保存指标与独立复算一致，且确认预测样本、mask和pack标签逐条对齐。smoke仅含2个train、2个val站点样本且val均为厚云，误差不作科学解释。
- 12:39 BJT更新：FD-107 full pack 已访问1,000/17,600个有CPP支持的时刻，写出1,051个监督站点patch，读取约18.5 GB；进程继续运行，尚无完整数据门禁结果。
- 下一门禁：完整pack完成后核验train/val样本数、源错误/缺失分类、数组shape/hash、0值样本和网格映射；将pack转至PBS后再核验拷贝hash。随后实时复查全账户PBS队列与node22资源，提交单卡新R训练并做checkpoint重载及独立FP64验证复算。test保持关闭。

## 2026-10-13 13:02 BJT：完整数据包仍在NFS读取

- FD-107 PID `179560` 仍运行，日志最后一次flush为1,000/17,600 CPP支持时刻、1,051个监督站点patch；`/proc/179560/io`累计读取约26.8 GB（12:39时约18.5 GB）。当前等待点为`wait_on_page_bit_common`，表示正在等待读取页完成；尚无完整数据包验收结果，不能提交正式训练。
- 最新PBS账户队列仅1个未结束作业：旧R的`213623.tc6000`（R）；已完成smoke `213643.tc6000`为C，不计入并发上限。node22 GPU0为约571 MiB/80 GiB，其余7卡约2 MiB，利用率均0%。训练仍等待数据包完整审计和传输hash门禁。
- 13:04 BJT更新：pack日志已推进至1,500/17,600个CPP支持时刻、1,571个监督patch；FD-107进程读取约29.8 GB，仍运行。正式验证审计新增按站点、COT强度分层的FP64误差统计；更新后本地/PBS脚本SHA-256一致（`557c42dc66297f19fa420d616479ed2457d4e7488d21651e536cdbc0c4ab45c9`）。
- 13:10 BJT Luna只读复核：pack推进至2,500/17,600（2,613个监督patch），累计read_bytes约45.0 GB；NFS I/O仍持续。PBS仍仅旧R诊断任务`213623`未结束，新R训练尚未提交。
- 13:17 BJT：FD-107只读SSH命令开始明显延迟，短命令及日志查询均无法在正常窗口返回；上次可确认状态仍为2,500/17,600且I/O继续增长。PBS节点和`qstat`通道正常，仅`213623`运行。未重启pack进程。
- 13:23 BJT Luna复核：FD-107 SSH恢复，pack继续到4,000/17,600支持时刻、3,996监督patch，read_bytes约75.1 GB；进程仍处于NFS I/O等待/读取状态。全量包及训练尚未完成，PBS仍仅`213623`运行。
- 13:27 BJT Luna复核：pack推进至4,500/17,600（4,339监督patch），read_bytes约84.2 GB，PID存活；PBS仍仅`213623`运行。
- 13:30 BJT：FD-107本轮只读检查超时，暂未拿到新flush进度；最近已知pack PID仍存活且I/O增长。PBS qstat正常，仅`213623`未结束。
- 13:39 BJT Luna复核：FD-107 pack推进至6,000/17,600（5,472个监督patch），累计read_bytes约107.1 GB，PID存活。PBS仍只有旧R诊断作业`213623`未结束。
- 13:43 BJT Luna复核：新一轮监控确认pack到6,500/17,600（5,916个监督patch），read_bytes约114.2 GB；FD-107 PID存活，PBS仍仅一个旧诊断作业`213623`未结束。
- 13:46 BJT Luna复核：pack推进至7,000/17,600（6,468个监督patch），read_bytes约124.9 GB；PID仍运行，PBS仍仅`213623`未结束。
- 13:50 BJT Luna复核：pack推进至7,500/17,600（6,984个监督patch），read_bytes约133.8 GB，日志无错误；PBS仍仅`213623`未结束。
- 13:54 BJT Luna复核：pack推进至8,000/17,600（7,475个监督patch），read_bytes约144.1 GB，日志正常；PBS仍仅`213623`未结束。
- 14:07 BJT：FD-107连续两次只读状态查询无输出超时，暂不能取得新进展。最后确认仍为8,000/17,600、PID存活且读盘；PBS正常，仅旧R诊断作业`213623`运行。
- 14:09 BJT：FD-107 SSH shell仍有延迟，但SFTP成功读取到pack日志；进度已到10,000/17,600支持时刻、9,669监督patch。无完成标记；未重启任务。PBS仍仅旧R诊断作业运行。
- 14:18 BJT Luna经SFTP复核：日志到11,000/17,600（10,701监督patch），没有失败/完成提示；PBS仍仅旧R诊断作业`213623`处于R。继续等待完整pack状态与审计。
- 14:24 BJT：SFTP日志查询恢复，pack已到12,000/17,600（11,707监督patch），日志mtime为14:24，无错误/完成提示。PBS旧诊断作业`213623`状态C，新R训练仍等待数据门禁。
- 14:28 BJT：SFTP日志推进到12,500/17,600（12,189监督patch）；`pack_status.json`尚未出现。PBS当前唯一未结束作业为213666 `hunan_detail_MH`（其他实验），账户并发数1；新R仍未提交。

- 2026-10-13 14:27 BJT：实时复核 `213623.tc6000` 已从队列退出；远端 `bank_manifest.json` 状态为 `COMPLETE`，55,733/55,733帧齐全（train 51,236、val 4,497），源错误0、errors为空、test载荷未打开；日志含 `MONTHLY_REAL_AGRI_COT_BANK_COMPLETE`。当前PBS队列仅见另一作业`213666.tc6000`，不属于本bank任务。

- 2026-10-13 15:54 BJT：新月度R的train/val pack通过transfer后审核（16,812监督patch，train 15,414、val 1,398；CPP网格误差0.0°；13个train时刻NetCDF/HDF读取错误，单独作为来源错误统计）。PBS `213712.tc6000`完成seed42新R训练及FP64验证：best epoch 20/28（zero-based），CPP支持白天像元RMSE 9.074、MAE 4.646；厚云(COT>=30) RMSE 14.129；train-only归一化、test载荷0。新R COT bank PBS `213716.tc6000`已在node22 GPU2运行，目标唯一train/val 55,733帧。R训练包单次加载入内存；R训练profile选batch 1,284、13 updates/epoch。

## 2026-10-13 16:15 BJT：新R bank与GHI对照准备

- Luna只读核对PBS `213716.tc6000`仍在node22 GPU2运行；账户另有`213666.tc6000`，共2个未结束任务。新R bank日志进度6,400/55,733（约11.5%），source errors=0，test_payload_opened=false。速度受共享盘逐时次小文件读取限制；没有再启动同路径读盘任务。
- 完成3×3 GHI训练数据连接器 `experiments/monthly_R_station16_20261013_v1/prepare_ghi_head_pack.py`：按`(station,target_time_utc)`连接train/val GHI标签、目标时刻AGRI、真实太阳几何和新R bank的3×3 COT；从16×16 COT裁`[7:10,7:10]`，不池化；仅保留主标签掩膜，缺失项不按行序错连。AGRI及log1p(COT)统计量从train样本计算。
- 完成配对训练器`train_ghi_heads.py`与单卡PBS入口`run_ghi_heads_seed42.pbs`。训练固定同一3×3样本、逐epoch样本顺序、30轮预算与GHI物理误差损失；A/AC共用初始化主干与读出，AC/AP约等参数；PBS入口先在单卡做三臂profile，再依吞吐与每站至少12次更新/epoch选择batch，顺序训练两站六臂seed42。Python源码本地编译通过并已同步；远端`swc`环境Python 3.10编译通过。profile/训练作业等bank完整、连接pack完成后再提交。

## 2026-10-13 识别诊断：新月度 R 云/厚云固定阈值验证

- 对已完成并经FP64独立复算的新月度R验证预测做云/厚云识别评价；有效CPP验证像元356,169，1,398个站点样本，test载荷读取0。参考COT标签与预测均按物理单位比较，预测`log1p(COT)`仅逆变换一次。
- 固定阈值：COT≥0.1为云COT代理，COT≥30为厚云。合并结果：云POD/precision/CSI=`0.957/0.871/0.838`，厚云=`0.677/0.663/0.504`；厚云漏报16,425、虚警17,526。按站点结果见`COT_CLOUD_DETECTION_VALIDATION_20261013.md`和JSON原始计数。
- 解释范围：CPP支持的白天验证像元；CPP是检索参考而非独立真值。该云标签由COT阈值定义，不等于独立CLP云掩膜验证；仅seed42，不能推出多seed稳定性或GHI增益。未读取test，也未提交新训练。
- 复算脚本：`evaluate_cot_cloud_detection.py`（SHA-256 `FD062D1CCD75915B76AF0F2E1D1784DC9F3D8113E31233468E81188330DAEB84`）；服务器结果在`experiments/cot_value_first_20261013/results/cot_cloud_detection_monthly_R_seed42_validation.json`。

## 2026-10-13 CPP CLP 同样本云识别及文献口径对照

- 从704个唯一验证时刻的CPP NetCDF逐站裁取CLP；1,398/1,398验证样本均与COT有效掩膜完整重合，共356,169像元，覆盖率100%，经纬网格最大误差0°。候选清单SHA与grid_static SHA与pack审计一致，test载荷读取0。
- 固定预测为物理COT≥0.1，参考为同像元CLP∈{1,2}（有云）/CLP=0（晴空）。合并云POD 0.925、晴空POD 0.740、云FAR 0.123、HR 0.863、KSS 0.665；四里/竺家分项及全部混淆矩阵见`cot_cloud_vs_cpp_clp_monthly_R_seed42.json`。
- 公开FY-4B CLM质检（2024-12全圆盘、与MYD35匹配，92,528,101点）报告云POD/晴空POD/云FAR/HR/KSS=`0.922/0.816/0.045/0.901/0.737`。本项目的云POD相近，但参考产品、范围、COT阈值定义及日间站点样本不同，故仅作指标量级上下文，不能声称达到或优于业务CLM。
- COT RMSE对照也仅作背景：本项目9.074（CPP验证参考）；FY-4A全天MODIS基准研究报告12.79；FY-4B夜间CALIPSO基准报告不同太阳角条件1.26/1.99。卫星、参考、条件和分布不同，不能直接排序。未找到同阈值COT≥30公开CSI/FAR数字；厚云CSI=0.504保留为项目内部指标。
- 结论与严格限制见`COT_CLOUD_LITERATURE_COMPARISON_20261013.md`。CPP CLP与COT同属CPP产品链，非独立真值；真正业务产品头对头评估还需相同时次FY-4B CLM。此次没有读取test，也没有提交训练。
- CLP提取脚本`extract_validation_clp.py` SHA-256 `22BFA129173AF104130765DDB0F2F2C4C26B6C3ADBCE9B16FE7C57CEAF0B216C`；评分脚本`score_cot_cloud_vs_cpp_clp.py` SHA-256 `6F798B6ACCA161416AA9F693C217CF68B93797597E196792564D543FF9BBF3C3`。原始CLP数组及提取审计保存在`clp_reference_monthly_R_seed42.npz`和`.audit.json`。
- 17:00 BJT实时检查：`213716.tc6000`仍运行，账户未结束作业2个；bank到23,424/55,733，source errors=0。为减少下一阶段共享盘流量，pack生成器按唯一AGRI帧分组，同一时刻两站patch只打开一次AGRI/geometry源文件；同时记录3×3 AGRI有效像元比例、数组hash。修订版hash已重新同步并通过服务器Python编译和PBS shell语法检查。
- 17:05 BJT：PBS `213716.tc6000`继续运行，进度25,472/55,733，source errors=0，PBS账户仍有2个未结束作业。pack job继续等待，不与bank竞争同源I/O。
- 19:48 BJT：新R bank完成并核验：55,733帧完整（train 51,236/val 4,497），source errors=0，`test_payload_opened=false`；COT bank输出绑定R checkpoint SHA `342596…f001`。PBS账户实时查询只有`213666.tc6000`未完成，于是提交CPU数据连接作业`213781.tc6000`，开始生成同键GHI/AGRI/几何/COT 3×3 pack；账户当前未结束任务数2。待pack审计后再提交单卡profile/seed42训练。
- 2026-10-13 20:14 BJT：PBS 213781 live check: 8-core CPU pack is R, elapsed about 7m34s, log reached 6,001/54,937; pack manifest is not yet present. Account has 2 unfinished jobs (213666 and 213781). Do not overlap another shared-data reader; after completion audit counts/hashes/support then submit single-GPU GHI-head profile/training if queue remains below the 3-job cap.

## 2026-10-14：T15/T16局部退化深入诊断（不改变T16覆盖或T17结论）

对四个冻结JJA案例及完整train/val做18个已冻结head的CPU重放、梯度/输入替换和错误分解。病例AC相对A平均下调约38–42 W/m²，竺家A原已低估，AC进一步放大低估；两个竺家窗口共享13/16目标，不能作为两个独立事件。关键失败子群是R中心COT≥3但实测区间kt≥0.9：train AC−A四里+12.466、竺家+16.820，val分别+16.822/+8.357 W/m²。低kt子群仍有改善，解释总体收益和局部退化并存。这是使用目标的事后诊断，不是可部署门控或独立天气分类。

保持head冻结，将R替换同刻CPP不能修复竺家主要退化（同CPU病例RMSE仅−0.196/−0.666）；替换直接H太阳几何的病例影响≤0.218 W/m²，未重算R几何，不全局排除该通路。COT排列响应很弱，±15 min/±2像元探针没有共同一致的修复方向。51个原HDF元数据路径在PBS均不可读；缓存扫描行偏移约四里+184 s、竺家+159 s，待源链核验，不能直接移动标签。未读test、未提交训练或PBS任务。

新证据：`experiments/monthly_R_station16_20261013_v1/results/t16_deterioration_forensics_20261014/T16_DETERIORATION_CAUSES_20261014.md`、逐目标JSON/CSV、两组真实数值图、SPA/扫描元数据/优化日志审计。优先闭合扫描时刻和15min支持，再对照时间序列/站点作用表示；T18来源控制仍必要。整体三seed验证增益和T16 4/16覆盖边界保留。

### 2026-10-14追加：四例原始扫描时刻已闭合

FD-107源HDF可读；绕开NFS用户site依赖后，按真实HDF行读取`NOMObs/NOMObsTime`，51/51成功。原始行中点为四里标称时刻+184.3515–184.3565 s、竺家+159.467–159.470 s，与aux最大差7.45微秒。确认云图实际观测在标签结束约3min后；这是时间支持差异，未发现aux偏移计算错误，未移动标签。竺家严重低估点在标签区间及随后区间GHI均约400，AC约223，时间差不足以解释全部退化。原始receipt、缓存比对及图03见同诊断目录；下一步明确全cohort区间表示和公平合同，保留空间作用/表示不足的主线。
- 2026-10-14 BJT：补充完成T19单窗物理反归一化→冻结R回放见证。PBS `213855.tc6000` 单A800运行成功；固定val窗 `hunan_202404181745`，+15…+240 min、两站共32幅16×16 COT，观测AGRI→R输出与已保存新R bank最大绝对差 `0.0012164`（log1p COT）。本地FP64对保存数组重算得到同一差值。SimVP输入仅历史8帧和24帧几何；未来AGRI只作离线R回放参考。复现代码 `experiments/monthly_R_station16_20261013_v1/witness_simvp_to_R_replay.py`，SHA-256 `9227aa5ca98da5df9436f806be4e536fe5e66bc3edcd0d851888284b4818f94`；结果与压缩数组已复制到 `results/simvp_to_R_replay_witness_20261014_val0/`，witness SHA-256 `BD835E0070AC914F1B1A96026DFDB292B84E2C78F77FAF76083E84F3BFA46FCC`、NPZ SHA-256 `D8A518F73C4AC8CDDF0CFAB5F81F12B84FEA3CE4B7169F4393A97EC983C0C2C3`。这不评价SimVP外推精度；下一步按T19先完成100窗单卡吞吐profile。提交前全账户1个未结束作业，提交后2个，低于3个上限；当前 `213855` 已完成，test载荷未打开。





T34 PBS 213860已提交，账户全队列核验：213850运行、213859完成、213860运行；未结束任务2个，低于3个上限。任务只做冻结SimVP→R验证推理并附加缓存预测AGRI站点3×3裁块，不训练模型，输出目录独立且将按原bank dataset index/sequence/time keys精确核验；test不读取。



T34核验：本地下载副本与服务器NPZ SHA一致；271个序列和全部 target_time_utc/dataset_indices 与T20原bank一致，预测AGRI裁块为物理通道值，13通道、3×3，无target-support过滤。T35 PBS 213861已提交；提交后账户未结束作业为213850和213861两项，低于3项上限。


## 2026-10-14 T35 固定头 2×2 输入归因

PBS 213861完成：固定AC头在同7,005个 forecast instances 比较真实/预测AGRI × 同期R-COT/预测R-COT。FP64两站等权四组RMSE TT=75.29、FT=87.02、TF=80.43、FF=97.16 W m⁻²。整块BJT日期配对5,000次bootstrap（8个日期）：AGRI只替换+11.73 [2.27,17.54]；真AGRI时COT替换+5.14 [−3.98,11.91]；预测AGRI时COT替换+10.14 [−3.18,18.28]；联合FF−TT +21.88 [1.09,34.55]。说明在冻结真实输入头中，预测AGRI输入偏移影响更明确，COT外推误差呈变差趋势但区间跨0。由于头未用forecast输入训练且仅8个BJT日期、窗口重叠，这些都不是端到端GHI预报成绩；T17的训练门仍未解除。图与详细限定见`refine-logs/cot_value_first_20261013/results/fixed_head_2x2_agri_cot_attribution_20261014_v1/`。test未打开。

|T36|B3 all-validation extension|当前会话|T19,T20|冻结SimVP→R对全3140个验证窗口生成预测AGRI、同期AGRI参考与持续性COT；不按CPP筛窗|forecast_agri3x3_allval_20261014_v1/forecast_cot_val_bank.npz|GPU inference + CPU FP64|PASS；PBS 213864 node22单A800；3140/3140验证窗口，独立FP64审计通过，test关闭。|
|T37|B3 attribution extension|当前会话|T36,T03|固定AC seed42头对全val共同键作真实/预测AGRI×同期/预测COT 2×2离线替换|fixed_head_2x2_allval_agri_cot_attribution_20261014_v1/|GPU inference + CPU FP64|PASS；PBS 213865完成，66,590条共同记录/52个BJT日期块，独立FP64复算通过；固定头诊断而非forecast-head成绩。|
|T38|验收敏感性|当前会话|T14,T18|用同一T18保存输出复现共享52日期主bootstrap及站点分开抽样敏感性|results/ci_method_reconciliation_20261014/T18_BOOTSTRAP_METHOD_RECONCILIATION_20261014.json|CPU FP64|PASS；两CI各自精确复现对应口径，主结果沿用预注册共享日期CI；validation only，test读取0。|
|T39|交付|当前会话|T30,T38|按冻结验证证据同步修订英文正文与中文对照，保持科学主张边界|paper/main.tex；paper/main_zh.md|CPU/LaTeX|CONTENT_SYNC_PASS；中英文结果、单位、区间和限制一致；内置LaTeX编译器启动失败（platform standard directories unavailable），编译未验证；test未读。|
|T40|交付|当前会话|T39,T14,T20|生成并渲染核验同期GHI增益与COT外推边界双面板论文图|paper/figures/fig1_cot_value_and_forecast_limits.pdf；plot_cot_value_validation_summary.py|CPU|PASS_FIGURE_DATA_RENDER_AND_REVIEW；初审后删除小字注释、补全控制边界/聚合/无量纲口径，复审无阻塞；中英稿已插图；整稿LaTeX编译仍受T39环境问题阻塞。|
|T41|B3来源审计|当前会话|T21|实时核验旧Motion+Source bank能否作为新月度T21公平对照|results/MOTION_SOURCE_REUSE_AUDIT_20261014.md|CPU/SSH只读|PASS_EXCLUSION_DECISION；zjnu-hpc可达，slfu队列1个运行中任务；旧bank存在但绑定旧R且`source_NC_generation_checkpoint_binding=UNVERIFIED`，不并入新合同；未提交PBS、未读test。|
|T42|交付验证|当前会话|T39,T40|重试当前英文稿内置编译|paper/main.tex|内置LaTeX|BLOCKED_PLATFORM；2026-10-14重试仍仅返回`Unable to find standard directories for platform`，未进入TeX诊断；不重复尝试，待编译器环境变化后重试。|

## 2026-10-14 T36 全验证窗forecast-COT bank扩展（运行中）

原CPP支持bank仅有271/3,140个验证序列，完整双站16lead覆盖仅8个目标日，不能代表月度验证cohort。新任务对冻结SimVP月度验证集全部3,140窗运行预测AGRI→R、同期真实AGRI→同一R及持续性→R；CPP只按键稀疏附带，不参与选窗或主比较。输出用于误差传播/链路一致性诊断，真实AGRI→R只是同一检索器参考，不是独立COT真值。

提交与错误记录：PBS 213862因脚本DATA路径漏写`/ttzhou`退出1，未产出；213863计算完成但旧“必须与CPP子集bank完全同键”断言不适用，未产出。修正数据根路径及键校验后，213864于node22单卡完成3140/3140验证窗口。forecast AGRI与旧271窗bank逐元素完全相同；COT最大log1p差0.000454；CPP支持mask完全相同。NPZ哈希`0acc9e4095b8d401e609b50905f7252b6c897c93698cd1e3a774e32ec84ecaaf`；本地独立FP64审计PASS，test读取0。

T37 PBS 213865经afterok依赖完成。全val固定AC头2×2在真实/预测AGRI×同期/预测COT四组合上共66,590行，覆盖2,817序列和52个BJT目标日；远端/本地CSV SHA一致（`462f3ab7b23f82ca0d581b40b5becb62e94454830eb653a61489684b03e20129`）。本地FP64复算：站点等权RMSE TT/FT/TF/FF=118.14/126.46/125.42/139.39 W m⁻²；FT−TT +8.32 [3.01,14.14]，TF−TT +7.28 [4.52,10.54]，FF−FT +12.93 [9.73,16.58]，FF−TT +21.26 [14.22,29.14]（5,000次配对BJT日期块bootstrap）。所有区间均为固定头离线输入替换诊断；AC头未在预测AGRI/COT上训练，TT含未来真实AGRI，不能称为可部署GHI成绩或COT独立物理验证。结果详见`results/fixed_head_2x2_allval_agri_cot_attribution_20261014_v1/ALLVAL_FIXED_HEAD_ATTRIBUTION_20261014.md`。两任务仅用validation，test未读；现有证据不越过T17/B4训练门槛。

## 2026-10-14 T36/T37 完成后的阶段判断

T36扩展到全部3,140个validation窗口，CPP仅在原稀疏支持像元上评分：跨站/lead简单平均的forecast→R、持续性→R、同期真实AGRI→R对CPP RMSE分别为26.11、27.73、26.04 COT；该跨lead等权汇总仅作补充，原预注册主口径仍是逐站逐lead等权，需引用原v3报告。forecast与真实未来AGRI→R输出的全窗口差异，是同一R的响应差，不是真实COT误差。T37全验证固定头诊断表明预测AGRI和预测COT误差都可传到GHI头；预测COT单独影响在预测AGRI下更大。因为GHI头训练分布不含forecast输入，且对T17主收益尚无支持，暂不启动B4成组训练。后续先完成全验证窗口COT逐lead/天气分层及真实误报/漏报案例复核，再按T17停止规则决定是否有充分依据进入预测输入公平训练。test保持关闭。


T36季节/lead差异已复核并保存到`results/forecast_agri3x3_allval_20261014_v1/ALLVAL_R_OUTPUT_DISCREPANCY_20261014.md`。旧表按UTC月份得到的3×3输出差RMSE为DJF 17.24、MAM 15.28、JJA 13.52、SON 19.86；它曾误标为BJT。2026-10-14独立复算后，正确BJT分组见下方更正记录。两者均为冻结R响应差，不是COT真值误差。后续T18已完成，B3未显示稳定forecast-COT优势，不启B4。

### 2026-10-14：T36季节分组UTC/BJT口径纠正

复核T36原脚本发现旧季节表实际按UTC目标月份分组，却标为BJT。新增独立FP64 NumPy复算，384个站点×季节×lead×空间尺度组逐项重现旧CSV（RMSE最大绝对差6.4e−14），再将目标时刻转UTC+8生成正确BJT季节表。更正后的两站×lead等权3×3输出差RMSE：DJF/MAM/JJA/SON=`17.29/15.23/13.52/19.86`，偏差=`−2.30/−1.07/−1.55/−3.62`。全时段总体指标和CPP稀疏支持评分不变；本项仍是冻结R对预测AGRI与同期真实AGRI的响应差，不能称为物理COT误差或外推准确率。test载荷读取0。报告、纠正后CSV、脚本及复算JSON均在`results/forecast_agri3x3_allval_20261014_v1/`。


T18合同已冻结候选层、局地crop、AL/AP参数匹配、train-only特征归一化和三seed同监督对照；文档为`T18_RAL_CONTRACT_20261014.md`。下一步实现特征提取/键连接与目标环境单batch梯度见证，之后单卡profile；本轮没有提交训练。

### 2026-10-14 T18 更新：AL 特征键审计完成，特征 bank 作业排队

- 新增只读冻结R特征提取器 `experiments/monthly_R_station16_20261013/extract_r_al_feature_bank.py`，从同一16×16站点AGRI输入的 `dec1`、即COT head之前捕获 `[B,32,16,16]`，按站点中心 `[7:10,7:10]` 输出32×3×3；不从GHI包3×3 AGRI伪造输入，不改现有A/AC/AP实现。代码SHA-256 `A3C5BEDD81597632A27455CA4C15EDD2DD52F15356BD547F05A746A8400E7C2B`。
- 独立核对GHI pack 54,937条 `(station,target_time_utc)` 均唯一；train 50,458、val 4,479、唯一时刻29,655。逐行与111,466条候选、55,733帧R-COT bank核对split、AGRI路径和站点坐标，0重复、0缺键、0错配。审计记录 `results/r_al_feature_key_audit_20261014.json`，状态`PASS_KEY_COVERAGE`，test载荷未打开。
- PBS脚本 `experiments/monthly_R_station16_20261013/run_extract_r_al_feature_bank.pbs` SHA-256 `8ED9040A3B616A942A3AB2D3B14B947380A358B3FCB3F0616428B0B89D7E122C`。本地Python编译及PBS bash语法通过；目标`swc` Python 3.10编译通过。首次以系统旧Python编译产生语法错误，确认是解释器选错，切换至训练环境后通过；该次未提交。
- 新增独立 `head_r_al_candidate.py` 与 `witness_r_al_head.py`，未改动历史A/AC/AP模型文件。目标服务器 `swc` 环境已编译两文件；运行时参数量为 AL=AP=24,613，符合容量匹配。实际数据梯度见证待特征bank完成后运行。
- 新增AL-only训练脚本 `train_r_al_head.py`，复用原30 epoch、batch=2000、seed配对初始化/样本次序、AdamW+cosine schedule与物理GHI MSE；脚本仅本地及目标环境编译，未启动训练。
- 资源：提交前全账户仅`213850`一个未结束任务；单卡请求后提交`213866.tc6000`，提交后账户未结束任务2个。作业在node22运行；最新只读进度为24,384/29,655个GHI关联时刻（82.2%），运行42分钟，R输出对已存COT bank的最大绝对差`0.0008007`，低于合同阈值0.002。分配GPU为node22卡1（80GB A800）。目标输出 `results/r_al_feature_bank_t18_20261014_v1/`。此作业只做train/val冻结R前向和隐藏特征bank，不训练GHI头。完成条件还包括54,937条键覆盖、特征有限值和train-only归一化清单。之后再做AL前反向梯度及单卡profile，未通过前不启动六组训练。

### 2026-10-14 T18 更新：AL 特征 bank 完成，真实梯度见证与单卡 profile 已提交

- Feature bank 作业`213866.tc6000`已写出`T18_R_AL_FEATURE_BANK_COMPLETE`及`state=COMPLETE`清单。覆盖54,937条GHI键（train 50,458/val 4,479）、29,655个唯一时刻，键为`(station,target_time_utc)`；R重放相对COT bank最大绝对差0.0008007288，source_errors=0，test_payload_opened=false。特征归一化仅拟合50,458条train行。远端核对特征数组、键表、归一化文件哈希分别为`19cf09eb0e3df444aed618ff78a33be4180e6d09dc680cb4b47b2ddda588acd5`、`e4aae0f616849570eba1838e05be8578a4fbca138b823200121f4045455a8371`、`85e4c757f03010d545227d5a00c6a9b1a397c1963d2d8700f68a15923a514826`，均匹配manifest。作业历史记录已清除，因此不推断Exit_status；完成标记、manifest与哈希已现场核验。
- 目标环境运行时参数量AL=AP=24,613。新增`run_r_al_witness_profile.pbs`将在一张node22 GPU上依次运行两站真实train batch梯度见证及12-update吞吐/峰值显存profile。提交前全账户仅CPU任务`213850.tc6000`，提交后共2个未完成任务；`213867.tc6000`当前排队。脚本已在`swc` Python 3.10.18环境编译通过。正式AL训练尚未提交；等待两站见证与profile通过后，再按实测batch决定是否启动六组匹配训练。

### T18 更新：梯度见证和profile通过，六组匹配AL训练已排队（2026-10-14）

- PBS 213867以exit_status=1结束，原因是第一步辅助编码器梯度为零。回查发现这符合配对初始化：AL读出层的辅助特征列先置零，以便初始输出复现A；首步应先更新读出层，第二步才会把梯度传入辅助编码器。已修正见证顺序并复跑，不把该实现见证失败隐去。
- PBS 213868在node22单张A800上exit_status=0。Sili/Zhujia真实train batch均通过：AL/AP参数数各24,613；首步辅助读出梯度L1分别0.09740/0.25482；首步更新后辅助编码器梯度L1分别0.005672/0.016782；R参数梯度全为None，缓存特征已detach。test_payload_opened=false。
- 12-step/batch单卡profile均成功。候选batch=2000峰值torch分配约419 MB、16 updates/epoch（Sili 24,155 train rows），满足稳定显存和每轮更新数条件。profile测得batch=1600吞吐最高，但为保持与A/AC/AP的原batch合同一致，正式AL仍使用batch=2000；不借profile改变训练合同。该轻量头显存占用很低，profile不支持继续扩大batch作为科学收益来源。
- `run_train_r_al_t18.pbs`已在目标环境做bash语法检查，资源查询后单卡提交六个站点×seed训练为PBS 213869；提交前未结束作业1个、提交后2个，低于3个上限。当前排队。六组完成后再做共同验证行FP64独立评价和按日期块配对CI，期间test保持关闭。

### T18最终结果（2026-10-14）

六组AL训练PBS213869 exit_status=0。共同验证行4,479条、两站等权RMSE：A=123.980、AC=120.240、AP=123.822、AL=119.807 W m⁻²。日期块配对2000次CI：AL−A=-4.173 [−7.137,−1.390]；AL−AC=-0.433 [−1.476,+0.677]；AL−AP=-4.015 [−6.914,−1.265]；AC−A=-3.740 [−6.074,−1.647]。本地第二实现从保存FP64预测重算，所有点估计和CI最大差0.0。结论：AL相对A/AP有收益，但与AC差异未分辨；不证明等效，也不能把AC收益唯一归因于显式COT物理语义。详细限制与两站指标见`results/T18_RAL_RESULTS_20261014.md`。T18完成；test保持关闭。

### T38 Bootstrap口径核对（2026-10-14）

从T18保存的FP64预测与键表独立汇总按日SSE，并以2,000次bootstrap复算两种抽样设计。52个并集BJT日期共享抽样、seed 20261013时，AC−A 95% CI为[-6.1795768090,-1.5972587519] W m⁻²，与T14主分析逐位一致；分别按Sili 44日和Zhujia 50日抽样、seed 20261014时为[-6.0743489555,-1.6473778775]，与T18复算逐位一致。两者point estimate同为-3.7402514464 W m⁻²、共同支持4,479行。主结论继续采用预注册共享日期区间；T18分别抽样区间只作敏感性，不能替换主区间。无指标/数据缺陷。JSON和输入哈希见`results/ci_method_reconciliation_20261014/T18_BOOTSTRAP_METHOD_RECONCILIATION_20261014.json`；复算脚本同目录。只使用validation保存输出，test载荷读取0。

## 2026-10-14 人工论文主张对齐复核（非独立盲审）

T30阶段的只读比对发现当时 `paper/main.tex`、`paper/main_zh.md` 尚无具体数值，仍写“提升仍是假设”，且结果/结论以未来时态表述。该历史状态已由T39中英稿修订更新。T18显示AC相对A有验证收益，但容量匹配的R隐藏特征AL也优于A，AL−AC日期块CI跨0，因此不能把全部收益唯一归因于显式COT语义，也不能声称AL与AC等效。T37全验证2×2是固定头离线输入替换，不得写成可部署GHI预报成绩；CPP只作稀疏参考，不是独立真值。T39已同步英文与中文正文，明确验证范围、主区间、R隐特征控制与COT forecast-chain边界；编译器未启动，编译状态未验证。

发现并修正文档漂移：`paper/PAPER_PLAN.md`原主体仍把完整16×16 COT空间输入和IFS增量作为主线，与当前COT价值优先合同冲突。已在文件顶部增加日期明确的当前方向/历史状态说明，保留旧文本作为历史记录，避免误导后续写作。参考 `paper/PAPER_PLAN.md`、`results/T18_RAL_RESULTS_20261014.md`、T37验证固定头诊断及`EXPERIMENT_PLAN.md`末尾的当前边界。本次未读取test载荷、未启动训练或PBS任务。

限制：这是本地人工对照，不是正式独立claim audit；当前协作规则不允许启动skill所要求的新鲜审稿线程。若之后要形成正式审稿产物，需另行满足该skill的fresh-review流程。

## 2026-10-14 COT外推文献可比性与独立掩膜来源复核

- 直接读取COT验证评分JSON/逐lead CSV并与协议表交叉核对：主3×3 RMSE S→R=25.538、持续性=28.252，差−2.714；8个CPP支持目标日上的95%区间[−6.476,+0.493]仅作描述；厚云CSI为0.082/0.189（S→R/持续性）。全验证3,140窗口的扩展不改变CPP参考仅8日这一事实。
- 通过出版社Wiley/AGU页面核实Cloud-FNO论文的独立测试COT RMSE=9.01为其6小时AHI全天CPP任务结果；数据网格约5 km/30 min、广域经纬范围，训练2016年3–12月、2017年测试。其MODIS第三方基准是另一口径（表列COT RMSE 12.64、Cloud-SimVP 17.02、真实全天产品11.84）。这两组数均不与本站点局地、FY-4B/CPP稀疏验证形成可比排名；已把官方DOI来源和该边界补入`results/cot_forecast_eval_20261014_v3/COT_FORECAST_LITERATURE_PROTOCOL_20261014.md`。
- 本地`solar-cot-ifs`文件检索未发现FY-4B CLM独立云掩膜bank，只有CPP生产工具；因此M04无现成独立CLM可跑。CPP CLP仍不可称独立验证。当前决定：维持GHI/forecast训练门控，不训练新模型；不以扩大预测窗数冒充增加独立参考日期。后续若独立CLM数据到达，再按M04已有合同做诊断；否则聚焦已存全验证bank的非CPP proxy-response attribution，并继续标注它不是物理真值评估。
- 复核范围仅train/validation结果与元数据；没有读取test载荷、提交PBS或启动训练。

## 2026-10-14 T40：论文结果图1

新增`paper/figures/plot_cot_value_validation_summary.py`，直接从三seed GHI FP64验证审计JSON与COT forecast FP64指标JSON读取点估计、日期块区间、日期数和厚云CSI，并保存源哈希清单。左面板呈现AC−A/AC−AP同期站点GHI RMSE配对差（4,479行、52个BJT目标日期、3 seeds）；右面板呈现3×3 S→R−持续性COT RMSE（8个CPP支持日期）和厚云CSI，图注明确该区间仅描述且非GHI forecast score。初次审查指出图内小字在稿件缩放后偏小，已删除灰色图内说明、增大字号，并在中英图注补充AP控制范围、AL归因边界、无量纲COT及逐站/逐lead聚合口径；复审无阻塞。PDF为单页矢量图，字体已嵌入；PNG已检查无裁切。`main.tex`和`main_zh.md`已同步纳入同一图与中英图注，PAPER_PLAN同步更新。整稿编译仍因内置编译器找不到平台标准目录而未验证；没有读取test或提交训练。

## 2026-10-14 T41：Motion+Source旧bank复用审计

实时只读核对 `zjnu-hpc`：主机可达，slfu账户有1个运行中任务 `213850.tc6000`；未提交PBS。旧 `geometry24_v2/evaluation/four_backbones_cpp_cot_val_20260923_geometry24_v2/motion_source/forecast_cot_log1p.npy` 存在，但其元数据指向旧 `formal_e12` backbone和旧 `cot_repaired_seed42_v1` R，且 `source_NC_generation_checkpoint_binding=UNVERIFIED`，不是当前新月度R/COT来源。因此不把旧Motion+Source bank拼入T21当前合同结果；T21保留为SimVP/持续性有限验证诊断并明确缺少可比输运基线。审计见`results/MOTION_SOURCE_REUSE_AUDIT_20261014.md`。只读元数据与文件stat；未读预测数组/test载荷、未启动训练。

## 2026-10-14 当前工程交接更新（T32部分完成）

在`docs/END_TO_END_HANDOFF_20261003.md`顶部增加当前合同/证据状态，明确旧主体是2026-10-03区域模型快照，并记录8→16 S→R→H链、当前3×3 A/AC/AP/AL同期头、日期划分、验证结论和B4停止边界。M0/M1 tracker也同步记录M03完成、M04无现成独立CLM、M05按M1结果暂缓。只更新工程文档与tracker；test仍关闭，未训练、未提交PBS。T32整体仍为PARTIAL：内置编译器启动失败使整稿编译未验证，且test未授权；不把验证版交接描述成最终test交付。

## 2026-10-14 T42：内置编译器重试

对当前打开的`paper/main.tex`再次调用内置编译器，仍在平台初始化阶段返回`Unable to find standard directories for platform`，没有生成TeX源代码诊断。源文件未变更；不尝试替代终端编译，也不在环境未变化时重复调用。整稿编译继续标为未验证。


## 2026-10-15 用户授权：已有历史COT外推的GHI直接配对

用户要求用已有历史COT外推结果检验GHI预报有无COT，覆盖此前该旧候选停止进入GHI的条件。独立方案与跟踪在`../history_cot_forecast_ghi_20261015/`，不改变新月度主成绩或开启test。复用旧四里S/R/输运和v5 update96混合修正；排除任何24帧窗口触及新test/buffer日期后保留旧train/val角色：train332序列4041行，val449序列5855行。固定3×3、A/AC-T/AC-H、seed42/43/44、同参数量/初始化/样本顺序/预算各1200更新，共9臂。输出所有16lead、1/2/3/4h及日期块CI，报告负结果而不择优。

214125在训练前因np.isin传set导致空索引退出，已单点修复为list；旧源码/日志保留。实时全账户仅214108未结束时提交214126.tc6000；后者node22 GPU3单A800运行，PREPARE train已开始。结果目录`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/history_cot_forecast_ghi_20261015_v1/results_attempt2`。例行监控交Luna；不重复提交，等待实际profile/训练/完成证据。此为旧合同探索性预报试验，不能替代新合同独立验证。

### 历史COT→GHI配对完成（2026-10-15）
214126.tc6000 exit_status=0，旧合同四里三seed九臂完成并从保存预测独立FP64复算。5855共同验证行/60目标日：A157.303、AC-T161.689、AC-H161.983 W/m²；AC−A分别+4.386 [+0.080,+9.476]、+4.680 [+0.715,+9.450]，退化2.79%/2.98%。不支持这两条已有历史COT外推路线的GHI收益；不与同期新月度3.02%混算。见`../history_cot_forecast_ghi_20261015/RESULTS.md`。新test/buffer触及窗口均排除，旧角色保留，未读test；不再为正向结果调参或加seed。

### 2026-10-15 历史COT→GHI退化诊断修订
全5855预测键一致，四例96帧回源AGRI时空/几何/运动/COT与pack差0。原验证每100步约隔203遍数据，5/6 COT臂较好模型早于首次验证；原前100步重放214162逐组精确复现step100。补查早期检查点后A157.091、输运155.372、混合156.125，点估计改善1.09%/0.61%，日期CI跨0，仅事后诊断。先前总体RMSE退化结论限于原稀疏检查点，不能据此否定COT预报价值。实际云图同时显示消散/增厚漏报；近零参考案例也有训练相关退化。详见`../history_cot_forecast_ghi_20261015/diagnosis_20261015/DIAGNOSIS.md`，无新方法训练、无test读取、论文未改。

### R07 corrected full-validation screen v2 (2026-10-17, completed)

- The prior `214497` attempt failed before inference and produced no bank. The revised evaluator now accepts the missing checkpoint geometry metadata only when backed by a separately generated source-and-manifest receipt; it does not edit checkpoint metadata or relax hash/split gates.
- Local CPU witness passed: exact source-keyed CPP join and paired-statistic checks on the repaired 271-window audit bank; all three v2 Python files compile in the target `swc` environment.
- Geometry receipt passed against 35,320 train / 3,140 validation / 3,202 test-metadata rows, frozen normalization and source hashes, and all three checkpoint hashes; test payload reads=0.
- PBS 214498 inference and PBS 214500 final FP64 scoring completed. The baseline bank is read-only; output is versioned under `simvp_supervision_cot_allval_20261017_v2`.
- The completed full-validation result and paired intervals are recorded immediately below. No test payloads were read; manuscript claims are now updated to match that evidence.

### R07 full-validation COT forecast result (2026-10-17)

PBS 214498 inference and PBS 214500 final FP64 scoring completed successfully. Final common-support output is `results/simvp_supervision_cot_allval_20261017_v2/scored_final/summary_fp64.json` with lead/station metrics in `cot_metrics_by_station_lead_scale.csv`. All 3,140 validation windows were scored; CPP supported 46 BJT target dates and 5,224,460 valid pixels in the 16x16 bank (183,564 local 3x3 pixels); test payload reads=0.

For the local 3x3 station neighbourhood, station-and-lead-macro COT RMSE was 10.664 (original SimVP), 10.750 (MSE fine-tune), 10.756 (temporal-change), 10.759 (spatial-gradient), and 20.487 (last-frame persistence). Candidate-minus-original paired date-block intervals cross zero: +0.086 [-0.078,+0.240], +0.092 [-0.074,+0.248], and +0.095 [-0.067,+0.252]. Thick-cloud CSI was 0.423, 0.431, 0.430, 0.428, and 0.306. Persistence minus original RMSE was +9.823 [ +7.722,+12.154 ], favoring the original forecast. No fine-tuning candidate passed the predeclared advance rule; this is not evidence against downstream COT value for GHI.

The corrected scorer uses the original SimVP bank's saved R(observed future AGRI) output as a shared proxy reference for all arms. Repeated R outputs differed slightly for two arms, so the proxy-response comparison is explicitly not independent COT truth. Manuscript Figure 7 now shows only CPP-supported COT RMSE and thick-cloud CSI by lead and station; matched forecast-input GHI remains unmeasured.

### 2026-10-17 full-cohort COT trajectory response audit and paper update

Completed a source-keyed, validation-only temporal-change audit in `results/cot_temporal_change_audit_20261017/`. Original, MSE, temporal-change and spatial-gradient forecast banks matched exactly on sequence IDs, target timestamps and dataset indices; current CPP values/masks were exact across R07 banks. On 52 BJT dates (423,900 local 3×3 pixel transitions per station), original SimVP→R changes had Pearson correlation 0.533/0.593 and through-origin slope 0.334/0.392 against R(observed future AGRI) changes. Relative to persistence, change RMSE fell by 1.737 [−2.396,−1.212] at Sili and 2.184 [−3.076,−1.494] at Zhujia. Temporal-change fine-tuning reduced proxy change RMSE by 0.046 [−0.085,−0.013] at Sili but 0.013 [−0.047,+0.018] at Zhujia. This is same-retriever proxy response only, not physical COT truth.

On the paired adjacent-lead CPP support, only seven BJT dates per station remain; forecast→R change correlations with CPP were 0.133/0.019, versus 0.650/0.608 for observed-future-AGRI→R. This suggests, but cannot establish, attenuation of cloud-physical temporal information in the predicted-image chain; CPP retrieval variability, sparse support, and R response are confounded. The analysis does not overturn the aggregate CPP-referenced result or establish GHI forecast gain. Bilingual manuscript results/discussion/conclusion were updated with this scope and limitation; no chart was added. Report: `results/cot_temporal_change_audit_20261017/RESULTS.md`. Script and bank hashes are recorded there. No training or PBS job was submitted; test payload reads remain zero; existing B4 gate unchanged.

### 2026-10-17 unique thick-event cohort and manuscript revision

The source-keyed full-validation audit in `results/event_cohort_decomposition_20261017/` identified seven unique station-pixel CPP thick-cloud events (COT≥30 for at least two consecutive valid 15-min targets): four at Sili, three at Zhujia, on only three BJT dates. Each event was scored once using the longest-lead validation forecast covering the full event and common current CPP/R-valid support in the local 3×3. SimVP-forecast-AGRI→R mean event RMSE was 19.53/13.48 and pooled thick CSI 0.149/0.000; the observed-future-AGRI→R hindsight diagnostic was 6.87/7.38 and 0.814/0.617; the direct CPP/R-hybrid predictor was 24.47/26.79. The observed-future condition is nondeployable and still differs from CPP. The older bank's embedded CPP values differed from exact current compact-pack values by up to 68.20 COT, so they were quarantined; exact labels/masks were rebuilt from current validation frame indices and intersected with current R frame-valid support. Forecast/R source hashes and keys passed; test payloads read=0.

Updated English and Chinese Discussion, `PAPER_PLAN.md`, and `PAPER_CLAIM_AUDIT_20261017.md` to replace stale references to the two-event forecast comparison. These descriptive events motivate a later one-factor SimVP COT-auxiliary-loss test, but do not establish a validation-wide causal effect, physical COT truth, or GHI forecast gain. No figure or bar chart was added, no training or PBS job was started, and test remains closed.

### 2026-10-21：湖北未来COT持平/辅助监督历史结果复核
历史对照与机制范围见 [HISTORY_AUDIT_PHYSICS_CONSTRAINT_AND_PERSISTENCE_20261021.md](../hubei_solar_geometry_repair_20261021/hubei_cot_auxiliary_retrain_attempt13_retry4/HISTORY_AUDIT_PHYSICS_CONSTRAINT_AND_PERSISTENCE_20261021.md)。历史已测直接COT预测 vs 持平、时间变化/梯度损失、COT辅助监督；显式物理定律约束未测。attempt13 job 215355 误将进度条合计282看成train updates，已提前取消；282=256 train+26 val，不是模型失败。保留attempt13目录，按原合同新目录retry4提交PBS 215356，node22单A800，当前R，尚无模型结果；test/buffer读取0。

2026-10-21 12:36 CST live progress: `qstat -u slfu` shows only `215356.tc6000` running (1/3), allocated `node22-gpu/1` on node22; elapsed 00:09:07. Retry4 preflight reports test/buffer reads=0 and single-GPU profile PASS (batch16, peak 69.291 GiB, 17.62 s). The matched `mse_control` arm has entered epoch 0 and was at 267/282 combined train+validation batches (validation 11/26) when polled; `cot_aux` has not started. This is startup progress only, not a checkpoint or COT skill result.

### 2026-10-21 12:40 CST：COT forecast-pair scorer preflight

- Added preflight evidence at `../hubei_solar_geometry_repair_20261021/hubei_cot_auxiliary_retrain_attempt13_retry4/SCORER_PREFLIGHT_20261021.json`. Both inference/scoring sources compile locally; scorer CLI loads; synthetic bootstrap and complete end-to-end scorer pass on 407 synthetic sequences × 16 leads (6,512 paired keys, 224 metric rows across all-patch, station-pixel and station-3×3 scopes). The synthetic known-delta result and equal-station/date-block aggregation are recovered exactly; test/buffer reads are zero. This verifies the evaluator only, not forecast skill.
- Live PBS at 12:40:05 CST: account has 2/3 unfinished jobs. `215356.tc6000` is R on node22 GPU1, elapsed 00:14:13; log has reached `START_ARM mse_control`, with no epoch result yet. `cot_aux` has not started. `215358.tc6000` (`cot_observed_ref`, normal queue) is also running; no action taken on it. No task was submitted by this update.

### 2026-10-21 12:43 CST：retry4 training and evaluator update

- `215356.tc6000` remains R on node22 GPU1; live account has 2/3 unfinished jobs (`215358.tc6000` is the separate `cot_observed_ref` CPU job). The retry4 log has moved beyond startup: `mse_control` is training epoch 1 at about 256/282 batches; epoch-1 validation has not started at the last log read, and `cot_aux` has not begun. This is progress only, not a scored result.
- Static review confirms the pair runner holds the shared SimVP/data/R/normalization/CPP-sidecar/lambda provenance and optimizer-update count as invariants; the completion receipt will reject mismatched initialization or updates. Local scorer py_compile, CLI, bootstrap smoke, and 407×16 synthetic end-to-end scoring all pass; receipt and hashes are in `hubei_cot_auxiliary_retrain_attempt13_retry4/SCORER_PREFLIGHT_20261021.json`.

### 2026-10-21 12:44 CST：MSE control enters epoch 2

- Live `qstat -u slfu` at 12:44:02 CST shows 2/3 unfinished jobs: `215356.tc6000` R on node22 GPU1 and `215358.tc6000` (`cot_observed_ref`) in the normal queue. No new job was submitted.
- Retry4 `mse_control` completed epoch 1 and began epoch 2 (about 16/282 batches at the log read). Its current best validation checkpoint is epoch 1, with `val_cot_rmse=15.87298`; Lightning explicitly reports the checkpoint monitor is `val_cot_rmse` (the filename's `val_loss` component is only a filename metric). This is an early single-arm validation value, not a paired result or skill conclusion. `cot_aux` remains pending.

## 2026-10-21 13:13 CST：配对训练进展与后续验证评分已排队

- 实时 `qstat -u slfu` 显示账户未结束任务 2/3：`215356.tc6000` 在 node22 GPU1 运行；`215359.tc6000` 为本轮验证推理/评分任务，状态 H，依赖 `afterok:215356.tc6000`。未终止或改动其他任务。
- `215356` 当前仍在 `mse_control` 臂，epoch 2 验证已完成，日志显示物理 `val_cot_rmse≈15.20`；epoch 3 约 34%，`cot_aux` 臂尚未开始。该值仅是单臂中途验证读数，不是成对结果或最终 COT 技能结论。
- 为避免训练结束后再等待准备，已新增 `run_attempt13_retry4_infer_score.py` 和 `attempt13_infer_score_retry4.pbs`，本地 Python 编译、远端 Python 编译及远端 `bash -n` 均通过。`215359` 将在配对训练成功后，对两臂各自锁定的最佳 SimVP checkpoint 仅推理 407 个 validation 窗口，再用同一冻结 R 生成物理 COT，并在完全共同 CPP 支持上进行 FP64 逐lead/站点/厚度评分，和历史持平及离线真实未来图参照对照。CPP 仍为检索参考；不读取 test/buffer。
- 推理任务请求 node22 单卡；执行时会检查配对完成收据、数据缓存和验证行键，缺任何必需项即失败退出，不覆盖已有结果。当前主要训练仍运行，尚无 attempt13 最终评分。

### 2026-10-21 13:15 CST：attempt13实时状态复核
PBS `215356.tc6000`仍在node22 GPU1运行；依赖评分`215359.tc6000`为H，账户共2/3未结束任务。MSE-control在epoch 3约65%，COT-aux尚未开始，尚无配对结果。详见 `refine-logs/hubei_solar_geometry_repair_20261021/hubei_cot_auxiliary_retrain_attempt13/EXPERIMENT_PLAN.md`。
### 2026-10-21 13:16 CST：lead偏差诊断与attempt13进度
attempt12已保存逐时效偏差显示，预测COT相对CPP的平均偏差由+75至+105 min的负偏转为+195至+240 min的正偏；+240 min偏差+4.43、RMSE 14.26，高于历史持平11.43。现有证据不支持“所有时效都只是预测幅值过弱”；过平、错位与演变误差仍需分解。详见 `refine-logs/hubei_solar_geometry_repair_20261021/hubei_cot_auxiliary_retrain_attempt13_retry4/HISTORY_AUDIT_PHYSICS_CONSTRAINT_AND_PERSISTENCE_20261021.md`。
Luna于13:16:55 CST实时核对账户2/3未结束任务：`215356.tc6000`运行于node22，`215359.tc6000`处于H等待训练依赖；MSE-control为epoch 3的241/282合并batch，best物理验证COT RMSE 15.24987，COT-aux尚未开始。尚无配对结果。详见 `refine-logs/hubei_solar_geometry_repair_20261021/hubei_cot_auxiliary_retrain_attempt13/EXPERIMENT_PLAN.md`。
### 2026-10-21 13:18 CST：attempt13单臂训练进展
实时qstat账户2/3：`215356.tc6000`运行、`215359.tc6000`等待afterok。MSE-control完成epoch 3，进入epoch 4约15%；epoch 3物理验证COT RMSE 14.00365，成为当前单臂best。COT-aux未开始，无配对结论；test/buffer读取0。此值仅为单臂早期选模指标。

### 2026-10-21 13:46 CST：attempt13新鲜只读状态
Luna实时核对全账户队列为2/3未结束：`215356.tc6000`在node22 GPU1，`215359.tc6000`因`afterok:215356.tc6000`挂起；`215369.tc6000`已完成。`215356`的MSE-control完成epoch 6、global step 1792，epoch 7已完成训练并进入验证（约9/26 batches），当前best物理验证COT RMSE为11.81630。COT-aux仍未开始；未发现配对预测、COT bank、评分结果或完成标记。未提交/取消任务，test/buffer读取0。该结果仍是单臂中期状态，不能说明COT辅助监督或预报技能。

### 2026-10-21 13:50 CST：attempt13 实时训练状态更新
Luna 只读复核 `qstat -u slfu`：账户 3/3 未结束作业，`215356.tc6000` 在 node22 单卡运行，`215359.tc6000` 仍等待 `afterok:215356`，另有 `215372.tc6000` 运行。MSE-control 臂 epoch 8 进度 122/282 batches（43%）；epoch 7 COT RMSE 13.20，当前 best 仍为 epoch 6 的 11.81630。COT-aux 臂尚未启动，尚无配对bank/预测/评分/完成或失败标记。因队列已达3个未结束上限，本轮不提交新作业。test/buffer读取0。

### 2026-10-21：对“预测 COT 太平/太弱”的证据边界
当前全量 attempt12 验证只证明 SimVP→R 的 COT RMSE 高于历史持平（16 leads 中 patch 15/16、站点像元和3×3各47/48比较更差）；这不能单独证明空间场方差被压平。验证预览文件保留 407×16×16×16 的预测COT，但本地配套工件没有完整共同支持的历史R-COT场与CPP逐像元标签，因而本地尚不能给出无偏的预测/持平/参考空间方差或纹理能量比。一个被挑选的验证事件（2025-09-27 13:30 BJT，非总体样本）在牛首站点轨迹 +120 min 为预测/离线R参考 17.48/39.79，+240 min 为47.90/24.51，表现为先漏掉增厚、后期过冲；不符合“所有时效都只是弱/平”的简单叙述。CPP和离线未来AGRI→R仍是检索/反演代理，不是独立COT真值。attempt13的COT辅助监督臂只有完成后才能检验辅助目标是否修复预测COT技能；若要验证“过平”机制，还须用同一val键逐像元对齐 forecast COT、初始化持平COT与未来参考场，计算空间标准差、梯度能量、厚云命中/虚警及云团到达时间，且注明参考来源。
