## 2026-10-30 执行简化：不默认追加内部隔离与审查层

当前执行以[简化说明](../docs/execution_simplification_20261030/CURRENT_POLICY.md)和[核查报告](../docs/execution_simplification_20261030/AUDIT.md)为准。默认复用完整train资产；不把U/H/OOF、嵌套日期、预报头重训或新观测数据作为普通GHI试验的通行证。最新湖南SimVP冻结头验证127.421→125.340 W/m²（1.633%）已可按实际范围报告。以下日期段保留为历史计划，不自动恢复旧任务、重训资产或追加门槛。数据完整性、测试关闭与现行资源约束不变。

---

## 2026-10-20 云掩膜文献核查与独立输入方案

用户要求先核查已有云掩膜实验并调研文献。已完成[历史去重与旧门控重评分](cloud_probability_input_20261020/HISTORY_AUDIT_20261020.md)、[文献方法选择](cloud_probability_input_20261020/LITERATURE_METHOD_DECISION_20261020.md)、[四组详细计划](cloud_probability_input_20261020/EXPERIMENT_PLAN.md)和[结果报告](cloud_probability_input_20261020/B2_RESULTS_20261020.md)。太阳几何与54,895个站点-帧键完成核验；分类器沿用旧几何，本轮没有宣称修正几何分类器。PBS `215069.tc6000` 在浙师大 node22/A800 GPU4完成24次配对训练。等站等seed验证RMSE：AGRI 89.485、+COT 88.264、+判云 88.186、+COT+判云 87.795 W/m²；主比较联合−COT为−0.469 W/m²（0.531%），日期块95%CI [−1.007,+0.047]跨零，且四里小幅退化，按冻结条件判为**未证实增量**，不启动二值B3。厚云参考分层有探索性改善，但CPP/CLP仅匹配1,265/4,226行、六个共同支持块且未作多重校正；高GHI子集CI也跨零。总体/分层/高GHI评分与配对差图已保存；真实连续云事件图仍待整理。test未读取。

## 2026-10-13 详细计划：COT反演主实验与并行推进

最新详细合同：[COT价值优先计划](cot_value_first_20261013/EXPERIMENT_PLAN.md)；[32项任务表](cot_value_first_20261013/EXPERIMENT_TRACKER.md)。本轮用户要求先制定，状态PLAN_ONLY。先18次真实图像反演对照，条件通过后24次预报头训练；最多12次条件诊断。PBS账户最多3未结束任务，更多并发转复旦；GPU按新授权与实时资源扩展。六个逻辑agent职责由单一heartbeat协调、Luna例行检查，当前最多3个worker同时活跃。尚未创建自动化、启动训练或放行test。
## 2026-10-13 用户最新方向：COT反演价值优先

权威修订见 `docs/COT_VALUE_FIRST_DIRECTION_20261013.md`（相对于仓库根目录）。先用真实目标时刻AGRI做有无COT的同期GHI反演配对对照，再独立评价COT外推精度，最后在固定外推方法下评价GHI预报收益。外推创新留给后续工作，IFS后置。采用小邻域方向；不恢复全图均值/p90或旧四统计量实验。下方旧模型与优先级仅作历史，不覆盖本修订。新月度划分、来源与test门禁继续有效。本次不代表新训练已经开始，不取消运行作业。
# 实验计划：完整 COT 空间场进入 GHI 网络

## 2026-10-03 SimVP监督目标改进计划（待执行）

用户要求按相同新划分checkpoint、相同预算制定 A=MSE、B=MSE+时间变化、C=MSE+空间梯度的独立微调筛选。详细合同见 [SimVP监督改进计划](simvp_supervision_20261003/EXPERIMENT_PLAN.md)，状态见 [执行清单](simvp_supervision_20261003/EXPERIMENT_TRACKER.md)。首轮固定采样，每臂4000更新，先验收损失/掩膜及来源；全年COT排名须通过新R日期来源门禁。本次仅制定计划，未提交训练、不打开test、不自动替换冻结骨干；后续采样、COT监督和GHI重训均按各阶段证据放行。

## 2026-10-02 最新划分合同

用户已批准覆盖全年季节的按月连续2天val、另2天test、留块前后1天缓冲的新合同，见 `../docs/MONTHLY_BLOCK_SPLIT_20261002.md`。全期资料重新划分，旧整段未来test不再是新合同test；新test载荷与正式评价仍关闭。首阶段保持13+3输入和geometry24_v2 SimVP结构，仅新归一化与随机初始化重训；后续R/H按同一日期重建。下面旧日期、冻结旧权重和历史状态不作为此合同的训练或独立测试依据，第一篇不自动跟随改变。
## 2026-09-24 新候选方案：GHI 指导的 COT 运动与演变

用户要求给出“如何用GHI目标保留重要信息、先反演历史COT再预测运动与变化”的方案。独立设计见 [COT dynamics实验方案](cot_dynamics_20260924/EXPERIMENT_PLAN.md) 与 [对应tracker](cot_dynamics_20260924/EXPERIMENT_TRACKER.md)。状态为 **B0执行中**：train/val 时间与行合同通过、P16覆盖和R视场诊断已完成；修复版SimVP训练、GPU pilot及后续配对GHI实验尚未完成，test未读。先完成geometry24全链路和oracle信息价值，再按纯搬运→COT监督的E→GHI梯度逐门禁推进。旧AFNO-COT bank存在几何错配，其配对GHI数值不得作为新版本的增益证据。以下历史实验与合同保留。

版本：2026-09-19（湖南站点 GHI 头与因果 COT 轨迹修订）；状态：H14空间图与站点感知空间头均完成探索性seed42筛选，未通过正向扩展门槛。取代旧版四统计量主方法计划。历史结果与 tracker 保留。

**2026-09-18 用户决定**：区域均值、p90 等 COT 全局平均类统计特征全部从本计划移除，不再作为任何未来实验的输入或对照组；新头不再保留四统计槽。旧四统计量 AC（三 seed 已完成）仅作历史负结果锚点只读保留，不重训、不扩展、不进入任何新对照表；空间图 `[8,8]` 中心像元由 v2/v3 头的站点读出覆盖，不构成统计槽。

## 2026-09-19 最新方向：湖南站点头上的因果 COT 时空轨迹

用户明确要求不以全图平均或静态 COT 图解释站点 GHI。原 `AC_map`、
`AC_gate` 和 `AC_station` 仅保留为静态空间对照，不能再作为主方法或用其结果
判断“COT 时空信息无效”。主候选改为 `causal_cot_trajectory_solarresnet_v5`：每行在目标
lead `k` 只使用 (a) 初始化前八帧真实历史 AGRI 经冻结 R 反演的 COT，及
(b) 初始化时刻由冻结 S→R 得到的预测 COT 的 `0..k` 因果前缀；后续 COT 图在
网络前遮蔽。GHI 头为湖南每个站点从头训练的残差图像主干，含 CBAM、中心像元分支及
5 维 metadata MLP；COT 分支保留时间因果 3-D 空洞卷积，并在图像/metadata 融合特征
之后接入。禁止全 patch
平均、四统计量和静态单帧 readout。

这里的“复用”只指网络结构和输入组织，**不复用湖北 checkpoint、归一化、样本、
站点模型或任何结果**。湖南四里、竺家各自从头训练独立 SolarResNet-style GHI
head；每个站点内的 `A_ST/AC_H/AC_F/AC_ST` 使用相同初始化、样本顺序和优化器。
这样既保留湖北已验证的站点预测结构，也不把湖北训练权重冒充为湖南证据。

每个目标时刻的图像输入为 `13 AGRI + cosSOZ + cosRAA + day_mask` 的 16 个逐像元
通道；metadata 严格按湖北结构使用 5 维 `[cosSZA, hour_sin, hour_cos, doy_sin,
doy_cos]`。它们都由目标时刻、站点/像元坐标和确定性太阳几何计算，不读 GHI、
CPP 或任何未来观测。B0 必须逐项验收这 16 个图像通道和 5 维 metadata 的时间基准、
方向、归一化及 train-only 统计。

新 B1-T 的配对组为 `A_ST/AC_H/AC_F/AC_ST`（无 COT/仅历史/仅预报前缀/两者），
固定 SolarResNet-style 头部、样本、可用性掩码、每站初值 seed、损失和有效 batch
2048。历史 COT 必须
由八帧历史 AGRI 重新通过冻结 R 生成，不能把未来真实 CPP 或单帧缓存伪装成
轨迹；未来 COT 直接从现有冻结 forecast cache 按 `(sequence, station)` 复用。
先执行无 test 的 seed42、10 epoch pilot，保存逐样本预测并独立审计后才决定是否
多 seed。详见 `docs/COT_CAUSAL_TRAJECTORY_DIRECTION_20260918.md`。本段覆盖下文
关于 H14、GAP、全 patch 背景及 B1/B2 主对照的当前执行指令；历史记录不删除。
实现脚本已同步服务器并通过服务器 Python 的 CPU 因果见证；侧车 PBS
`209579.tc6000` 在 Gold pre.sh 中止，未进入 Python、未创建输出，故不进入
pilot、多 seed、IFS 或 test。

## 2026-09-18 用户授权的站点感知空间头

经冻结湖南网格合同和独立审计核验，Sili、Zhujia的最近网格像元均为各自16×16 patch的`[8,8]`。用户授权把逐像元对齐的AGRI/COT联合处理明确为站点感知编码：14通道先进入同一卷积，再保留站点像元、其5×5邻域和全patch背景三种联合特征。该`station_aware_spatial_v2`是独立候选，A/AC使用相同18,521参数头，旧H14保持原样。

首次吞吐筛选的seed42结果为A_station 151.1860、AC_station 151.2517 W/m²（两站等权；四里退化0.7752、竺家改善0.6437 W/m²），但自动batch16384使10epoch更新数降为原B1 batch2048合同的八分之一，不能作为正负表示结论。batch2048重跑PBS209558在Python启动前因PBS Gold服务异常退出（exit−2、零资源时间），等待调度恢复后才可重跑；在此之前不进入多seed、IFS或test。详见`docs/STATION_AWARE_SPATIAL_V2_20260918.md`；不将该未定结论写入论文正文。

## 2026-09-18 用户授权的逐像元 COT 门控

为避免把COT仅作为并排的第14通道，新增独立`cot_gated_station_v3`：每个像元的COT经1×1映射生成缩放/偏置，直接调制同位置AGRI特征；随后保留站点`[8,8]`、5×5邻域和全patch汇集。`A_gate/AC_gate`参数完全相同，唯一变化为COT图置零或输入。远程CPU前反向见证通过；GPU B0 PBS209559因Gold服务异常在Python启动前退出，尚无GPU或性能结果。调度恢复后按B0→batch2048 seed42配对pilot顺序继续，不进入多seed、IFS或test。详见`docs/COT_GATED_STATION_V3_20260918.md`。

**2026-09-18 修订**：按用户决定，v3 读出中原有的四个 COT 统计量拼接槽随全局平均类特征一并移除；B0 重跑前须同步修改 `scripts/cot_gated_station_head_v3.py` 并重做 CPU witness（输出维度与梯度路径变化），站点像元/邻域/全patch三尺度读出不受影响。

## 1 主张、范围与冻结合同
- C1（优先）：空间 COT 在固定外推/反演与公平头部对照下改善 GHI，且不能仅用参数量或 CPP 监督解释。
- C2（后置）：因果可用 IFS 在空间 COT 之上提供增量收益；超加性是额外待检验命题。
- 湖南 Sili/Zhujia，8历史帧→16未来帧，init=历史末帧，lead=15:15:240 min。训练2024-04-02～2025-06-30，验证2025-07-01～09-30，测试2025-10-01～2026-06-30；所有输入/标签窗口完整归属所在 split。
- 标签为区间 (target-15min,target] 的 GHI 均值 / 同区间晴空均值；5min区间末记时取 target-10、-5、0 三条。实测不作输入。
- 固定湖南 S/R 权重、输入和归一化合同。未来 AGRI、COT 均由冻结模型预测；未来真值只用于隔离诊断。CPP 是参考产品。
- 来源、权重、时间、空间和 split 门禁闭合后冻结 manifest；本版不宣布门禁已通过，不打开 test。

## 2 最小网络与命名
新主模型 ID 为 `causal_cot_trajectory_solarresnet_v5`，不能覆盖 H14、
`station_aware_spatial_v2` 或旧四统计量 head。它在湖南从头训练残差图像主干、CBAM、
中心像元分支和 metadata MLP；输入为 16 个图像通道，输出头融合 32 维因果 COT
轨迹特征。所有参数均随机初始化，不加载湖北 Direct-GHI checkpoint 或其归一化文件。

每行对应唯一 `(sequence, station, lead)`：当前目标预测 AGRI 的 13 通道与
`cosSOZ/cosRAA/day_mask` 逐像元对齐，组成 `[batch,16,16,16]`；完整 COT 轨迹
独立输入 3-D 分支。缓存 COT 已是 `log1p`，禁止再次取对数；其尺度仅用训练序列
拟合一个标量 mean/std，不逐图或逐像元独立标准化，也不以未来 CPP 有效性筛主表。
metadata 为目标时刻确定性的 5 维太阳/日历量；对周期量使用原始 sin/cos，连续
量只以训练期统计标准化。新头不含 COT 全局统计槽。

| 组 | 第14空间通道 | IFS槽 | 作用 |
|---|---|---|---|
| A_ST | 历史/预测轨迹均为零 | 零 | 同结构、无COT基线 |
| AC_H | 仅历史轨迹 | 零 | 历史云厚度增量 |
| AC_F | 仅截至当前 lead 的预测轨迹 | 零 | 外推云厚度增量 |
| AC_ST | 历史与因果预测前缀 | 零 | 主候选 |
| AN_ST | A_ST + 因果IFS图像通道 | 因果IFS辐射 | IFS基线，仅B3 |
| ACN_ST | AC_ST + 因果IFS图像通道 | 因果IFS辐射 | 融合候选，仅B3 |

四组的 SolarResNet 图像主干、metadata MLP、COT 3-D 分支、参数量、FLOPs 和
优化步骤完全一致；A_ST 仅将 COT 数值置零，不能删除该分支或减少参数。COT 轨迹
分支的有效性由零轨迹对照、空间排列对照和同 CPP 监督隐特征对照检验；这些控制
在 B1 成功后再预注册实现。IFS 分支未通过发布时间/单位门禁前保持禁用。

## 3 统一训练、统计与预算
保持已批准 GHI 损失 mean(w*((pred_kt*clear_sky-ghi)/1000)^2)，原固定 w、白天掩膜和样本顺序合同；不换回 kt MSE。
原始 cohort LR=3e-4；QC1 LR=1e-3。SolarResNet-style v5 使用 AdamW、wd=1e-4，
有效batch2048，最多50epoch，patience8；B0 先实测显存和吞吐，若结构使该有效
batch 不可用，记录事实并在所有配对组采用同一个经预先登记的有效 batch。统一按
验证主终点选 checkpoint。同组使用配对 seed42/43/44；不每seed调LR，不增seed追逐显著性。
原始 cohort 为主分析，QC1 为固定稳健性分析。现有验证99849行仅为待复核锚点，必须实际验收完整键、顺序、真值、mask及哈希后才能声称一致。
主要终点：先算每站全16 lead GHI RMSE，再两站等权平均。报告每seed、均值、样本std、分站RMSE/MAE/MBE、分lead曲线、n；附样本加权总体RMSE但不替代主终点。
配对目标日期块bootstrap2000次，所有模型/seed共享抽样日期，跨站同日期一起抽；1日块主分析，3/7日块敏感性。CI对seed平均配对差计算，seed变异单列，不将全部行当独立样本。测试仅在冻结后一次；已有跨项目测试接触须披露，不能宣称从未见过。
学习方法最多三类：无COT的AGRI系（含IFS与容量控制）、COT空间系（含IFS）、同监督latent系。IFS原始通量作非学习参考，仅在相同有效样本评价。
先单卡profile实际峰值显存、每epoch时间和吞吐，提升单卡利用率后再考虑扩卡。GPU小时按实测每次训练×预算计算，当前不编造工时。每次qsub前全账户qstat，最多3个未结束PBS/8卡，串行提交重查，不终止他人任务。持续监控关闭；后续需监控时交给Luna。

## 4 五个实验块

### B0 来源、对齐和实现验收（必须；支撑C1/C2有效性；补充材料合同表）
数据：train/val限定固定样本，覆盖两站、多lead、晴/厚云及缺测边界；不读test。
工作：闭合CPP生产者→脚本/权重/归一化→实际AGRI输入→输出经纬度/时间；核对S/R湖南合同；从旧分片cot_log1p逐行绑定head pack，保留16×16图；为 SolarResNet v5 重新打包 16 个图像通道和 5 维目标时刻 metadata；手工样本核对中心、方向、坐标、log变换、目标时间和训练归一化。
对照/指标：缓存与新读入值误差、键与标签完全一致、禁用通道为零、S/R无梯度、CPU reload一致、参数表。完整列出容差，不能只凭loss下降验收。
通过：来源链可核查、像元/时间/样本合同无冲突且数值验收通过；失败则停止训练并登记缺口。IFS未闭合只阻止IFS组。
成本：元数据/小样本CPU验收，单卡短profile；不重新生成全部预测场。

### B1 表示主对照（必须；C1；Table 1与lead曲线）
原因：直接检验保留空间场是否优于无 COT 基线；压缩统计（区域均值、p90 等）已被用户于 2026-09-18 从计划移除，旧四统计量 AC 三 seed 结果仅作历史锚点列于补充材料，不与新头同表对比。
数据：固定原始train/val，A_ST、AC_H、AC_F、AC_ST；相同归一化来源、16图像通道、5维metadata和标签。
pilot：seed42、两站各自四组最多10epoch（不是主结果），先验收再做完整预算。以 AC_ST 相对 A_ST 为主比较；若其退化或任一站RMSE退化>2%，暂停扩展查失败原因；不切换QC1寻找正结果。不满足稳定结论不等于排除方法。
完整：四组均做42/43/44、相同50epoch上限/早停规则。短pilot权重不混作完整种子结果。
成功判据：AC_ST 相对 A_ST 主终点平均相对下降≥1%，3/3seed方向有利、配对差95%CI排除0，且任一站seed均值退化不超过1%。AC_H 与 AC_F 单列报告，用于区分历史和预测轨迹贡献。
失败：任一条件不满足就不宣称稳定空间优势；保留所有结果，不自动扩架构/调loss/开test。
成本：B1 pilot 为两站×四组共8次短训练；完整阶段为两站×四组×三种子共24次训练上限。只有pilot门禁通过才展开完整组。

### B2 容量、监督和空间排列（必须；C1归因；Table 2）
原因：排除“多参数/CPP监督/仅厚度分布”的替代解释。
数据：B1相同cohort/split。`A_ST_proxy` 用同一 SolarResNet 和同维可学习非COT轨迹槽替换 COT 分支；`AL_ST` 用同一冻结 R 的时空隐特征经预注册固定投影替换物理 COT，三投影×三seed均报告。
空间对照：`AC_ST_shuffle` 在每一历史/预测时间帧按预注册坐标置换 COT，保留中心像元并破坏周边位置关系；置换只由行键和固定seed决定，与标签无关。训练和推理使用同一变换规则，三seed重训。冻结 `AC_ST` 的置换响应如另做须单列为分布外敏感性，不等同重训结果。
成功：`AC_ST` 对容量和每个预注册 latent 投影均有可重复优势，且排列对照支持空间关系作用；分别报告差值/CI，不以某项显著代替其余。若latent相当，收窄为“CPP监督表示有效”；若shuffle相当，不能归因空间排列。
成本：最多15次完整训练（3容量+9latent+3shuffle）；仅B1通过再开展。不增加其他泛化模型大赛。

### B3 IFS增量与交互（条件必须；C2；Table 3）
前置：可验证release_time≤init、cycle/valid/step/单位、同cycle差分前一步、区间重叠积分、产品版本与来源绑定。mtime不是release证明；缺步长不补造。
数据：冻结共同可用IFS cohort。若与B1不同，A_ST/AC_ST必须在同一新训练/评价合同重训，不能拿不同cohort旧指标拼表。
对照：A_ST、AC_ST、AN_ST、ACN_ST，三配对seed；IFS-only原始通量作参考。标准化IFS为区间平均flux/同区间clear_sky。
成功：ACN_ST优于AC_ST才支持IFS增量；同时优于AN_ST才支持IFS背景下COT增量；分别报告主终点及CI。站点等权MSE的I=MSE(A_ST)-MSE(AC_ST)-MSE(AN_ST)+MSE(ACN_ST)，仅I<0且CI排除0称超加性。
失败：IFS门禁未过则该块blocked，论文保留待检验，不降低发布时间标准；增量不显著就不写“协同有效”。
成本：同cohort可复用已冻结A/AC时新增6次，否则12次；明确记录复用依据。

### B4 稳健性、误差传播与冻结测试（必须；C1/C2边界；Fig. 2/3及补充表）
数据：QC1固定敏感性cohort、验证CPP参考子集与最终独立test；QC1只做 B1 主比较 `A_ST/AC_ST` 的两组×三seed，不按结果替代主cohort。IFS失败时不强行加入。
诊断：相同target及CPP掩膜比较R(real AGRI)与R(forecast AGRI)的COT/log1p误差、厚云偏差及lead变化；该参考子集不反向筛选GHI主表。分站/lead/因果可用云况分层，分层阈值用train冻结；不按误差挑案例。
oracle：未来真AGRI/CPP只作单列诊断，不作部署输入、选模或严格性能上界。误差相关不是因果证明。
成功：原始与QC1差异有完整解释，所有seed与站点均报告；冻结方法/统计协议后统一test，结论与证据边界一致。测试失败如实收窄结论，不返回测试集调参。
成本：QC1最多6次头训练 + 一次批量冻结测试/诊断推理。test门禁以完成前述必需控制为准。

## 5 里程碑与停止规则
M0：B0闭合、实现smoke和单卡profile；否则停在来源/实现缺口。
M1：B1四组、两站短pilot；按上述判据决定是否进入三seed完整比较。
M2：B1完整→B2归因；仅“空间输入改善”不足以跳过容量和监督对照。
M3：来源允许则B3并完成B4验证稳健性；对照不支持时收窄主张，不能堆模块补故事。
M4：独立审计、最近邻查新、协议冻结→一次test→双语正文填实数和作图。
SolarResNet v5 的总GPU小时只在 B0 profile 后按实测填写；B1、容量/同监督/排列和 IFS 均为分阶段门控，不是一次性提交指令。
