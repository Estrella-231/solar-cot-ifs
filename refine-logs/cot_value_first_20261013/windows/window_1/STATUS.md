> 2026-10-13：此窗口状态归档；分工、交接和独立定时要求撤销，scope receipt不再要求。当前状态见../../EXPERIMENT_TRACKER.md。

# 执行窗口1状态

更新时间：2026-10-13 BJT。负责人：执行窗口1（数据与COT）。

## 当前判定

窗口1持续推进T03/T04/T05/T06/T20。总控最新 `W0_AUDIT_20261013.md` 仍将T03列为PARTIAL，但已静态核实观测区间记录与晴空中点是同一组三个5分钟区间；端点瞬时晴空不再作为冻结门槛。train/val候选键也通过独立CPU审计。当前仍没有最终GHI cohort、合格新合同R、COT bank或服务器本地缓存。未读取test载荷；本窗口没有PBS/GPU请求或远程进程。已为当前窗口建立30分钟heartbeat（automation `cot-value-window1-progress`，ACTIVE）；无变化时保持安静。

## 任务状态

|任务|状态|已有交付/证据|未过门禁与下一步|
|---|---|---|---|
|T03|PARTIAL|`target_candidate_cpu_audit_20261013.json`独立审计PASS：111,466候选键唯一、train/val交集0、UTC/BJT差8小时、标签时间字符串为target-10/-5/0；未读GHI/AGRI/CPP/test。`legacy_geometry_formula_witness_20261013.json`对12帧历史验证AGRI输入重算cosSOZ/cosRAA，最大误差分别5.93e-8/8.90e-8，day mask不匹配0，3×3裁剪为[7:10,7:10]。join脚本已增加scope receipt门禁，拒绝未预筛选标签源。|历史几何见证不代表当前月度cohort。仍缺带有效scope receipt的train/val标签切片、旧标签cohort逐键核验、站点观测有效支持统计、连接到当前月度样本的逐帧几何见证。因此候选键仍不是正式有效GHI cohort。|
|T04|WAITING_T01|`retrieval_contract.json`收录本地2026-10-08 R/CPP来源审计、CPP监督mask及支持限制；新合同正式R状态明确标记NOT_VERIFIED_LIVE。|总控T01尚未验证远端checkpoint、manifest、训练日期和哈希。旧 `cot_repaired_seed42_v1` 属于旧cohort；DirectCOT的M1 smoke也不是同期AGRI→COT反演器。取得T01收据后判断是否需要单独重建R，并先报告CPP季节覆盖门槛。|
|T05|BLOCKED_T02|无节点缓存。|T02尚未选择并验收主机、项目数据路径、节点本地盘余量和源哈希；不得猜路径或擅自启动远程缓存。|
|T06|BLOCKED_T03_T04_T05|无同期真实AGRI→R COT bank。|标签cohort、有效R来源和数据缓存均过门后，按同一unique station/target键从真实AGRI运行原生R，再裁站点3×3；检查缺失不得静默删样。|
|T20|WAITING_T19|无预测COT bank。|窗口3的T19预测来源和验收未到；后续只使用合规预测AGRI，不接入未来真实AGRI/CPP或test。|

## 已冻结与未冻结口径

- 日期合同：`hunan_monthly_block_2d_buffer1d_seed42_v1`；candidate builder审计manifest SHA-256为 `f6703ed5e42f1ae00aac6538b108dc4d9361197ad57ad86a0f407767aa2e4cb2`，train/val分别有51,236/4,497个唯一AGRI时刻，展开后102,472/8,994个站点target候选；跨split重复为0，test载荷读取为0。这是元数据候选，不是有效GHI样本数。
- 站点patch元数据：四里patch中心(8,8)、真实HDF像元(694,1554)；竺家(8,8)、(612,1549)。来源为项目静态grid合同审计，并非新现场测量。
- 标签候选：`(target-15min,target]`，观测为target-10/-5/target三条5分钟区间末记录的均值；主mask还要求站点太阳天顶角<90°及区间平均晴空GHI>0，>=100 W/m²作敏感性分析。CPP不得筛选GHI共同样本。
- 晴空取样：Ineichen在target-12.5/-7.5/-2.5分钟取值，分别对应观测三条target-10/-5/0区间记录的中点。总控静态审计已确认时间窗一致；端点瞬时值并非同一时间平均口径，不再列为冻结门槛。`audit_clear_sky_sampling.py`只保留为可选采样敏感度诊断。
- 几何候选为逐像元 `cosSOZ`、`cosRAA`，使用与AGRI/COT相同3×3；`day_mask`仅作样本准入。12帧历史验证见证支持公式和裁剪实现，但没有本轮月度候选的逐帧aux几何载荷，因此新合同映射仍未冻结。
- T04引用的CPP覆盖及R审计是本地2026-10-08证据，不代表当前服务器。CPP参考监督mask为COT有限且[0,100]、CLP∈{0,1,2}、13通道AGRI有效、day_mask>0.5、cosSOZ>cos80；通过联合mask的零COT保留。CPP缺失/读取失败不能转为晴空，也不能缩小GHI主样本。

## 窗口1产物

- `target_candidates_train_val.csv`：候选键，不含观测标签或test。
- `data_contract.json` 与 `target_candidate_audit.json`：候选日期、站点索引和mask合同及元数据审计；几何字段同时标注历史见证通过、新合同逐帧核验待完成。
- `legacy_geometry_formula_witness_20261013.json`：12帧旧验证样本公式/裁剪见证，仅作代码链佐证。
- `audit_clear_sky_sampling.py`：仅读train/val目标时间；需要项目pvlib环境，不读站点GHI、AGRI、CPP或test。
- `join_unique_station_labels.py`：仅接收带scope receipt、已按新train/val候选键预筛选的站点标签切片；receipt需绑定候选与标签文件哈希，并确认不含新test标签。脚本拒绝候选外key；对重复target核验一致性、保留新旧split。代码安全审查收据为 `ops_receipts/20261013_label_join_scope_guard.json`；这不是实际标签切片的scope receipt。
- `retrieval_contract.json`：本地R/CPP来源审计和限制；不是远端资产收据。

## 下一推进点

1. 等数据端提供严格按候选station/target键过滤的train/val标签切片及绑定哈希的scope receipt；不能把全量旧canonical manifest交给脚本。之后复核重复窗口标签一致性、旧/新split迁移、缺标签/无效GHI及逐键记录时间。不得因旧split不一致拒绝新月度归属。
2. 可访问train/val标签源后，统计两站、月份/季节、主mask与>=100 W/m²敏感mask的有效支持；CPP不得筛选GHI cohort。
3. 以新合同数据源核验逐像元太阳几何；冻结与3×3 AGRI/COT完全同格的两张cos图，并确认day_mask仅筛样本。
4. 收到T01/T02/T19后分别解锁T04/T05/T20；T06等T03/T04/T05全过再进行。GPU/缓存提交仍由总控唯一入口处理。

## 证据边界

本窗口没有对复旦或PBS服务器做新的连接尝试。2026-10-13的W0收据显示当时PBS查询无账号作业，但GPU节点有其他作业占用，且复旦SSH连接关闭；该快照不证明当前资源状态。heartbeat只唤醒并推进窗口1自己的本地审计/依赖检查，不代表服务器状态已刷新，也不提交任务。收到新的来源或资源收据后才继续相应任务，不以旧路径、旧队列或本地审计代替实时证据。
