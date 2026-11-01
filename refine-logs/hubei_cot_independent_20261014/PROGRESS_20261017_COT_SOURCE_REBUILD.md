# 湖北三站 COT 输入包来源重建进度（2026-10-17）

## 2026-10-17 20:39 BJT：审计结果口径冲突复核

- 复核发现独立test-freeze审计给出的同期A/AC/AP（74.361/71.429/73.894）与当前机器收据中的等站点、等seed汇总（74.33/70.99/73.84）不一致；目前尚不能判断是聚合定义不同，还是审计读取了不同运行结果，因此不把差异归因于旧版来源/归一化合同。需核对其逐样本预测来源、RMSE聚合口径及run metadata后统一报告。当前机器收据为`audit_receipts/same_time_three_seed_normreload_20261017/independent_three_seed_validation_audit.json`。
- forecast-GHI方面，当前normreload机器收据为FA/FAC=77.150/78.155 W/m²、FAC−FA=+1.005 W/m²；test-freeze审计引用的77.250/78.581也需查清运行来源后再判断是否为旧运行。`SAME_TIME_THREE_SEED_VALIDATION_20261017.md`已改为引用当前normreload汇总。
- 旧test-freeze审计的FAIL/NO-GO仍作为保护性状态执行，但其指标和运行谱系判断不能未经复核沿用。独立复核完成前，test/buffer继续关闭；未提交新作业。

## 2026-10-17 test-freeze 审计与可复现性修复

- Fresh reviewer 对同刻与forecast-GHI验证谱系完成只读审计，结论 `FAIL / NO-GO`；当前held-out test/buffer继续关闭。审计报告、JSON输入哈希与reviewer响应留痕见 `audit_receipts/test_freeze_audit_20261017/`。本地留痕不是平台级模型trace，审查为同系列provisional。
- 审计确认forecast-GHI三seed验证FA为77.249908、FAC为78.580996 W/m²，FAC−FA为+1.331088 W/m²，当前验证不支持加入预测COT带来收益。该结果不是test结论。
- 同刻结果的唯一当前谱系为重建pack `e8c5…`：A/AC/AP为74.361233/71.429292/73.894113 W/m²。旧pack `7990846…`的AC=71.428不得与新pack结果混用；正文和claim ledger仍需同步修订。
- 针对checkpoint未绑定精确train-only normalization及缺少checkpoint reload推理见证，已修改 `../../scripts/train_hubei_ghi_heads.py`：新训练将写出规范化合同JSON与SHA，并将其纳入checkpoint；保存后从checkpoint重载，再复跑validation并核验输出一致性。`python -m py_compile`通过。旧checkpoint没有因此自动补齐或获得新见证，必须在合规新运行中产生新的证据。
- 当前未提交PBS作业，也未打开test/buffer载荷。距离test放行仍有fold-specific OOF训练清单证明、真实分析汇总脚本hash绑定、fail-closed test推理/评分器及synthetic/validation-only dry-run、论文/计划/追踪器同步等工作；预测头arms与test评价目的也须先冻结。训练不因账户有空闲GPU而绕过这些科学门禁。

## 2026-10-17 19:14 BJT：6—7月缺档与原工程口径再次核实

- 用户补充的节点凭据仅用于只读访问；FD-172可登录，FD-201由已配置密钥登录。对FD-201两处登记HDF年度根及预处理树按16个合法UTC键精确复查无命中；原工程`GHI/data/hubei`、`archive`、`preprocession`中出现的8个同名NPZ全是湖南`hunan_truth_only_labels`标签文件，不是湖北AGRI源，未读取载荷或挪作候选。完整细节见`AGRI_GAP_RECONCILIATION_20261016.md`。
- FD-201当前代码再次确认：原始Excel第6列“总辐射1瞬时值”直接读取、记录时间取第17列；历史标签按`[BJT,BJT+5min)`筛正值平均；GHI按1300归一化。该操作合同不是独立仪器校准证明；本轮15分钟右闭标签定义不同，必须继续分开表述。
- 六月至七月14日数据仍是2,644/2,660个合法影像时次；没有新原始帧可用于建立补充manifest。此前A/AC/AP同期与forecast-GHI FA/FAC三seed训练已在冻结可用数据上完成，故本轮没有重复训练或提交新PBS作业。当前`zjnu-hpc`全账户队列快照只有`214221.tc6000`一项排队；held-out test/buffer继续关闭，因新审计为NO-GO。
- 下一步：若用户/数据管理员能提供可追溯原始HDF或其他有生成血缘的副本，按原处理流程逐像元比对后另建manifest及run；否则保留16个缺档，不用插值补造。先修复NO-GO审计列出的OOF来源证明、预测头归一化/重载见证与fail-closed评测器，再讨论一次性test评分；不为缺失时次或追求正向COT结果重复训练。

## 目标与范围

重建同时间 A/AC/AP/AL 试验使用的冻结 R-COT train/validation 特征包，逐项绑定 AGRI 路径清单、R 预测、站点特征和标签包；不覆盖旧 pack/模型输出，不读取 test/buffer 载荷。此工作是来源审计和输入包重现，不等同于新的GHI训练或独立测试结果。

## 2026-10-17：路径命名空间问题与重跑

- 首次重建作业 `214605.tc6000` 在 `node21` 获配单卡后约48秒失败（exit status 1）。错误为路径找不到：manifest 中的历史路径 `/home/Data_Pool_3/chenyi/.../data16_label/...` 不在 PBS `zjnu-hpc` 文件系统命名空间内；不是确认到源样本缺失。作业未生成新的R bank、station feature bank、pack或比较报告；日志未显示 test/buffer 读取，但失败运行没有闭合的计数收据，因此不宣称已证明其读取计数为0。
- PBS阶段目录 `/public/home/slfu/ttzhou/swc/irradiance_forecast/hubei_al_same_time_20261017/data16_label/` 已有 train/validation 输入。新增路径映射脚本仅将manifest中已审计的相对月份/文件名映射到该阶段根，保留原 `time_bjt` 和 `split`，拒绝test/buffer行，并在输出前逐项检查文件存在。
- 预检收据 `source_rebuild_20261017_v2/manifests/agri_paths_trainval_pbs_report.json`：24,705行、24,705唯一时刻；train 22,660、val 2,045；原 `data16_label` 来源22,061行，已审计补充来源2,644行；映射文件检查24,705/24,705存在，缺失0、非train/val行0。原冻结split SHA仍为 `6ef8cbc7d2bb7d1b1d3defdfcba473cb171e06db0115c8206f7c289d8bad19c6`。清单与报告SHA分别为 `df2a858c…e493cf`、`5b796a38…e1b6b6`。
- 重跑脚本已通过PBS服务器侧Python语法编译及 `bash -n`；使用新的 `source_rebuild_20261017_v2/` 输出目录，旧pack及已训练输出只读保留。提交前账户队列为1个非终态任务（214221）；提交 `214608.tc6000` 后为2/3，未超项目上限。
- 作业 `214608.tc6000` 已于PBS记录中自然退出，exit status 0，walltime 3分09秒，node21物理GPU3单卡。产物存在于远端 `source_rebuild_20261017_v2/`：R bank 24,705行、station feature bank 24,705行、共同pack 29,351条（train/val）。R预测均有限非负、范围0–74.919；R/feature/pack报告均标明test/buffer闭合计数为0。
- 独立manifest元数据审计通过：原清单SHA `26d1e586…03534`、remap清单SHA `df2a858c…e493cf`；冻结day assignment SHA `6ef8cbc7…19c6`；24,705键与划分逐行保留，按日期映射再次核对，train 22,660、val 2,045，24,705路径全部存在。只读元数据审计未读载荷值。
- pack与旧 `ROOT/pack` 比较：AGRI、geometry、GHI标签、clear-sky及rows.csv均逐字节相同；**COT数组不相同**，旧SHA `b401de7e…326b5`，重建SHA `3f25e82d…61d4e`，264,159个COT值中9,495个不同，最大绝对差0.226608。差异分布于train和val三站；新pack逐行与本次重建feature bank完全一致，并绑定本次R报告/预测哈希。旧pack报告没有保存R checkpoint哈希，故旧COT数组的来源无法由现有报告闭环。详细分站/划分差异见本目录 `audit_receipts/cot_pack_source_rebuild_20261017_v2/cot_pack_difference_audit.json`。
- 结论：旧A/AP所用AGRI、几何、标签和晴空数组与重建pack相同；旧AC使用的COT源不能确认与冻结R一致，故旧AC点估计及AC-A/AC-AP比较不再作为来源闭合的证据。保留旧训练产物，不覆盖。为获得受审计的同期比较，按同一pack对A/AC/AP整体重训三seed；不把本次重建或pack差异本身描述为技能结果。
- 新pack上的A/AC/AP单卡profile作业 `214612.tc6000` 已提交；提交前全账户1个非终态（214221），提交后为2/3。Luna最新快照为状态R、node21物理GPU3；尚未看到profile产物，等待自然结束。

## 2026-10-17 profile复核与重提

- 对 `214612.tc6000` 的终态核查显示 `exit_status=2`。pack预检通过（29,351条、test/buffer闭合），但profile调用漏传训练器必需的 `--output`，因此没有profile JSON；该作业不构成任何batch/吞吐/显存测量，也不放行训练。
- 已修复并同步PBS脚本 `tooling_20261017/profile_hubei_same_time_heads_rebuild_v2_20261017.pbs`，新增一个仅profile模式使用的占位 `--output` 路径，不会创建训练结果。远端脚本SHA-256=`b6930930dced32f596824f02ecee8e348a909746e2724e4464db7ee92044b1b9`，与本地一致。
- 新profile作业 `214613.tc6000` 已提交并确认运行；提交前账户非终态任务1个（214221），提交后为2/3。分配资源为单节点8核/1 GPU；等待profile结果后再选共同batch，不启动正式训练。
- 六月至七月14日train/val AGRI载荷当前可用2,644/2,660个合法时次；其余16个为登记来源中的真实归档缺口，未经验证的数组文件不纳入。这里的“补齐”是补充新增月份并按来源核验后进入有效支持，不代表16个缺口已恢复或整段日历全覆盖。
- 随后的profile job `214613.tc6000` 再次未通过：train/val预检通过、获得物理GPU3，但Python加载时发生cuDNN动态库符号冲突（exit 127）。对照成功的重建PBS确认其显式把当前Conda环境的 `nvidia/cudnn/lib` 放在 `LD_LIBRARY_PATH` 首位；profile wrapper此前遗漏此设置。
- 已按同一成功作业的环境初始化修复profile wrapper，并核对本地/远端脚本SHA相同（`a3f813031fcb4daf50150f85605a849ff41d06266300bb5706b4ba322b6f7f37`）。第三次profile `214614.tc6000` 自然完成，PBS exit 0，walltime 4:56；有效profile JSON已留存到本地 `audit_receipts/cot_pack_source_rebuild_20261017_v2/profile_same_time_heads.json`。
- 共同batch选择512：三臂在Mizhuang最小train支持集8,832行上均达到17个完整updates/epoch；吞吐为A 183,118、AC 190,801、AP 152,469 examples/s，三臂各自最高吞吐均出现在512；profile峰值显存约0.151 GiB。该profile是小型局地回归头的单卡吞吐见证，不表示模型训练已经达到GPU高利用率，亦不需要为了显存占用盲目增大batch。
- seed42正式同时间训练已提交为 `214616.tc6000`，PBS依赖 `afterok:214614.tc6000` 已满足；单卡node21物理GPU3，A/AC/AP、三站、30 epochs、batch512。Luna首个启动快照确认run_metadata已写入且 `PROFILE_AND_PACK_GATE_PASS`，test_payloads_read=0；但1分多钟后尚无epoch行或stdout更新，GPU约550 MiB/40.5 GiB、util 0%，进程处于低CPU `lock_blkgrp_sync` 等待。当前只能说训练入口已启动、尚无训练/验证结果；须继续检查数据加载是否推进，不能把此状态称为正常训练吞吐。

## 6月—7月14日数据覆盖与单位边界

- 冻结日期上的AGRI覆盖仍为2,644/2,660个日历合法时次；16个已登记源目录均无原始HDF命中。已知2+6个 `.npy` 仅格式兼容、源血缘未验收，不计为补齐帧；不插值、近邻替代或热改当前训练manifest。
- 原工程 `build_label.py` 对第6列“总辐射1瞬时值”直接取数值；旧标签是 `[BJT,BJT+5min)` 内正值均值，并将 `ghi_obs` 除以1300训练、评估时乘回1300。此为旧工程操作合同；未找到仪器/导出字典，不能称为独立单位认证。
- 新标签按右闭15分钟窗 `(target−15min,target]` 从分钟记录构建，保留有效零值，缺测窗口mask。标签量纲的独立计量来源仍是训练/发布前置门禁；当前任务不借由特征重建绕过它。

## 2026-10-17 15:20 BJT：seed42正式训练完成，seed43/44接续

- `214616.tc6000` 已正常结束：PBS `job_state=C`、`exit_status=0`，walltime 8分45秒；node21 物理GPU3。run metadata为`HUBEI_GHI_HEADS_SEED_COMPLETE`，seed42、30 epochs、batch512，pack SHA=`e8c5b2dc587e8a938bba43fc564be43aec777788aca223c054341f75dd09a515`，trainer SHA=`eed9c4aa761ae8fd5e42454d2b267e2ed9e2e58a9ba454289adf0e3067c89125`，`test_payloads_read=0`。
- 三站9个A/AC/AP臂均产生best checkpoint、validation JSON、带键逐样本预测和metrics；每臂30 epoch，站点级90条metrics，合计270条。验证RMSE（W/m²，A/AC/AP）：Niushou 75.996/74.682/75.312；Wangzhai 67.896/66.107/67.338；Mizhuang 81.591/76.146/81.242。AC在本seed的三站RMSE都低于A，但仅是一个seed的固定validation观察；尚未完成独立指标复算、配对日期块不确定性与其余seeds，不能作论文效应结论或预报技能结论。
- 新建seed43/44 PBS脚本`tooling_20261017/train_hubei_same_time_seed43_44_v2_20261017.pbs`，本地与远端SHA-256均为`27377b80079116793fdbabf3291ebbbba8cb8f2f2c93f2887269358488ff8e1c`，远端`bash -n`通过。作业`214618.tc6000`（seed43）与`214619.tc6000`（seed44）现均运行于node23物理GPU2/3；15:27 BJT快照时均持续产生epoch日志。提交前后逐次核验全账户队列；当前3/3非终态为`214221`、`214618`、`214619`，达到项目上限，不再提交新PBS作业，直至有任务完成。
- 已把seed42的9组validation JSON、逐样本带键预测包和metrics拉取到本地`audit_receipts/same_time_seed42_independent_recheck/`，独立脚本`tooling_20261017/audit_same_time_seed_predictions.py`逐行验证三臂键、实测真值和晴空值一致，并从已保存预测复算RMSE/MAE；9组都与训练端元数据在1e-8内吻合，test仍未读。按目标日期进行10,000次配对bootstrap：AC−AP的RMSE差在三站分别−0.630、−1.231、−5.096 W/m²，95%区间分别[−2.717,1.621]、[−2.663,0.034]、[−11.468,0.662]，均跨0；单seed不能证明COT优于容量匹配对照。报告为`audit_receipts/same_time_seed42_independent_recheck/independent_metric_recheck.json`。本地均值方向与CI只用于验证诊断，最终论文结论须等三seed和后续门禁。
- 按用户新授权密码只读登录FD-172成功；五张RTX 3090当前显存分别约5.0、14.9、15.6、1.6、18.0 GiB，均有非零显存占用，因此不将其视作可自由分配卡、不启动本项目计算。FD-201/FD-107已登记状态仍见前次只读清点：FD-201 GPU0–3仅为空闲迹象、未取得调度分配；GPU4–5及FD-107各卡按他人/现有任务占用处理。密码不写入工程文件。
- 6月1日至7月14日已验收AGRI覆盖仍为2,644/2,660；16个真实时次缺档在已登记原始HDF根及已知预处理树均无通过验收的来源。不能把“补齐该时段可用数据包”说成全日历无缺；本轮训练使用冻结pack，不做插值或热替换。
- seed43/44现已完成并独立复算；HB04的三seed同时间A/AC/AP第一阶段完成。完整对照与下一步分层分析见后续补记。同期验证完成不代表独立test结论；test/buffer仍封闭。五折forecast S/R OOF和预报链路尚未完成。

## 2026-10-17 15:40 BJT：三seed同期结果独立复算

- PBS `214618.tc6000`（seed43）和`214619.tc6000`（seed44）均已`exit_status=0`，分别使用node23物理GPU2、GPU3，walltime约5分07秒、5分00秒；三站九臂各30 epoch完整完成，metadata及每个validation结果均记`test_payloads_read=0`。全账户此刻仍有`214221 Q`一项非终态；Hubei三seed作业已终止完成，未碰其余任务。
- 保存预测、9组键和值、每seed run metadata均下载到本地`audit_receipts/same_time_seed{42,43,44}_...`。独立审计脚本`tooling_20261017/audit_same_time_three_seeds.py`核验各方法与各seed逐样本 `(station,target_time_bjt)`、真值和晴空值完全一致；训练端RMSE/MAE与预测重算均匹配；三个run metadata一致绑定pack SHA=`e8c5b2dc…09a515`、split SHA=`6ef8cbc7…8bad19c6`、trainer SHA=`eed9c4aa…c7c89125`。独立报告：`audit_receipts/same_time_three_seeds_recheck/independent_three_seed_validation_audit.json`。
- 预定主要终点（每seed三站等权平均，再对3个seed等权）validation RMSE差：AC−A=`−2.932 W/m²`，配对BJT日期块95%CI `[−5.069,−0.941]`；AC−AP=`−2.465 W/m²`，95%CI `[−4.598,−0.453]`；AP−A=`−0.467 W/m²`，95%CI `[−0.689,−0.265]`。连续两日块敏感性区间同样均低于0：AC−A `[−5.197,−0.626]`、AC−AP `[−4.687,−0.168]`、AP−A `[−0.724,−0.250]` W/m²。
- 这是用于best-epoch选择的固定validation，因此这些区间仅为验证阶段的描述性/稳定性证据，**不当作独立确认性test结果**，也不是未来GHI预报skill。结论范围仍是湖北三站同时间GHI估计、原工程W/m²操作合同和当前15分钟右闭标签；仪器校准/字段字典未闭环。
- 扩展服务器盘点：FD-172用用户授权凭据只读登录成功，5张3090均显存非零，占用处理；FD-107 6张A6000均显存非零；FD-201 GPU0–3呈空闲迹象（各5 MiB），GPU4–5占用。FD-201未发现可用调度器命令/本项目GPU分配；按执行计划不得在未获授予的卡直接启动。248未访问。故本轮实际并行使用PBS node23两张已调度GPU，没有占用FD节点空闲迹象卡。
- 6–7月14日覆盖没有变化：可验收2,644/2,660；16个缺失HDF时次已在登记源上精确搜索，FD-172也无新命中。本次新密码只补足FD-172登录，不会恢复缺失产品；数据manifest保持冻结，不插值或替换。
- 下一步：先完成validation固定类别（月份、太阳高度、云况）及预先规则案例报告，再完成COT来源/模型冻结；之后才推进五折forecast S/R OOF和GHI预报头。test/buffer按计划保持关闭，除非既定验证流程完成并明确一次性开启。

## 2026-10-17 15:58 BJT：验证分层覆盖修订

- 复查分层代码时发现旧太阳高度区间从0°起，导致每站少量负太阳高度键被`pd.cut`转为NaN并静默跳过。脚本已修订为明确纳入`below_0`组，并断言所有验证预测键恰好归入一个高度箱；重跑后覆盖为Niushou 856/856、Wangzhai 784/784、Mizhuang 816/816，未分箱0。低于0°的样本分别9、7、9条，其实测GHI均值约4.97、5.19、5.04 W/m²，作为边界组记录，不作有效日间技能解释。
- 新增机器报告来源闭环字段：冻结pack SHA、父级三seed validation audit SHA、每个seed/站点/方法预测NPZ SHA，以及每站分箱覆盖。报告位于`audit_receipts/same_time_three_seeds_recheck/month_solar_subgroups_validation.json`；方法摘要位于`HUBEI_VALIDATION_SUBGROUPS_20261017.md`。
- 太阳高度分层仍显示站点异质性：Niushou在15–45°点估计基本持平，Wangzhai在45–60°AC较差约1.90 W/m²，Mizhuang在15–45°AC低于A约6.88–13.02 W/m²、但45–90°较差约1.89–8.10 W/m²。月份层面三站AC−A方向分别在8/11、8/11、6/11个有支持月份为负，日期支持稀疏。所有为被用于checkpoint选择的validation描述结果，不作独立确认或普遍条件结论。
- 云况门禁未通过：GHI共同pack无独立CLP/CPP云覆盖字段；已知fixed_v2 CPP参考时间支持只到2026-03-02，且现有审计尚未证明其可依预注册≤20%/≥80%云覆盖阈值生成云况类别。没有用GHI、误差或R输出倒推晴/阴。云况分层记缺项，先核对预先规则案例图与本次冻结pack/新三seed来源链；不复用来源未闭环的旧图作结果。
- 六月至七月14日train/validation AGRI仍只有2,644/2,660个合法产品时次通过原始HDF验收，16个缺档未获得同源可验收原件；既有标签继续沿用原工程操作口径并与新15分钟标签窗口明确区分。A/AC/AP三seed同期训练已完成，但forecast-GHI训练仍需满足五折S/R OOF银行及pack门禁；本次没有提交PBS、没有读取test/buffer载荷。用户补充节点密码未保存；248节点未访问。
- 下一步：核查旧案例图的pack/checkpoint/预测哈希；若不匹配，按预先哈希/观测事件规则基于新预测重建validation示例图，并明确云况标签缺项。另继续完成五折S/R OOF merge与forecast pack门禁；不在未获调度分配的FD空闲迹象卡上直接运行计算。

## 2026-10-17 16:05 BJT：新三seed预测示例图重建

- 已核验旧`validation_cases_b2_20261016_v2/FIGURE_AUDIT.json`：其中A/AC/AP三seed 27个预测文件SHA与本次冻结pack的27个文件全部不匹配，旧图不能代表本次审计结果；旧资产保留，不覆盖。
- 复用预先冻结、只依观测事件和哈希选日期的24案例清单，另建GHI折线图`validation_cases_b2_20261017_rebuilt_v2/`。脚本`tooling_20261017/plot_hubei_validation_cases_rebuilt_20261017.py`按带键NPZ检查三臂/三seed的目标时间和真值完全一致，绘制观测与A/AC/AP三seed均值及样本SD；15分钟缺测保留断线。报告绑定冻结pack、父级审计、清单和全部预测文件SHA，test/buffer读取计数均为0。
- v1布局检查发现总标题与图例重叠，已保留并标记为superseded；v2调整后已人工查看Niushou图，标题、图例、坐标轴和注释未重叠，三站PDF/PNG均已生成。
- 图只呈现GHI折线，不包含同位AGRI/COT图像；云况标签仍缺，故它是validation示例而非完整天气案例或最终论文图。没有生成晴/阴天气分类，也没有利用模型误差重新挑日期。
- 未训练、未提交PBS、未触及test/buffer。下一步若要完成计划中的全案例分析，需取得来源可核验的AGRI/COT局地sidecar，并继续核实全时段独立CLP/CPP云况字段；同期推进仍未闭合的五折S/R OOF merge门禁。

## 2026-10-17 16:31 BJT：五折全验证集复载审计启动

- 用户补充了节点登录口令。本轮仅用于获授权的只读/本项目操作，没有把口令写入工程文件或命令脚本；未访问FD-248。
- 为修复此前因使用缺少`cv2`的错误Python环境导致的审计启动失败，新增独立版本runner `tooling_20261017/run_hubei_fullval_audits_parallel_20261017_v2.sh`，固定调用服务器`swc`环境，逐折核验冻结index/norm/checkpoint/stage SHA，完整验证407条validation键并断言test/buffer载荷计数为0。脚本本地SHA-256：`8c5cc4469116844db062ac0266a799098a2a76bee85a0f2d5cef6ed741eb9ae2`；远端`bash -n`通过。旧目录不覆盖，新产物仅写入`fullval_audits_parallel_v2_20261017/`。
- 最近一次可确认的远端快照：runner PID `2025973`仍存活；fold0/1/2分别在FD-201 GPU0/1/2启动，三折各自的validation索引均为407条、test计数为0，尚未产生最终完整审计收据，因此目前只能记为“运行中、待验收”，不能说全验证集审计通过。该快照来自约16:28 BJT；随后本机尝试的`zjnu-hpc`别名实际落在PBS登录节点而非FD-201，FD-201当前密钥认证也未通过，故本段记录不将过期快照冒充当前状态；例行只读刷新交由Luna监控任务继续确认。
- 本地五折合并单元测试在正确包路径下9/9通过；这只证明合并器的门禁测试通过，不代表真实五折数据已合并或forecast pack已生成。
- 六月至七月14日的16个AGRI缺失时次在已登记FD-107、FD-201、FD-172来源根中均无精确文件命中；覆盖仍为2,644/2,660，没有插值或改写冻结manifest。
- 16:31 BJT Luna只读刷新：PID `2025973`仍存活，fold0–2三路推理正在运行；每折validation `pred/true` memmap均已达到完整预期大小`86,687,872 B`，geometry为`20,004,992 B`，读写计数仍增长。尚无keys、`BANK_REPORT`或审计JSON，fold3/4尚未启动，也没有`COMPLETE`标记。三个launch manifest均记`test_payloads_read=0`、`buffer_payloads_read=0`、`test_or_buffer_rows=0`。因此审计确有进展但仍未通过最终门禁。
- 16:34 BJT Luna再次刷新：fold0、1、2已分别通过完整验证集复载审计，均覆盖407/407条validation键，keys/bank/hash收据与审计JSON已落盘；launch manifest记录`test_payloads_read=0`、`buffer_payloads_read=0`、`test_or_buffer_rows=0`，并有对应`FOLD_n_FULLVAL_AUDIT_PASS_TEST_CLOSED`状态。PID `2025973`仍存活；fold3/4已启动并进入推理，初始进程状态处于等待、数组尚未写出，不能判失败。当前fold3/4使用GPU0/1；没有COMPLETE marker，五折整体审计仍未闭合。
- 16:36 BJT Luna刷新：fold3/4均已进入推理，各自train键约3,277/3,279、validation为407、CSV test为0；预测/真值memmap与geometry文件达到预期大小，PID `2037925/2037926`存活，read_bytes约34 MB，暂未生成最终审计JSON。GPU0/1显存约3.7/4.1 GiB；同一时点GPU2空闲迹象，GPU3–5有其他用户进程。FD-107的GPU0/1/2/4/5被占用，GPU3仅为空闲迹象且尚无分配；FD-172本次身份认证失败，无法核验。以上仅为瞬时快照，不把空闲迹象当可分配资源，也未启动/停止任何任务。
- 本机在`solar-cot-ifs/scripts`目录重跑合并器与forecast-pack门禁测试：13/13通过。测试验证五折键映射/覆盖、拒绝重复或缺失fold、完整validation来源收据、R折外隔离、test/buffer计数、OOF lineage和5折pack入口；仍只是软件合同测试，不代表真实bank已合并。
- 16:39 BJT Luna对同一runner的再次复核：PID `2025973`与fold3/4及DataLoader子进程都仍在运行，无退出或traceback；两折memmap尺寸不变、mtime更新到08:39 UTC，显示输出仍在写。审计JSON与`COMPLETE.txt`仍未产生，manifest闭合计数继续为0。GPU瞬时显存回落到5 MiB且利用率0%，因此当前只确认进程/输出时间戳仍活动，不把它描述为GPU满负载；未重启或更改任务。
- 16:41 BJT Luna最新快照：fold3/4推理已各写出407条完整validation bank报告，fold3归一化全场MSE/MAE=0.2409054/0.3084052，fold4=0.2387806/0.3001511；两折完整预测/真值/geometry数组及`keys.csv`已生成。fold3/4审计JSON仍未出现，`COMPLETE.txt`仍缺，故尚不能进入五折合并。启动清单和bank报告仍闭合test/buffer计数为0；PID `2025973`及推理进程继续运行，未见错误。
- 下一步：等三折全验证审计产生成功/失败收据；确认fold0–4完整复载与共同键通过后，再执行真实S/R OOF合并和forecast-GHI pack门禁，随后才开始预先批准的forecast head训练。test/buffer继续关闭；PBS提交前仍须重新查全账户队列并服从最多3个未结束任务限制。

## 2026-10-17 17:29 BJT：五折 OOF GHI 训练包闭合，单卡 profile 排队

- 完成此前待办：五折完整验证集复载审计已全部通过；真实 forecast OOF 训练bank合并完成，包含4,086条训练序列，五折数量823/797/850/809/807。合并报告状态为`MERGED_HUBEI_SIMVP_FORECAST_GHI_OOF_TRAIN_BANK_TEST_CLOSED`，test和buffer读取计数均为0。
- 预报GHI训练包已建立于`/home/Data_Pool/zjnu/hubei_cot_independent_20261016/forecast_ghi_pack_oof_20261017_v1`，样本70,171（train 63,448；validation 6,723），覆盖Niushou、Wangzhai、Mizhuang；输入为SimVP预测AGRI→冻结R预测COT，train侧5折OOF、validation侧冻结full-train S/R。报告明确target为BJT右闭15分钟区间结束时刻，lead为+15…+240 min，标签沿用原工程W/m²操作合同；传感器校准/导出元数据仍未独立认证。pack报告记`test_and_buffer_values_read=0`。
- 逐文件迁移到PBS共享目录`/public/home/slfu/ttzhou/swc/irradiance_forecast/hubei_forecast_ghi_20261017/pack`，7个包文件SHA-256均与来源文件及`pack_report.json`一致。训练器、OOF pack gate、合同见证和profile PBS脚本同步到同目录`scripts/`。
- 目标SWC环境合同见证通过：拒绝非OOF输入、接受五折OOF lineage、拒绝错误validation来源；FA/FAC无/有COT前向输出有限且形状正确；OOF pack gate标准库测试4/4通过；profile脚本`bash -n`通过。合同测试不是科学性能结果。
- 新增profile作业脚本`tooling_20261017/run_hubei_forecast_ghi_profile_20261017.pbs`。首次`214644.tc6000`因运行时cuDNN库冲突退出（详见下方17:34更正）；修复后`214646.tc6000`申请1 GPU、walltime 45分钟，只测FA/FAC batch吞吐并写profile报告，不训练、不读取test/buffer。
- 数据补齐核查没有找到16个缺档时次的可验收同源AGRI文件；截至目前仍为2,644/2,660。仍不插值、不拿邻近时次替代、不改冻结manifest。若后续发现原始发布文件，必须重新按原HDF验收并单独版本化数据包，当前已冻结OOF和pack不静默热替换。
- 用户补充的节点认证范围仅用于必要远程操作；本轮迁移通过PBS登录节点密钥完成，未将密码写入脚本/文档，未访问248。
- 下一步：profile作业完成后核对真实batch、吞吐、峰值显存和test关闭收据；按结果先提交FA/FAC seed42，再逐次查询全账户队列并按同一预算推进seed43/44。若profile未达训练更新/吞吐门槛，先修loader或batch选择，不直接开正式训练。

## 2026-10-17 17:34 BJT：profile运行环境修复并重提

- 首次profile作业`214644.tc6000`已由PBS记录`exit_status=127`；stdout证明包门禁通过、物理卡为node23 GPU3 A100 40GB，但Python在CUDA初始化前因SWC环境cuDNN动态库加载冲突退出。该次没有profile测量或训练更新，不能报告GPU训练结果。
- 按仓库此前验证过的SWC启动方式修订profile脚本，显式将`$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib`置于`LD_LIBRARY_PATH`前端。服务器`bash -n`通过；同一变量设置下登录节点`import torch`成功（`torch 2.3.1+cu118`，登录节点CUDA不可见属预期），脚本SHA-256=`d694bdb552d0c9b6c36a87b46920a5273d6b019abf551fd9d3958f84f3a0ac2f`。
- 按项目资源规则重新查队列后，仅有`214221.tc6000`一个未结束任务；故重提单卡profile `214646.tc6000`，当前状态R、node23 GPU3。当前全账户未结束任务数为2；未结束他人任务、未读取test/buffer。作业仍在初始化阶段，尚未产出profile JSON，正式FA/FAC训练不启动，等待profile收据。

## 2026-10-17 17:36 BJT：单卡 profile 通过，FA/FAC seed42 已启动

- 修复后的profile `214646.tc6000`由PBS记为`exit_status=0`、node23 GPU3（A100-SXM4-40GB），walltime 1:17。profile站点为训练支持最小的Mizhuang（20,611 train样本），test读取计数0。batch候选128/256/512/1024/1536中，batch1536为FA和FAC各自最高实测吞吐，并满足每epoch至少12次更新（实际13次）；吞吐分别557,331与518,443 samples/s，峰值分配约0.167 GiB。batch1536被选为两臂共同batch。该指标只针对小型局地GHI头profile，不能解读为GPU已满载或端到端S/R推理速度。
- profile JSON已存本地`audit_receipts/forecast_ghi_profile_20261017/forecast_ghi_profile_fa_fac.json`，SHA-256=`569295277ca9c7f64b9b1d9eccd0296d3ce22fad748e6f914bfd5e9d4ae85de0`；远端原报告位于本项目PBS run root的`profile/forecast_ghi_profile_fa_fac.json`。profile脚本源哈希及首次cuDNN失败/修复记录如上。
- 按计划先启动FA/FAC seed42，不启动容量对照FAP：`214647.tc6000`，状态R，node23 GPU3，30 epoch、batch1536，run root `/public/home/slfu/ttzhou/swc/irradiance_forecast/hubei_forecast_ghi_20261017/results/forecast_FA_FAC_seed42`。启动前全账户仅`214221.tc6000`一个未结束任务；提交后两个未结束作业，符合最多3个限制。训练输入绑定五折OOF pack，validation仅作选模；test/buffer未读。
- 训练尚未完成，当前没有预测指标、技能结论或论文结论。下一步等待seed42独立收据，复核run metadata、每站FA/FAC全30 epoch完成、checkpoint重载与保存预测；通过后才依次评估seed43/44提交时的全账户队列，不提前开启test。

## 2026-10-17 17:43 BJT：seed42独立复算通过，seed43运行中

- 更正上节中的历史状态：seed42作业`214647.tc6000`现已自然完成，PBS exit status 0，FA/FAC六个站点模型均完成30 epoch，训练元数据记录test读取为0。逐站预测和checkpoint已保存。
- 独立FP64复算收据`audit_receipts/forecast_ghi_seed42_20261017/seed42_validation_audit.json`通过，SHA-256=`D994284998EA166B570F916ABB53332EC5A5D5FEFED4229BCBAA4B6911726C5B`。各站FA/FAC验证RMSE（W/m²）：牛首74.318/75.080，王寨73.884/73.928，米庄84.757/84.835；FA/FAC使用完全相同的验证键和真值。FAC−FA分别为+0.762、+0.043、+0.078 W/m²，表示本seed的COT条件没有改善验证RMSE；该结果仅是选模验证证据，不能外推成多seed结论或test结论。
- 预注册seed43作业`214649.tc6000`已在PBS运行（node/训练日志由只读监控继续核实），同一冻结forecast-OOF pack、batch1536和30 epoch合同；截至本记时点未取得最终验证收据。当前账户另有`214221.tc6000`排队，合计未结束任务2/3；不提交seed44，待seed43完成并重新查询全账户队列后再按计划决定提交。
- 预测包绑定SHA=`9421c0d5e19094454e5b94b98de797ac90248d3e5b56fdef6dacda339df359dd`，train侧为五折日期外S→R预测，validation侧为冻结full-train S/R；预测时效+15…+240 min，实测值按15分钟目标窗评估。训练和验证过程test/buffer保持关闭。
- 数据与口径：6月1日至7月14日冻结train/val的AGRI数据已纳入独立补充包，2,644/2,660个合法产品时次通过来源验收；另外16个精确产品时次在已登记归档/实时根检索未找到，8个同名NPY候选因缺乏原始HDF逐像元血缘仍未验收。没有插值、邻帧替代或热改manifest。原工程直接将“总辐射1瞬时值”作为辐照度数值，旧站点标签为[BJT,BJT+5min)正值均值，Direct-GHI以1300归一化并按W/m²输出；本轮新标签是明确采用的右闭15分钟窗(target−15min,target]、有效零值保留、缺测窗mask。W/m²是原工程操作合同而非独立仪器校准证书，时间窗口也是本轮新定义，不冒充旧标签口径。
- 当前工作：已在冻结、可用数据上继续运行预测头实验；未声称16个真实缺档已恢复，也未声称seed42证明COT有效。下一节点是seed43保存结果的独立复算，通过后再查队列并提交seed44。

### 2026-10-17 17:45 BJT：seed43只读运行快照

- Luna只读监控确认`214649.tc6000`仍为R，node23 GPU3；Niushou与Wangzhai的FA/FAC、Mizhuang FA已完成，Mizhuang FAC正在训练（快照约epoch14），run metadata仍是`TRAINING_STARTED`，`test_payloads_read=0`。
- 日志：`/public/home/slfu/ttzhou/swc/irradiance_forecast/hubei_forecast_ghi_20261017/logs/train_seed43_stdout_214649_tc6000.log`；metadata：`/public/home/slfu/ttzhou/swc/irradiance_forecast/hubei_forecast_ghi_20261017/results/forecast_FA_FAC_seed43/run_metadata.json`。
- 中间训练日志中的验证数值尚非完整seed结果，暂不用于比较或作结论。等待全部站点FAC完成、checkpoint与预测保存后再下载并做独立复算；不提交seed44，直到账户队列刷新且seed43终态可查。

## 2026-10-17 17:47 BJT：seed43复算完成，seed44已排队

- seed43作业`214649.tc6000`已由PBS确认exit status 0，六个站点×方法组合均完成30 epoch，checkpoint、逐样本验证预测和指标已保存；metadata及六份验证结果均记录test读取0。远端stdout：`/public/home/slfu/ttzhou/swc/irradiance_forecast/hubei_forecast_ghi_20261017/logs/train_seed43_stdout_214649_tc6000.log`。
- 已将seed43 FA/FAC结果复制到本地审计目录，并运行同一独立FP64验证审计脚本；报告状态`HUBEI_FORECAST_GHI_SEED_VALIDATION_AUDIT_PASS_TEST_CLOSED`，本地审计JSON SHA-256=`F1AAD875126FB33AB1177EB0D823F3FD971077C7CF510C369C291C5EC0ABC219`。三站FA→FAC验证RMSE分别从76.10→78.32、71.42→74.11、80.96→82.14 W/m²；各站两臂键和值完全一致，test/buffer仍关闭。此次是第二个独立训练seed，不能单独代表最终估计。
- 提交前PBS账户快照仅有`214221.tc6000`一个未结束作业；同一runner已核验哈希`0f98844523a5e25ee33af69ed83e3825aa545ba7c3a2c57b59e37449a74762f8`，输出目录未存在后提交预注册seed44为`214651.tc6000`。提交后队列显示`214221 Q`及`214651 Q`，共2/3；未终止或干扰其他作业。seed44等待调度，完成前不做三seed最终汇总。
- 下一步：只读监控`214651`运行及自然结束；完成后拉取预测、独立复算，再按冻结协议做三seed站点等权汇总与日期块配对不确定性分析。若三seed验证结果完成但不能支持COT预报增益，仍继续执行计划内的COT预报误差评价与规定证据整理，不通过反复调参追正向结果。

### 2026-10-17 17:49 BJT：seed44已开始运行

- PBS刷新：`214651.tc6000`状态R、node23，运行约2分钟；账户未结束任务为`214221(Q)`和`214651(R)`，2/3。该作业申请1 GPU，沿用已通过profile的batch1536与固定训练预算。登录节点没有直接运行GPU作业，其他节点/GPU状态不作为可用资源分配证明；训练仍经PBS调度。
- 本快照无seed44最终指标；待Luna确认三站六臂全30 epoch及退出码，再拉取预测作独立复算。test/buffer未打开。

### 2026-10-17 17:52 BJT：seed44开始正常产出epoch

- 更正等待阶段的判断：seed44训练进程仍存活，并已在Niushou FA/FAC上完成各30 epoch，随后继续Wangzhai/Mizhuang；早先日志短时空白不能证明卡死或归因于共享存储。
- Niushou中途完成结果为FA RMSE/MAE=79.05/52.10、FAC=78.41/51.70 W/m²；尚待其余四臂完成，不能作为seed44终值。metadata仍为`TRAINING_STARTED`，test读取计数0。

## 2026-10-17 seed44完成与三seed独立验收

- `214651.tc6000`已自然结束，PBS `C/exit_status=0`，node23 GPU3，runtime约390 s；三站六臂均完成30 epoch。run metadata与六份validation JSON均`test_payloads_read=0`，六个checkpoint和逐样本预测文件齐全。期间出现的短时无日志后自行恢复，不能据此归因于特定I/O故障。
- seed44结果已复制到`audit_receipts/forecast_ghi_seed44_20261017/forecast_FA_FAC_seed44/`。独立FP64审计`seed44_validation_audit.json`通过，SHA-256=`BA44C7086524FC2A3F5EB23D5438DA205BE05FD0E34A820DB02066AAEFA51D23`；所有站点FA/FAC键与真值一致，test/buffer未打开。
- seed44验证RMSE/MAE（W/m²）：牛首FA79.048/52.104、FAC78.413/51.696；王寨FA71.808/44.967、FAC74.354/47.324；米庄FA82.953/52.365、FAC86.040/54.948。此种子仅牛首FAC略好，王寨与米庄变差；按计划现在对三seed作站点等权聚合、lead分层与共享日期块bootstrap，不能选择性报告单站改善。
- 三个预注册seed42/43/44均已完成训练和逐样本独立复算。账户队列刷新后仅剩`214221.tc6000`为非终态任务；没有提交额外训练。下一步是完成三seed验证集统计与完整预测链路检查，再决定并记录一次性打开test的冻结清单/评价程序；当前test/buffer仍封闭。

## 2026-10-17 三seed forecast-GHI validation 汇总

- 新增分析入口`solar-cot-ifs/scripts/analyze_hubei_forecast_ghi_three_seed_validation_20261017.py`，只读三个独立seed审计通过的验证预测包；强制验证seed之间、站点内FA/FAC样本键和真值完全相同，并检查lead/init/target一致性。以同一抽样日期权重对站点、两臂和三个seed进行10,000次单日块bootstrap及连续两日块敏感性；test/buffer读取0。
- 汇总JSON及CSV见`audit_receipts/forecast_ghi_three_seed_validation_20261017/`，人读报告见`FORECAST_GHI_THREE_SEED_VALIDATION_20261017.md`。
- 三站等权、三seed等权FA RMSE=77.25，FAC=78.58 W/m²，FAC−FA=+1.33 W/m²（COT条件相对高1.72%）；seed42/43/44总差分别+0.29/+2.03/+1.67，0/3个seed总体降低RMSE。单日配对日期块95%区间[−0.42,3.82]，连续两日敏感性[−0.30,3.82] W/m²，均跨0。lead均值差在16个时效均为正(+0.87至+2.00 W/m²)。这是validation选模观察，不能作为独立论文确认性结论；方向上不支持“加预测COT改善未来GHI”。
- 资源方面，seed44按单GPU PBS合同完成；账户队列当时仅余`214221.tc6000`未终态任务。节点GPU4–7的瞬时空闲迹象没有作为分配授权，未越过PBS调度或占用他人GPU。本轮三seed均完成，未为追求正结果追加seed或改变模型合同。
- 下一步继续冻结forecast链路和结果图/分层；随后依据预注册split做独立test一次性评分，期间test/buffer保持关闭。若test同样未证实收益，需据实报告负/未确定结果，不把validation差异宣传为论文已证实结论。

## 2026-10-17 18:00 BJT：三seed逐时效验证图冻结

- 新增可复现绘图脚本`solar-cot-ifs/scripts/plot_hubei_forecast_ghi_validation_per_lead_20261017.py`，只读已审计的`three_seed_per_lead_metrics.csv`；不读取任何test/buffer载荷。PDF为矢量输出，PNG为300 dpi，并已目视检查版式与数据曲线。
- 产物：`audit_receipts/forecast_ghi_three_seed_validation_20261017/validation_per_lead_FA_FAC.pdf`，SHA-256=`8D0B6F8A8605994FEBA60CDFACE794C79F7E5ED6F72E554CBA150A14A1A57E82`；PNG SHA-256=`CFBFE4CEE7C5A46804D25E0191DCB435DD00B6E9FAA6A0B2168C442CF65D77E6`；脚本 SHA-256=`CA245CA679A21465543FAE38B6FA3CD337241DB802550306208E0A6735AEC355`。
- 图注建议：**湖北三站forecast-GHI验证集上FA（预测AGRI）与FAC（预测AGRI+COT）的逐时效RMSE。**左图显示三站等权、站点内按seed等权均值，阴影为三个seed的最小—最大范围（仅描述训练seed变异，不是置信区间）；右图显示各站及三站均值的配对差FAC−FA，正值表示FAC误差更大。全16个时效的三站均值差均为正，整体跨时效平均差约+1.36 W/m²。该图只支持validation选模描述，不是独立test结论；配对样本按时效共同支持，test/buffer读取仍为0。
- 图表解释与三seed总汇总一致：不能写成COT带来预报提升。下一项仍是冻结天气/月份/太阳高度分层、预先规则案例和最终评价代码/输入链provenance，再独立打开test一次性评分；不因负向validation结果加seed或改样本。

## 2026-10-17 18:18 BJT：预报真值到训练包逐键绑定

- 为回应独立代码审计提出的真值溯源缺口，新增 `tooling_20261017/audit_hubei_forecast_label_binding_20261017.py`，并从PBS产物目录只读复制forecast-head pack的`rows.csv`、GHI/晴空数组和pack report到 `audit_receipts/forecast_ghi_label_binding_20261017/`。只审计train/validation样本；没有读取test/buffer载荷。
- 逐行将70,171条预报头样本与冻结的原始15分钟标签表及晴空派生标签表按站点、目标时刻、split核对：70,171/70,171匹配，0差异；GHI最大绝对差`2.85e-5 W/m²`、晴空最大绝对差`3.06e-5 W/m²`，为pack `float32`存储精度范围内。报告绑定原始标签CSV SHA `729771d2…ccdb2f2c`、晴空标签CSV SHA `326e16c1…d199a72`、训练包文件SHA和生成器SHA。详细收据 `audit_receipts/forecast_ghi_label_binding_20261017/forecast_ghi_label_binding_audit_v2.json` SHA-256=`047E428CD8D4DE5E0A7F8942DDD24FA46B7D6CD92B094F3F6CA2233D67F78E32`；审计脚本SHA-256=`24B82C69C5949279DEDA34F3362B0B3D8C1208E04D9F257671705D1DEEC251F2`。
- 标签表与报告明确采用 `(target−15min,target]` 的15个右闭分钟箱等权均值；来源字段是工作簿第6列“总辐射1瞬时值”，按原工程W/m²操作合同处理。供数/仪器校准元数据仍未独立确认，因此这次闭合的是“冻结标签CSV→预报训练pack”的值与哈希链，不是仪器计量溯源。
- 独立实验审计另发现旧路径`al_20261017/baseline_runs/`留存seed42–44 run metadata指向旧pack SHA `7990846a…`，而新审计所用的重训回执指向重建pack SHA `e8c5b2dc…`。前者来自更早训练记录、不得与后者混用；等待独立复核明确其历史/失效标记及论文声明一致性后，再更新正式结果索引。此冲突不通过改哈希或覆盖旧产物解决。
- 当前test/buffer仍关闭。现有forecast-GHI三seed是validation选模结果；只有在权威资产谱系、论文声明和一次性test评分程序全部冻结并复核后，才继续测试阶段。

## 2026-10-17 19:25 BJT：节点资源复核与归一化重载复现训练

- 按用户补充的新凭据只读登录FD-172；仅查询主机名、时间、GPU占用和相关进程归属，没有保存或回显凭据。该机5张RTX 3090均已有其他账号的计算进程，GPU利用率/显存占用不一；未启动任务、未占用他人设备。FD-107快照为6张A6000，其中GPU0–2高利用率，GPU3仅空闲迹象，GPU4–5有显存占用；FD-201为6张4090，GPU0–3空闲迹象、GPU4–5占用。没有把瞬时空闲当作调度授权，仍使用SWC PBS分配GPU。
- PBS全账户提交前复核仅见`214221.tc6000`一个未结束任务。由于先前训练器在内存中拟合train-only归一化却未把精确值序列化进checkpoint，已将修订版训练器单独上传为`train_hubei_ghi_heads_normreload_20261017.py`（SHA-256=`fd15bf6f939cb81d09bd05e6fc889f88e53e8a9e40366d417fa195848caf5999`），并新增隔离输出的runner `tooling_20261017/run_hubei_forecast_ghi_normreload_20261017_v1.pbs`。该runner已通过远端`bash -n`，并验证pack为5折OOF、70,171行train/validation、test/buffer读取计数0；不触碰旧结果目录。
- 新复现训练`214659.tc6000`已提交，申请1张PBS GPU、seed42、FA/FAC、30 epoch、batch1536，输出至`forecast_FA_FAC_seed42_normreload_v2`。提交后全账户未结束任务为`214221`与`214659`，2/3；提交时处于Q，尚无训练结果。训练代码会在每站每臂保存train-only normalization及哈希，重载best checkpoint重跑validation并检查预测一致性；本次若通过将补足一个seed的运行时证据，不能自动替代seed43/44，也不构成test结果。
- 16个缺档HDF此前已在FD-172登记的历史/实时年目录按精确UTC键复核，本轮没有找到新的登记来源线索；仍为2,644/2,660，不插值或近邻替代，16个缺档在获得并逐像元验收原始HDF前不纳入补齐声明。
- 下一步：待`214659`自然完成后，独立复算新增FA/FAC验证预测、检查每站checkpoint归一化哈希和reload-parity；随后按提交前队列及最多3个任务规则，决定seed43/44复现。并行整理OOF日期隔离凭证和test冻结文档；test/buffer仍关闭。

## 2026-10-17 19:46 BJT：fold日期排除实证与seed42复现通过

- 新增元数据审计脚本`tooling_20261017/audit_hubei_simvp_oof_date_exclusion_20261017.py`，在FD-201逐行读取五折SimVP训练/OOF CSV索引，不打开任何AGRI数组。审计按文件名UTC转BJT（+8h），与冻结全局日期表及折分配对照，结果`HUBEI_SIMVP_OOF_DATE_EXCLUSION_AUDIT_PASS_TEST_CLOSED`：5/5折S训练日期与各自held-out折重叠均为0；训练日期均属于全局train；固定验证日期未越界；OOF日期均属于对应held-out折。每折50个分配日期中46–48个有可用OOF序列，差异是有效数据覆盖而非补造样本。收据`audit_receipts/simvp_oof_date_exclusion_20261017/simvp_oof_date_exclusion_audit.json` SHA-256=`4e8a30f04175b0e779ca851fd9a0e677eb2bffa6363d5a1a9c817c7437370a0`；脚本SHA-256=`90020cfd0cc04106ae3b3dd8fd59970cb7f064fa358d6c1c2f1766ba7f1c6a88`。这补上旧审计中“没有直接读折级S训练日期索引”的证据缺口，但不是test放行审计。
- 修订归一化保存/重载代码的forecast-GHI seed42作业`214659.tc6000`已自然完成。run metadata记录修订训练器SHA、train-only归一化合同SHA=`b29586b76400c119f50ade3fd252b4700cdf3d013d047ae64f3cace2890240bd`、1张A100、训练耗时104.4s、test读取0。六个station×arm均保存了norm绑定checkpoint；训练端重载后复算的最大差异均为0.0 W/m²。独立FP64验证复算通过，收据`audit_receipts/forecast_ghi_normreload_seed42_20261017/seed42_validation_audit.json`报告state PASS，三站FA/FAC RMSE分别为牛首74.318/75.080、王寨70.823/72.654、米庄81.846/83.324 W/m²。仅为修订合同下单seed验证结果；历史seed43/44不得与其混算，待同合同重跑并重新汇总。文件可见于该审计目录中的`run/`，test/buffer仍关闭。
- 提交前PBS队列为`214221`一个未结束任务，之后seed43复现作业`214665.tc6000`提交成功，申请1张GPU、同一固定pack、FA/FAC、30 epoch、batch1536，使用隔离输出`forecast_FA_FAC_seed43_normreload_v2`；提交后队列2/3，状态Q。未启动第二张卡任务，也未直接运行未调度节点GPU。
- 下一步：seed43自然完成后验收保存重载、逐样本复算及共同键，随后同样复现seed44；完成后只用三个修订合同seed重新汇总/画图。再补齐同刻A/AC/AP checkpoint/归一化的冻结方案，统一双语论文和PAPER_PLAN/claim audit，最后重新做test-freeze独立审计。16个AGRI缺帧仍未找回，不插值；test/buffer payload仍未读取。

## 2026-10-17 19:52 BJT：湖北同期三种子归一化重载复现推进

- 远端复查确认同期seed42、seed43的隔离输出目录均已生成run metadata、train-only normalization contract及三站A/AC/AP逐臂验证文件；seed44目录尚不存在。它们使用重建pack `e8c5b2dc587e8a938bba43fc564be43aec777788aca223c054341f75dd09a515`，与早期旧pack结果隔离。
- 当前PBS全账户未结束作业为既有`214221.tc6000`和本轮seed44作业；提交前队列仅有`214221`一个未结束作业，低于3个上限。seed44作业`214670.tc6000`已通过PBS提交，使用1张调度GPU、batch512、A/AC/AP三臂、30 epoch，当前`R`运行于node23；没有直接占用未分配GPU。
- 本地独立复核脚本`tooling_20261017/audit_same_time_three_seeds.py`已确认包含10,000次配对日期块bootstrap、跨seed pack/split/trainer/normalization/run provenance一致性断言、checkpoint重载和归一化绑定校验；`py_compile`及目标文件`git diff --check`通过。
- 下一步等待seed44自然结束，收集42/43/44新合同预测与metadata至隔离审计目录，运行独立FP64重算和三seed配对分析；此前任何同刻B1点估计均不与此批重建pack结果混报。test/buffer仍关闭，test有效载荷读取数仍应保持0。

## 2026-10-17 19:58 BJT：同期A/AC/AP新合同三seed审计通过

- seed44作业`214670.tc6000`已自然完成，PBS `exit_status=0`，分配node23 GPU3（A100-SXM4-40GB），walltime约4分29秒；run metadata记录`test_payloads_read=0`。与seed42/43（`214668/214669`）一起，均使用重建包`e8c5…`、冻结split SHA、同一训练器及同一train-only normalization contract。
- 保存预测、每站/每臂JSON和run metadata已复制到`audit_receipts/same_time_normreload_runs_20261017/`。独立审计脚本先发现并修复了一处审计断言错误：不同seed的run metadata哈希本来就应不同，跨seed一致性仅要求pack、split、trainer、normalization合同一致。修复后脚本通过`py_compile`及10,000次配对bootstrap审计，生成`audit_receipts/same_time_three_seed_normreload_20261017/independent_three_seed_validation_audit.json`；三seed验证目标键、真值、晴空值一致，逐样本预测指标复算一致，checkpoint reload parity和归一化绑定通过。
- validation主结果（三站等权、seed等权）：RMSE A=74.33、AC=70.99、AP=73.84 W/m²；AC−A = −3.34 W/m²，22日配对日期簇95%区间[−5.80,−1.12]；AP−A = −0.49 [−0.80,−0.26]；AC−AP = −2.85 [−5.15,−0.72]。连续两日块敏感性区间分别[−6.14,−0.73]、[−0.85,−0.25]、[−5.50,−0.35]。详细站点/seed结果和限制见`SAME_TIME_THREE_SEED_VALIDATION_20261017.md`。
- 解释边界：这是validation选模期、22个目标日期上的三种子结果，不是独立test。AP控制参数容量但不能控制R监督或空间感受野；不应把差值解释为COT物理量唯一因果效应。预测COT的FA/FAC validation仍未显示预报增益。test-freeze当前仍NO-GO；所有test/buffer载荷继续关闭。
- 下一步：按冻结规则补完同期与预报两部分的预先确定天气/月份/太阳高度分层和案例摘要，锁定报告/代码/权重范围，再重新执行独立test-freeze审计；审计通过前不读取test/buffer payload、不提交test评分任务。

## 2026-10-17 20:06 BJT：forecast-GHI归一化重载汇总与图表冻结

- 用当前normreload三seed逐时效CSV重绘FA/FAC图，保持旧图不覆盖；绘图脚本增加输入CSV和输出目录参数，新增图位于`audit_receipts/forecast_ghi_three_seed_normreload_20261017/figures/`。目视确认 +15…+240 min曲线完整，阴影仅表示seed范围。PDF SHA-256=`88A0683590BE76B4FB90ABF4DD9ED2B59487F45459ADE4586EE784482BE6A317`，PNG SHA-256=`1BE49CA578616A6FEB678470F829DF2893835825B8A99A55A94431FD9AA46BB8`。
- 新三seed汇总FA/FAC=77.150/78.155 W/m²，FAC−FA=+1.005 W/m²；单日日期块95%区间[−0.722,+3.780]，两日块[−0.681,+3.813]，均跨0。16个lead的三站等权RMSE差都为正（+0.45至+1.83 W/m²）。报告见`FORECAST_GHI_NORMRELOAD_THREE_SEED_VALIDATION_20261017.md`。旧77.25/78.58指标已在旧报告顶部标注为被取代，不与新run混合。
- 本轮绘图脚本修改后`py_compile`通过；报告明确validation范围和图中seed范围含义。test/buffer未读，冻结门禁仍待剩余事项与独立复审。

## 2026-10-17 20:12 BJT：同期新合同月份/太阳高度分层

- 依据冻结的目标月份与太阳高度区间，对新normreload A/AC/AP三seed保存预测作了键连接复算；仅使用训练/验证pack中的GHI与几何数组，全部27组seed×站点×方法预测键与pack逐项一致，真值逐行匹配，test/buffer载荷读取0。
- AC−A RMSE分层点差显示：太阳高度0–15°为−3.91、15–30°为−5.77、30–45°为−6.43 W/m²；45–60°转为+2.49；60–90°接近0（−0.28）。按月份正负混合，且每月支持有限；不作子组显著性推断。晴/多云/厚云标签仍无独立一致来源，不从GHI或预测COT推断。
- 新报告`HUBEI_VALIDATION_SUBGROUPS_NORMRELOAD_20261017.md`和机器收据`audit_receipts/same_time_three_seed_normreload_20261017/month_solar_subgroups_validation.json`已生成；旧分层报告顶部注明旧run历史状态。所有分层仍是validation选模后描述，不是独立确认性结论。
- 下一步按误差无关的冻结案例清单重建新合同三站图，随后汇总剩余test-freeze门禁；不得根据分层结果挑日期或重开test。

## 2026-10-17 20:32 BJT：冻结案例清单在新合同下重绘完成

- 使用`validation_case_selection_20261016_v1/validation_case_manifest.csv`中的24个预先冻结案例；清单基于观测GHI ramp和固定hash日期，不查看模型误差。新绘图程序按(station,target_time)把seed42/43/44三臂NPZ与train/validation-only pack逐键连接，逐目标真值回配，三站所有案例支持数与冻结manifest完全一致；test/buffer读取0。
- 三站新图及逐点表见`validation_cases_normreload_20261017/`，审计收据`FIGURE_AUDIT.json`登记pack、case manifest、全部预测bundle和图表哈希。目视检查牛首、王寨、米庄图版式正常，缺失目标仍断开；阴影为三训练seed的样本标准差，不是置信区间。
- 图只包含15分钟站点GHI曲线，没有AGRI/COT图像，也没有独立天气分类；观测GHI ramp选例不等于“阴天/晴天”标签。该交付是描述性validation图，不作预报技能或独立test结论。
- 分层、逐时效预报图和新案例图均已保存；下一步进行fresh integrity review，并按仍未闭合的项目更新PAPER_PLAN/论文与test-freeze清单。未获审计放行前test/buffer仍关闭。

## 2026-10-17 20:50 BJT：冷启动归一化审计修正重提

- 首次冷启动审计作业`214679.tc6000`于20:41:17以`Exit_status=1`结束。日志显示失败在PBS前置检查：同期包报告没有`input_mode`字段，原runner错误地将预报包字段要求用于同期包；模型推理尚未开始，不代表checkpoint或数据校验失败。
- 已修正runner，依据同期包真实合同校验其`state`、共同键标志与test/buffer读取计数，并单独校验预报包`input_mode=forecast`。本地脚本未改动模型/数据；远端`bash -n`通过。修订后提交`214689.tc6000`，使用1张调度GPU、最多30分钟；提交前全账户队列仅有`214221`，提交后为2/3。20:50查验状态为R，尚无审计结论。
- 该任务仅从已有train/validation pack重建归一化输入并复算预测，不训练、不访问test/buffer。训练资产与六月至七月14日已验收AGRI清单均未变；2,644/2,660有效源帧，16个原始时次缺档仍未补齐，不插值、不替换。
- 用户最新提供的其他节点口令未复制进日志、脚本、命令参数或记忆；现阶段继续使用已配置的SSH密钥及PBS分配访问，明确避开248节点。
- 后续待作业自然结束后只读取一次最终日志与审计收据；无新增可信AGRI源文件时，不重复提交训练或声称16个缺档已补齐。现有训练已在冻结可用数据上完成，任何新增真实帧需先新建manifest/数据包和独立重训，不热改旧run。

## 2026-10-17 20:57 BJT：冷启动审计浮点口径修正

- 第一次修正后的冷启动任务`214689.tc6000`通过train/validation-only前置检查，随后在同期seed42首站发现saved bundle的晴空数组与pack数组逐位不等。只读对照发现GHI真值逐值完全相同；晴空最大绝对差为`3.06e-5 W/m²`，来自pack为float32、bundle保留float64的存储精度差异。对27组同期站点×种子×A/AC/AP bundle定点复核后，所有真值差为0，所有晴空最大差约`3.04–3.06e-5 W/m²`；station/target键顺序一致。
- 同时发现预报冷启动清单误含FAP，但现有forecast-GHI训练对照只有FA/FAC。修正审计代码：预报组只核FA/FAC；真值和晴空均采用显式`5e-5 W/m²`绝对容差（零相对容差）并记录最大差，checkpoint预测仍按原严格复算门槛比较；不变更训练数据、预测或指标。
- 修订脚本SHA-256=`8adf6c0763c27b24a9227232896e3b6deae81ad4ce35ea1667bfe239571ea182`，本地`py_compile`通过，远端两份副本SHA一致。作业`214691.tc6000`已提交为1-GPU冷启动只读核验；提交前账户未结束队列为`214221`，提交后为`214221`与`214691`（2/3）。截至20:57作业完成态，尚未取回其最终日志/收据，故审计仍未通过或定论。
- 数据状态未改变：六月至七月14日train/validation补充包仍为2,644/2,660个可验收合法产品帧；16个精确时次在已登记HDF源及已查预处理树中未发现可信原件。正式训练资产已在这一冻结支持上完成；缺档若有新原始源才另立manifest和重训，不做插值、不覆盖旧包。test/buffer仍关闭。
- 按规则由Luna作一次非重复状态核对：`214691.tc6000`仍为R（node23/GPU3），日志停在train/validation共同键与test关闭前置检查PASS；尚无两个冷启动JSON结果文件或最终退出状态，不能提前判作审计完成。

## 2026-10-17 21:08 BJT：冷启动审计批次复核与runner路径修正

- 取回`214691.tc6000`最终产物后确认：同期A/AC/AP 27组checkpoint-bundle冷启动重算通过，最大预测差`1.831e-4 W/m²`、test读取0；forecast阶段在`42/Niushou/FA`失败，最大差`0.1660 W/m²`。原审计器导入通用trainer而非checkpoint所记录的精确normreload trainer；因此另建v2隔离复核任务，不改模型/结果。
- v2作业`214698.tc6000`使用run metadata所指的精确trainer SHA `fd15bf6f939cb81d09bd05e6fc889f88e53e8a9e40366d417fa195848caf5999`。其同期阶段仍通过，但forecast首站同样出现`0.1660 W/m²`差异；这排除了“仅因导入了错误模型类”这一解释。检查发现审计器以整站一次推理，而训练回放按run metadata中的batch size 1536分批；现将冷启动审计改为复用该批次，作为待验证的数值复现修正，不预先视为根因已证实。修订审计器SHA-256=`ad1f8f855cd6caf6371dd53cb8f6e970e9add1c208eeff206ce980242659b38f`。
- v3首次提交作业`214703.tc6000`在PBS前置检查退出，原因是runner把forecast项目根误当成pack目录，拼出了不存在的`hubei_forecast_ghi_20261017/pack_report.json`；正确清单实际位于`.../pack/pack_report.json`。该失败没有读取pack载荷，也没有产生模型结论。已将前置检查参数改为显式`PACK_FCST`，确认远端`bash -n`通过、两个pack目录与其`pack_report.json`存在且此前v3目录没有审计JSON后，重提`214705.tc6000`。提交前账户未结束PBS任务为`214221`，提交后为`214221`与`214705`（2/3）；当前状态Q，等待自然运行。
- 本轮没有重训、没有改预测或标签；所有复核仍只用train/validation pack，test/buffer读取为0。用户提供的节点凭据未写入脚本、命令日志或工程文档，FD-248保持未访问。
- 下一步由Luna只读核查`214705.tc6000`最终状态和v3回执。若仍有forecast预测差，继续逐项对比trainer与复核器的输入张量/分批顺序及环境，而不放宽误差阈值；只有A/AC/AP及FA/FAC均通过才关闭冷启动核验。6月1日至7月14日AGRI清单仍为2,644/2,660，16个缺档未补齐；本次没有找到或引入新源数据。

## 2026-10-17 21:51 BJT：checkpoint 冷启动/归一化重载审计通过

- 修正pack参数路径后，PBS作业`214705.tc6000`在node23分配GPU3自然完成，`exit_status=0`，walltime 89秒。同期与预报审计均使用run metadata指向的精确trainer SHA `fd15bf6f939cb81d09bd05e6fc889f88e53e8a9e40366d417fa195848caf5999`，从checkpoint内train-only归一化重建输入，并按每个run的batch size 1536逐批重算。
- 同期A/AC/AP共27组及forecast FA/FAC共18组均通过；checkpoint重算预测与保存bundle逐项最大差均为`0.0 W/m²`。两报告均记录test/buffer载荷读取0，pack分别绑定`e8c5b2dc…`（同期）和`9421c0d5…`（预报），故不是不同数据包混算。此前forecast `0.1660 W/m²`漂移由一次性整站推理与训练时分批推理口径不一致引起；按相同批次复算后消失，严格门槛未放宽。
- 本地回执：`audit_receipts/coldstart_normreload_20261017_v3/same_time_coldstart_audit.json` SHA-256=`EB72F580DABC3C523F7651CB1F70B936A296DC516046EBC35CE3AC42D1F65CDE`；`forecast_coldstart_audit.json` SHA-256=`54FBFCE21CE01FD5E775EB23888869468C8DE591C231FEA8CBDFAF0E67E00286`。远端日志另确认v3完成标记。该结果只验证train/validation checkpoint冷启动和归一化重载可复现，不是独立test精度证据。
- 冷启动复现门禁已闭合，可继续独立准备论文/测试冻结审计；但不得据此直接读取test。缺帧状态不变：6月1日至7月14日有效AGRI为2,644/2,660，16个缺失原始时次仍无登记来源命中；没有在本轮补帧或重训。下一步优先处理test-freeze审计遗留的预测/论文来源口径差异，随后经独立放行再执行一次性test评价；test/buffer继续关闭。

## 2026-10-17 22:35 BJT：论文结果修订与原始标签绑定审计

- 依据fresh experiment audit，将英文结论中残留的旧forecast-GHI数值替换为normreload三seed结果：FA/FAC=77.150/78.155 W/m²，FAC−FA=+1.005 W/m²，单日块95%区间[−0.722,+3.780]；同时在中英文结果段同步记录单日/连续两日块均为10,000次重采样（较通用2,000次默认值的协议精度调整），保持测试集关闭与验证范围限定。
- 修订`PAPER_PLAN.md`及`PAPER_CLAIM_AUDIT_20261017.json`：checkpoint冷启动与归一化重载18组forecast配对审计已通过，最大预测差0，test/buffer读取0；并将标签绑定列为已通过而非待办。
- 扩展标签绑定审计脚本v3，增加原始15分钟标签表→daylight/clear派生表→forecast pack三段直接数值断言。执行结果：70,171条训练/验证pack行均匹配；30,013条有效daylight/clear派生标签与原始15分钟GHI最大差0.0 W/m²；pack与原始标签最大差2.85e-5 W/m²；测试与buffer读取均为0。收据为`audit_receipts/forecast_ghi_label_binding_20261017/forecast_ghi_label_binding_audit_v3.json`。
- `main.tex`已由Codex桌面LaTeX编译器编译成功；编译结果未导出覆盖工作区`main.pdf`，故PDF页面视觉检查仍待完成。`main.tex`与`main_zh.md`的指标、区间和bootstrap次数已同步；旧78.58/77.25结论数字在当前正文/计划中已无残留。claim audit JSON结构校验通过。
- 标签绑定脚本SHA-256=`ee3445dde8f106b2091c8393abb9aee17d550956e4cabca9504c395aea49aef`；v3标签绑定回执SHA-256=`c6648e56fd86507555fe787d7bd6504a4d025f9644ed839108e7ede984563c89`。
- test-freeze fresh review仍需重做，独立test没有解封。AGRI缺帧未补，训练未启动；期间用户提供的节点登录信息未写入工程文件或命令脚本，248节点未访问。

## 2026-10-17 23:00 BJT：AL容量对照profiling与三种子训练启动

- 先前AL特征提取作业`214716.tc6000`并非提取失败：冻结R dec1特征已生成。退出原因为runner把GHI pack需要的12,301个唯一`(time_bjt, split)`键误要求成完整AGRI manifest的24,705帧。
- 新增只读键审计`audit_hubei_al_feature_bank_keys_20261017.py`：逐项核验29,351条GHI站点行、12,301个唯一时间/划分键与12,301条R特征行；键集合完全相同，特征形状`[12301,3,16,3,3]`且全部有限。回执`feature_bank_key_audit_20261017.json`记录pack报告SHA、split SHA和`test_payloads_read=0`；R特征生成过程未读GHI标签或CPP。
- 使用已生成特征库完成PBS单卡profile：作业`214718.tc6000`在node23 A100 GPU2上成功，AL头约24.5k参数，AP约24.6k；batch 128/256/512峰值吞吐约19.3k/43.6k/106.2k样本每秒，峰值已分配显存约0.156 GiB。512满足每epoch至少12次更新且达到测得吞吐峰值，因此选作训练batch。低显存占用是此容量对照头较小的实测结果，不据此人为堆载或声称GPU利用已饱和。
- 修复训练runner中的profile batch选择语句语法错误，远端`bash -n`通过。之后发现AL审计脚本使用`ROOT.parent/frozen_split_20261015_v1`，即要求链接位于实验根目录而非AL子目录；本地runner已改为正确目标，但修正同步发生在训练作业启动后。
- 三种子AL容量对照训练已提交：PBS`214719.tc6000`，node23，1×A100，30 epochs/站/seed；训练仅用当前train/validation pack，不读test/buffer。最后一次日志检查时seed42、seed43已完成，seed44正在训练，未见错误；独立汇总审计尚未开始，不能报告AL结果或科学结论。
- 用户提供的节点口令未存入日志、脚本或工程文件；当前集群通过既有SSH密钥可访问，没有必要使用或持久化该口令。248节点未访问。
- 数据支持未因此改变：6月1日至7月14日仍为2,644/2,660个已验收AGRI帧；16个精确时次缺档保持原样，未插值。这里的AL是解释AC增量的同期容量/特征控制，不是预报实验。test/buffer仍关闭。

## 2026-10-17 23:05 BJT：AL训练完成、审计路径修复待复核

- PBS`214719.tc6000`完成三站×三种子共9个AL训练，日志逐一标记`AL_SEED_COMPLETE_TEST_CLOSED`；退出码1发生于训练后的独立审计阶段，仅因冻结`day_assignment.csv`链接目标放错层级。此条待办已于23:31修复并审计通过，详见下文。

## 2026-10-17 23:31 BJT：AL三种子结果审计通过

- CPU审计作业`214720.tc6000`以`exit_status=0`结束；seed42/43/44、三站共9组训练均通过metadata预检，独立FP64复算和共同时间敏感性审计完成。test/buffer读取0。
- 完整22日validation、站点等权/种子等权RMSE（W/m²）：A 74.333、AC 70.993、AP 73.839、AL 71.090；AC−A −3.340，AC−AP −2.846，AL−A −3.244，AL−AP −2.749，AC−AL −0.097。三站共同目标时刻敏感性支持为486个时刻、21日：A 76.075、AC 72.867、AP 75.551、AL 72.785；AC−AL +0.082 W/m²，连续两日块95%区间[−1.490,+1.713]跨0。AL与AC表现无法据此区分，不能声称AC包含独立于冻结R隐特征的增益；AL只是容量/特征控制，仍是same-time validation，不是预报或test。
- 机器结果及说明已从服务器复制到`audit_receipts/al_normreload_20261017/`；主要审计JSON SHA-256=`16d3edf9b8f89d5fa612843d5eb3af4474bb65c4f71d43d9ff3f0929c347f40c`，共同支持JSON SHA-256=`be784ec129adf5c59dba0baf19105d901cc0cd584d80ad410cbca777da23b89a`。profile和键审计也已归档。
- 目前仍不能宣称湖北论文独立测试结果。2026-10-17旧test-freeze审计早于本轮normreload和AL结果，且引用已被替换的旧同刻/forecast值；不能作为当前放行证明。下一步需刷新唯一结果provenance和双语主张、冻结test evaluator，然后重做独立test-freeze审计；通过前test/buffer仍关闭。

## 2026-10-17 23:51 BJT：AL normreload 结果跨运行独立复核

- 从项目PBS服务器只读定位并复制了`al_normreload_20261017/runs_maskfix_v2/`的3种子原始验证产物、run metadata、模型权重和norm文件；同时复制当次pack报告/rows及AL特征报告/键审计。未读取test/buffer载荷，也未把密码写入本地文件。
- 新增`tooling_20261017/audit_al_normreload_crossrun_20261017.py`，独立校验36个带键预测包（A/AC/AP/AL × 3种子 × 3站），与当前冷启动审计回执、同一pack SHA=`e8c5b2dc…09a515`、split SHA=`6ef8cbc7…bad19c6`、特征报告SHA=`317e6a47…1cc40`绑定；AL键、真值、晴空值和预测NPY/NPZ一致，站点样本数为牛首856、王寨784、米庄816；测算RMSE与各保存验证指标在0.03 W/m²内一致。
- 独立重算三站×三种子等权验证RMSE：A 74.333、AC 70.993、AP 73.839、AL 71.090 W/m²；AC−AL为−0.097 W/m²，几乎无差别。这只说明当前validation上未显示显式COT相对冻结R解码特征控制的可区分提升，不是test或未来预报结论。机器收据`audit_receipts/al_normreload_crossrun_audit_20261017.json`，状态`HUBEI_AL_NORMRELOAD_CROSSRUN_KEYED_RECHECK_PASS_TEST_CLOSED`。
- 复核仍不等于完整AL冷启动重载审计：现AL checkpoint仅保存权重，train-only normalization在同目录JSON；原验证重载用的是已标准化内存张量。故需将checkpoint与norm合同绑定，并从原始train/val特征重建输入做独立复跑，才可关闭AL归一化来源门禁。此前的旧本地AL目录仍是pack SHA=`7990846…`，禁止与新normreload比较。
- 加固旧`al_20261017/audit_al_results.py`：现在先强制核对本地pack与current normreload回执的pack SHA，再校验基线split及AL station/arm/checkpoint/bundle哈希；在本地旧pack环境下会明确拒绝跨pack评分。`py_compile`通过，预期拒绝测试通过；没有覆盖旧JSON/Markdown指标报告。
- 实时PBS队列仅有`214221.tc6000`一项排队任务，未提交新作业。对FD-107、FD-172、FD-201的只读SSH连接均被服务器在认证前关闭；没有登录成功、启动进程或触碰FD-248。用户补充的认证信息未持久化。当前能继续使用的项目PBS仍是后续训练入口；按项目门禁不在未分配节点直接启动GPU任务。
- 下一步：先做AL checkpoint+normalization冷启动推理见证（保持相同训练合同、独立输出目录），再刷新总claim/provenance审计和中英文正文；同步核对当前forecast折次/OOF实际作业状态。test/buffer继续关闭。

## 2026-10-17 23:56 BJT：AL原始输入冷启动重载见证通过

- 运行新增审计器`tooling_20261017/audit_al_coldstart_normreload_20261017.py`，仅载入重建pack和冻结R特征库中的train/validation内容，逐站从原始train行重算AGRI与R dec1均值/标准差，再用已有AL checkpoint独立重跑验证集。9个checkpoint均通过身份、pack/feature/split SHA、station/target键和truth检查；train-only归一化统计最大差0，test/buffer值读取0。
- 跨CPU/GPU的逐点最大预测差为0.299 W/m²；9组RMSE相对已保存GPU结果最大差0.0112 W/m²（预设容差0.03），因此判定冷启动指标复现通过，并明确这是跨设备数值差异而非逐点完全一致。归一化文件仍独立于checkpoint保存，但本审计回执同时绑定checkpoint、norm文件、输入包及特征哈希，可作为联合推理合同复现证据。
- 新回执：`audit_receipts/al_normreload_coldstart_20261017/AL_COLDSTART_NORMRELOAD_AUDIT_20261017.json`，SHA-256=`a17e73415ec78a9781707c1ed5c8e5abe31b6efbdd22524839eea337d34a6b3b`，状态`HUBEI_AL_COLDSTART_NORMRELOAD_AUDIT_PASS_TEST_CLOSED`。没有重训，也没有读取test/buffer载荷。
- 下一步转为刷新总provenance/claim审计与英文LaTeX、中文Markdown同步，再重新审阅test evaluator与freeze gate；此前FAIL/NO-GO审计已过时，不能用于放行，但test/buffer在新审计通过前继续关闭。

## 2026-10-18 00:07 BJT：AL结果纳入双语稿；非248节点仍不可达

- 用户提供了非248节点通用密码后，只尝试已配置的FD-107、FD-172、FD-201，均在TCP/22认证前由服务器关闭连接，未显示密码提示；没有认证成功或启动远程进程。FD-248未访问，密码未写入文件/命令行/日志。学院PBS只读状态仍为单项无关CPU队列任务`214221.tc6000`，无湖北GPU任务；test/buffer payload仍关闭。
- 根据AL跨运行与冷启动复核回执，英文`main.tex`和中文`main_zh.md`已同步增加湖北AL特征控制结果及对COT特异性主张的限定：共同支持486条站点—目标时刻、21日期，AC/AL RMSE 72.867/72.785 W m⁻²，AC−AL +0.082，连续两日块95%区间[−1.490,+1.713]跨零。湖北COT特异性增益仍未分辨。摘要、研究问题、结果、讨论、结论及`PAPER_PLAN.md`均更新；未改指标口径、未访问test数据。
- `latexmk -pdf -outdir=build -interaction=nonstopmode -halt-on-error main.tex`首轮成功生成20页PDF，但渲染抽查发现英文摘要中的AL限定句重复两次；已移除重复句，需重新编译并复查PDF首页与结果页。现有overfull hbox为此前论文内容的小幅排版警告，不是此次AL段编译错误。
- 下一步：重编译并检查英文摘要/结果页；核对双语AL数值/表述和引文；刷新完整claim/provenance以及test-freeze复审。未通过新的独立门禁前不打开test/buffer、不提交新训练。

## 2026-10-18 00:14 BJT：双语方法口径补齐；门禁仍NO-GO

- 按独立lineage review校正中英文训练设置：湖南同期batch=2,000；湖北同期A/AC/AP与AL batch=512；湖北FA/FAC预报输入batch=1,536。bootstrap次数按用途写明：湖北同期A/AC/AP主区间和两日敏感性均10,000次；AC-AL共同支持两日块2,000次；预报GHI单日/两日块均10,000次；通用其他分析2,000次。
- 正文AL支持口径明确为三站共同的486个目标时刻、每站486条、合计1,458条站点—目标记录，跨21日期；描述为冻结R的16通道解码特征分支。英文和中文均写AC/AL=72.867/72.785 W m⁻²、AC−AL=+0.082、95%区间[−1.490,+1.713]。
- 修订`PAPER_PLAN.md`：Table 3仍对应湖南容量辅助特征控制，湖北AL结果在§5.2独立报告；Figure 7仅为预测COT RMSE/CSI，不画GHI增量。更新机器可读claim audit。
- `latexmk`成功，生成20页、1,018,242字节PDF；页面1摘要与页面11湖北结果已渲染检查，无重复AL摘要句、无未定义引用/引用键。两条轻微overfull hbox警告位于既有资源诊断/预报段，不影响本次新增内容。
- 额外尝试用现有`audit_same_time_three_seeds.py`重建汇总回执时，它因旧输入合同要求`checkpoint_reload_parity == PASS`而在normreload metadata处失败，未生成新统计回执，未触碰test/buffer。需先基于当前cold-start回执重建并冻结分析器/输入绑定，不能把旧脚本失败视为指标不变的证明。
- 两份独立review仍是`FAIL / NO-GO`：结果分析源码哈希未与收据闭合、test evaluator未冻结为fail-closed、完整validation S/R bank日期排除证明不足、pack/checkpoint元数据强校验未闭合。学院PBS当前唯一非终止项仍为无关CPU任务`214221.tc6000`，没有湖北GPU任务。非248三节点密码登录仍在TCP/22认证前被关闭；248未访问。下一步优先修复provenance/evaluator门禁；此前不得解封test/buffer或提交test作业。

## 2026-10-18：同刻三种子验证汇总与冷启动回执重新绑定

- 旧汇总器要求每个逐臂metadata含`checkpoint_reload_parity=PASS`，与新版冷启动/归一化回执字段合同不兼容，因此首次重跑在断言处停止、未产出统计。现更新`tooling_20261017/audit_same_time_three_seeds.py`：可显式接收冷启动回执，并逐一检查其状态、pack、split/run metadata SHA、种子/站点/臂、checkpoint SHA、预测bundle SHA、批次大小、预测复算差值及test/buffer关闭计数；未放宽指标容差。`py_compile`通过，使用新版normreload目录重算完整三站×三臂×三种子汇总成功。
- 新验证回执：`audit_receipts/same_time_three_seed_regen_20261018/independent_three_seed_validation_audit.json`，SHA-256=`df0374c16ace142b5acb11b5d0f9b92b9138483ac1d6638bd06e3b1a6383ea6f`；分析器SHA-256=`719eeb731ac3a628c594423a9708031ec8b604019b05b105f4b24448fb5537fd`。主配对RMSE差（AC−A）为−3.340 W/m²，22日日期簇95%区间[−5.800,−1.118]；AC−AP为−2.846 W/m²，区间[−5.147,−0.717]。连续两日块敏感性区间也未跨零。AP−A为−0.494 W/m²，提示容量/辅助信息臂自身亦有小幅改善，因而结果只支持“当前验证口径下加入COT的总输入方案优于AGRI基线和该辅助特征容量臂”，不单独识别COT语义机制；仍非独立test结论。
- 该回执明确标记`test_payloads_read=0`，对应冷启动回执同时为`test_payloads_read=0`、`buffer_payloads_read=0`；输入限制在22个validation日期。该新机器回执尚待新鲜独立审查通过后再纳入正式论文主张或刷新claim audit。
- 数据核对没有发现可接受的新帧来源：2026-06-01至2026-07-14仍是2,644/2,660个合法产品时次有验收文件；16个精确时次已列入已登记历史/实时源联合缺失清单。先前8个`.npy`候选依然缺逐帧HDF来源/处理血缘，未计入补齐数；没有插值、近时次替代或热改训练manifest。单位/时间口径继续采用旧工程操作合同（“总辐射1瞬时值”直接用作W/m²量；当前新标签是右闭完整15分钟窗），独立仪器校准/字段字典仍未取得。
- 实时PBS复核（tc6000，约2026-10-17 23:59 +0800）为单个无关CPU排队作业`214221.tc6000`，没有湖北GPU任务；非248三节点本轮依然在TCP/22认证前关闭，尚未到密码认证环节；FD-248未访问。没有新增训练作业。B3预测GHI训练仍等待经审计的五折S/R OOF来源、同键输入pack与冻结训练门禁；不能为消耗GPU绕过这些条件。
- 下一步：等待本次同刻三种子验证汇总的独立只读审查；随后修复/重建forecast OOF与GHI输入pack来源哈希链和fail-closed evaluator，再运行新的test-freeze独立审计。测试/缓冲数据仍保持关闭；若可验收缺档来源或PBS可达性出现实质变化，再另行更新manifest/排程。

## 2026-10-18：独立审查、文档绑定与节点认证探测

- 新汇总的fresh read-only审查结论为**验证完整性WARN、test unseal FAIL/NO-GO**：独立重算27组bundle结果、三站键和truth配对均通过，RMSE/区间一致；旧说明与claim JSON最初仍指向旧回执/旧分析器哈希。本轮已将当前receipt、analysis SHA、coldstart SHA及执行来源补充回执写入`SAME_TIME_THREE_SEED_VALIDATION_20261017.md`和`paper/PAPER_CLAIM_AUDIT_20261017.json`，并保存审查报告/机器JSON/审查文本于`audit_receipts/same_time_three_seed_regen_20261018/`。
- 另对远端PBS冷启动脚本作只读核验：作业`214705.tc6000`日志标记v3成功关闭test，日志SHA=`e31c04a50985b17d00ad5ae62d9df6ac2c764a994541bf4a824e09be3a510608`；same-time和forecast远端审核器当前SHA一致=`ad1f8f855cd6caf6371dd53cb8f6e970e9add1c208eeff206ce980242659b38f`，文件mtime分别早于该作业完成。补充`audit_receipts/coldstart_normreload_20261017_v3/COLDSTART_EXECUTION_PROVENANCE_20261018.json`（SHA=`f004411d081464d3896f795c1e12fd1f6a1a4c7e6287f07f96d072d450fce3d6`）；它明确保留一个限制：原始回执未记录执行时脚本SHA，事后哈希与mtime不能作为执行时字节级证明。
- 对用户提供的非248节点密码认证信息，FD-201本轮显式禁用公钥并指定password认证进行SSH协议探测；TCP连接建立后在`kex_exchange_identification`阶段即由远端关闭，未进入密钥交换/认证，也没有发送密码。FD-107/172此前同样在认证前关闭；FD-248未访问。训练仍只通过学院PBS调度，直接节点SSH不可达不是绕过PBS或动用未分配GPU的理由。
- 当前全账户PBS只有`214221.tc6000`一个无关CPU排队任务，没有湖北GPU训练/复评分作业；未提交、取消或改动他人作业。已有湖北A/AC/AP/AL与FA/FAC验证训练结果不需重复开跑。新增的下一步是等待claim-artifact复核，随后将剩余S/R五折OOF来源链、同键forecast pack和fail-closed evaluator逐项闭合；通过后再准备一次性独立test审计。6月1日至7月14日仍为2,644/2,660验收帧，16个缺档没有新原件命中；test/buffer载荷继续关闭。

## 2026-10-18：新鲜审查复核当前同期验证绑定

- 第二位新鲜只读审查员复核修订后的当前文件绑定：同期验证回执、分析器、冷启动回执、来源胶囊、27组bundle/metadata/checkpoint、3份run_metadata及9组配对键均一致；旧回执已明确标作历史。同期验证证据链当前判为PASS，但总体仍WARN：审核器源码SHA为运行后采集，不能证明执行时精确字节，且完整paper-claim审计尚未重跑。
- 已把复核结果同步到`PAPER_CLAIM_AUDIT_20261017.json`和本次`EXPERIMENT_AUDIT.md/json`；机器记录区分“当前验证绑定通过”和“完整论文冻结待办”。测试解封仍为FAIL/NO-GO，test/buffer读取数维持0。
- 当前可继续推进的实质工作是审计剩余forecast COT的五折S/R OOF排除证明、同键输入pack哈希闭环及fail-closed test evaluator；不是重复训练同期臂。测试载荷门禁通过前不得查看test/buffer或启动测试评分。

## 2026-10-18：fail-closed test scorer 原型与服务器重查

- 新增`solar-cot-ifs/scripts/evaluate_hubei_forecast_heads_frozen_test.py`及合成合同测试。评分器在打开任何预测包之前，必须验证独立审计放行状态、test/buffer预检读取计数为0、精确评分器SHA、冻结split SHA、test pack/model manifest SHA、预注册站点/种子/FA-FAC臂和test日期清单哈希；随后检查目标日期、init/target/lead一致、跨模型/种子共同键及真值/晴空值一致，并保存所有预测包SHA和按目标日配对区间。
- 合成测试6/6通过，覆盖未放行时先拒绝且不生成结果、评分器SHA不匹配、split不匹配、非test日期、臂间support不一致，以及完整合成包成功评分。该测试未使用真实test/buffer载荷。
- 该文件仅为评分器，不含test pack构建和模型推理；真实test仍未解封，不代表test-freeze gate已经通过。还需实现/冻结测试包与推理入口，并由新鲜独立review检查整个builder→inference→scorer链后，才能重新判断是否进行一次性test评价。
- 再查`zjnu-hpc`全账户队列：只有无关CPU排队任务`214221.tc6000`，无湖北训练作业；未提交新PBS作业。FD-107/FD-172本轮SSH在认证前关闭；FD-201连接未提供认证提示。用户刚给的凭据未输入、未记录，FD-248未访问。PBS GPU任务不因节点空闲迹象而在调度器之外启动。
- 六月至七月14日仍为2,644/2,660个验收AGRI时次，16个缺档不以近时次或未验收`.npy`替代。已有A/AC/AP/AL、FA/FAC三seed训练已完成；没有科学理由重复训练。当前工作转向闭合test数据构建/推理来源、真实test单次评价协议及完整paper-claim审计。

## 2026-10-18 01:20 BJT：test scorer 独立审查与加固；服务器实况

- Fresh read-only scorer review 判定 **FAIL / NO-GO**。除原型仅评分既有预测、缺少pack builder/inference入口外，审查指出其未实际校验split/pack/model文件哈希，ground truth与任意预测包同源，容许缺失时效/部分日期，lead可能被有损截断，数组广播可造成错误指标，bootstrap默认值与冻结计划不符，亦无一次性执行账本。故该评分器及旧6项合成通过结果都不能放行真实test。
- 本地对评分原型作最小加固：要求冻结收据提供分站期望key数量和key SHA；拒绝非一维/不等长数值数组、有损浮点lead、非15分钟对齐时间；将默认共享日期bootstrap改为计划口径2,000次、seed 20261013；输出改为严格JSON并原子替换。因实际split/pack/model哈希、test真值来源和一次性执行账本尚未实现，原型现强制拒绝任何非`SYNTHETIC_ONLY`评估范围。扩展的合成测试10/10通过，包含拒绝真实test范围、部分支持、广播形状和小数lead错误。此修改不补齐builder/inference或真实artifact哈希链，test/buffer仍不可读。
- 通过既有密钥只读登录`zjnu-hpc`，实时`qstat -u slfu`仅有无关CPU作业`214221.tc6000`处于Q，无湖北GPU作业；该作业仍占PBS并发名额。登录节点没有`nvidia-smi`，未以其输出推断GPU空闲。FD-201/FD-172仍在SSH认证前断开；对ZJNU密码入口的一次尝试被拒绝，未重试；凭据未写入文件、命令或日志，FD-248未访问。
- 数据核查没有新来源：2026-06-01至07-14仍为2,644/2,660个已验收AGRI时次，16个原始产品时次仍无合规补帧。A/AC/AP/AL和FA/FAC验证训练此前已完成，不重复启动；下一项仍是闭合五折OOF来源、forecast pack/模型manifest与builder→inference→scorer完整审计链。独立test-release复审通过前不启动最终评分；远端Fudan训练资源恢复后按既定单卡profile和PBS≤3规则推进，不越过数据/模型门禁。

## 2026-10-18：冻结湖北test时间支持的元数据重算

- 仅连接冻结日期表与 `target_timestamp_support.csv` 的分钟完整性字段，独立重算22个test日期的完整15分钟候选目标支持：Niushou 1,620、Wangzhai 1,642、Mizhuang 1,521，三站共同980个时刻；各站目标时间集合SHA及方法记录于 `TEST_TARGET_SUPPORT_METADATA_ONLY_20261018.md`。与原支持汇总一致，GHI值、AGRI像元、CPP/COT与buffer载荷读取均为0。
- 实时PBS队列仍为单个无关CPU作业`214221.tc6000`（Q），无湖北GPU任务；远端已暂存Hubei目录仍只提供train/validation输入与checkpoint，未发现test stage/test pack。已有三种子FA/FAC训练及复算齐全，因此本轮不重复训练或为了占卡启动无关作业。
- 下一步必须补齐test端到端输入构建与独立来源绑定：冻结test forecast key/连续序列目录、从授权原始输入构造预测AGRI→COT→GHI头输入及独立15分钟GHI真值、绑定18个seed/station/arm权重和归一化合同，再完成独立静态审计与一次性release。未完成之前test/buffer仍关闭；该元数据核验不是test成绩。

## 2026-10-18 01:13 BJT：FA/FAC冻结模型清单完成

- 从学院服务器`zjnu-hpc/tc6000`重新读取FA/FAC三seed、三站共18个best checkpoint SHA，并核对3份`run_metadata.json`、3份共享内容的`normalization_contract.json`、训练器源码和训练pack报告；本地冻结清单`audit_receipts/forecast_model_manifest_20261018.json`通过远端逐文件SHA复核，26项均一致、0 mismatch。
- 清单绑定训练pack报告SHA=`9421c0d5…359dd`、冻结split SHA=`6ef8cbc7…bad19c6`及五折OOF训练银行SHA=`fdba5b91…c86fc4e`，明确`test_payloads_read=0`、`buffer_payloads_read=0`。它是模型provenance，不是test解封收据或测试结果。
- 复旦FD-107/172/201/248本轮BatchMode只读SSH均在认证前由远端关闭；未使用密码、未登录、未启动任何GPU进程。PBS全账户仍只有`214221.tc6000`无关CPU排队任务。由于现有训练seed齐全且无test输入可用，本轮不重训、不越过PBS调度器占卡。
- 下一步：获取可访问的test原始AGRI/GHI来源后，按冻结test日期构造独立test pack；实现只接受冻结manifest/一次性release receipt的推理和评分入口，再做新鲜独立审计。当前测试/缓冲数据仍关闭。

## 2026-10-18 01:15 BJT：模型清单结构复核与test源位置确认

- 本地对冻结模型清单做结构合同检查：3 seeds × 3 stations × FA/FAC 共18个组合齐全，SHA格式有效，test/buffer读取数为0；通过学院远端重新计算18 checkpoint、3 normalization、3 run metadata、训练器源码和训练pack报告共26项SHA，全部与清单一致（0 mismatch）。
- 只读检查训练阶段复制清单发现，原始`/home/Data_Pool/zjnu/hubei_cot_independent_20261016/`及`/home/Data_Pool_3/chenyi/Auxiliary_data/HuBei_AGRI_Preprocessed/`在当前`tc6000`均不存在；实际保留下来的仅是`/public/home/slfu/ttzhou/swc/irradiance_forecast/hubei_independent_trainval_stage_20261017/`的train/validation stage。其历史stage manifest只列train/val来源，不能据此构造冻结test输入。
- 本轮FD-107/172/201/248均被TCP/22远端在SSH认证前关闭；未提交密码或启动节点进程。学院PBS队列仍只有`214221.tc6000`无关CPU Q作业。
- 该证据把下一步明确收敛为取得有权限访问的test源存储/接通原始数据服务器；收到可读test原始输入后再运行数据stage和test pack，不用验证数据替代，也不回收未分配GPU直接运行。

## 2026-10-18 01:42 BJT：test评分链合成门禁加固

- 对 FD-201 的无凭据 verbose 握手确认 TCP 可连，但远端在 SSH banner/key-exchange 前关闭；FD-107、FD-172同样在密码提示前关闭。本轮没有发送用户提供的密码、没有验证其正确性、未尝试 FD-248。
- `evaluate_hubei_forecast_heads_frozen_test.py` 新增每个初始化时刻必须覆盖完整 +15…+240 min 共16个lead的检查；合成测试改为完整时效夹具，并新增缺lead拒绝用例。测试共11项全部通过。
- 该改动只验证评分器的输入时效完整性，未补测试包构建器/真实推理入口，也未改变 synthetic-only 范围；test/buffer仍未读取。最近一次可访问的 ZJNU 队列快照（01:18 BJT）只有无关CPU排队任务`214221.tc6000`，本轮未提交PBS/GPU作业。

## 2026-10-18 01:34 BJT：test AGRI 文件名覆盖盘点与认证状态

- 重新用已配置的`zjnu-hpc`只读核对：账户队列仍只有无关CPU排队作业`214221.tc6000`；湖北训练/验证产物和FA/FAC权重仍可见，旧原始GHI工程及原始湖北站点数据根在该服务器不存在。
- 在可读目录`/public/home/slfu/ttzhou/Auxiliary_data/HuBei_AGRI_Preprocessed/data`，只用冻结test日期列举`*.npy`文件名，计算文件槽位和同一自然日连续24帧候选窗；未读任何AGRI数组、test GHI、CPP/COT或buffer载荷。22个冻结test日中20日有某些帧文件，另2026-06-14和06-15两日目录不存在；已有日的文件数为76–95，纯文件名连续24帧候选窗为17–49/日。该统计尚未应用白天掩膜、三站真值键或模型输入有效性，不能当作预测支持量。相关表格已加入`TEST_TARGET_SUPPORT_METADATA_ONLY_20261018.md`。
- 按用户提供的非248节点凭据条件，分别尝试FD-107、FD-172、FD-201；均在SSH认证前由远端关闭，未显示密码提示，因此密码没有被发送、正确性仍未知；FD-248依指示未访问。没有登录复旦节点，也没有启动任何未调度GPU进程。
- 下一步是找到可审计的缺日原始帧来源、完成全窗口key元数据交集，并闭合端到端test pack/inference/scorer；独立release审计通过前继续保持test/buffer数值载荷封存。已有A/AC/AP/AL、FA/FAC训练完整，不为消耗GPU重复训练。

## 2026-10-18 02:01 BJT：审计修正、合成测试修复及旧日期接触提示

- 按 `experiment-audit` 技能调用新鲜只读审查员复核Hubei frozen-test评分链；完整请求/回复已存于 `.aris/traces/experiment-audit/2026-10-18_run01/`。审查结论仍为 **FAIL / NO-GO for actual independent test scoring**：真值与预测同包、缺builder/inference/一次性账本、manifest哈希未实体验证，完整论文指标也未齐全。没有读取test/buffer数值载荷。
- 审查员复跑发现此前“11项全通过”记录已过时：当时代码变为要求 `split_path: Path`，而夹具仍传hash字符串，实际为1项通过、10项异常。现已修正夹具生成真实临时split CSV并绑定文件SHA，错误scope/split用 `np.load` 哨兵证明在开预测包前被拒绝；评分输出标签也改为 `SYNTHETIC_CONTRACT_TEST_PASSED` / `synthetic_proxy`。当前重跑11/11通过，仅是软件合同见证，不能作为test科学成绩。旧记录保留为历史，不追改。
- 元数据层面新增潜在既往接触提示：旧六月回放工程曾为2026-06-14建立44个起报时刻的运行任务，该日与当前冻结test日期重合；其区域及是否使用相同真值/参与选择尚未确认。只读了运行日志开头与路径/配置元数据，没有打开输出/观测载荷；不自动移除该日，已登记在 `TEST_TARGET_SUPPORT_METADATA_ONLY_20261018.md`，需独立决定是否敏感性处理。
- 六月至七月14日数据补齐仍未过来源门禁：已知学院可访问预处理AGRI目录缺2026-06-14/15日，其余冻结test日也有文件槽位缺失；此前湖北训练已经结束，不因这项缺档重训。已尝试用提供的非248节点认证条件连接 FD-201，但远端在认证前关闭，未出现密码提示；凭据未发送、未留存，无法用它补取数据。未提交PBS/GPU任务；最近ZJNU只读队列快照仍为无关CPU任务 `214221.tc6000`（Q）。
- 下一步：仅从源目录清单、作业脚本与工程配置定位6月—7月14日缺档原始AGRI/站点分钟记录来源；不借用旧test输出作标签或补帧。能取得有来源证明的train/validation缺档后，重建相应stage并审计单位/时间窗口；若涉及test/buffer，保持封存并先过test-freeze review。已有训练权重不重复训练。
- 02:04 BJT补充精确名检索：在学院个人工程目录`/public/home/slfu/ttzhou`按16个规范UTC时次同时匹配精确`.npy`文件名和HDF时间戳文件名，命中0；只做文件名搜索，没有打开payload。服务器当前账户全队列仍为`214221.tc6000`一个CPU作业（Q），未提交新作业。认证节点FD-107/172/201均在SSH认证阶段前关闭，密码仍未发送，无法通过这些节点取得缺失原件。

## 2026-10-18：湖北论文结果闭环与可用算力现场核验

- 重新读取根规则、错误预防规则、`solar-cot-ifs/AGENTS.md`、湖北执行计划及B0审计。核对结论：湖北A/AC/AP与FA/FAC模型训练已完成；当前欠缺不是训练算力，而是独立test的真实标签/输入来源和可审计 builder→inference→scorer。已有验证值不能代替论文最终独立测试。未因“多用卡”要求重复训练已完成的模型。
- 实时查询学院PBS：账户 `slfu` 当前仅有 `214221.tc6000`（CPU作业，Q），占并发1/3。GPU节点 `node21/22/23` 均报8张卡，但节点作业记录显示已有其他作业分配，`node22`另有维护标记；未在登录节点或未分配GPU上启动进程，也未提交无明确科学任务的占位训练。
- 重新以显式禁用公钥的密码认证尝试FD-107、FD-172、FD-201，三者仍在SSH banner/key-exchange前关闭；再经`zjnu-hpc`作ProxyJump连接FD-201，远端在banner阶段超时。没有出现密码提示，因此用户提供密码未发送，认证结果未知；FD-248按用户约定未访问。该现象是访问路径/服务端握手阻断，不是密码错误的证据。
- 元数据检查了旧六月回放工程 `baseline_cot_20260601_20260714_20260723/inputs`：当前可见的是AGRI预处理帧和太阳几何目录及传输清单，未找到原始站点工作簿或独立GHI真值包。旧脚本也明确对目标缺帧做线性填补、目标帧未要求同日；因此旧结果不能直接填入当前正式测试链。测试日 `2026-06-14` 仍保留为潜在既往接触风险，未打开预测/观测结果。
- 下一项优先级：恢复能读取 `/home/Data_Pool_3/chenyi/GHI` 及原始FY-4B存储的认证/网络路径；随后从授权原件补齐可用AGRI和独立15分钟GHI真值，生成彼此分离且哈希绑定的test pack与prediction artifact；完成同时间A/AC/AP与forecast FA/FAC各自完整支持上的一次性评分、独立复核和论文LaTeX/中文MD同步。资源可用后将GPU用于明确的构包推理或必要重训，不为占满GPU制造无关实验。

## 2026-10-18：旧项目派生观测表候选与测试覆盖核验

- 发现本地旧项目15分钟站点表 `three_day_ghi/runs/quarter_hour_observations_20260401_20260714.csv`，只载入时间、站名、观测分钟数和有效标记，不选取/评分GHI数值。它覆盖Niushou/Wangzhai/Mizhuang三站，2026-04-01至2026-07-14；生成代码显示该表格式可由三站Excel分钟记录按右闭15分钟窗口均值构造，观测有效阈值为至少10分钟。但该文件实际生成命令、原始工作簿、仪器单位证明均未找到，故登记为候选标签源，不纳入正式test pack。
- 与冻结22个test日期做时间/站点元数据交集：目前表内只有2026-04-29/30、05-15/16、06-14/15共6日；六日的有效观测条数为Niushou 523、Wangzhai 542、Mizhuang 374，三站共同有效时刻315个；另16个test日不在表的覆盖范围。该统计不代表可评价样本数或模型成绩。6月14日的旧回放接触风险依然未定。
- 另发现三站的逐小时观测汇总表覆盖2025-08-01至2026-07-14（每站8,338行），但其时间分辨率与冻结15分钟测试合同不符，且源文件/构建记录未闭环；本轮只读时间和站点元数据，不用于评估。
- 远端 `hubei_forecast_ghi_20261017/pack/pack_report.json` 元数据明确 `HUBEI_GHI_FORECAST_HEADS_TRAINVAL_PACK_TEST_CLOSED`，仅含train/val共70,171条；stage报告的`test_sequences=0`且`test_or_buffer_payloads_staged=0`。没有把已有训练/验证包误当成测试包。
- 网络路径补查：FD-201从本机仍在SSH认证提示前断开；从可达的`zjnu-hpc`以`maont`用户访问10.158.205.201时TCP连接超时，密码没有发送。SSH本地别名当前指向10.158.205.201但配置用户是`zjnu`；这既不能证明该密码错误，也未建立复旦数据/GPU访问。未访问FD-248、未提交PBS或启动未调度GPU进程。
- 复核本机旧SSH进程时发现一个此前遗留的湖北冷启动审计提交尝试：远端shell的命令续行不完整，`qsub`无作业文件参数并等待stdin；同一时刻`qstat -u slfu`无该作业，只有`214221.tc6000`。已先核对本地PID的命令行、再断开该本地SSH会话；随后远端`ps`不再显示qsub/shell，`qstat`仍只有原CPU排队任务。该清理没有取消或终止任何PBS作业/他人进程，之后提交需使用完整脚本路径并在提交后核对队列。
- 下一步：拿到/找到可追溯的分钟Excel原件及生成记录，先只校验来源、单位与测试覆盖；为其余16个冻结test日取得原始标签与AGRI源后，独立构建test pack及推理链，再申请新鲜审计。A/AC/AP与FA/FAC既有训练无需重复；训练已完成这一事实不替代独立测试。详细字段级范围见`TEST_TARGET_SUPPORT_METADATA_ONLY_20261018.md`。

## 2026-10-18：密码节点连通性复测与本地缺帧精确检索

- 收到用户补充的非248节点凭据适用说明后，仅复测已配置的FD-107、FD-172、FD-201 SSH端点。三者TCP均可建立，但远端在SSH key-exchange/banner前主动关闭，尚未到用户名/密码认证阶段；本次未发送凭据，因此不能判断密码有效或无效。FD-248未访问，凭据未复制到命令行、脚本或工程文件。
- 为继续追查6月至7月14日16个AGRI合法时次缺口，在本地`F:\`对16个规范UTC时间戳做文件名精确搜索，未发现匹配文件。搜索范围是本地F盘文件名，不覆盖网络盘、离线归档、改名打包文件或其他计算机，不能据此称全盘/全源穷尽。
- 当前冻结train/validation补充包仍为2,644/2,660个已验收合法产品帧；16个来源缺档未补。已有训练资产不因搜索结果重训；无可验证的新输入源时不生成替代帧，不打开test/buffer载荷。
- 下一步维持本地数据来源追查和test链路的synthetic-only修复；复旦网络需先恢复到SSH认证阶段，密码才有验证意义。若取得可靠原件，再以独立manifest和逐文件/逐像元QA接入，不热换已训练run的数据。
- 新增官方原始数据候选路线：国家卫星气象中心风云数据服务目录列有FY-4B/AGRI L1全圆盘4 km产品，目录日期覆盖至2026-10-08；门户说明下载需先完成用户注册/审核。尚未登录或检索具体缺失时次，也未下载，故仅作为可追溯来源路线，不计为补齐证据。若获得授权账号，按16个规范UTC时次检索并逐文件校验HDF时间、产品类型及SHA；涉及冻结test/buffer的载荷仍须先过独立release门禁。官方入口与注册要求见NSMC数据门户：`https://data.nsmc.org.cn/DataPortal/cn/data/dataset.html?satelliteCode=FY4B`及其[英文数据服务说明](https://satellite.nsmc.org.cn/DataPortal/en/data/dataset.html?dataTypeCode=FHS&instrumentTypeCode=AGRI&satelliteCode=FY4B)。

### 2026-10-18 复旦节点重测与官方数据检索条件

- 再次通过SSH别名直连FD-107/172/201，三台仍在key-exchange/banner阶段被远端关闭；没有用户名/密码提示，未发送凭据。PBS队列于02:07 CST仍仅有`214221.tc6000`（无关CPU作业，Q），本轮没有Hubei GPU作业或提交。
- 核对NSMC官方目录后，明确可检索的目标产品是“FY-4B成像仪L1数据（全圆盘/4KM）”；目录给出的覆盖区间截至2026-10-08。门户要求先在中国气象数据网注册并通过审核后才能在线下载；尚未用账号检索16个具体时间，因此产品总体覆盖不能替代逐时次文件命中证明。
- 可执行下一步为恢复FD节点SSH握手或使用已注册的中国气象数据网账号，从官方目录查询16个缺失UTC文件名并下载。拿到文件后先做HDF元数据与SHA核验，再仅按现有冻结train/val来源门禁决定是否构造新manifest；不把不明来源`.npy`候选或相邻时次当作补齐。

## 2026-10-18：6—7月站点原件哈希复核与数据补齐状态

- 在本机微信文件目录找到`天气数据_2026-06-01-2026-07-06.xlsx`与`天气数据_2026-07-07-2026-07-14.xlsx`。两份文件SHA-256分别为`7b06d7931c6079b21e6d2b744879bc59c0a693f1251ab663d688a376bfe94398`与`37793b3b5d70f974b9a7e9c703c348bd9c8a1d1ee671fa19f0f066be325fad5f`，与冻结标签构建报告中对应源文件哈希完全一致。仅检查了工作簿名、大小、表名、表头/列位置并计算哈希；未读取测试日期的辐照度值。
- 冻结train/val标签文件`trainval_labels_20261015_v1/ghi15_train_val.csv`哈希为`729771d288df1092cd26660bb641e958125ba4684c6d7c0e7bbc6b64ccdb2f2c`，对应构建状态为`TRAIN_VAL_LABELS_BUILT_TEST_VALUES_CLOSED_UNIT_PROVISIONAL`；报告记录75,414个train/val目标槽、test辐照度单元格解码数0。既有六月1日至七月14日切片每站含2,392个train和190个validation目标键，完整窗有效数见`PROGRESS_20261017.md`。这确认该时段站点train/val标签已经进入冻结数据合同，不需要为补同一批标签重复训练。
- 来源字段为第6列“总辐射1瞬时值”，第17列为“时间”。原工程将该字段直接作为辐照度操作值、不做累计量差分或物理换算；旧资产5分钟正值均值与当前右闭完整15分钟标签不同。工作簿未给仪器型号、计量单位字典或校准记录，因此W/m²仍是继承原工程的操作合同，不能称独立计量溯源已完成。
- AGRI来源缺口未变：6月1日至7月14日仍为2,644/2,660个已验收合法产品时次，16个缺失时次未补。NSMC公开目录的元数据复核未找到这16个精确产品文件名（0/16）。冻结22个test日按BJT→UTC窗口检查到2,056/2,090个目录文件槽，缺34个；这是文件名/目录元数据，不是模型输入或测试支持结论，也未下载或读取测试载荷。在线下载仍需获批的中国气象数据网账号。
- 复旦节点访问：本机直连FD-107/172/201都在服务端SSH版本横幅返回前断开；从可访问的ZJNU-HPC到三节点TCP/22也均不可达。未到账号认证阶段，密码没有发送；FD-248未访问。ZJNU-HPC实时队列仅有无关CPU排队任务`214221.tc6000`，没有湖北GPU作业。本轮没有提交或启动训练。
- 下一项继续闭合独立test的ground-truth、builder→inference→scorer及一次性账本门禁；test/buffer数值载荷保持封存。现有同时间A/AC/AP及FA/FAC训练结果不重训；只有取得经审计的新AGRI源、构成新的合法训练合同后，才评估是否需要独立重训。

## 2026-10-18 02:43 BJT：独立评分数值核心与认证重试

- 按用户最新说明再次尝试FD-201 SSH密码认证。TCP建立后仍在SSH key-exchange/banner阶段被远端关闭，未出现用户名/密码提示；因此没有发送口令，不能据此判断密码是否正确。FD-248未访问。
- 新增与数据载荷隔离的评分基础模块 `solar-cot-ifs/scripts/hubei_forecast_metrics.py`：RMSE/MAE/bias/R²、以同支持实测均值为分母的nRMSE/nMAE、显式晴空敏感性筛选、逐lead统计、共同键唯一性检查、共享目标日期块配对bootstrap；对不合法日期、不同seed支持、truth/key不一致及无效抽样显式报错，不静默跳过。同步新增纯合成单元测试 `test_hubei_forecast_metrics.py`。
- 本地Python执行8项synthetic tests全部通过，`py_compile`通过。模块SHA-256：`D1CB78C45D33C0A43AB3D84539C6B638343CD573F1534A11084D87705C7905EF`；测试SHA-256：`22EB39634EE4907F27DE059D7C895F031D50D55D8B3E37C1017646A2417503E7`。这是评分代码的单元测试，不是实际test结果；未读取test/buffer数值，也没有改动既有评分器的test封锁。
- 数据结论未变：六月1日至七月14日站点标签及单位/时间处理已经核验并纳入既有train/validation合同；AGRI合法预处理时次仍为2,644/2,660，16个正规产品时次未找到可追溯原件。已有训练基于已冻结支持完成；缺失产品不能靠重训补出，不插值、不替换。测试评分链仍为NO-GO，test/buffer保持关闭。
- 下一步：把该纯数值核心纳入经审计的train/validation重算验证，不接触测试载荷；并继续寻找16个合法缺帧的原始来源。若获得源文件，先独立验来源、时间、产品类型、单位/变换与SHA，之后另建数据合同；只有数据及推理/评分门禁通过才讨论新增训练。

## 2026-10-18 02:50 BJT：评分器指标闭环与服务器资源复核

- 将新评分核心接入仍然synthetic-only的冻结test评分器：每组增加RMSE、MAE、bias、R²、按观测均值归一化的nRMSE/nMAE、逐lead指标及晴空均值≥100 W/m²敏感性；另输出三站共同键敏感性。日期块bootstrap现在使用共用日期抽样、核对主点估计，并对无效重复显式失败，不再静默跳过。release/split门禁仍在任何预测包读取之前执行，实时范围仍会被拒绝。
- 本地回归运行冻结test评分器与数值核心共20项synthetic tests，全通过，`py_compile`通过。评分模块SHA-256=`9D179C803CB0A3A1FA4B6119497F1D5146A54933B077CB586D4019C0EEE89223`；评分器SHA-256=`64A88A3A2FA345AEAEC774FFFEB1FA7B388EEAA487016A039A19754CED58D3A9`。本轮仍未评分任何湖北test数据，也没有解除封存。
- 通过可访问的ZJNU-HPC作只读复核：`qstat -u slfu`仅见无关CPU队列作业`214221.tc6000`；`node21/node23`记录有其他作业分配，`node22`标记维护。没有湖北GPU作业或已分配给本工程的GPU；未提交新任务。对2026六月至七月14日16个缺失UTC `.npy`键再次在已登记预处理树精确查名，命中0。此搜索不包含其他未挂载/离线备份。
- FD-172当前SSH也在交换密钥/banner前断开；用户提供口令仍未发送，未能确认任何复旦节点认证。FD-248未访问。此前文件记录中的“当时可登录FD-172”是历史证据，不代表此刻可达。
- 数据/训练结论未变：现有训练已在冻结可用输入上完成，标签合同沿用原工程直接读取第6列、将第17列作时间，并按新定义构造右闭15分钟均值；单位是原工程W/m²操作合同，非仪器校准证明。16个AGRI缺帧尚未补，且不能靠重复训练生成。下一步优先完成验证集真实预测包回算见证并获取可追溯AGRI原始源；真实独立test仍需独立真值包、完整构建/推理/ledger与fresh release audit后才可读取和评分。

## 2026-10-18：湖北预报验证集独立复算复核

- 对三站、3个随机种子、FA/FAC共18个验证预测包作独立复算。所有目标日期均落在冻结验证日期内；包哈希与各自验证元数据一致；不同种子和方案的键、实测值及晴空值完全配对一致。累计复算40,338条包内记录，test/buffer数值载荷读取数均为0。
- 复算与保存JSON中的RMSE、MAE逐项完全一致（最大差值均为0）。三种子、等站点RMSE均值：FA=77.1502，FAC=78.1553 W/m²，FAC−FA=+1.0050 W/m²；三站共同支持1,387个键上差值为+1.16 W/m²。此为冻结验证集结果，FAC没有显示优于FA；不能代替独立test或作为确认性论文结论。
- 复算报告：`audit_receipts/forecast_validation_recalc_20261018/validation_recalc_v2.json`，SHA-256=`434DCD84F6F915E1A3D796F781154777CBA763476C7BFC1BC2F7C342D9FD09AE`。复算脚本：`../../scripts/recalc_hubei_forecast_validation.py`，SHA-256=`E343F57539F83D715D19D74EFF0B68785FCFFE7D764918B51786BAD3258A5E31`。
- 本地冻结评分器与指标核心共20项synthetic tests再次全部通过，复算脚本/评分器/指标模块`py_compile`通过。这只验证软件合同与执行，不是测试集科学结果。
- 按用户补充的节点口令信息复试FD-107、FD-172、FD-201，均在SSH banner/key-exchange前断开，没有进入密码认证阶段，故未发送凭据、不能判断口令是否正确；FD-248未访问。服务器障碍仍是握手/网络可达性，不是已证实的密码拒绝。
- 六月至七月14日16个AGRI合法产品时次仍未找到原始文件。既有各方案训练已完成；本轮没有重复训练或提交PBS任务。test/buffer继续封存，只有获得可追溯输入/标签来源并通过独立test释放审计后，才推进正式评分。

## 2026-10-18 02:57 BJT：登录路径与缺失影像来源推进

- 复核工程记录后确认当前FD别名配置为`zjnu@10.158.205.107/.122/.201/.248`。使用用户授权的非248节点口令说明测试107/172/201，三台均在SSH banner/key-exchange阶段关闭，未弹出密码提示，未发送口令；248仅做本机配置密钥的只读握手测试，也在同阶段关闭，未使用口令。通过可达的ZJNU-HPC测试到107/172/201的TCP/22均不可达。此前10月17日FD-172曾有成功会话记录，当前失败说明需要恢复/确认网络登录路径；不可据此认定密码失效。
- ZJNU-HPC实时资源快照（02:37 BJT）：账户当前仅`214221.tc6000`一个无关CPU排队任务；node21/22/23记录着其他作业GPU分配，node22仍有维护标记。此为ZJNU-HPC状态，不代表复旦GPU节点可用；本轮未提交作业、未在未分配GPU运行负载。
- 官方NSMC目录检索再次确认FY-4B AGRI L1全圆盘4 km产品目录包含该历史区间；在线检索/下载需通过中国气象数据网注册并审核。匿名公开页面没有返回16个精确目标时次的下载记录，不能视为已经取得数据或补齐。来源页：https://data.nsmc.org.cn/DataPortal/cn/data/dataset.html?satelliteCode=FY4B 。
- 当前训练结论与test门禁不变：已有湖北A/AC/AP及FA/FAC训练资产不重复训练；16个train/val产品日历HDF缺档和独立test真值/推理释放链仍未闭合。收到访问信息后继续查可追溯数据源；不以插值/邻时次替代，也不读取test/buffer数值。

## 2026-10-18 03:20 BJT：评分器真值与预测文件拆分

- 按独立test审查指出的同包来源问题，重构synthetic-only评分原型：预测包现在只允许保存预测、clear-sky和时空键；若预测包包含`truth_ghi_wm2`即拒绝。评分器改从独立的逐站真值包读取GHI/clear-sky/键，并要求与每个seed/arm预测键和clear-sky逐项一致。
- 新增真值manifest文件SHA绑定、逐站真值文件SHA复核、站点集合校验、manifest路径限制及目标日期/时效验证；真实执行范围仍在release receipt中被明确拒绝，`np.load`仅用于synthetic fixtures。没有读任何真实test/buffer载荷。
- 回归测试23/23通过，评分器与测试文件语法编译通过。评分器SHA-256=`09AF8A9A29855FE8897B75DF6DA7371118FA1882F59F1543CFE6923ABD78EE1D`；测试SHA-256=`62E777C015EC324CF38B1AC19DA7AC4ECCB0EBBCA0466FAFC59AB862976458A1`。这只关闭“真值和预测被放在同一NPZ”的原型设计缺陷，不代表真实数据来源或test链已验收。
- 仍待补齐完整test pack builder、真实源标签manifest、模型/归一化/推理代码及checkpoint的实际绑定、推理回执、原子一次性ledger和fresh independent release review；数据端16个HDF缺档与复旦节点访问故障仍在。因此不启动新训练、不提交GPU任务、不解封测试集；既有训练/验证成绩仍只能作为validation结果。

## 2026-10-18 03:05 BJT：synthetic评分接入一次性执行账本

- 新增`hubei_test_ledger.py`：用`O_CREAT|O_EXCL`原子占用账本文件，绑定release receipt、split、真值manifest、模型manifest、evaluator与test日期哈希；写账本并fsync后才允许对真值载荷做SHA或读取预测/真值数组。崩溃/评分失败仍保留占用状态，不自动清除，不允许第二次使用。
- 将账本接入冻结test评分原型；新增并发争抢仅一方成功、绑定错误不占位、评分后重试在`np.load`前被拒绝等合成见证。该原型仍拒绝任何真实test scope，不证明正式PBS执行环境或共享文件系统的一次性语义已经验收。
- 三组合成测试总计28/28通过，相关模块`py_compile`通过。SHA-256：评分器`DF9AD4C7001ADA3EB93B1A9D33EFF6292FDC11FB601F6E61C33AC8A6039AA8A6`；评分器测试`2B46397EA4B50EB599375D4EA6120A56FD5CCBC2D8F50154584E730E43E906D9`；账本`8B0026E4912EA7DEC4C208C5AA394322FBE100361CA8BE5A1DA36DD229A24A5B`；账本测试`0B130A5E3B1D4DD0314A43D060CFA1ADD9FCAC8F1C7980E2C623A41747B65FB3`。
- 正式链仍未达到放行条件：实际test pack builder、可追溯标签来源、模型/归一化/训练与推理产物实哈希绑定、推理回执以及fresh independent audit尚缺；16个AGRI时次原件未补，测试/缓冲数值未读取。既有训练不重跑、无新PBS/GPU任务提交。

## 2026-10-18：学院镜像中的缺失时次副本复核

- 收到用户补充的节点密码适用范围后，再次尝试FD-201；连接仍在SSH认证前关闭，未出现密码提示，本轮未发送凭据，因此不能判断该密码对复旦节点是否有效。此前可达学院节点 `zjnu-hpc` 上，账户队列快照仍仅见无关任务 `214221.tc6000` 排队；这不是复旦服务器的队列状态。
- **时区勘误：** 首次在学院镜像检索时误把 `.npy` basename 当作BJT，并将 `202606111615.npy`、`202606111630.npy` 错报为两个缺口候选。与冻结索引及 `FY4B_wuhan_256.py` 的UTC键合同交叉核对后确认，basename是HDF UTC时次；这两帧对应BJT 2026-06-12 00:15/00:30，和缺失的UTC 2026-06-11 08:15/08:30无关。此前的候选结论撤回。
- 按16个冻结缺失UTC键逐个检查学院镜像路径 `/public/home/slfu/ttzhou/Auxiliary_data/HuBei_AGRI_Preprocessed/data/2026/YYYYMM/YYYYMMDD/<UTC-key>.npy`，16/16均不存在；目标日期下另有的其他UTC帧不计作补档。仅取文件名/存在性与mtime/大小，未读取数组值。对学院 `Auxiliary_data` 下对应日期检索也未发现原始 `.hdf/.h5` 文件名，尚无可接入的逐像元原件或来源manifest。
- 数据补齐结论不变：16个合法源产品槽仍均未获得合规原始产品，覆盖仍为2,644/2,660；现有训练资产不因此重训。下一步是恢复可达复旦SSH认证通路，或通过已获批的官方FY-4B数据账户检索并下载这16个UTC时次。test/buffer载荷继续封存。
- 继续确认替代路径：当前可达的 `zjnu-hpc` 未挂载两处已登记原始HDF归档根 `/home/Data_Pool/data/FY/FY4B/AGRI/4KM/2026` 与 `/home/Data_Pool/data/FY/FY4B/real_time_data/AGRI/4KM/2026`，故不能从该节点扩展原始文件检索。国家卫星气象中心公开目录当前列出 FY-4B/AGRI 全圆盘4 km L1数据覆盖至2026-10-14；这是产品级目录覆盖，不证明任一缺失UTC槽存在可下载文件。在线下载需通过已注册/审核账号；本轮未登录或下载。官方目录：`https://data.nsmc.org.cn/DataPortal/cn/data/dataset.html?satelliteCode=FY4B`。

## 2026-10-18：本机与学院工作簿的站点真值覆盖复核（仅结构/时间列）

- 在本机微信文件及可达学院用户目录核对湖北站点工作簿清单、SHA-256、工作表名、记录数和第17列时间范围；仅读第6列字段标题与时间列，没有读取任何辐照度单元格。可访问学院目录中仅找到2026-04-01至2026-05-31、2026-06-01至2026-07-06、2026-07-07至2026-07-14三份原始工作簿；没有发现2025-08至2026-03的同类分钟工作簿。
- 另发现文件名为`天气数据_2025052500-2026060110.xlsx`的工作簿及同哈希副本。文件名起始年写2025，但三站第17列实际时间均从2026-05-25开始，分别有9,908、9,694、10,096条时间记录，末时刻为2026-06-01约10:18–10:20；该文件SHA-256=`C1B7CA8CD3081A49ED9A643B68B35868D15B7108ECF3C7A90D1331A472241B25`，两个副本字节相同。它只增加2026-05-25至06-01附近的分钟记录，不是2025年历史观测。
- 对照的三份工作簿SHA-256与此前冻结标签来源记录一致：2026-04-01至05-31为`D4781696F42231BD2EBF54E05C43EED98ECBF084506BE1AB78A0C14B3631D50A`；2026-06-01至07-06为`7B06D7931C6079B21E6D2B744879BC59C0A693F1251AB663D688A376BFE94398`；2026-07-07至07-14为`37793B3B5D70F974B9A7E9C703C348BD9C8A1D1EE671FA19F0F066BE325FAD5F`。表头仍只写“总辐射1瞬时值”，没有单位标注；时间覆盖复核不构成计量单位证明。
- 结果：可验证的15分钟站点标签来源目前覆盖2026-04-01至07-14的若干工作簿，不能为冻结test中2025-08至2026-03各月份提供同源原始分钟真值。六月至七月train/validation既有标签不需重建；没有因此打开test值或启动新训练。下一步需恢复原服务器/归档访问，或取得相应月份可追溯的站点原始分钟记录和官方数据产品。

## 2026-10-18 03:21 BJT：湖北已训练模型现场复核与复旦认证阻断

- 依用户新给的非248节点密码说明，显式禁用公钥尝试FD-107、FD-172、FD-201；三台均在SSH banner/key-exchange阶段由远端关闭连接，未出现用户名/密码提示，故未发送密码，不能判断密码是否正确。FD-248未访问。
- 经当前可访问的学院登录节点 `zjnu-hpc` 复核，湖北同刻模型的 norm-reload 目录有27个 `A/AC/AP_best.pt`（3站×3种子），外推模型 norm-reload 目录有18个 `FA/FAC_best.pt`（3站×3种子×2方案）。三份外推训练日志（214659/214665/214666）以及初次训练日志（214647/214649/214651）均以相应 `...SEED_COMPLETE_TEST_CLOSED` 行正常结束；这些产物属于既定训练与validation阶段，不是独立test结果。
- 学院PBS实时队列仍只有无关CPU排队任务 `214221.tc6000`（Q，账户并发1/3）；没有湖北GPU作业。登录节点对node21/22/23的只读快照显示节点GPU已有PBS作业占用/独占分配，不能把单卡瞬时低利用率当作可绕过调度的空闲卡。未提交重复训练，也未在未分配GPU上运行负载。
- 因而本轮确认：湖北现有获批A/AC/AP和FA/FAC训练已经有完整的三种子checkpoint与正常结束日志；当前没有需要“再训一遍”的阶段。接下来应闭合有来源的独立test真值与AGRI输入、builder→inference→scorer和release审计，再按门禁运行一次性test评价；如果数据审计发现新的合法train/validation载荷或合同改变，再单独评估重训。

## 2026-10-18：微信归档工作簿与冻结训练/验证标签来源绑定

- 在此前未展开的本机压缩包 `按采集点聚合时间(1).zip` 中找到按月工作簿。对2025-08至2026-03共8个ZIP成员计算原始字节SHA-256，与 `trainval_labels_20261015_v1/REPORT.json` 冻结来源清单逐文件比对，8/8完全匹配。该结果将本机归档副本与训练/验证标签来源绑定；SHA计算没有解析或使用工作簿中的辐照度单元格。
- 只解析各表的F列字段标题和Q列时间戳。三站表头对应 `总辐射1瞬时值` 与 `时间`。2025-08至2026-03各月均含三站sheet；王寨2026-02仅覆盖26个日历日，其余月度覆盖仍须以逐窗完整性为准。没有从本归档重建或改写标签，也没有读取test/buffer辐照度数值。
- ZIP中的2026-04成员仅784,967字节，三站时间列都只覆盖4月1日至3日；该成员不能代表完整4月。ZIP无2026-05至07工作簿。2026-06-01至07-14仍以已审计的两份分段工作簿及冻结train/val标签表为准。
- 口径沿用原工程核验结果：第6列 `总辐射1瞬时值` 直接作为辐照度操作值，第17列为时间；新标签按目标时刻右闭的15分钟窗 `(target-15min,target]` 计算，归一化沿用原工程W/m²操作合同。原始表未提供仪器单位字典/校准记录，故不能将操作单位描述成独立计量溯源结论。
- 数据补齐边界不变：六月至七月14日train/val站点标签已构建并用于现有模型；仍缺16个FY-4B AGRI原始合法时次，不能用站点工作簿、邻帧或重复训练代替。A/AC/AP与FA/FAC模型的三种子训练已经完成；本次未提交PBS，也未重复训练。下一步应继续补取这16个有来源的AGRI帧，并完成独立test数据/推理释放审计。

## 2026-10-18：test 预测支持与稀疏lead评分合同修订（不解封）

- 收到更新的非248节点凭据说明后，单次重试FD-201连接仍在SSH认证前被关闭，没有出现密码提示；本次未发送凭据，不能判断密码是否正确。FD-248未访问。学院 `zjnu-hpc` 实时PBS快照为无关CPU任务`214221.tc6000`排队（1/3）；node21/22/23均列有其他PBS作业/维护状态，无本项目GPU任务。未绕过调度器或提交训练。
- 复核test支持时纠正一项口径：连续24张真实AGRI只对未来卫星场真值评价必要；站点GHI预测推理是8帧历史AGRI→SimVP预测16帧→R/GHI头，未来真实AGRI不是预测输入。依据原预处理器“UTC文件名→BJT加8小时”，只列学院镜像文件名，并与冻结test日期、三站`complete15`时间支持相交；共读取2,773个UTC文件名，换算后20/22个test日存在8帧历史初始化候选。元数据上界为1,305个初始化、10,020个三站共同`(init,target,lead)`键；未应用太阳几何/日间mask、GHI数值质量和6月14日既往接触裁决，不能作为正式评分支持。6月14/15在该镜像树仍无可用历史文件名；4月29日没有三站共同完整15分钟窗。没有读取AGRI像元、GHI数值、CPP/COT或buffer载荷。
- 新发现的评分原型要求每个初始化都必须覆盖16个lead，与正式train/validation pack因真值mask仅保留部分lead的合同不符。移除该额外限制；每个bundle仍须使用规范+15…+240中的lead，严格等于独立真值keys，并通过release receipt冻结的每站key数量与SHA-256核验。新增“稀疏但共支持”合成fixture。`test_hubei_forecast_heads_frozen_test.py`、`test_hubei_test_ledger.py`、`test_hubei_forecast_pack_gate.py`、`test_hubei_forecast_metrics.py`共32项标准库合成测试全部通过。测试只证明评分软件合同，不代表真实test来源/推理器/释放审计通过。
- 当前真实test仍未解封。尚缺test pack builder、可追溯test真值包、test推理入口和实际checkpoint/归一化/模型清单哈希绑定；6月14/15输入源和旧回放接触裁决未解决。新增该评分器源码后，必须更新精确SHA并重新做独立release审查；不提交重复训练，不把10,020候选上界写成模型精度结果。
- 本机额外数据检索：`F:\Markdown\HuBei_AGRI_Preprocessed`仅含2025-01数据，冻结测试期无重叠；按冻结test日期检索F盘原始产品文件名无命中。对`C:\Users\29728`、D盘、E盘同样按常见影像扩展名检索无命中，但一个Steam缓存路径拒绝访问，因此只代表已遍历路径，不宣称所有本机/网络位置穷尽。没有解码test/buffer数组。
- 2026-10-18 03:32 BJT学院只读快照：PBS仅有无关CPU排队任务`214221.tc6000`（全账户并发1/3）；`/home/Data_Pool_3/chenyi/GHI`、旧`/home/Data_Pool/zjnu/...`及`/home/Data_Pool_3/chenyi/Auxiliary_data/HuBei_AGRI_Preprocessed`在当前登录节点均不存在，仅发现`/public/home/slfu/ttzhou/Auxiliary_data/HuBei_AGRI_Preprocessed`。没有湖北GPU任务，未重复训练。
- 本机Python 3.14按脚本入口复跑4个湖北评分/门禁测试：16+4+4+8=32项全部通过。先前模块名调用因`scripts/`导入路径配置报`ModuleNotFoundError`，改正确入口后通过；不视为代码失败。测试仅为synthetic contract，不改变真实test仍NO-GO。

### 2026-10-18 04:55 BJT：官方缺帧申请路径细化

- 对照国家卫星气象中心当前公开目录与FAQ，确认FY-4B/AGRI全圆盘4 km L1目录覆盖到2026年10月；缺失16个时次落在目录产品的历史范围内，但目录汇总不能证明逐文件存在。FAQ明确数据免费下载、实名审核后检索/下单；普通账户每日限额30 GB和3万文件，网站使用UTC，客户端才提供BJT转换。具体精确UTC键和核验步骤已记入`AGRI_GAP_RECONCILIATION_20261016.md`。
- 未登录外部账户、提交订单或下载；因此没有新原始帧进入项目，冻结训练数据仍是2,644/2,660。当前`zjnu-hpc`实时PBS查询为一个无关CPU任务`214221.tc6000`处于Q；没有Hubei GPU训练任务，也没有为已完成模型重复提交训练。
- 结果边界未变：FA/FAC三seed验证已完整复核，但预测COT没有显示增益；真实test仍因来源/释放链审核NO-GO，不能写成论文独立测试结果。下一步是用已有实名账号对16个UTC键做逐文件检索/最小订单；若没有可用账号，需由用户提供已批准的官方账号/订单访问途径。拿到HDF后先逐文件验源、再决定manifest与重训范围，不热换当前训练合同。

## 2026-10-18：测试GHI原始来源容器补充核验（未读取测试单元格）

- 新增 `scripts/audit_hubei_test_truth_source_inventory_20261018.py` 并生成 `audit_receipts/test_truth_source_inventory_20261018/source_inventory.json`。该流程只哈希来源文件/ZIP成员、核对冻结split日期与文件月份容器映射，不解码辐照度值、不构造test真值包。
- 22/22冻结test日期均能映射到候选原始站点工作簿：2025-08至2026-03的16个test日来自按月ZIP；2026-04/05的4日来自`天气数据_2026040100-2026053123.xlsx`；2026-06的2日来自两份分段工作簿。8个ZIP月成员SHA全部匹配train/val来源报告；Apr–May工作簿SHA=`D4781696…3631D50A`匹配已记录来源审计值；Jun–Jul两份SHA与train/val报告匹配。脚本SHA=`91224dac8b107409537422455c7debe85ec4e0b6fcfbc514a806f70714f45ff7`，JSON SHA=`e173c85795502a038c514850ea9c4e6d8ee72a8d7162d661f98b486e66678878`。
- 边界：来源容器齐全不等于每个冻结目标窗口有效；也不证明仪器单位认证、6月14日既往接触风险已解决，或真实test可释放。机器回执明确`test_radiation_cells_decoded=0`、`test_truth_pack_created=false`、`test_evaluation_authorized=false`。下一步应由新鲜只读reviewer审核来源清单与test访问门槛，再实现并审核受release receipt控制的一次性truth builder/inference/scorer；不得直接运行标签读取或测试推理。

### 2026-10-18 来源收据审计更正（仅元数据）

- 初版收据经独立审核指出三项不足：把月份映射说成来源闭环；Apr–May SHA由运行参数提供且未绑定B0/冻结train-val报告；JSON未含原件绝对路径，且未扫描原工作簿时间列。故上文初版“来源容器齐全/全部通过”的语义撤回，不作为测试解封证据。
- 已更新脚本`solar-cot-ifs/scripts/audit_hubei_test_truth_source_inventory_20261018.py`，现在收据写入全部输入绝对路径、字节大小和SHA；月度ZIP成员仍逐项与冻结train/validation`REPORT.json`对照。Apr–May文件只与已有进度文档中的固定SHA比较，明确标注`matches_trainval_source_report=null`，不声称该digest来自独立B0审计。当前脚本SHA=`bc83eef6c0c6106b3811658f52c1844444055cdff5b773f0850b0f4ef5e4bd54`，JSON SHA=`8f1b513c35f12502c01a50f4cfdf0ab06e63b4ef1c1600c1d7fc07241af6cde9`。
- 修订后的脚本仅扫描原工作簿第17列时间戳，22个冻结test日期均在其映射的三站工作表中有时间记录；这只证明日期时间戳出现，不证明分钟箱完整或辐照度值有效。7月7–14工作簿三站时间列最晚仅到7月14日08:53–08:55 BJT，不能称7月14日全天齐全。
- 与冻结`target_timestamp_support.csv`的test日期元数据交集重算为每站1,620/1,642/1,521个`complete15`候选目标，三站共有980个时间点；GHI数值读取仍为0。来源收据仍`metadata-only`，test真值包未构建、评分未授权。
- 这一轮只修订来源收据与工程说明，未读取测试/缓冲GHI或AGRI载荷，也未进行test评分。一次新鲜只读复核对修订后的收据给出PASS：确认输入路径/哈希一致、8个月ZIP成员匹配冻结报告、22个日期均在对应三站工作表时间列出现、摘要没有越界声称标签完整或授权评分。审查结论见任务审计消息，证据键为`source_inventory.json`中的`inputs`、`monthly_archive_members`、`apr_may_workbook_digest_status`、`date_to_source_window`、`source_workbook_timestamp_column_q_metadata`、`date_to_station_timestamp_presence`和`timestamp_support`。
- PASS仅表示这份来源候选元数据收据准确，不是测试解封PASS。Apr–May digest仍缺独立来源报告绑定；单位/仪器元数据、6月14日既往接触裁决、正式forecast key冻结以及builder→inference→scorer/release gate仍阻断真实test评分。未启动重复训练。

### 2026-10-18：扩展可达共享盘缺帧精确检索

- 在可达`zjnu-hpc`只读查询当前挂载：`/public`为Parastor，`/home/Data_Pool_3`不存在。对用户可见的`/public/home/slfu/ttzhou/Auxiliary_data/`、`data_TC/`及`swc/irradiance_forecast/`依16个规范UTC时次搜索准确`.npy`basename和4KM HDF时次文件名，新增命中0；只访问路径/文件名/大小，没有读数组。当前训练数据合同仍为2,644/2,660可验收AGRI时次。
- 当前尝试的FD-107/172/201/248 SSH端点均在认证前关闭，未提交密码或PBS作业；不能据此判断FD节点无数据。zjnu-hpc队列只有无关CPU排队任务`214221.tc6000`，本工程无活动GPU训练。
- 查官方数据补齐路线：NSMC目录列出的FY-4B/AGRI全圆盘4KM L1产品覆盖2022-06至2026-10；FAQ要求实名注册/审核后才能下单下载，当前没有逐时次查询结果或下载文件。本轮未触碰test/buffer值。官方入口和访问要求已记入`AGRI_GAP_RECONCILIATION_20261016.md`。
- 结论仍是已有训练已完成、验证结果可复核，但缺帧和独立test门禁未闭合；没有新数据足以触发重训，所以未开空转GPU任务。下一步应拿到NSMC账号订单/FD访问后针对16键精确补源，验收后再建新数据manifest与对应新run，而不是覆盖历史checkpoint。

### 2026-10-18 05:12 BJT：公开逐文件目录和节点口令范围复核

- 通过NSMC公开DataPortal只读检索接口，对16个缺失UTC键所在的12个日期逐日取4KM文件目录并作精确起始时间匹配，16/16均无匹配档案名；每日目录文件数为93或94，与95帧/日产品日历扣除该日期冻结缺档数一致。没有登录、下单、下载或读取影像。详见`AGRI_GAP_RECONCILIATION_20261016.md`新增条目。
- 用户新提供的节点口令范围用于连接诊断；FD-107、FD-172、FD-201仍在认证会话前关闭，跳板至FD-201在banner交换阶段超时，没有出现密码提示，未向任何节点发送新口令；FD-248未尝试。因此本次无法验证口令，也不能说口令错误。不得把SSH握手问题当作数据缺失证明。
- 学院登录节点实时PBS快照（2026-10-18 04:47 CST）：账户仅有无关CPU排队作业`214221.tc6000`；未发现湖北训练提交。本项目已有A/AC/AP及FA/FAC三种子训练已完成，未重复训练；validation复算显示FAC相对FA没有稳定提升，仍是validation范围。真实test/buffer载荷仍关闭。
- 下一步不再重复探测同一不可达FD节点：等待管理员恢复连接或提供可用VPN/跳板路径；并请求已实名审核的NSMC账号/订单来源，或其他具有权限的历史归档入口。出现新的合规数据后才验收并评估新manifest/重训；否则继续推进不依赖缺帧的独立test builder/release审计和论文证据整理。

### 2026-10-18 05:19 BJT：替代影像归档定向检索

- 对NSMC官方FY-4B产品目录/DataPortal及公开网页按产品、精确文件名检索，未发现第二个可直接下载16个缺失AGRI L1 4KM文件的权威公开镜像。NSMC公开产品目录仍有FY-4B AGRI L1全圆盘4KM产品，但具体下载需已审核账号。此项为定向公开检索，不能证明私有科研镜像或离线归档不存在；细节和链接见`AGRI_GAP_RECONCILIATION_20261016.md`。
- 数据和训练结论不变：公开逐文件目录16/16未列出；活动训练manifest仍为2,644/2,660；已批准的同刻和外推头三种子训练已完成；没有重复开GPU/PBS作业。复旦节点未完成SSH认证，因此没有在那里用卡。
## 2026-10-18 05:12 BJT：新共享目录检查与连接/队列复核

- 新查学院共享盘的`swc/Data_Pool_3/chenyi/data/FY/FY4B/AGRI/`和`swc/Data_Pool/data/FY/FY4B/AGRI/4KM/`，可见目录结构分别只到2023年L2/CLP和2023年4KM数据，未发现2026年L1归档分支；没有读取影像载荷。具体边界已补记到`AGRI_GAP_RECONCILIATION_20261016.md`。
- 收到用户更新的非248节点口令范围后，本轮对FD-107、FD-172、FD-201的连接仍均在SSH认证前由远端关闭；通过`zjnu-hpc`转连FD-172也超时。没有出现口令提示，因此没有发送口令、不能判定口令有效性；FD-248未尝试。
- `zjnu-hpc`队列实时快照（2026-10-18 05:11 CST）：仅有无关CPU排队任务`214221.tc6000`，无本项目GPU任务。湖北既有A/AC/AP与FA/FAC训练已完成；本轮无新源帧，仍不重复训练。现有独立测试链保持NO-GO、test/buffer载荷继续封存。
- 本机重跑冻结test评分、一次性ledger、forecast pack gate及指标的32项synthetic合同测试，全部通过；这仅是软件见证，不代表真实测试放行或模型结果。当前数据仍为2,644/2,660；后续需通过复旦网络/归档授权或经审核的NSMC下载路径取得精确缺失帧，再验源并评估受影响的训练范围。

## 2026-10-18 05:37 BJT：验证指标冲突完成收敛

- 对照旧的`test_freeze_audit_20261017`与后续重建收据，确认先前报告的74.361/71.429/73.894和77.250/78.581属于较早审计所引用的旧运行，不能覆盖当前regen/norm-reload来源；不再把这组旧差异描述为尚未解决。
- 重新从当前same-time regen收据的逐seed×站点指标聚合，得到A/AC/AP=74.33330054/70.99308242/73.83922214 W/m²；AC−A=−3.34021812，AC−AP=−2.84613972。收据SHA=`df0374c16ace142b5acb11b5d0f9b92b9138483ac1d6638bd06e3b1a6383ea6f`，其fresh review也复核了27个预测包、checkpoint和元数据哈希；范围仍为22日validation，test/buffer未读。
- 重新由forecast validation recalculation收据的三个seed RMSE等权平均，FA/FAC=77.15023022/78.15526938 W/m²，FAC−FA=+1.00503916。收据SHA=`434dcd84f6f915e1a3d796f781154777cba763476c7bfc1bc2f7c342d9fd09ae`，`test_payloads_read=0`、`buffer_payloads_read=0`。这与当前norm-reload报告一致；旧77.249908/78.580996仍只作为旧test-freeze审计引用留档。
- 因而当前验证性结论可以引用regen/norm-reload两份收据，不再引用旧运行数字；仍不能称为独立test或确认性泛化结论。测试链审计仍FAIL/NO-GO，须先完成真实builder、推理回执、实际模型manifest绑定和fresh独立release复核。

## 2026-10-18 05:59 BJT：节点认证通路与正文审计更正

- 用户提供了非FD-248节点的新口令范围说明。随后对FD-201作password-only SSH尝试，但TCP/SSH连接仍在认证前被关闭，未出现口令提示，故未发送口令，无法判断凭据有效；经`zjnu-hpc` ProxyJump连接则在banner exchange阶段超时。没有尝试FD-248，也没有把凭据写入命令、脚本或工程记录。
- 根据湖北结果JSON和AL三种子回执，同步修订论文英文LaTeX与中文对照：校正AL/COT摘要指标归属顺序；更新已完成的湖北预报输入验证描述；删除“没有匹配预报GHI实验”的过时句，明确该图仅展示COT；将AL/AC参数量更正为24,549/24,489；补充湖北W m⁻²属于旧工程操作口径、仪器单位/校准元数据未独立核验的限制；结论改为湖北三站不代表广泛区域气候样本并删除中文重复段落。
- `main.tex`经桌面LaTeX编译器成功编译。编译与正文修订没有读取test/buffer载荷，没有新训练或PBS提交。湖北A/AC/AP与FA/FAC验证模型已完成；独立test仍NO-GO，缺少来源和释放链审计，16个合法AGRI训练/验证时次仍未补入。
- 通过可达学院账号的实时`qstat -u slfu`快照为2026-10-18 05:43 CST：仅有无关CPU排队作业`214221.tc6000`（账户1/3），没有湖北GPU任务。未终止或改动该作业。

## 2026-10-18 05:56 BJT：节点连接复试与测试释放链独立审计

- 用户说明非FD-248节点使用同一口令。本轮仅尝试已配置的FD-201、FD-107、FD-172及经`zjnu-hpc`跳转至FD-201；三个直连均在认证前由远端关闭，跳转连接在SSH banner阶段超时。没有出现密码提示，因此未发送口令、无法判定口令有效性；FD-248未尝试，凭据未写入文件或工程记录。
- Fresh read-only release audit结论为`FAIL / NO-GO`，未访问远程主机或封存载荷，也未改文件。主要阻塞：真实truth builder、推理回执及完整模型绑定不存在；清单仍标记`test_truth_pack_created=false`、`test_evaluation_authorized=false`；forecast manifest仅是验证模型清单；当前评分脚本自报`SYNTHETIC_ONLY`字段可伪造，模型manifest SHA仅校验格式而未打开比对，buffer读取计数由代码硬编码，命令行可指定新的ledger路径绕过单路径一次性限制，ledger也在数据包/预测生成之后才claim，不能保护首次测试数据读取。
- 审计另发现论文方法与执行计划来源不一致：Hubei FA/FAC验证段落邻近文字仍写Hunan SimVP，而计划要求湖北独立S/R OOF链；当前冻结测试脚手架仅覆盖FA/FAC，未覆盖论文主问题的同步A/AC/AP/AL。10月17日审计绑定的论文哈希已过期，须基于当前文件重新做claim audit。完整审计要点在本轮只读审计消息中；以上均未触碰test/buffer数值。
- 已同步修订`paper/main.tex`与`paper/main_zh.md`方法段：明确预测器按区域训练/冻结，湖北使用独立湖北预测器与冻结R；湖北forecast-head训练采用五折按日期分组的S→R OOF输入，验证采用全训练S→R预测bank。桌面LaTeX编译成功。修订后SHA-256：英文`D41AE70AEAD79FD28B62E84D43CCAF3DA7F44AE27132C4EC404271A61D860D6E`，中文`43AE5546A2C0200F0A2F40E4F3707CD5301C105F34D51AA98065BC49FB05CCD9`。
- `zjnu-hpc`实时队列快照（2026-10-18 05:56 CST）：仍仅有无关CPU排队作业`214221.tc6000`（账户1/3）；无本项目GPU任务。未提交作业或终止他人任务。
- 下一步：基于新论文哈希做新鲜claim审计；真实test release工程不得沿用目前可伪造的JSON gate。需重新设计跨builder→推理→评分、测试前置、路径固定且可认证的一次性授权链；在该链经新审计及明确放行前，继续封存test/buffer，不做真实评分。

## 2026-10-18 06:20 BJT：湖北工程与论文数字定向复核

- 通过可达学院登录节点实时查询PBS：只有无关CPU排队任务`214221.tc6000`（状态Q，账户1/3），没有湖北训练任务。既有湖北A/AC/AP及FA/FAC三种子训练仍是已完成状态；16个AGRI时次尚未补齐，当前没有满足重训条件的新数据，因此未重复提交训练或占用GPU。
- 复核论文claim审计中直接COT数值：原始结果JSON中的`12.162478360`按三位小数应为`12.162`。英文LaTeX与中文对照已同步统一；LaTeX编译成功。审计记录C49已改为`rounding_ok`，总审计仍为`FAIL`（25组缺少合格原始证据、3组映射含糊、2组不受当前证据支持），不是论文通过或独立test结果。
- 复旦FD-107/172/201连接此前均在SSH认证前被远端关闭，未出现口令提示；所以新提供的口令没有被发送，登录有效性未能验证。没有访问FD-248。当前不得据此声称复旦GPU已使用或启动训练。
- 本轮不读取test/buffer数值，不执行独立test评分。下一步仍是取得经授权的缺失AGRI源帧或可用服务器通道，验收后建立新manifest；对test则先修复并独立审计一次性数据释放链，再决定是否开展最终评价。

## 2026-10-18：论文变化指标符号修正（针对性复核）

- 新鲜只读审阅将相邻时次 COT 变化结果与 `cot_temporal_change_summary.json`、`RESULTS.md` 对照，发现正文把 RMSE 差值写成“降低 1.737/2.184”，但配套区间为负值；同样，微调的 `0.046/0.013` 与区间符号混用。源结果定义的是“模型减持续性”或“微调模型减原始SimVP”，负数代表更低RMSE。
- 已同步英文与中文正文：明确报告“预报减持续性/微调减原始模型”的带符号差值，保留对应日期块区间，并说明负值代表RMSE较低；没有改动底层数值或把代理反演响应说成物理COT真值。LaTeX重新编译成功。
- 更新后正文SHA-256：`main.tex`=`E708C4F985354A7ACB883FE8F466E8C8F827CF502CB97AAEB9D6D8DF94E62463`；`main_zh.md`=`80D4EA4CBAD32A50153F4F1BB19601892B5F65774866D2E6A214A555D02B3917`。原始摘要JSON SHA-256=`E2F75753BA01BCB480B84AD4F384E90E1E69D2845D289AE209F85BED83EED79D`；结果说明SHA-256=`43867EB0899BD3C835D401C8C72E11A0AC12E45BD334381A89833B20083EB0AA`。
- 该针对性复核还确认C49的12.162478结果应四舍五入为12.162；C53当前稿件所报2,817序列/52日期结果仍未在所审阅的允许产物中找到支持。整体论文claim审计仍为FAIL，不能因本轮修正称为审计通过。

## 2026-10-18 14:28 BJT：补数与重训可行性复核

- 按用户最新提供的认证说明，仅对FD-201做一次password-only连接尝试。TCP已建立，但远端在SSH key-exchange阶段关闭；没有进入认证流程或出现密码提示，因此未发送凭据，不能判断口令是否有效。未访问FD-248。
- 通过可达的学院登录节点复核PBS：`214221.tc6000`为无关CPU排队任务，账户未结束任务1/3；没有本项目GPU任务。未提交训练作业。
- 当前冻结train/validation输入仍有2,644/2,660个应有AGRI时次；精确UTC缺口仍为16个。已登记FD与学院镜像目录、NSMC在线逐日目录中均未找到这16个精确文件；其中2个同名NPY只通过数组格式检查，缺少原HDF/同源像元血缘，不可作为补帧。未改manifest、没有读test/buffer载荷。
- 原工程口径现阶段依据既有源码审计：`build_label.py`直接将“总辐射1瞬时值”作为辐照度数值，按UTC+8转BJT，旧标签取`[BJT,BJT+5 min)`正值均值；旧工程以W/m²报告并用1300归一化/还原。当前新标签则为`(target−15 min,target]`的分钟箱均值，保留有效零值并对缺测mask。仪器型号/校准及字段字典仍未独立取得；本轮因FD连接失败未重新读取原服务器源码。
- 已批准的湖北同刻A/AC/AP/AL及外推FA/FAC三种子训练均已完成，使用的是当前已验收的数据合同。新增源帧尚未取得，故既不能声称数据已补齐，也没有依据在相同manifest上重复训练。获得有来源的原始HDF后，须逐帧按原处理链验收，另建manifest并重算受影响样本；只有新合同检查通过后才按共享日期映射重训。

## 2026-10-18：预报容量对照 FAP 单卡 profile

- 按计划中“若需排除预报阶段容量因素，则补FAP九配置”推进；FAP以匹配参数的AGRI特征分支替代COT分支，保留原FA/FAC方法与训练预算。当前仅先做单卡profile，不等同于启动九配置训练。
- 核实训练器来源：当前通用文件`train_hubei_ghi_heads.py`哈希与历史checkpoint声明不一致，但历史FA/FAC任务实际调用的是版本文件`train_hubei_ghi_heads_normreload_20261017.py`；其远端SHA与冻结模型清单及seed42 run_metadata记录一致（`fd15bf6f939cb81d09bd05e6fc889f88e53e8a9e40366d417fa195848caf5999`）。本轮FAP profile固定调用该确切训练器；pack gate本地/远端SHA一致（`b01a5017c0342ebe36dc08e18a376bfd88c16ea5ba03164287c940573400232b`）。
- 首次profile作业`214743.tc6000`自然结束但`exit_status=2`：pack预检通过并短暂获配node21一张A100，启动器漏传训练器必需的`--output`，因此没有profile产物、没有开始正式训练。失败日志确认参数缺失；没有测试载荷访问证据。
- 已在启动器中加入仅profile模式所需的占位`--output "$ROOT/profile/FAP_PROFILE_ONLY_DO_NOT_CREATE_20261018"`，不创建正式训练输出；本地`bash -n`通过，修订后SHA-256=`37ecc9e6e52ddeb03447299f9d2464cd7e3eb857314209f1c24f3a8843241c55`，已同步且远端SHA一致。pack gate核验70,171行、5折OOF输入、validation来自冻结全训练S→R，`test_and_buffer_values_read=0`。
- 提交前实时账户队列为2个非终态计数口径内条目（无关CPU作业`214221`及历史已完成`214743`被qstat列出；按状态剔除C后实际为1/3）；重提`214744.tc6000`后实时队列为2/3（`214221`与本作业）。Luna在node21确认`214744`为R，单A100-SXM4-40GB；日志目前只有pack gate与GPU分配行，profile JSON尚未产生，尚无profile结果或最终exit status。当前仅跑profile，不是九配置训练。
- Luna随后核验`214744.tc6000`自然完成，PBS `exit_status=0`、node21单A100、walltime 40秒。profile JSON `forecast_fap_capacity_20261018.json` 已下载到 `audit_receipts/forecast_ghi_fap_profile_20261018/`，SHA-256=`cb0d0805656a366fc1ca3f52c51c30aefe95d6f6faee1d1d6ca1f39609b4e151`；状态`HUBEI_GHI_HEAD_SINGLE_GPU_PROFILE_COMPLETE`，FAP参数量24,613，profile_rows=20,611，`test_payloads_read=0`。batch 128/256/512/1024/1536的profile吞吐约27,024/54,398/112,928/235,725/350,510 examples/s，最小支持集每epoch更新数161/80/40/20/13，均过profile门槛。此为profile吞吐，不是正式训练速度或模型精度证据。
- 对照三份FA/FAC `run_metadata.json`，seed42/43/44均为30 epochs、batch 1536，并绑定同一forecast pack SHA=`9421c0d5e19094454e5b94b98de797ac90248d3e5b56fdef6dacda339df359dd`及训练器SHA=`fd15bf6f939cb81d09bd05e6fc889f88e53e8a9e40366d417fa195848caf5999`。FAP runner保持同pack、同seed/epochs/batch与训练器，仅运行FAP单臂，输出到新目录 `forecast_FAP_seed{seed}_normreload_20261018`。本地/远端PBS脚本SHA一致=`bc39ebc15b5f9917afc7f3960198aac5320de0cc3796b5ca71b63f53c34c5417`，`bash -n`通过，提交前seed42目标目录不存在。
- 提交前实时qstat有1个未结束的无关CPU任务`214221`；提交FAP seed42作业`214745.tc6000`后为2/3，状态Q，等待GPU调度。该作业只训练/验证三站FAP seed42，不读取test/buffer；seed43/44须等其完成后逐个重新核查队列并提交。六月至七月14日16个原始AGRI缺档未解决，现有pack仍仅包含已验收样本；不会声称整段时次无缺。
- 整体独立test仍`FAIL/NO-GO`，不因FAP profile或训练解除；test/buffer保持关闭。后续并行修复主结果证据闭环（forecast配对bootstrap收据、AL及R反演结果JSON、独立真值来源manifest、不可绕过的真实test release链），再由新鲜审计决定能否一次性开test。

### 2026-10-18 FAP三seed训练与validation独立复算完成

- FAP seed42/43/44依次由PBS单卡完成：`214745.tc6000`、`214746.tc6000`、`214747.tc6000`，均`exit_status=0`，分别使用node21一张A100，均跑完Niushou/Wangzhai/Mizhuang三站30 epochs、batch1536。每个run metadata绑定同一forecast pack SHA=`9421c0d5e19094454e5b94b98de797ac90248d3e5b56fdef6dacda339df359dd`及norm-reload trainer SHA=`fd15bf6f939cb81d09bd05e6fc889f88e53e8a9e40366d417fa195848caf5999`，并记录test payload reads=0；未读取test/buffer载荷。
- 三个FAP结果目录已拉取到本地`audit_receipts/forecast_fap_seed{42,43,44}_20261018/`。独立审计脚本`tooling_20261018/audit_hubei_forecast_fap_seed_20261018.py`逐站核对FA/FAC/FAP的有序(init,target,lead)键、truth和clear-sky数组完全相同，从保存预测重算RMSE/MAE/bias，并核验FAP元数据、pack/trainer hash及test/buffer计数；三个seed均通过。逐seed等站RMSE（FA/FAC/FAP，W/m²）：seed42 75.663/77.019/79.033；seed43 77.612/79.318/78.662；seed44 78.176/78.129/79.244。
- 三seed审计脚本`tooling_20261018/audit_hubei_forecast_fap_three_seed_20261018.py`对九个seed×站点组执行严格共支持/真值/clear-sky哈希核对，并按等站后等seed汇总；validation RMSE FA/FAC/FAP=`77.150/78.155/78.980`。10,000次配对目标日期cluster bootstrap：FAC−FA=`+1.005 W/m²`，95% CI `[−0.722,+3.780]`；FAP−FA=`+1.830`，95% CI `[−0.517,+4.778]`；FAP−FAC=`+0.825`，95% CI `[−2.657,+4.402]`。三组区间均跨0；FAC−FA的当前正文区间由本次新收据复算吻合，补齐了其预测包SHA及逐seed/station共同键、truth和clear-sky摘要。FAP点估计没有优于FA或FAC。该结果只说明当前湖北验证支持下未观察到额外AGRI容量分支收益，不能升格为独立test或确认性结论。
- 机器收据：三seed审计`audit_receipts/forecast_fap_three_seed_20261018/forecast_fap_three_seed_validation_recheck.json`；seed级收据分别位于各自forecast_fap_seed目录。当前PBS队列复核只剩无关CPU排队作业`214221.tc6000`，没有本项目运行任务。tracker HB09已更新为27头完成、test blocked。
- 新鲜独立审计仍判论文冻结FAIL、真实test NO-GO：FA/FAC当前配对CI与预测包收据现已闭环；仍需闭合AL与R反演主结果JSON/逐键来源、独立真值manifest及不可绕过的一次性release/inference链。下一步优先补可从现有train/validation产物完成的证据，再对test流水线做新鲜独立审计；期间不读test/buffer载荷，也不因FAP结果改变日期、样本或对照。
## 2026-10-18 匹配预报输入GHI三臂收口

- 完成FA/FAC/FAP三个种子×三个站点验证结果的共同键、真值、clear-sky值与支持核验；三臂共享预报输入包和trainer哈希，test/buffer payload读数为0。
- 三站等权、种子等权RMSE：FA 77.150、FAC 78.155、FAP 78.980 W/m²。单日目标日期块bootstrap（10,000次）：FAC−FA +1.005 [−0.722,+3.780]，FAP−FA +1.830 [−0.517,+4.778]，FAP−FAC +0.825 [−2.657,+4.402] W/m²；全部跨0。只有FAC−FA原有连续两日区间，不将其外推到FAP。
- FAP结构为预测AGRI主干特征32→4投影，参数量24,613，与FAC相差0.51%。训练为三种子×三站点、30 epochs、batch 1536。单卡profile只证明该配置可运行，不作性能结论。
- 已将三臂结构、预算与验证结果同步写入英文LaTeX和中文对照Markdown的实验设计、讨论与结论，并更新论文计划与研究发现。摘要不加入FAP；结论只说“未能分辨增益”，不称等效或显著变差。
- 独立review：forecast-COT/额外AGRI分支带来GHI预报收益的正向主张未获支持；验证结果属条件性validation，独立test仍NO-GO。密码信息未用于登录、命令或持久化；如仍有效，请更换。
- 英文稿已由`latexmk -pdf -interaction=nonstopmode -halt-on-error`编译成功，产物为`solar-cot-ifs/paper/main.pdf`（20页，1,035,537 bytes）；最终日志无未定义引用/文献键。存在少量既有overfull hbox警告，未阻止编译。一次初编因新增Unicode负号失败，已改为LaTeX负号后重编通过。
- 中英文对应段落已核对FA/FAC/FAP数值、三个配对差值区间、validation/test边界一致；FAP单日bootstrap与FAC既有两日区间明确分开。PDF文本抽取确认三臂数值和结论均进入成稿。
- 下一步：继续修订完整test evaluator与独立test-freeze审计，审计通过前保持test/buffer关闭；若之后要汇报FAP两日块敏感性，再单独运行预先定义的两日块复算，不从FAC区间类推。

## 2026-10-18 test evaluator one-use ledger 加固（synthetic only）

- 重新读取测试链FAIL/NO-GO审计与当前评分源码后，确认CLI允许通过`--ledger`改变一次性账本位置，且默认账本跟随输出文件名，换输出路径可能绕过同一次性限制。
- 修复`solar-cot-ifs/scripts/evaluate_hubei_forecast_heads_frozen_test.py`：移除`--ledger` CLI选项及函数级ledger路径覆盖，账本固定锚定在冻结split文件的同目录/同名旁侧文件（`<split>.hubei-test-once.ledger.json`），不再由score输出路径决定。第二次调用换输出文件也复用同一ledger并失败关闭。
- 更新合成测试`solar-cot-ifs/scripts/test_hubei_forecast_heads_frozen_test.py`：新增CLI无`--ledger`以及ledger路径锚定检查，并更新一次性与失败消耗测试。`python -m unittest -v test_hubei_forecast_heads_frozen_test.py`通过17/17；`py_compile`通过。仅为合成软件契约见证，未读取test/buffer载荷或任何真实测试预测。
- 当前评分源码SHA-256=`8bb1c009b1c8d536b202996df0f598f8cf9858bf3050b8ce0cf5281c576b01e7`；测试源码SHA-256=`3b157a35118a53453a86d7b6d8654f9ed9eaae3f9b6e7407ef26b1a6338646a5`。此处哈希只绑定本地源码文件，不构成服务器端冻结/签名证据。
- 独立test仍NO-GO：真实test pack builder、预测推理入口与inference receipt、实际checkpoint/normalization manifest校验、release receipt外部授权及fresh independent review未闭合。当前PBS快照仅有无关CPU作业`214221.tc6000`排队，湖北训练已完成，不重复训练或抢占未调度GPU。
- 下一步优先实现并审计“模型/归一化清单—冻结测试预测包—inference receipt—scorer”实际哈希绑定；随后再冻结FA/FAC/FAP是否均进入一次性test，并补独立标签/原始来源合同。期间test/buffer保持关闭。
