# 湖北独立COT实验执行表

## 2026-10-18：冻结评分器覆盖FA/FAC/FAP

- 按三seed验证结果与预先登记的最终范围，把synthetic-only冻结评分器的冻结臂从FA/FAC扩展为FA/FAC/FAP；主输出现对FAC−FA、FAP−FA、FAP−FAC分别计算配对目标日期bootstrap，并在三站共同键敏感性中给出同样的三组比较。release receipt必须精确列出三臂，否则在打开预测/真值载荷前拒绝。
- 本地 `py_compile` 通过，`test_hubei_forecast_heads_frozen_test.py` 17/17通过，`git diff --check`通过。评分器SHA-256=`6839bdba23114005748841810166a6d42a8ac44cb0f9005de8cbbcbe32e55ecb`；合成测试SHA-256=`bc6328c222bcf2d8648485dea9268d86f8f932d0e938dd2e00e15d17a9b02fc4`。
- 对`audit_receipts/test_truth_source_inventory_20261018/source_inventory.json`中7个输入重新做文件级校验，全部存在且SHA-256与清单一致（包括冻结split、时间支持表、8个月月档ZIP及3份独立工作簿）；核验仅读文件字节/元数据，没有解析或解码测试辐照度、AGRI、CPP/COT或预测值。冻结22个测试日期对应6147条时间支持记录；时间戳完整窗候选为牛首1620、王寨1642、米庄1521，三站共同目标980。它们只是时间完整性候选，不能当作已验证真值或推理键。
- 以上仅是合成软件合同验证，不是实际test评分，也不代表release gate通过。真实test/buffer继续关闭；仍需完成可追溯真值pack builder、实际checkpoint/归一化/训练与推理代码哈希绑定、端到端推理回执及新的独立test链审计。未提交PBS或访问真实test载荷。

## 2026-10-17 19:58 BJT：同期A/AC/AP三seed归一化重载复核完成

- 同一重建pack `e8c5…`、冻结split、同一训练器及train-only归一化下，seed42/43/44（PBS `214668/214669/214670`）均成功结束。三个站点、三臂验证键/真值/晴空值严格一致；checkpoint reload parity通过，test/buffer载荷读取0。
- 三站与seed等权RMSE：A 74.33、AC 70.99、AP 73.84 W/m²。AC−A = −3.34 W/m²，paired date-cluster bootstrap 95% CI [−5.80,−1.12]；AC−AP = −2.85 [−5.15,−0.72]；AP−A = −0.49 [−0.80,−0.26]。连续两日块敏感性三个区间也均未跨0。详见`SAME_TIME_THREE_SEED_VALIDATION_20261017.md`和收据`audit_receipts/same_time_three_seed_normreload_20261017/independent_three_seed_validation_audit.json`。
- 这是22个validation目标日期上的选模期证据，不是独立test。AP控制容量但不控制R监督/空间感受野等差异；不能宣称唯一物理因果。预测COT的FA/FAC验证仍为负向点估计，故同期增益不能外推为未来预报增益。test-freeze仍`FAIL / NO-GO`，test/buffer保持关闭。
- 同合同月份/太阳高度分层已按预测键重新复算，月别差异正负混合，太阳高度0–45°点差偏向AC而45–60°反向；无独立云况标签，不推导晴/阴类别。见`HUBEI_VALIDATION_SUBGROUPS_NORMRELOAD_20261017.md`。
- 冻结、误差无关的24案例清单已用新预测重绘，输出`validation_cases_normreload_20261017/`；只作站点GHI示例，缺测断线、seed阴影为样本SD，不是CI，也不是天气分类。

## 2026-10-17 test-freeze审计（覆盖下方旧状态快照）

- 当前test/buffer门禁：`FAIL / NO-GO`，继续关闭。最近独立审计快照：`audit_receipts/test_freeze_audit_20261017/EXPERIMENT_AUDIT.md`；新同刻A/AC/AP复现尚未纳入新的独立test-freeze复审。未读test/buffer载荷，未启动test PBS作业。
- 同刻A/AC/AP归一化重载复现（pack `e8c5…`）已由顶部新三seed审计取代：RMSE分别74.33/70.99/73.84 W/m²。旧训练器run的validation值仅作历史留档，不与重载复现混算。
- Forecast-GHI旧未保存归一化版汇总已由重载三seed复现取代：FA/FAC=77.150230/78.155269 W/m²，FAC−FA=+1.005039 W/m²；日期块95%区间[−0.721941,+3.779833]跨0。该三站、22验证日结果仅为validation选模证据，不支持宣称预测COT改善未来GHI。
- 已修订训练器：精确train-only normalization写入合同文件并hash绑定checkpoint，新增checkpoint reload validation parity。语法通过；这是未来重跑的新合同，既有run不被追认为具备该见证。
- 放行前仍须：将S折日期排除收据、实际配对bootstrap/逐lead汇总、normreload训练器与最新结果纳入统一冻结清单；完成fail-closed test builder与推理入口，并对当前评分器`solar-cot-ifs/scripts/evaluate_hubei_forecast_heads_frozen_test.py`开展独立审查（其6项合成合同测试已通过，但不能代替端到端dry-run）；冻结FA/FAC/FAP范围及报告目的；完成paper-claim审计与英文LaTeX/中文Markdown一致性核对；由新鲜独立审计重新评估是否允许一次性test。test/buffer保持关闭。

## 2026-10-17 08:13 BJT 状态增补

- HB01：按原工程UTC文件命名与BJT→UTC读入逻辑复核16个缺档的规范UTC `.npy`键；已知原预处理树和baseline包均未找到，旧的2+6个同数字候选不对应这些缺档。验收覆盖仍2,644/2,660，当前训练manifest不变，详情见`AGRI_GAP_RECONCILIATION_20261016.md`。
- HB08：实时qstat `2026-10-16 23:55:26 UTC`显示fold3/4同步验证训练`214526/214527`均R，Epoch1验证全局`val_loss`为0.5520809/0.5345694且各rank一致，Epoch1 best/last已落盘；队列3/3，不提交新作业，等待自然结束后复评分。test/buffer继续关闭。

## 2026-10-17 07:58 BJT 状态增补

PBS实时快照`2026-10-16 23:39:29 UTC`：fold3/4同步验证重训`214526/214527`仍R，账户非终态3/3（另有`214221 Q`）。两折Epoch0均完整286/286步并完成407条validation；全局`val_loss`分别为fold3 0.68824、fold4 0.66779，所有三个rank的best值一致，checkpoint已落盘。同步验证源码SHA匹配预期。两作业仍训练中，无最终退出/完成收据；full-validation独立复评分未提交，等训练成功结束和名额释放。test/buffer关闭。

## 2026-10-17 07:53 BJT 状态增补

PBS `23:35:06 UTC`快照：fold3 Epoch0 212/286（74%），fold4 223/286（78%），仍未到首个完整validation；作业`214526/214527`运行，队列3/3。407样本复评分脚本的时间戳日志门禁已修正，更新后本地/远端语法检查通过，远端SHA=`7f50e54d37b8d536e28915d4e1fcf16cd5ed93ddbd6d56c29f2124851784c06e`；未提交，等待训练正常结束及队列空位。test/buffer关闭。

## 2026-10-17 07:48 BJT 状态增补

PBS报告时间`23:30:49 UTC`时，fold3/4同步验证重跑仍在Epoch0，分别132/286（46%）与139/286（49%）；未有完整validation或可审计best checkpoint。PBS全账户3/3。单卡407样本full-validation rescore脚本已本地/远端通过bash语法检查，远端SHA=`93e03ca91d4c2d37c6dedb26fbc047edf66fd8c7dc00486e3dd95569e4be2b3e`；当前未提交，等训练成功结束并有资源名额再运行。test/buffer关闭。

## 2026-10-17 07:45 BJT 状态增补

PBS服务器快照时间`23:27:48 UTC`：`214526`/fold3 Epoch0 77/286 batch，`214527`/fold4 79/286；两作业仍R，队列3/3，暂无完整epoch验证值或可审计best checkpoint。等Luna检查完整validation与同步聚合后再推进OOF；test/buffer关闭。

## 2026-10-17 07:44 BJT 当前状态

同步验证修正版fold3/4重跑已进入训练：PBS `214526`为40/286 batch，`214527`为43/286；sanity validation通过，尚无完整epoch验证指标。两作业launch manifest记录模型SHA与预期同步验证源码一致，未见训练异常。PBS全账户3/3（另有`214221 Q`），暂无新增作业；等自然完成后统一复评完整407条validation，再放行OOF。test/buffer继续封闭。详见`PROGRESS_20261017.md`。

## 2026-10-17 07:41 BJT 当前状态更正

覆盖下方06:10训练快照：fold3/4旧PBS作业已结束，但rank-local val checkpoint选择不进入正式OOF。已建立隔离同步验证实现，只改validation epoch聚合为`sync_dist=True`；在原fold配置/日期/norm/seed上启动新run `214526.tc6000`（fold3，node23 GPU3/4/5）与`214527.tc6000`（fold4，node22 GPU1/3/4）。两作业刚获配资源，尚无sanity/完整validation指标。全账户PBS 3/3，不再提交；由Luna只读监控。AL控制已完成但validation非test，16个AGRI缺档本轮仍未找到原始HDF。见`PROGRESS_20261017.md`首节。

## 2026-10-17 07:35 BJT 当前状态

AL三种子九配置已完成并通过本地FP64键/真值/指标复算；三站、seed等权RMSE为A 74.361、AC 71.428、AP 73.894、AL 71.089 W/m²。连续两日块bootstrap显示AC−AL为+0.339，95% CI [−0.244,0.990]跨0；不能将AC增量归因成显式COT物理值优于R隐藏表征。结果仍只是validation，模型选模与结果集相同。FD-172真实`YYYYMMDD`路径上的16个精确HDF时次复查无命中，覆盖保持2,644/2,660。PBS fold3/4训练已结束，但选模指标为rank-local、checkpoint仅保留best/last，暂停正式OOF bank；先修正分布式验证聚合并重训/复核。PBS当前全账户仅1项排队；test/buffer关闭。完整数值和限制见`PROGRESS_20261017.md`、`findings.md`。

## 2026-10-17 06:10 BJT状态增补

FD-107两处已登记HDF源根按16个准确缺档UTC键定点复查，新增原始HDF命中0；正式可验收覆盖仍为2,644/2,660，8个格式兼容`.npy`候选不计入补齐。fold3/4 PBS作业`214517/214521`继续运行，账户非终态3/3；fold3最近完整验证Epoch8未刷新best，fold4 Epoch7刷新best。数据缺档不通过插值解决；五折S/R forecast-OOF QA前不开始FA/FAC。详见`PROGRESS_20261017.md`及`AGRI_GAP_RECONCILIATION_20261016.md`。

## 2026-10-17 05:58 BJT 状态更新

fold0 的 823 条 SimVP 预测AGRI OOF已完成，随后由折0专属冻结R生成823条 forecast-COT OOF。GPU烟测8条通过；全量bank独立QA确认keys与SimVP来源逐行一致，数组形状`[823,16,1,16,16]`、有限非负、day-mask关闭像元精确为0，范围`0–65.3454`。bank报告SHA-256=`6b53be3cc32d3e700d52ed80254cf2c220a14dcbddebb28ddebd73fa685a8d1e`，预测SHA-256=`bc25279366e43506781e0738ba226d71997545c3fcbce7b9b1f932ba8c670882`；test/buffer读取0，GHI标签未读。这是训练日期held-out OOF输入，不是COT精度或GHI技能结果。fold1/2的AGRI/COT OOF此前已通过审计；fold3/4仍训练，PBS全账户3/3，不新增作业。实时细节见`PROGRESS_20261017.md`。

## 2026-10-17 05:41 BJT 状态更正

下方原摘要和HB08表格行为此前快照，现由本段覆盖：fold1的797条及fold2的850条预测AGRI OOF bank均已审计通过；专属R分别完成且训练监督日期与对应S折外日期交集为0。fold2 forecast-COT 850条、fold1 forecast-COT 797条均已生成并独立核验：输出键与AGRI来源逐行一致、日期全部属于相应冻结折、数组有限非负，test/buffer读取0且GHI标签未读。fold0真正OOF索引823条，held-out日期与S/R训练日期均无交集，8条折外烟测通过；823条全量S预测目前在FD-201 GPU0运行，memmap已建立，尚无最终报告。fold3/4仍训练，PBS为214221(Q)、214517(R)、214521(R)，3/3，不提交新任务。下一步完成fold0 S OOF并审计，再生成其预测COT；保持test/buffer关闭。实时详情见`PROGRESS_20261017.md`首节。

2026-10-17更新。已回查原工程`build_label.py`和Direct-GHI归一化/评估链：瞬时总辐射字段直接作为GHI使用，工程操作量纲按W/m²处理，1300仅为归一化常数；传感器校准/供数平台字段字典仍未独立确认。新标签继续按真实分钟记录构造右闭15分钟窗口，不冒称旧5分钟均值。train/val扩展AGRI已纳入可取得的2644帧；按原产品日历每天95帧，28个冻结日期应有2660帧，已审计归档中仍缺16个合法时次（具体见`AGRI_GAP_RECONCILIATION_20261016.md`），未插补。共同pack 29,351个日间晴空样本。HB04/05三站A/AC/AP共27配置、每配置30 epoch训练完成，best预测指标已独立复算，checkpoint重载推理与保存预测最大绝对差≤0.000184 W/m²，测试/缓冲载荷读取0。B2共享日期bootstrap完成但当前仅为用于选模的validation证据，不能代替最终test。B3 OOF S fold0训练/validation复载审计通过，fold1训练/验证及797条OOF bank审计通过；fold2训练自然早停，Epoch6/8/9/11统一407条validation复评中Epoch6最低（normalized MSE 0.250026），850条fold2 held-out train-date OOF预测正在FD-201四卡分片运行；fold3/4仍在PBS训练。全折OOF银行、R预测COT及FA/FAC GHI头尚未完成，test/buffer载荷仍关闭。实时状态见`PROGRESS_20261017.md`；下方旧记录仅作历史，若与本段及执行计划冲突，以本段及执行计划为准。
权威方案：[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。

| ID | 阶段 | 任务 | 数量 | 状态 | 下一证据 |
|---|---|---|---:|---|---|
| HB00 | 数据 | 原始实测/CPP/grid及覆盖 | 三站 | PARTIAL | train/val标签质量、逐键时间支持和日间/晴空样本支持已核；仪器/字段单位元数据待补 |
| HB01 | 数据 | 湖北日期划分、标签/归一化 | 1套 | TRAINVAL_DATA_READY_TEST_CLOSED | 6–7月14日可取得的标签及AGRI train/val补充已进入冻结数据包；原始字段/旧5min与新15min窗口口径见`PROGRESS_20261016.md`；原44个差值中28个是UTC 17:00常规无产品时次，合法产品日历下仍缺16帧且已审计归档无命中；test载荷未读 |
| HB02 | R | 用修正版CPP重训湖北R并生成新split COT bank | 1个候选 | FULL_TRAINVAL_BANK_QA_PASS | seed42 R训练、目标环境真实batch/梯度/保存重载、FP64独立复算及单卡profile通过；best epoch12、val COT RMSE13.35694/R²0.43579（对检索参考，不是独立真值）。完整bank共20,206 train +1,855 val，GPU1推理；测试载荷0。逐行核对冻结manifest通过；COT支持配对和科学门禁见`cot_bank_full_trainval_v1/BANK_AUDIT.json`。 |
| HB03 | 适配 | 去两站硬编码、三站见证与profile | 3臂 | PROFILE_QA_PASS | 三站真实batch、有限梯度、参数量、共同支持与单卡profile已核；完整结果见`PROGRESS_20261016.md` |
| HB04 | 对照 | seed42 A/AC/AP | 9头 | TRAINED_QA_PASS | 同键保存预测、重载复算通过；test载荷0 |
| HB05 | 对照 | seed43/44 A/AC/AP | 18头 | TRAINED_QA_PASS | 三seed 27配置齐全；保存预测与checkpoint重载吻合；test载荷0 |
| HB06 | 评价 | 三站/省等权、共享日期CI、天气/案例 | 1包 | VALIDATION_GHI_CASE_FIGURES_COMMON_SUPPORT_QA_PASS_B2_FREEZE_PENDING | 2000次共享日期bootstrap、连续两日块敏感性、月份/太阳高度分层及三站486个共同目标时刻敏感性已算；结果仍仅validation且受best-epoch选择影响。AC低于A/AP，但AC−AL区间跨0，不能归因于显式COT值优于R隐藏表征。24个预选案例已按冻结manifest绘制三站GHI曲线（观测+A/AC/AP三seed均值、seed间SD），本地PDF/PNG、数据表、哈希审计见`validation_cases_b2_20261016_v2/`；图只表示描述性验证曲线，未把案例称天气类别。无独立CPP/CLP天气标签，B2最终freeze审计仍待完成；7月无val支持，test/buffer仍关闭。共同支持结果见`al_20261017/THREE_STATION_COMMON_TIME_SENSITIVITY_20261017.md` |
| HB07 | 归因 | 条件AL同R隐特征 | 9头 | TRIGGERED_AFTER_VALIDATION_PENDING | AC−A、AC−AP验证集共享日期95%CI均低于0，按预注册条件触发AL；结果仍受validation选模影响，尚非独立test证据。需同R隐特征、同支持、同三seed补9头后再讨论物理语义归因 |
| HB08 | 预报准备 | 五折S OOF预测AGRI及预测COT bank | 1套 | FIVEFOLD_FORECAST_OOF_PACK_QA_PASS_TEST_CLOSED | 五折全validation复载和OOF来源/键审计通过；正式训练pack为4,086条train序列，OOF折数量823/797/850/809/807，另有6,723条validation行；全pack共70,171行，test/buffer读取0。pack与train-only标签绑定审计已完成，见`PROGRESS_20261017_COT_SOURCE_REBUILD.md`及相应收据。 |
| HB09 | 预报 | FA/FAC/FAP三站三seed | 27头 | FA_FAC_FAP_THREE_SEED_VALIDATION_COMPLETE_TEST_BLOCKED | FA/FAC/FAP均完成三站三seed训练与预测重载/逐键复算。等站等seed validation RMSE：FA/FAC/FAP=77.150/78.155/78.980 W/m²；FAC−FA=+1.005（配对目标日期bootstrap 95% CI [−0.722,+3.780]），FAP−FA=+1.830（[−0.517,+4.778]），FAP−FAC=+0.825（[−2.657,+4.402]），三组区间均跨0。FAP点估计未支持额外AGRI容量分支改善forecast；当前也未显示预测COT增益。仅validation证据，不是独立test。完整键/真值/数组哈希和10,000次配对日期簇bootstrap见`audit_receipts/forecast_fap_three_seed_20261018/forecast_fap_three_seed_validation_recheck.json`；其中FAC−FA CI复算与正文现值一致。真实test链仍被fresh review判FAIL/NO-GO，test/buffer继续关闭。 |

批次执行不等于PBS任务个数；HB02实际作业`214178.tc6000`（GPU runtime）、`214179.tc6000`（单卡profile）、`214180.tc6000`（seed42 R）和`214181.tc6000`（train/val COT bank）均已结束，退出码均为0。最近一次全账户`qstat -u slfu`无未结束任务；以后每次qsub前仍须实时重查。湖南作业/结果独立保留。

## 2026-10-15 当前门禁

- 正式日期表：`frozen_split_20261015_v1/day_assignment.csv`，250 train / 22 val / 22 test / 44 buffer，SHA-256 `6ef8cbc7d2bb7d1b1d3defdfcba473cb171e06db0115c8206f7c289d8bad19c6`。R/S/H共享该日期成员。
- 全期31,257个卫星目标槽位中，Niushou/Wangzhai/Mizhuang分别25,015/24,593/24,180个有完整15分钟时间窗。test分别有1,620/1,642/1,521个完整时间窗；三站共同支持980/2,049。该统计包含昼夜，尚未应用日间mask，也未读GHI值。
- 4–5月旧标签结果不代替扩展合同；扩展全期train/val 15分钟标签已重建并与时间支持逐键对齐，日间/晴空主支持已算出。单位元数据与清洁pvlib运行环境仍是正式归一化门禁。
- R真实batch/反向、保存重载、FP64复算、单卡profile和seed42正式训练完成；现有COT bank仅为CPP配对子集，完整AGRI train/val bank尚未完成。GHI单位/时间语义和晴空运行环境尚未闭环，正式归一化及A/AC/AP GHI训练仍关闭；test GHI/CPP载荷保持关闭。
- 新增湖北R train/val pack CPU完整性审计：`r_path_audit_20261015_v1/PACK_QA.json`；train 3,178 / val 301，16通道输入和fixed_v2显式mask形状一致、哈希全部匹配、归一化train-only。`r_supervision_trainval.csv`显式区分R监督子集（train 2,811、val 259）和全CPP无效行（367/42）；这只影响R监督/选模，不减少之后COT bank的AGRI键，也不改GHI共同样本。
- 学院swc的A800目标环境已完成GPU runtime见证及seed42训练，训练产物已归档本地；bank覆盖范围不足已发现并降级标注，正在补完整AGRI映射。验证只对`cpp_16_fixed_v2`参考产品，不作独立真值精度主张。test CPP/GHI均未读取。

> 上述“完整bank尚未完成”等为较早快照，已由下方2026-10-15全量bank补齐记录覆盖；当前状态以新增段为准。

## 2026-10-15 全量COT bank补齐

- FD-201文件名索引按冻结`day_assignment.csv`生成全train/val清单：train 20,206、val 1,855；test只统计路径元数据（1,859条），未打开test载荷。清单SHA-256=`f6280abbbd4c5ae990e9f18d27ad0880be1a4a69adfc031d6c5d89734cddf479`。
- 冻结R checkpoint `1e6c58...f7d83`在FD-201空闲RTX 4090 GPU1完成全部22,061条AGRI推理（不是PBS任务），输出`[22061,1,16,16]`，有限非负，范围0–74.9189；`test_payloads_read=0`。完整bank在`experiments/hubei_r_runtime_gate_20261015_v1/cot_bank_full_trainval_v1/`。train COT为R训练输入上的域内推理，**不是OOF**；预测参考CPP仅为检索监督。
- 发现该R训练子集的`day_mask`均值1、标准差下限1e-6；夜间0若直接沿用z-score会变成−1,000,000并使前向溢出。全量bank推理对`day_mask<=0.5`像元使用中性归一化输入，并将COT填0作为“无日照/不在R反演支持域”哨兵；共2,857,609/5,647,616个像元。该哨兵不是夜间反演真值，正式GHI头必须同时使用几何/日间mask。
- 独立QA逐行验证bank与冻结AGRI manifest的时间、路径、split完全一致，bank数值有限非负，test边界保持关闭。再按train/val晴空日间GHI标签与bank时间键做**键/支持计数**（不读取GHI数值、不据此断言精确时刻语义）：Niushou train/val匹配8,013/766，Wangzhai 7,893/694，Mizhuang 7,800/735；三站同一目标时刻均有标签和AGRI/COT键的时次5,351。A/AC/AP仍须共享站点内同一支持键。
- 审计产物`cot_bank_full_trainval_v1/BANK_AUDIT.json`及`ghi15_daylight_clear_cot_common_support.csv`；审计脚本`solar-cot-ifs/scripts/audit_hubei_cot_bank_full_trainval.py`。B0来源表原“16×16 crop内行列”表头勘误为256×256 crop坐标，并补列本地16×16像元位置Niushou(8,6)、Wangzhai(9,8)、Mizhuang(7,10)；HDF行列仍直接来自grid_static映射。
- 按修正坐标已构造三站站点级3×3 AGRI/几何/冻结COT原始特征侧车：`station_3x3_features_trainval_v1/station_features.npy`，shape `[22061,3,17,3,3]`。独立QA核对66,183个中心COT与全幅bank完全相同、所有特征有限，night/daymask关闭处的COT零哨兵一致；无GHI标签值、无归一化、无test载荷。报告`FEATURE_REPORT.json`、`FEATURE_AUDIT.json`。
- 代表性的2025-08、2026-06、2026-07聚合工作簿只读属性与首行表头，没有观测值解码；表头区分瞬时总辐射与日累计，但没有单位/设备/区间边界元数据。JSON：`aggregate_header_audit_20261015_v1/aggregate_headers.json`。
- 下一步：继续追溯湖北供数平台/仪器字段单位和时间定义，并核验清洁pvlib版本与晴空可复现性；只有这些门禁闭合后才构造正式train-only归一化及运行A/AC/AP。此侧车不能被称为训练或指标结果。此前PBS `214181.tc6000`只是CPP监督子集bank，保留为历史产物，不覆盖。

## 2026-10-15 较早快照（已由上方“当前门禁”覆盖）

- HB00：已查明实际旧标签生成器和原工程W/m²合同；设备元数据补证仍单独记录。所查原始data16为28,231帧、最晚6月12日，实测最晚7月14日；继续找旧六月至七月preprocessed数据，不把这一个data16根当作全服务器库存。
- HB01：新增扩展候选日期表，226 train/20 val/20 test/40 buffer；保留4–5月已有归属，8月1日不分test。9项合成边界测试PASS；真实时间列支持、标签包和归一化待生成。
- HB02：已修正R沿用旧split的冲突，必须与最终GHI日期表一致；候选CPP路径计数train3,178/val301/test296/buffer535，只读元数据。
- 下一轮先定位旧交接`forecast_20260601_20260714_20260723/preprocessed/`的当前迁移路径、核验时间/空间/通道合同，然后完成全期时间列支持盘点；不能直接拿旧处理标签或forecast AGRI补真实输入。

## 2026-10-18 测试门禁加固

- 将合成测试评估器的一次性ledger改为固定实验路径，避免复制冻结split后得到另一份ledger；新合成测试覆盖复制split后的重复读取拒绝。
- 加入连续日历两日块配对bootstrap敏感性；主评分和三站共同支持均输出，且无相邻日期块时会明示不可用。
- 语法编译和评估器/指标两个测试模块通过，合计26/26。代码哈希及完整记录见`PROGRESS_20261018_TEST_GATES.md`。
- 该进展仅为软件门禁见证，不是test指标，也没有打开真实test/buffer载荷。真实独立测试仍FAIL/NO-GO；真值来源绑定、实际模型与推理回执哈希及新鲜独立审计仍未完成。湖北三站A/AC/AP与FA/FAC/FAP三seed训练已完成，本轮不重复训练。

## NEW_USER_OBSERVATION_SOURCE_QC_20261021

- 用户新增工作簿采集点天气数据_2026-07-15_2026-08-18.xlsx，SHA256=6fc4659bf382c21f5db87e27b17762acb8588929e4580502b7a3283c4e5ee300，全部三sheet只做来源/质量审计，未加载模型或评分。王寨46002条/35天、米庄33960条/26天至08-09、牛首46743条/35天；与既有冻结清单至07-14无交集。此处不据日期无交集替代历史接触审查。
- F列瞬时总辐射为GHI候选；典型间隔60秒但带秒且有长缺口。G/H散射/直接辐射三站全部0，不可作有效辅助观测。时区、W/m²单位、采样还是平台接收时刻待用户确认，尚未构造正式15分钟标签或新模型pack，不使用湖南权重评价。
- 新来源报告audits/new_hubei_ghi_20260715_20260818_20261021/REPORT.md（repo相对路径），完整源副本/逐行审计保留本地不纳Git。新时段AGRI来源仅完成少数精确路径metadata探测，缺少完整源绑定；不据此释放测试或训练。后续确认须绑定修正R和全链模型日期/哈希、共同键与一次性协议。本文仅登记新数据候选，不改既有湖北任务或结果；论文正文和旧test/buffer保持关闭。
