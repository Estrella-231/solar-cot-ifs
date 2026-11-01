# 未来预测 COT 的日期块 OOF-GHI 对照计划

## 2026-10-20 GHI 头多卡并行纠偏

用户指出可用GPU应转化为实际提速。此前单卡决定只基于小型IntervalHead的`DataParallel` profile（两卡17.22 steps/s、单卡146.51 steps/s），它证明单模型拆分后更慢，却没有检验不同随机种子独立训练并行。该结论不应外推为整个GHI头阶段只用一张卡。

现将待运行的GHI头PBS改为申请node22两张卡：seed42结构门仍单卡运行；若通过，seed43和seed44各自使用一张卡、独立输出目录并并行训练。两进程沿用既有未修改训练器、同一OOF pack、seed、样本顺序、三组配对初始化、30轮、batch1792及模型选择规则；合并器仅在两边完整且合同字段一致后复制逐样本预测/权重/指标并汇总manifest，随后执行原有配对审计。若主门未通过、触发AC50单因素支线，其seed42先训练，seed43/44同样分卡并行运行四组配对训练，合并后由原AC50审计器评分。合成合并见证覆盖三组主对照和四组AC50均已通过。该改动是运行调度并行，不使用DataParallel，不改变科学比较或预算。

2026-10-20 07:19 CST已完成服务器同步和替换：旧单卡held job `215044`从未运行，已撤销；双卡job `215045`已按`afterok:215042`重新排队，资源为node22两卡。替换前队列核验为3个未完成任务，撤销后确认≤2才提交，提交后仍为3/3；没有增加并发任务总数。服务器SHA清单、`bash -n`、Python编译和合成合并见证全通过。该新作业仍处于H，尚未分配GPU或实训。fold4 SimVP job `215036`当前运行两卡，到epoch9/step11100；OOF预测`215042`仍等待`215036`成功，随后才运行双卡GHI头。test/buffer仍关闭。小头两卡DataParallel profile旧数字保留为历史证据，但不再作为拒绝独立任务并行的理由。

## 2026-10-20 条件式 AC50 后续试验

固定五折 OOF A/AH/AC 主筛选完成后，当前 GHI head PBS 流程会先执行已登记的 AC−AH 门禁：若两站、日期块95%区间、+15–60分钟门均通过，则不运行AC50；若未通过，则在同一OOF pack上立即训练完整配对 A/AH/AC/AC50，并保存独立结果目录与日期块审计。AC50只把未来标准化COT异常乘0.5，其他输入与训练合同不变。这样不新增PBS任务/槽位，也不修改正在运行的SimVP/OOF基线。OOF shrink gate、训练器、审计器和合成见证通过服务器`swc`环境测试；原held GHI作业`215043`已替换为条件runner `215044`，仍依赖fold4 OOF预测`215042`。

## 2026-10-20 06:43 CST 最新状态

fold3 OOF预测AGRI→冻结R-COT bank已完成5271/5271，PBS `215041=C`且Exit_status=0；5271条唯一索引与fold OOF ID清单逐行一致，AGRI/COT数组shape、float32 dtype、所有array/index SHA均通过独立复核。对fold0–3的预测数组/index SHA均匹配各自manifest；未来真实AGRI、CPP、buffer与test读取标志均为false。fold4训练PBS `215036=R`，已到epoch7/step8900，global batch20；fold4预测作业 `215042=H` 依赖训练完成。完整GHI头作业 `215043=H` 已按 `afterok:215042` 排队，使用已存在的一卡流程，预计届时自动执行五折bank核验/合并、因果pack构建、A/AH/AC头训练和日期块配对审计。当前队列3/3。尚无全五折GHI输出与效果结论；继续以AC−AH为未来COT主对比，不打开test/buffer。

## 2026-10-20 06:38 CST 执行状态与算力决策

服务器队列 `215036=R, 215041=R, 215042=H`（3/3）。fold4固定12轮SimVP的2卡DDP训练到epoch7/step8600，global batch20；fold3的两卡预测AGRI→冻结R-COT bank为4832/5271，仍在生成。fold4预测作业按训练完成依赖挂起；fold0–2 bank已完成。本轮尚无完整五折bank、GHI A/AH/AC配对评分或增益结论，后续仍以 `AC−AH` 日期块bootstrap和两站短时效门判定。

多卡策略基于实测而非卡数：40GB A100上两卡用于维持SimVP global batch20。固定小型IntervalHead的合成profile显示单卡146.51 steps/s，两卡DataParallel为17.22 steps/s（0.118×）；因此不把两卡DataParallel用于GHI头。当前GHI训练实现保持单卡合同，若实测证明头阶段成为耗时瓶颈，再以同初始化、同epoch行序的独立A/AH/AC进程做配对吞吐profile，不能更改科学合同。当前pack因果见证的合成OOF元数据已按新provenance字段和数组/index SHA要求修复并通过；未读取项目载荷。测试集、buffer及未来真实AGRI/CPP推理输入继续关闭。

日期：2026-10-19。状态（22:17服务器复核）：折1已完成12轮；折0 `215028`从安全epoch2 checkpoint恢复后workers=6，推进到epoch3/step3500；折2 `215021`完成epoch1/step1333，进入epoch2/step1400。折0最近完整checkpoint仍为epoch2/step2584，折2已有epoch1 checkpoint。两项作业均继续运行，全账户未结束作业2个。两张A800分别训练独立折；GPU短采样波动，不能以单次空闲采样判停。node22 `/tmp`可用135GiB，两个活动缓存不得清理。尚无五折OOF预测、冻结R-COT bank或GHI A/AH/AC结果；test/buffer及未来真实AGRI/CPP推理输入保持关闭。完整筛选耗时须待后续预测和GHI头阶段首轮实测后更新，不把旧pilot速度当作本实验耗时。

噪声路径初审（22:17）：当前折训练没有AGRI/COT输入或目标的显式加性噪声和数据增强；无效AGRI像元经QA掩膜后置零，AGRI按train-only均值/方差归一化。模型普通dropout为0，但SimVP训练态含DropPath随机正则（概率由0.01递增至0.1），不可笼统称“完全无随机性”。未来预测AGRI在后续冻结R阶段会反归一化后直接按R统计量标准化，代码路径没有物理范围裁剪；这提示潜在的R输入分布偏移，但目前还没有本轮OOF样本的越界统计，不能把它当作噪声来源结论。折0从安全检查点恢复时会重置DropPath随机流，折内样本顺序保持`seed+epoch`；该折的随机训练轨迹与不中断训练不完全相同，需作为工程记录解释。

### 执行状态更新（2026-10-19 13:27 CST，服务器时钟）

首折PBS 214933已完成fold-only归一化和单卡profile，固定12轮训练于epoch 2、step 2000后约22分钟无新日志；只读现场显示主进程与四个worker的CPU/I/O计数未增、GPU利用率连续为0%、70.8 GiB仍被进程占用，3秒strace只见poll/futex等待。epoch 1的`last.pt`及优化器状态完整保存于12:56:51，含step 1292。基于该证据，只结束本项目的挂起作业214933，checkpoint仍在；恢复作业214946已提交并排队，固定12轮、从epoch 2续训至12。恢复代码记录中断前后代码哈希，并用确定性新dropout RNG继续，epoch sampler仍按seed+epoch；本次workers=0以绕开疑似DataLoader同步挂起。首折纯GPUprofile batch20为24.23 samples/s、峰值60.70 GiB；首轮完整输入25,839样本/2,440秒，约10.59 samples/s。fold1已通过顺序见证，13:43日志为14,000/38,454帧归一化，不触碰fold1。当前PBS 214221=Q、214945=R、214946=Q，未结束3/3。

214941与214944均在训练前因persistent-worker样本顺序不同而失败，没有生成fold1输出；已撤销persistent-workers路径。fold1改为每轮按`seed + epoch`重建DataLoader后，计算节点顺序见证通过，workers=6、batch20，fold1为train 26,172、heldout 7,181、bank OOF 6,430序列，帧交集0。fold1 PBS 214945当前在GPU ID 1（gpu2）做归一化；13:43日志已到14,000/38,454帧，仍未进入profile/train。首折214933经22分钟无CPU/I/O/GPU进展后结束；epoch1 checkpoint、优化器状态仍保留，恢复作业214946已排队，使用workers=0从step1292继续到12轮，需通过恢复metadata门禁后方可更新模型。当前PBS总数3/3；测试与缓冲载荷仍关闭。

## 问题与主张

当前 A/AH/AC/AC50 pilot 的上游 SimVP 对训练序列做了 in-sample 预测，而验证序列由 train-only 模型预测。三种子总 RMSE 中 AC−AH 的区间跨零。固定 AC/AC50 头的反事实审计又显示，清除或打乱未来 COT 会显著损害该固定头。两项证据合起来说明模型会使用对齐 COT，但尚不能证明单独训练的未来 COT 头有净收益；训练输入来源不匹配仍是待检验因素。

**主假设：** 若 GHI 头训练时接收由未见过相应目标日期的 SimVP 产生的未来 AGRI/COT，未来 COT 相对历史 COT 的增量会比当前 in-sample screening 更能泛化到使用全训练集 SimVP 产生的验证输入。

这是一项验证筛选。它不预设收益，不改变固定 SimVP/R，不把 CPP 当独立真值，不读取 test/buffer，也不修改论文正文。

## 证据与可行性

初始 M0 审计只读预测库索引，得到30,436条bank训练序列；M1随后将它们与正式月度 SimVP 训练清单逐行核对。正式清单含35,320条训练序列，其中1,967条24帧窗口跨越北京时间日期边界，须在五折实验中整体剔除，防止同一卫星帧同时进入训练折和留出折。余下33,353条按月分层分入五折；bank 所需30,436条全部属于可折分序列，且每条恰好留出一次。M1仅读取manifest与索引元数据，未打开影像或标签；不读取原validation、test或buffer。UTC帧路径先转北京时间后检查24帧均落在同一日期。实际生成的折统计如下：

| 折 | 上游 SimVP 训练序列 | 留出日期序列 | 其中bank OOF序列 | 训练/留出影像帧交集 |
|---:|---:|---:|---:|---:|
| 0 | 25,839 | 7,514 | 6,954 | 0 |
| 1 | 26,172 | 7,181 | 6,430 | 0 |
| 2 | 26,643 | 6,710 | 6,214 | 0 |
| 3 | 27,521 | 5,832 | 5,271 | 0 |
| 4 | 27,237 | 6,116 | 5,567 | 0 |

折按月分层确定性分配，映射、输入文件SHA和完整行数保存在 `results/future_ghi_oof_manifests_v1/manifest_build_audit.json`。较早的M0 v2审计仅评估预测库样本、未与完整官方训练清单联结；不得用其中的折大小估算正式训练。

## 实验组与数据合同

- **数据划分：** 按 `hunan_monthly_block_2d_buffer1d_seed42_v1` 官方 manifest 的全部train日期分五折；跨北京时间日期的1,967条24帧窗口在所有折中剔除。每折SimVP仅用其余四折的合格训练日期，bank中的留出序列不得出现在24帧训练窗口中。已生成的M1 manifest对各折实算0个影像帧交集。
- **上游生成：** 五折模型使用与当前月度湖南SimVP相同的架构、通道、损失、优化器和12轮预算，随机初始化；每折固定独立seed `42+fold_id`。该折归一化只使用fold-train AGRI帧；预测逆变换回物理AGRI后交给冻结R生成未来COT。R权重与R归一化沿用当前pilot。禁止以留出折做逐轮验证选模：训练固定12轮，OOF使用第12轮checkpoint，避免heldout日期进入训练选择。
- **配对输入：** 对每个GHI训练样本，三组均使用同一折OOF预测AGRI、真实可用的历史AGRI/COT、太阳几何、标签、站点和key。A不输入COT；AH仅输入历史COT；AC输入历史COT加同一OOF预测AGRI经R得到的未来COT。验证时三组统一使用现有全训练集SimVP forecast bank与同一冻结R；仅A/AH/AC分别训练的GHI头不同。
- **归一化：** SimVP fold内部采用fold-train AGRI统计；物理输出转换后，GHI头特征继续使用当前train-only head pack的固定AGRI/COT归一化。GHI标签、15分钟窗口、太阳几何、公共样本掩膜和站点映射均沿用当前pilot，不重新定义。
- **训练：** 先以seed42跑三组头作代码/数值筛选；若结果文件、keys和配对审计全部通过，再复用同一OOF bank训练seed43/44。各组使用完全相同初始化方式、样本顺序、优化器和更新预算。不得因验证结果调整COT缩放、厚度阈值或时效权重。

## 评价与停止门

- **主对比：** AC−AH，两站等权共同样本GHI RMSE，负值表示预测未来COT有增益。AC−A作为总输入对照，AH−A用于历史COT背景解释。
- **统计单位：** 目标北京时间日期块；三组、两站、三个GHI头seed共享同一批2,000次bootstrap重采样。另报两站、+15–60、+75–120、+135–240分钟、逐时效RMSE、MAE和bias。
- **进入完整三seed筛选：** seed42代码和manifest审计通过、OOF bank覆盖全部30,436个bank训练序列且无帧交集、标签与key配对率100%。
- **支持未来COT增量：** 三seed合并AC−AH点差为负，日期块95%区间上界小于0；四里和竺家各自总体点差均不为正；+15–60分钟的站点等权点差不为正。否则只报告筛选结果，不能称为两站稳定收益。
- **停止：** 任一折读到非train帧/标签、manifest或归一化来源不一致、模型训练预算与冻结合同不符、OOF序列缺失/重复、帧交集非零、共同样本key不一致时立即拒绝评分；候选未过预设门时不进行验证集调参，不开启test。

## 执行顺序和算力

| 阶段 | 工作 | 放行条件 |
|---|---|---|
| M0 | UTC→BJT、bank五折可行性元数据审计 | v2通过；30,436条bank训练序列全部唯一留出（旧折规模只作预审） |
| M1 | 关联官方完整train清单并生成折manifest | **已完成**：35,320官方train中1,967条跨BJT日期窗口剔除；33,353条可折分；30,436/30,436 bank行覆盖；五折帧交集均为0 |
| M2 | 完成fold-0固定轮次训练并保存epoch-12 checkpoint | profile通过（batch 20，峰值60.70 GiB）；fold0恢复点为epoch1/step1292，完整train cache已逐路径见证通过，复用缓存的恢复PBS已准备；队列名额空出且节点保留≥20 GiB后即可在独立卡上续跑，无需等待fold1释放缓存 |
| M3 | 完成五个相互独立的fold SimVP并生成OOF AGRI→R(COT)银行 | 按空出的PBS名额在独立A800并行运行不同fold；每折checkpoint、fold norm、预测银行及key manifest独立保存，保持种子/数据/训练预算不变，队列最多3个未结束任务 |
| M4 | 三seed A/AH/AC GHI头训练与FP64独立评价 | 共同key和日期块bootstrap通过；按停止门解释结果 |

五折训练合计133,412条训练序列，约为3.78个完整35,320序列的12轮训练预算（单折使用完整官方训练量的73.2%–77.9%）；另需OOF前向和冻结R推理。fold-0在一张80GB A800完成归一化及真实训练profile：batch 20吞吐24.23 samples/s、峰值显存60.70 GiB；batch 24吞吐22.77 samples/s、峰值72.81 GiB，超过85%安全门，因此选择batch 20。profile保存重载输出最大差为0，未读取test。固定12轮折0仍在node22一张A800训练；记录到的GPU采样利用率曾达100%、显存约70.9/81.9 GiB。214941及214944均在训练前因persistent-loader顺序见证失败，未生成fold1输出。当前代码已撤销persistent worker尝试，恢复每轮按`seed + epoch`新建DataLoader的fold0等价语义；接下来待计算节点前置见证通过后再训练fold1（单卡、batch 20、6 workers）。后续最多在PBS账户3作业总限制内使用额外卡；每次qsub前复核全账户队列。

实际训练前200步复核：step100→200用时146秒，即batch20约13.70 samples/s；同一时段分配A800监测约100%利用率、显存约65.5GB。profile不含完整DataLoader读取，因此其24.23 samples/s只用于硬件候选筛选，不能代替全训练吞吐。按每折约1,292步/轮外推，当前约31分钟/轮、12轮约6.3小时；以保存的epoch记录和真实墙钟更新此估算。

## 解释边界

若OOF训练后AC−AH改善，可说明训练期上游预测是否会影响GHI头对未来COT的利用；仍只是月度验证结果，不能代替独立test。五折SimVP训练数据约为完整模型的四分之三，因此OOF训练输入与全量模型验证输入仍有容量/样本量差异；R保持在全部train上冻结，不能据此证明R本身完全没有训练日期依赖。若筛选失败，结论是当前固定链路下没有观察到可靠的预测COT增量，而不是COT在物理上无用。

## 证据位置

- 初始bank日期折分审计：`results/future_ghi_oof_fold_feasibility_v2.json`，SHA-256 `bc6c2e94a4f34b88686678b8e15254c3aca185056532629e3e962222b9f2b827`；完整官方train联结后的权威M1结果见 `results/future_ghi_oof_manifests_v1/manifest_build_audit.json`。
- M1生成器及见证：`build_future_ghi_oof_manifests.py`、`test_build_future_ghi_oof_manifests.py`。
- 固定epoch训练器：`train_monthly_simvp_oof_20261019.py`；只构建train数据集的合成见证：`test_train_monthly_simvp_oof_20261019.py`。
- 通用单折PBS执行器：`run_future_ghi_oof_fold.pbs`，显式接收0–4折号、采用seed `42+fold_id`、拒绝覆盖旧norm/run，并对每折重复代码/manifest/test门禁；已通过远端`bash -n`，尚未提交后续折作业。
- 未来AGRI→冻结R OOF预测和合并：`predict_future_ghi_oof_fold.py`、`merge_future_ghi_oof_bank.py`；观测历史路径与OOF合并各有合成测试。预测程序只读8帧历史AGRI及未来太阳几何sidecar；未读未来AGRI/CPP。
- 原训练/验证GHI输入来源pilot：`results/future_ghi_cot_source_validation_audit_v1.json`。
- 固定头COT敏感度审计：`results/future_ghi_cot_counterfactual_v1.json`，其输入依赖性不可解读为单独的部署增益。





## 2026-10-19 22:31 status refresh: OOF input-shift gate and alternate-node capacity

Live `qstat -u slfu` at 22:31:38 CST showed only `215021=R` (fold2, elapsed 1:16:10) and `215028=R` (fold0 resume, elapsed 0:41:01), 2/3 unfinished. Fold0 was at epoch4/step4000; latest complete checkpoint epoch3/step3876. Fold2 was at epoch2/step1800; latest complete checkpoint epoch1/step1333. node22 `/tmp` had about 135 GiB free. GPU0/1 were assigned to our fold0/fold2 and sampled at about 96%/99% utilization with 70.8/66.1 GiB used; GPUs2/3 belonged to other PBS jobs. GPUs4–7 appeared idle in `nvidia-smi` but had no allocation to this project, so they are not available for direct use.

The requested frozen-R input-support diagnostic cannot yet be computed as an OOF result. `results/future_simvp_r_bank_v1/bank_manifest.json` describes a complete-count pilot bank but explicitly marks training forecasts `in_sample_train_forecasts=true`; its progress JSON still says BUILDING. `results/future_ghi_head_pack_v1/pack_manifest.json` is another pilot, not the five-fold OOF bank. Fold0 and fold2 are active, folds3/4 have not run, and no complete `future_ghi_oof_bank` exists. The frozen-R `normalization.json` contains train-only means/stds and counts but no empirical per-channel quantile limits. Required dependencies are the completed five-fold OOF forecast bank and empirical training-R input support statistics. No future-observed AGRI, test, or buffer payload was read.

Read-only scheduler/node inspection found node1/5/15 had no GPU query output. Nodes21/23 each had an A100 system and about 1.3 TiB free `/tmp`, with the shared project source path visible, but both had other PBS allocations and visible third-party GPU activity; their unassigned GPUs cannot be used outside a scheduler-approved allocation. They could be candidates for a later formally allocated fold if the scheduler grants resources. No job was submitted or changed.
## 2026-10-19 22:42 CST status: fold3 launched on node23

After the full pre-submit gates passed, submitted exactly one fold3/seed45/workers=6 job, `215030.tc6000`, with an explicit `node23:ppn=8:gpus=1` allocation. The pre-submit queue was `215021=R, 215028=R` (2/3). Fold3 manifest audit confirmed 27,521 train / 5,832 heldout / 5,271 OOF sequences, zero train-heldout frame intersection, zero test rows, `validation_test_used=false`, and 1,967 purged cross-BJT-date sequences; fold3 run/norm paths did not exist. node23 had 1,373,007,769,600 bytes free on `/tmp`, and shared source data was visible.

Only PBS GPU assertion/logging was adapted: fold3 requires and reports `node23-gpu/<id>` while existing folds still require node22. No model, data, loss, seed, batch-20, 12-epoch, or manifest contract changed. Updated PBS SHA-256: `e4ad590b19923694ff9132f3c3cc5627e226fce84910ac35b1e7f7cd3928e047`. Remote `deployment.sha256`, `bash -n`, py_compile, manifest-builder, train-only, and per-epoch loader-order witnesses passed; the job log reports `OOF_FOLD_PREFLIGHT_PASS 3`. Post-submit full queue is `215021=R, 215028=R, 215030=R` (3/3); `qstat -f` confirms `exec_gpus=node23-gpu/2` and `Resource_List.nodes=node23:ppn=8:gpus=1`.

At 22:42:18 the job had entered fold3 normalization with workers=6 and seed45. GPU2 was at 0 MiB/0% before profile, expected for this stage. Profile throughput and peak memory are not yet available. Its dedicated cache is `/tmp/slfu_future_ghi_oof_fold3_215030_tc6000`; fold0/2 caches are untouched. Heldout/test/buffer/future-observed AGRI payloads remain unopened.
### 2026-10-19 22:44 CST incremental progress

Full account queue remains `215021=R, 215028=R, 215030=R` (3/3). Fold0 reached epoch4/step4500 (latest complete checkpoint epoch3); fold2 reached epoch2/step2400 (latest complete checkpoint epoch1). Fold3 `215030` passed startup gates and is running train-only normalization with six workers on node23; log advanced from 1,000 to 3,500/40,286 unique train frames, about 2,500 frames in an 85-second interval (~29 frames/s, a short-window estimate and not training throughput). Assigned GPU2 remains at 0 MiB/0% before profile. node23 `/tmp` retains about 1.3 TiB free. Profile throughput and peak memory are still pending.
### 2026-10-19 22:46 CST incremental progress

PBS remains `215021=R, 215028=R, 215030=R` (3/3), so there is no fold4 slot. Fold3 node23 normalization reached 8,000/40,286 frames (~20%); it advanced 2,500 frames from 22:45:17 to 22:46:45, a short-window rate of about 27.8 frames/s. Assigned GPU2 remains at 0 MiB/0% before profile; node23 `/tmp` has about 1.3 TiB free. Fold0 reached epoch4/step4600 and fold2 epoch2/step2500. Normalization is progressing; profile throughput and memory remain pending.
### 2026-10-19 22:50 CST incremental progress

The full queue remains `215021=R, 215028=R, 215030=R` (3/3). Fold3 train-only normalization reached 16,000/40,286 frames (~40%), +2,500 since 22:49:30 (~33 frames/s short-window rate). GPU2 is still at 0 MiB/0% before profile; node23 `/tmp` has about 1.2 TiB free. Fold2 log reached epoch2/step2600; the latest sampled fold0 log is epoch4/step4700. No fold4 slot is available; continue monitoring for normalization completion and queue changes.
### 2026-10-19 22:52 CST incremental progress

Queue remains `215021=R, 215028=R, 215030=R` (3/3). Fold3 normalization reached 20,000/40,286 frames (~50%); GPU2 has not entered profile, and node23 `/tmp` has about 1.2 TiB free. Fold0 reached epoch4/step4800. Fold2 completed epoch2/step2666 (`train_loss_mean=0.2067571868484215`, `heldout_payload_read=false`) and entered epoch3/step2700. No queue slot is available for fold4 yet.
### 2026-10-19 23:02 CST stage transition

Fold3 `215030` completed train-only normalization (`40,286/40,286`, `NORMALIZATION_DONE`) and is entering profile. Normalization took about 20 minutes; no heldout/test payload was read. At the stage-transition sample GPU2 was still at 0 MiB/0%; profile metrics are pending. node23 `/tmp` has about 1.1 TiB free. Fold0 `215028` completed epoch4/step5168 (`train_loss_mean=0.17035663505041562`, `heldout_payload_read=false`) and entered the next training epoch; fold2 `215021` reached epoch3/step3000. Full queue remains 3/3, so no fold4 slot yet.
### 2026-10-19 23:04 CST profile gate deviation: fold3 is not a formal batch-20 fold

`215030.tc6000` is allocated to `node23-gpu/2`, an NVIDIA A100-SXM4-40GB (40,536 MiB). Profile results at 23:03:48: batch4 20.196 samples/s at 12.261 GiB peak; batch8 21.522 samples/s at 24.370 GiB peak; batch12 OOM; reload max delta 0; `test_payloads_read=0`. The profiler selected batch8. Source inspection confirms training reads `profile.json.selected_batch`, and `training/run_metadata.json` at 23:04:52 records `batch=8`, seed45 and 12 epochs. This violates the requested batch20 contract; quarantine this run as a hardware profile/non-formal fold and exclude it from formal OOF scientific comparisons. Heldout/test/buffer/future-observed payloads remain unopened. The process has entered `--mode train`, but no `TRAIN_STEP` or epoch checkpoint marker was present at 23:04:52. Queue remains `215021=R, 215028=R, 215030=R` (3/3); fold0 completed epoch4/step5168, fold2 was around epoch3/step3100.

Do not cancel the active job: project instructions prohibit stopping existing PBS jobs, and the fold3 run directory/run metadata now exists so the executor refuses an overwrite; cancellation would leave cache/partial artifacts without a safe original-path rerun. Preferred compliant path is a scheduler-granted 80GB A800 and batch20 (prior A800 profile peaked around 60.70 GiB). A two-GPU/global-batch20 or gradient-accumulation route would require separate approval, implementation, and witnesses for optimizer updates, sampler grouping, and stochastic regularization equivalence. A single 40GB A100 cannot meet batch20; batch12 already OOMed.
### 2026-10-19 23:06 CST: non-formal run and batch-20 resource path

Fold3 `215030` has started epoch1/step100; `run_metadata.json` records batch8, seed45, 12 epochs, and heldout/test flags false. Its allocated node23 GPU2 is an A100 40GB; current sample is 27,764/40,536 MiB and 100% utilization. Treat this as a batch8 hardware/profile training run only and exclude it from the formal fold3 OOF comparison. Queue remains 3/3; fold0 reached epoch5/step5300 and fold2 epoch3/step3200.

GPU queue ACL lists node21/22/23. Nodes21 and 23 have A100 40GB cards and about 1.3/1.1 TiB free `/tmp`; node22 has A800 80GB cards but only 135 GiB free. Active node22 fold0/fold2 caches are about 180/185 GiB and must remain untouched. Once one job is terminal and its dedicated cache is eligible under the authorized terminal-cache cleanup procedure, freeing that cache would raise node22 free scratch to roughly 315 GiB or more, exceeding the >260 GiB reserve. Then, with an account slot open, request one node22 A800 for a formal batch20 fold. A single 40GB A100 cannot support batch20 (fold3 batch12 already OOM); two A100s with global batch20 would need new DDP/gradient-accumulation implementation and optimizer-update, sampler, and DropPath equivalence witnesses.

215030 was not cancelled: the user instruction prohibits stopping existing jobs, and it has already entered step100 with run metadata written. Preserve the run and isolate it as a non-formal batch8 experiment; do not overwrite its fold3 output or include it in the formal OOF bank. Queue is full, so no fold4 job was submitted.
## 2026-10-19 implementation note: batch-20 equivalence

Gradient accumulation on a single 40GB card is not an equivalent substitute here: the trainer updates once per batch and clips gradients at each update, while SimVP uses BatchNorm2d and per-sample DropPath. A two-card DDP path would need SyncBatchNorm and explicit witnesses for global sample grouping/tail handling, masked-loss normalization, update/LR counts, clipping, and DropPath RNG semantics. Keep the formal batch-20 path on a scheduler-allocated 80GB A800 with the scratch gate; the active node23 batch-8 attempt remains a quarantined hardware/profile run. No changes to the protocol are authorized by this assessment.

## 2026-10-19 static source note: augmentation and DropPath

The current Hunan dataset loader applies train-only normalization and zeros invalid QA/nonfinite pixels; it contains no random-noise, flip, or crop augmentation. The Triad model sets drop=0.0 and drop_path_prob=0.1; SimVP uses BatchNorm2d and per-sample DropPath masks with rates from 0.01 to 0.1. This supports only a future single-factor DropPath ablation candidate after the fixed OOF baseline is complete. It does not show that noise or DropPath caused lost gains and does not alter the current baseline contract. Source SHA-256: hunan_data.py f4b4aab55101bde575f474d5daa4a184466c16399b401c2ac67bf666c5f28b91; triad_models.py 9273447d357bad92bcf3cd976b65fb4c440187da456fb56d5f36ded349c681eb; simvp.py 2e0d280f7962a77f0c9b7e4cc817fed21d142c1436a12eb858e3234485313d16.

## 2026-10-19 23:27 CST monitoring checkpoint

Fold2 wrote epoch 3 / step 3999 (	rain_loss_mean=0.1823841327911736, heldout_payload_read=false); fold0 latest complete checkpoint remains epoch4/step5168. PBS remains 3/3 (215021, 215028, 215030 all R). node22 /tmp has 135 GiB free, below the 260 GiB gate; fold0/fold2 caches remain untouched. The node23 fold3 batch8 attempt remains quarantined. No test/heldout/buffer payloads were read.

## 2026-10-19 23:29 CST monitoring update

Fold0's latest durable checkpoint is still epoch4/step5168 (written 23:00), but a 75-second sample found its main process and six workers active (workers about 77-94% CPU); allocated A800 GPU0 held about 70.8 GiB and sampled at 100% utilization. Treat as ongoing slow epoch/compute, not a confirmed hang. Fold2 completed epoch3/step3999 at 23:26 (	rain_loss_mean=0.1823841327911736, heldout false), with workers active. Quarantined fold3 batch8 job 215030 completed epoch1/step3441 (	rain_loss_mean=0.2925220619658681; metadata batch8, test_payloads_read=0). Queue remains 3/3; node22 /tmp 135 GiB with fold0/fold2 caches retained. No test/heldout/buffer payloads read.

## 2026-10-19 23:35 CST monitoring update

Fold0 resume job 215028 wrote epoch5/step6460 (	rain_loss_mean=0.15993890381957354, heldout false, resumed from epoch2). Fold2 remains epoch3/step3999 (	rain_loss_mean=0.1823841327911736, heldout false). Quarantined batch8 fold3 job 215030 remains epoch1/step3441 (	rain_loss_mean=0.2925220619658681, metadata test_payloads_read=0). Queue is 3/3. Node22 /tmp has 135 GiB free with fold0/fold2 caches retained; no slot or scratch gate for the formal batch20 fold3 attempt. No test/heldout/buffer payloads read.

## 2026-10-19 23:39 CST fold2 epoch-boundary diagnosis

For job 215021 (fold2), step3999 remains the last durable epoch boundary. From 23:36:19 to 23:38:51, PBS remained R on node22 GPU1; the main PID and all six workers persisted at roughly 74-100% CPU. Two workers' /proc/<pid>/io rchar increased about 14.6GB and 15.3GB in a 59-second sample; GPU1 held about 66.1 GiB and utilization varied from 0-100% across samples. This supports ongoing epoch4 data loading/compute and does not confirm a hang. PBS Output_Path is /dev/null, so no durable TRAIN_STEP print is available; latest saved marker remains epoch3/step3999. No intervention or prohibited payload access.

## 2026-10-19 23:44 CST fold2 follow-up

At 23:44:37, fold2 215021 remains R; epoch3/step3999 is still the latest durable marker. Main PID50548 and six workers persist at about 69-100% CPU. Two sampled workers' rchar rose about 100.8GB and 104.0GB since 23:37:52; allocated GPU1 held about 66.1 GiB and last utilization sample was 66%. This supports active epoch4 data/compute, not a confirmed hang. PBS stdout is /dev/null, so printed TRAIN_STEP markers are not persistently available; do not infer an unseen step. Queue remains 3/3; node22 scratch 135 GiB and caches retained. No prohibited payload access or intervention.

## 2026-10-19 23:52 CST fold0 activity check

For resumed fold0 job 215028, latest durable checkpoint remains epoch5/step6460 (23:35). Between 23:51:14 and 23:52:36, job remained R on node22 GPU0; main PID61574 and all six workers persisted at about 70-100% CPU. Three sampled workers' rchar increased about 18-19GB each; allocated GPU0 held about 70.8 GiB and utilization varied from 0% to 98%. One worker briefly showed rwsem D-state then returned R. This supports active epoch6 read/compute, not a confirmed hang. PBS stdout is /dev/null, so no persistent per-step log is available. Queue remains 3/3; node22 scratch 135 GiB and active caches retained. No intervention/prohibited payload access.

## 2026-10-19 23:54 CST monitoring update

Quarantined batch8 fold3 job 215030 wrote epoch2/step6882 at 23:52 (	rain_loss_mean=0.18869523024059284, heldout false). Metadata remains seed45, batch8, 12 epochs, 	est_payloads_read=0; PBS R on node23 GPU2, about 27.8/40.5 GiB and 100% utilization; node23 /tmp about 1.1 TiB. Do not include this run in formal OOF. Full queue remains 215021/215028/215030 all R (3/3); node22 /tmp135 GiB; active fold0/fold2 caches retained. Latest formal checkpoints: fold0 epoch5/step6460; fold2 epoch3/step3999. No slot/scratch gate, cleanup, or submission. No test/heldout/buffer payloads read.

## 2026-10-19 23:56 CST cross-date queue check

Cluster clock remains 2026-10-19 23:56 CST although local date has crossed into Oct 20. Full queue is still 215021=R (fold2/node22 GPU1), 215028=R (fold0 resume/node22 GPU0), 215030=R (quarantined batch8 fold3/node23 GPU2), 3/3. Latest durable markers remain fold0 epoch5/step6460 (23:35), fold2 epoch3/step3999 (23:26), nonformal batch8 fold3 epoch2/step6882 (23:52). Node22 /tmp has 135 GiB free with active fold0/fold2 caches about 180/185 GiB retained; node23 /tmp about 1.1 TiB and batch8 cache about 190 GiB. No terminal formal job, queue slot, or scratch gate; do not clean or submit. No prohibited payload access.

## 2026-10-19 23:57 CST PBS log-path correction

Correction: despite `#PBS -o /dev/null`, both active scripts execute `exec > >(tee -a "$LOG") 2>&1`, so durable logs do exist. Fold2 script SHA256 `e4ad590b19923694ff9132f3c3cc5627e226fce84910ac35b1e7f7cd3928e047`; fold0 resume script SHA256 `8045ef6976af525c8f75cb33242b7453ec3c6dc94652369671a2ac0d14e79d4c`. At 23:57:17 cluster time, queue remains 3/3. Latest log markers: fold2 epoch4/step5100 (log mtime 23:56:44); fold0 resumed epoch6/step7200 (mtime 23:55:02); quarantined batch8 fold3 epoch3/step7500 (mtime 23:57:03). Node22 scratch remains 135 GiB, active caches retained; node23 about 1.1 TiB. No terminal/slot/scratch gate, cleanup, or submission. No test/heldout/buffer payloads read.

## 2026-10-19 23:59 CST five-fold training status

Fold1 is a completed formal batch20 model: metadata seed43/12 epochs/batch20; 	raining_complete.json reports COMPLETE_OOF_SIMVP_FIXED_EPOCH_TRAINING, 15,708 steps, epoch_012.pt, heldout=false, test_payloads_read=0. The epoch_012.pt SHA matches the completion receipt: 13e120c5cbe2e937ceba6334ffee160c4df6d193c0256eb3ca339094b4c02616; log contains OOF_FOLD_FIXED_EPOCH_COMPLETE_PASS 1. It is ready for its fixed-manifest OOF prediction generation; the five-fold bank is not complete.

Active log steps: fold0 215028=R, seed42/batch20, epoch6/step7300 (log mtime 23:57:39); last full epoch took 2057.8s (~34.3min), with 1292 updates/epoch, yielding ~12min in current epoch plus ~206min for six remaining epochs (~3h38 total). Fold2 215021=R, seed44/batch20, epoch4/step5100 (mtime 23:56:44); last full epoch took 2153.8s (~35.9min), with 1333 updates/epoch, yielding ~6min current plus ~287min for eight remaining epochs (~4h53 total). These are linear estimates from the most recent completed epoch only. Quarantined fold3 215030 is seed45/batch8 (nonformal), epoch3/step7800; its latest full epoch took 1434.1s (~23.9min), rough current+remaining estimate ~3h53, but exclude it from formal OOF.

No formal fold0/2 terminal state; formal fold3 batch20 rerun has not started; fold4 directory is absent. Queue is 3/3. Node22 /tmp has 135 GiB free with active fold0/fold2 caches about 180/185 GiB retained. After a formal job is terminal, verify its receipt/checkpoint and only then clean that job's own cache; submit formal fold3 only if node22 reaches >=260 GiB free and fresh qstat is <3, using an independent output path on A800. Fold1 checkpoint is ready for OOF prediction but does not mean the OOF bank is complete. No cleanup/qsub or prohibited payload access.

## 2026-10-20 00:02 CST: low-sun forecast-COT noise hypothesis

Live PBS and durable logs at 00:02 CST: queue is full (215021 fold2/node22 GPU1, formal batch20; 215028 fold0 resume/node22 GPU0, formal batch20; 215030 node23 GPU2, batch8 and quarantined from the formal OOF bank). Durable log steps are fold2 epoch4/step5300, fold0 epoch6/step7400, and quarantined fold3 epoch3/step8200. Fold1 formal seed43/batch20/12-epoch checkpoint is complete and receipt-verified; fold4 has not started. No complete OOF prediction bank or downstream OOF GHI comparison exists yet. No heldout/test/buffer/future-observed-AGRI/CPP payloads were read.

A read-only analysis of the existing in-sample train/validation forecast-COT input-shift CSV found weighted mean absolute 15-minute COT changes of 0.880/0.899 in the sun band, 4.492/4.635 in low sun, and 7.607/7.510 at night (train/validation). The validation GHI head pack contains 66,590 rows: 21.63% low-sun (cosSOZ<0.3), 77.66% sun, and 0.71% night. The low-sun rows have mean GHI 49.396 W m-2 versus 325.761 W m-2 in the sun band. The existing AC implementation passes all three normalized COT context fields to the head; it does not gate forecast COT by solar geometry. The fivefold OOF experiment remains the source of any generalization claim; this older distribution diagnostic is in-sample for train and is not a COT-error or GHI-benefit score.

This is a targeted noise/stability hypothesis, not a conclusion: low-sun forecast-COT changes are about five times the sun-band changes, which could reflect real cloud evolution, low-sun retrieval sensitivity, or upstream forecast artifacts. After the fixed OOF A/AH/AC baseline and its key/provenance audits complete, run one separate paired follow-up arm: retain observed-history COT unchanged, but set only forecast-derived COT pixels with target-frame cosSOZ<0.3 to the train-standardized neutral value (zero). Keep the same OOF bank, samples, head initialization/order, seeds, optimizer, epochs, labels, and scoring; compare the gated arm against AC and AH with the existing date-block bootstrap, both stations, and the short-lead gate. Do not use the current in-sample pilot to claim benefit or alter the active fixed baseline. PBS is full, so no new job is submitted.

## 2026-10-20 00:11 CST: low-sun harm is not confirmed by paired pilot strata

To test whether the large low-sun COT changes coincide with GHI degradation, the saved validation predictions from the older in-sample forecast-COT pilot were joined to the exact validation-pack keys and stratified by target cosSOZ (same rows/truth for AC and AH). Across seeds 42/43/44, AC-minus-AH RMSE differences in low sun were Sili [-0.049, +0.911, -0.261] W m-2 and Zhujia [+0.685, -0.106, -1.382] W m-2. Sun-band differences were also mixed: Sili [-1.169, -2.542, +1.258] and Zhujia [-0.535, -1.467, +2.962] W m-2. Night strata contain only 218 Sili and 256 Zhujia rows per seed. This descriptive, non-independent, older in-sample pilot does not show that low-sun COT is a consistent cause of GHI degradation; it does not estimate date-block uncertainty and cannot establish benefit or harm.

The high low-sun 15-minute COT variation remains a retrieval/forecast stability clue, not proof of injected noise or the reason the GHI head fails. Do not prioritize or claim the proposed low-sun gate as a fix based on these rows. Complete the fixed five-fold OOF baseline first; then use its own paired GHI errors and OOF COT reference diagnostics to decide whether a low-sun gate, lead-dependent attenuation, or another single-factor noise-control experiment is justified. Active OOF samples/model contracts remain unchanged.

## 2026-10-20 00:26 CST：GPU allocation correction

用户明确允许使用多卡。实时核验后，账户队列此前为3/3：正式fold0 `215028` 与fold2 `215021` 各申请 `gpus=1`，均在node22 A800上训练；另有node23 `215030`，因实际profile只支持batch8而偏离正式batch20合同，已按用户先前“停掉”要求qdel并核验为C，不纳入正式OOF。node22本次nvidia-smi抽样显示GPU0/1由正式作业占用且100%利用，GPU2/3有其他作业活动，GPU4–7当时0%/约2MiB；`/tmp`仍只有135GiB可用，fold0/fold2活动缓存保留。PBS当前只剩两项正式作业运行。

资源利用不充分的直接原因是我按单GPU写了提交资源（`nodes=...:gpus=1`）；训练实现也是单进程 `.cuda()`，没有DDP。故不能给运行中作业热加GPU。多卡授权应落实为经过语义核验的后续多卡启动方案，不应以改变单折有效batch、样本顺序或seed来换吞吐。当前正式折继续按固定batch20/12轮运行；不在活动缓存期间追加node22工作。新提交前仍须满足全账户PBS<3及node22临时盘>=260GiB。下一轮须优先实测并审计多卡方案的global batch=20、每轮样本覆盖/顺序、更新数、checkpoint恢复与逐卡分配，再用于正式fold；否则先以独立fold并行方式用卡，保持每折合同不变。

## 2026-10-20 00:33 CST：启动fold1 OOF推理

GPU资源纠偏后，停掉的batch8隔离作业`215030`保持终态C；两个正式训练`215021` fold2/seed44 与 `215028` fold0/seed42仍在node22分别用GPU1/GPU0，日志在00:29分别推进至fold2 epoch5/step6100和fold0 epoch7/step8300。全账户队列2/3时，用释放的槽位提交已完成fold1/seed43的OOF推理`215032`，PBS分配node22 GPU4；训练折没有被中断。

推理前核验fold1 fixed-epoch checkpoint `epoch_012.pt`及`training_complete.json`、manifest：6,430个OOF ID、7,424张唯一历史AGRI帧、test序列数0、训练test载荷读取数0，推理输出目录此前不存在。按每帧约4.5MB估计节点缓存约32.6GiB，node22 `/tmp`有135GiB可用。先发现SHA清单里通用训练PBS的旧哈希与当前已记录的node23兼容调度脚本SHA不一致；对照当前代码哈希与此前工程记录后，仅修正SHA清单，不改变训练逻辑；完整`sha256sum -c`、bash语法、Python3.10编译、预测测试及`OOF_FORECAST_CAUSAL_PATH_WITNESS_PASS`均通过。首次错误提交`215031`仅因qsub不支持位置参数、作业以exit_status=2退出，未产生预测文件；已将预测PBS改为兼容`FOLD_ID`环境变量，并重新通过全部完整性门。

`215032`当前运行中，分配GPU4，已写入`OOF_FORECAST_PROGRESS 16/6430`。推理器只读过去8帧AGRI和未来几何侧车，经冻结R反演未来预测AGRI的站点patch COT；未来真实AGRI、CPP、test/buffer载荷均关闭。输出完成前不评分、不称为GHI收益。node22空闲GPU在作业申请前确有4–7，但当前训练代码本身不是DDP；本次先将空闲卡用于独立OOF折推理，后续训练多卡方案仍需守住batch/样本顺序与checkpoint语义。

00:33 CST现场增量：全账户`215021` fold2、`215028` fold0、`215032` fold1 OOF推理共3/3。正式训练日志推进至fold0 epoch7/step8600、fold2 epoch5/step6400；OOF推理从16推进到240/6,430，`progress.json`仍为BUILDING，未来实况AGRI/CPP读取标志false。node22 `/tmp`余量由135降至134GiB；GPU4分配给OOF预测，GPU0/1为训练折。推理耗时预估需观察后续斜率，当前不判失败、不重启。

00:35 CST核验：`215032`仍为R，fold1 OOF推理推进到512/6,430（约8%），完整bank尚未完成；fold0日志到epoch7/step8700、fold2到epoch5/step6500，两者继续运行。全账户正式活动作业3/3；node22 `/tmp`剩余132GiB，缓存未清理。当前没有GHI收益结果，后续等fold1产物完整性核验，再接续其他fold推理与GHI训练评价。

00:36 CST复核：预测作业`215032`在运行4分56秒时推进至656/6,430，`progress.json`保持BUILDING，因果来源标志仍为future_observed_agri_read=false、cpp_read=false。折0、折2仍为R，最近持久日志分别step8700、step6500；尚未看到轮次完成或终态。账户队列3/3，node22 `/tmp`约132GiB可用。按当前预测斜率只能粗估约45–50分钟完成fold1 OOF推理；该时长会随文件缓存和R推理变化，不视作保证。无需干预，继续监控同一作业。

00:37 CST现场复核：OOF作业`215032`运行6分21秒，持久进度864/6,430（约13.4%），作业仍为R、GPU分配持续有效；进度较00:36的656有增长。折0、折2仍为R，节点22临时盘约130GiB。当前按已观测斜率的粗略剩余时间约40分钟，I/O波动可能改变耗时；没有失败标记，保持该进程运行，不重启。test/buffer及未来真实AGRI/CPP读取标志仍false。

00:38 CST核验增量：fold1 OOF作业`215032`仍R、GPU4，`progress.json`从864增至1,040/6,430（16.2%）；PBS已运行约7分钟。fold0 `215028`和fold2 `215021`也有新训练记录，分别到step8800与step6600。节点22 `/tmp`剩余约130GiB；输出仍BUILDING，`future_observed_agri_read=false`、`cpp_read=false`。当前只确认管线推进，尚无COT/GHI评分或收益证据。

00:39 CST实时复核：折1 OOF推理`215032`推进至1,184/6,430（18.4%），PBS仍R、GPU4；test/buffer/future observed AGRI与CPP边界标志仍false。正式训练fold0 `215028`到step8800，fold2 `215021`到step6600，仍R。全账户3/3，node22 `/tmp`余量129GiB；没有完整银行或下游GHI结果。本轮只确认实时推进，继续保留固定合同与活动缓存。

00:40 CST现场复核：fold1 OOF预测`215032`推进至1,344/6,430（20.9%），状态BUILDING，尚未完成。fold2 `215021`完成epoch5/step6665并写入epoch记录，`heldout_payload_read=false`；fold0 `215028`推进到step8900。三项作业仍3/3，node22 `/tmp`余128GiB。没有完整预测银行、GHI头或收益分数；不把单折训练loss当成预测收益。test/buffer/future observed AGRI/CPP仍未访问。

00:41 CST实时状态：OOF推理`215032`推进至1,488/6,430（23.1%），仍在node22 GPU4运行；进度连续增长。fold0 `215028`仍运行step8900附近，fold2 `215021`进入epoch6并到step6700；全账户仍3/3，node22 `/tmp`余127GiB。推理边界标志仍未读取未来真实AGRI/CPP/test/buffer。当前完成的只是OOF预测子集生成，等待全量及收据后继续配对GHI验证。

00:42 CST现场复核：OOF预测作业`215032`仍R，完成1,648/6,430（25.6%）；与00:41核验相比由1,488继续上升。折0继续训练，折2在epoch6，三项作业占满账户3/3；node22 `/tmp`余126GiB。预测输出收据仍BUILDING且`future_observed_agri_read=false`、`cpp_read=false`。当前工作是完整生成折外预测，不据此提前运行或选择噪声处理组。

## 2026-10-20 01:38 CST DDP资源方案及fold3状态

node22临时盘仍低于260 GiB门限，因此后续fold3改为node23两张A100上的单个DDP任务，而非在node22增加缓存。新增runner及PBS仅用于新的fold3输出目录，不改fold0/2运行和旧batch8隔离输出。DDP以每rank 10、全局20为batch；用确定性全局排列切分rank，SyncBatchNorm复现batch20的归一化统计，跨rank按总有效像元数归一化MSE。验证通过：单卡DataLoader顺序与分布式重建顺序一致、pooled-mask梯度一致；实际两卡SyncBatchNorm输出/梯度/running-stat witness通过。真实数据profile为41.18样本/s、峰值30.46 GiB/rank，越过预设速度和显存余量门。

首次PBS `215033`因漏传数据根目录参数，在读数据前退出，未写模型或训练产物；已修正并以`215034`启动fold3/seed45正式12轮。`215034`占node23 GPU1/2并复用完成的fold3 train-only raw cache；01:38训练至epoch1 step200。状态与逐项验收证据见`EXPERIMENT_TRACKER.md`。此前fold1单折OOF预测银行虽已完成，仍没有完整五折银行和GHI A/AH/AC收益结论。

01:40 CST新鲜日志检查显示`215034`到epoch1 step300，node23 GPU1/2利用率约97%、各显存约36 GiB；node22 fold0/2也继续推进。DDP runner 与 PBS SHA-256 分别为`df4b51274cdb4b5eedd59af5bf7a7ff73b3a6b9aa60b74224eac3a7d821d593b`、`2bc0b2adb61ebe205f5708c4b4dc2ddc9f62677da129c8f48deb595ef66ffa72`，本地与服务器一致。须待epoch完成后测得训练吞吐再估算时间；profile结果不可替代完整训练速度，也不是GHI效果。

### 2026-10-20 01:56 CST：DDP DropPath RNG caveat and fold4 path

Static inspection found that formal fold3 job `215034` seeds both ranks' CUDA RNG identically before constructing the model. The SimVP implementation uses per-sample DropPath masks (maximum probability 0.1); equal call order and local batch shape imply corresponding rank-local positions may share masks. This does not change global batch, sample order, optimizer updates, pooled loss, or SyncBatchNorm, but it means the stochastic regularization stream is not exactly equivalent to a single-device global batch. Do not interrupt fold3; retain this caveat in its provenance and assess sensitivity if the final GHI result is marginal. The audit is recorded in `DDP_DROPPATH_RNG_AUDIT_20261020.md`.

For fold4, a separate rank-RNG runner, PBS file, CPU semantics test, and GPU mask-independence witness have been prepared and synced. Fold4 remains seed46/global-batch20/12 epochs; its audited split has 27,237 train, 6,116 heldout, and zero test sequences. Its train-only normalization/raw cache will be built inside the allocated node. Server `swc` compilation, PBS shell syntax, and CPU semantics passed; the two-GPU mask witness must pass in the future PBS allocation. Do not submit while the full-account queue remains 3/3.

2026-10-20 07:32 CST更新：AC50支线的seed43/44也已分配到两卡并行，合并器合成见证通过。经过服务器哈希、shell/Python与审计见证后，更新版runner替换未运行的held job 215045，当前双卡任务为215046，依赖215042。主实验结果与GHI收益仍待OOF链完成。