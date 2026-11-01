# Experiment Tracker — COT-aware SimVP

## 2026-10-20 07:02 CST — OOF降噪后续已接入同一PBS链

为避免固定OOF主结果之后再等待PBS槽，更新了`run_future_ghi_oof_heads.pbs`：完成三seed A/AH/AC配对审计后，自动执行fail-closed AC−AH门禁；门未通过时，同一因果OOF pack上运行预注册的AC50（完整A/AH/AC/AC50配对，未来标准化COT异常×0.5）并独立审计；门通过时记录跳过AC50。训练/样本/标签/归一化/预算都不变；SimVP和现有OOF银行不修改。条件支线占用同一PBS作业，不另增任务槽。旧pilot相同head阶段实际耗时：A/AH/AC三臂389 s、四臂含AC50 522 s，说明该后续训练本身约9分钟，不会给12小时下游作业带来明显预算风险（pack构建与评分耗时仍以本次现场记录为准）。

更新脚本SHA已纳入`FUTURE_GHI_OOF_SHA256SUMS`；服务器`bash -n`、主流程及AC50两个SHA清单、AC50训练/审计/gate合成见证通过。原挂起作业`215043`已由`215044`替换，保持`afterok:215042`和node22单GPU/12小时资源；复核时PBS为`215036=R,215042=H,215044=H`（3/3），旧`215043=C`。fold4在07:10推进到epoch8/step10500；后续8秒dmon显示GPU3/GPU4的DDP rank大多同时约90–99% SM，仅在批间同步时短暂共同回落，没有发现某张卡持续闲置；活动SimVP基线未改。当前无OOF GHI收益结论。

## 2026-10-20 06:53 CST — fold4 epoch7; multigpu throughput and noise evidence

现场复核PBS：`215036=R`（fold4 DDP SimVP），`215042=H`（fold4 OOF预测，afterok训练），`215043=H`（GHI A/AH/AC筛选，afterok预测），全账户3/3。fold4固定12轮已完成epoch7/step9534，train loss=0.147309，epoch耗时约22分钟；heldout payload读取标志为false，`last.pt`与epoch7记录于06:53更新。06:59日志到epoch8/step9900；短间隔未出现step9900前仍有持续DDP进程，之后step9900已落日志。计算节点五次dmon/pmon抽样确认本作业两rank分别在node23 GPU3/GPU4运行，SM约90–100%/65–97%；GPU0的100%负载属于其他作业，未计入本作业。

多卡实测补充：fold4两卡DDP、global batch20 profile为41.06 samples/s；此前同batch单卡profile为24.23 samples/s，两卡约1.69倍吞吐，故本折采用DDP确有速度收益。小型GHI头不适合简单DataParallel：合成profile单卡146.51 steps/s、两卡17.22 steps/s。独立A/AH/AC并行可作为后续提速方向，但当前固定PBS pipeline仍按单卡串行执行，不能把它描述为已经并行；当前队列3/3，不增加任务。

关于“预测COT过度扰动”的已有筛选证据：旧in-sample pilot中AC50（仅将未来标准化COT异常减半）相对完整AC的RMSE差为−0.0014 W/m²，日期块95%区间[−0.633, +0.644]，没有可辨认的净收益；该结果不是OOF证据，不能证明或排除噪声机制。按既定方案先完成五折OOF A/AH/AC，再用OOF配对误差决定是否做单因素低太阳高度门控/收缩试验；活动baseline不改。

## 2026-10-20 06:43 CST — fold3 bank complete; GHI stage chained

fold3的OOF预测AGRI→冻结R-COT bank已于06:41完成，`5271/5271`；PBS `215041=C`且Exit_status=0。独立复核fold0–3的OOF预测AGRI/COT文件及index SHA均匹配各自manifest，四折状态均为`COMPLETE_FUTURE_GHI_OOF_FOLD_FORECASTS`，`future_observed_agri_read/cpp_read/buffer_payload_opened/validation_test_payload_opened`全为false。fold3索引含5271条唯一训练序列，和fold OOF ID清单逐行相同；AGRI/COT形状分别为`(5271,2,16,13,3,3)`与`(5271,2,16,1,3,3)`，float32，SHA全部吻合。fold4 DDP训练PBS `215036=R`，06:43已到epoch7/step8900，global batch20；fold4预测 `215042=H`（依赖215036）。已提交正式GHI三臂作业 `215043=H`，依赖 `afterok:215042`，单卡执行完整五折bank合并、pack构建、A/AH/AC三seed训练及日期块审计。队列3/3，不再提交额外作业。尚无完整五折预测bank或GHI结果，不能宣称收益。

因果pack合成见证已更新到新的OOF manifest契约（显式五折来源、五个checkpoint SHA、数组与index SHA），本地及服务器运行均通过；服务器环境下COT来源臂、COT收缩、AC50门、OOF来源和DDP全局batch语义见证通过。固定IntervalHead两卡DataParallel profile仅适用于小头的合成负载：1卡146.51 steps/s，2卡17.22 steps/s（0.118×），故正式头训练保持单卡，避免已实测的慢速实现。该记录只用于工程推进，未改论文正文或科学合同。

## 2026-10-20 06:38 CST — 五折 OOF 与多卡执行复核

当前队列为 `215036=R, 215041=R, 215042=H`（3/3）。`215036` 正在 fold4 固定12轮 SimVP DDP 训练，申请2张A100 40GB，global batch=20；最新日志到epoch7/step8600，前6轮均有完整epoch记录，heldout payload读取标记为false。`215041` 正在fold3预测AGRI→冻结R→COT bank生成，申请2张GPU；进度4832/5271（91.7%），bank仍为BUILDING，`future_observed_agri_read=false`、`cpp_read=false`。`215042` 按 `afterok:215036` 挂起，等待fold4训练checkpoint；目前没有第三项运行中的GPU计算。折0–2 bank已完成，五折bank合并和GHI评分均未开始，因此目前不能报告COT/GHI收益。

为避免把“多卡”误当作统一提速，已完成固定IntervalHead的合成实现profile：单卡146.51 steps/s；两卡 `DataParallel` 17.22 steps/s，速度比0.118（约慢8.5倍），不读取项目样本或标签。当前采用2卡DDP是为了在40GB卡上保持global batch20的SimVP合同；未来GHI头沿用单卡实现，暂不套用DataParallel。四卡并行已用于两个可独立的重计算阶段（fold4训练、fold3推理）；额外GPU不能在运行中的DDP作业里热加入。下游合成因果pack见证曾因OOF来源字段/哈希fixture未更新而失败，已补齐五折checkpoint及bank数组/index哈希字段；本地与服务器同一脚本现通过 `FUTURE_GHI_CAUSAL_PACK_WITNESS_PASS`。同组来源臂、AC50门禁、OOF provenance及DDP batch语义见证均在服务器环境通过。无科学结果或训练合同变更。

## 2026-10-19 实验启动更新

PBS `214871.tc6000` 已于 00:09 BJT 提交并进入运行态，分配 node22 上一张 NVIDIA A800-SXM4-80GB（GPU 1）；账户当前未结束任务为 `214221` 与 `214871`，共2/3。提交前远端 `SHA256SUMS` 全部通过，候选结果目录为空且无重复运行锁。作业脚本先执行真实批次profile（4步，显存峰值须低于卡容量85%），通过后才进行seed42、batch 8、4,000步的配对COT辅助损失训练和验证审计。

截至 00:11 BJT，PBS仍为运行态，计算节点上的训练Python进程已启动；作业输出已通过代码散列检查并确认分配GPU，但显卡利用率仍为0%，`profile.json`及正式训练记录尚未生成，当前只能判为数据/配对侧车加载与初始化阶段，不能宣称训练更新已开始。下一次检查同一作业句柄 `214871.tc6000` 的profile和step记录；test载荷保持关闭，论文正文未修改。

00:13 BJT复查同一作业仍为运行态。节点本地AGRI帧缓存已生成111个文件、约265 MiB；训练进程约2分半，GPU显存约475 MiB、利用率0%。目前可见瓶颈在按需读取并缓存原始帧，真实批次profile尚未开始；继续观察缓存及profile记录，不重复提交。

00:15 BJT复查，作业`214871.tc6000`仍运行，Python训练器持续存活；节点缓存增至384帧，GPU显存约2.5 GiB、瞬时利用率0%。结果目录与profile仍未出现，推断仍在构造32个训练校准窗口需要的原始帧缓存；这一阶段属于输入准备，尚无训练更新或科学结论。已将每小时heartbeat切换为本目标，按同一作业句柄继续跟进并在结束后按审计结果决定下一项降噪实验。

00:18 BJT复核同一作业仍运行。节点缓存中的完整`.npz`帧增至245个（相较上次观察继续增长），训练Python进程约6分半且仍在执行；GPU约3.6 GiB显存、采样时利用率0%，进程采样显示处于文件系统等待。训练输出目录和profile尚未生成，确认当前主要耗时仍在原始帧读取/缓存及预训练准备；没有训练更新或结果。自动续接频率改为每30分钟，继续盯住同一PBS作业，不重复提交。

00:18 BJT续查：同预算MSE对照的远端`training_complete.json`记录4,000步耗时5,936秒（约99分钟，A100）；profile稳态吞吐2.29样本/秒、峰值29.21/39.59 GiB。这表明本候选8小时PBS预算足够，启动慢本身不构成停止理由。候选作业`214871`仍运行；本地完整AGRI缓存增至369帧，进程采样仍显示共享文件系统等待，GPU瞬时0%/3.6 GiB，候选profile和训练输出仍未产生。继续跟随当前作业。

00:20 BJT复查：同一作业仍运行，节点`.npz`缓存增至524帧；训练进程持续活跃，GPU约14.1 GiB/80 GiB，采样利用率0%。候选结果目录暂未出现，进程仍在真实批次校准前的数据加载/计算准备。未观察到退出或OOM；继续监控当前PBS句柄。

00:22 BJT：候选预处理与梯度比例校准完成，`preflight.json`报告PASS、冻结R、test读取0；仅用train日期32个窗口定出的`lambda_COT=0.005891630984842778`。结果目录尚未写出profile，PBS作业仍在运行；当前开始profile所需的4个真实训练批次读取/计时，不能提前宣称正式训练已开始。该数值用于本次固定试验，不根据验证结果调整。

00:25 BJT：真实数据单卡profile通过。A800-SXM4-80GB峰值分配30.42/79.33 GiB（低于85%门槛），稳态吞吐7.92样本/秒；四个profile步均有限，首批40.81秒（缓存冷启动），后三批0.79–1.35秒。PBS `214871`仍运行，训练器已写出`run_metadata.json`并进入step-0验证/正式训练切换阶段；正式更新步尚待`recent_steps.json`确认。仍为seed42、batch8、4,000步、单卡，与固定MSE配对。

00:27 BJT复核：PBS仍运行；正在执行完整验证集的step-0评估，缓存增至2,195帧，GPU峰值观测约34.5 GiB且利用率在采样间变化（最高观测40%）。`validation_step_0000.json`与`recent_steps.json`尚未生成，故正式训练步仍未确认。无OOM/退出迹象。自动heartbeat处于ACTIVE、每30分钟续接当前目标。

00:29 BJT复核：作业`214871`运行20分钟；step-0验证仍在读取/评估，节点缓存增至2,622个完整AGR​​I帧，4个DataLoader worker均存活且有CPU活动，GPU显存约34.5 GiB。step-0结果及训练步记录尚未出现；队列共2/3未结束任务。保持当前验证过程，不重复提交。

更新时间：2026-10-17 11:51 BJT。论文引言、讨论和结论已完成一轮双语收口；未引入新实验结果。

| 阶段 | 状态 | 证据 / 下一步 |
|---|---|---|
| 单因素假设与配对合同 | READY | AGRI-MSE与AGRI-MSE+COT辅助损失；相同seed42、训练行、batch8、4,000步和初始化，COT辅助仅用训练数据确定权重。|
| 论文双语写作 | PASS_COMPILE | 引言明确三项研究问题；讨论分开同期估计、预测COT和未测的预测输入GHI技能；结论同步收束。LaTeX内置编译成功。|
| 核心损失与冻结R反传 | PASS_REMOTE_WITNESS | `cot_auxiliary.py`：R权重梯度为0，SimVP预测输入梯度有限非零；CPP掩膜和R回退目标通过CPU合成见证。真实月度R CPU反传见证通过。|
| train/val日期与序列键 | PASS_REMOTE_AUDIT | 35,320 train、3,140 val；16个未来帧逐项按UTC帧键核验，初始化与样本时间索引吻合；test载荷关闭。|
| 固定训练行和对照权重 | PASS_REMOTE_AUDIT | 复用MSE对照固定32,000行、seed42、batch8、4,000步；step-4000权重存在，并在脚本启动前核验步骤。|
| 同掩膜COT比较 | IMPLEMENTED_AND_TESTED_SYNTHETIC | 候选、原始SimVP和等预算MSE step-4000均保存逐样本预测银行。独立FP64审计器按同一CPP支持重算patch与中心3×3宏平均，配对10,000次BJT目标日期bootstrap，并检查厚云CSI、近零虚警和AGRI护栏；合成统计见证通过。|
| PBS资源与运行 | GATED_QUEUE_LIMIT | 2026-10-17 11:51全账户有3个未结束任务：`214221`排队、`214517`与`214521`运行；达到上限，尚未提交有效训练作业。候选此前`214518`（校验清单目录）、`214519`（模块目录）、`214520`（Path参数/模型加载）均在训练更新前退出；修复后的训练器、独立审计器、PBS已同步并通过远端散列、真实月度数据合同CPU见证、R反传见证、合成配对审计及PBS语法检查。等待队列低于3后重新qsub；A800单卡实数profile门槛仍有效。|
| 论文结果更新 | WAIT_FOR_AUDIT | 训练结果须经独立复算、同掩膜对照和AGRI/厚云护栏检查后，才同步写入正文；test保持关闭。|

## 执行边界

该实验最多支持预测COT筛选结论，不能单独证明GHI预报收益。若通过预设门槛，再考虑seed43/44；未通过则记录为负结果，不按验证集调权重。不得覆盖冻结SimVP、MSE对照、原SimVP→R主结果或已有论文PDF。

00:30 BJT复核：PBS `214871.tc6000`仍为R，已运行约21分钟，队列总计2/3。训练进程PID 19003及4个DataLoader worker均存活；worker仍有约31% CPU活动，训练主进程处于等待状态，说明step-0验证的数据加载/批处理仍在进行。GPU上训练进程约34.5 GiB显存，采样利用率曾为0%；未见退出/OOM。结果目录仍只有preflight、profile和run_metadata，step-0验证文件及正式更新记录未生成，因此尚不能说训练更新已经开始。保持同一作业，不重复提交；下次核验重点看step-0产物和日志是否前进。

00:32 BJT复核：重复检查同一在运行句柄，缓存从5,980帧/15 GiB增至6,662帧/16 GiB，45秒内仍增加682帧；4个数据加载worker继续有CPU活动。训练主进程和GPU训练进程PID 19003仍存活，结果目录尚无step-0验证文件。进一步核对`eval_model`实现后确认，它只在整套验证集迭代完成后写一个step级JSON，没有逐批进度文件；因此此阶段虽无新JSON，缓存持续增长是有用的前进证据。队列仍2/3，继续等待当前全验证，不提交替代作业。

00:33 BJT复核：PBS `214871.tc6000`仍R，已运行约24分钟；队列2/3。step-0完整验证还未结束，结果目录未新增记录。节点本地输入缓存由16 GiB增至18 GiB（6,842→7,492个文件），四个loader worker继续累计CPU时间，主进程仍存活。验证函数没有逐批状态文件，故以缓存继续增加和worker运行作为明确的在途证据；不重新提交或中断当前运行。

00:35 BJT复核：PBS `214871.tc6000`继续R，运行约26分钟；全账户2/3未结束任务。缓存由19 GiB/7,685帧增至20 GiB/8,352帧，loader仍在处理，训练进程存活。step-0验证产物仍未出现；按代码逻辑只能在完整验证集结束后落盘。继续跟随该作业，不启动重复候选。

00:37 BJT复核：作业`214871.tc6000`仍R（运行约28分钟），缓存由20 GiB/8,352文件增至23 GiB/9,467文件；训练进程和4个worker继续存活。step-0验证JSON尚未生成，故正式更新步未确认。缓存持续增长，未见作业退出或OOM；等待同一全量验证结束。

00:38 BJT补充核验：节点缓存为24 GiB/10,201文件并继续增长。`nvidia-smi dmon`连续6秒采样显示本作业分配的GPU 1出现一次28% SM利用率，其余采样为0%；GPU 0/2的高负载属于节点其他卡，不能算作本作业。结合训练PID和loader活动，当前仍在验证批次/数据加载阶段，不是已停死；step-0 JSON仍待全验证迭代返回。

00:39 BJT复核：PBS `214871.tc6000`仍R，运行约30分钟、全账户队列2/3。节点输入缓存增至25 GiB/10,475文件；四个loader进程继续有CPU活动，训练器PID 19003存活。输出仍未出现`validation_step_0000.json`或任何正式step文件；无退出/OOM。继续保持当前完整验证，不重复提交。

00:40 BJT训练数据IO量级审计：按冻结`hunan_8to16.csv`逐行去重，val为3,140窗口/4,497个唯一AGRI帧，train为35,320窗口/51,236个唯一AGRI帧。当前节点缓存约26 GiB，已接近val全量帧体量；后续正式训练会逐渐冷缓存更多训练帧。单卡profile吞吐在4个热/半热批次达7.92样本/秒，但真实端到端时间还受共享AGRI存储和帧缓存写入影响；8小时预算需持续观察，不据profile单独承诺完成时间。当前仍不切换或并行重复启动。

00:41 BJT复核：PBS`214871.tc6000`在运行，运行约32分钟，缓存新增至27 GiB/11,233文件，队列仍2/3。完整验证JSON未生成；训练器PID19003和4个worker仍活跃，worker主要在节点缓存文件锁/存储等待中。缓存保持增长，现有磁盘余量充足，当前判断为慢IO而非停滞或故障；继续当前作业。

00:49 BJT：step-0全量验证已完成并保存，等季节均衡AGRI RMSE=0.285743（归一化单位，3,140个val窗口；DJF 0.16567、MAM 0.34405、JJA 0.42355、SON 0.20970）。训练器已进入训练DataLoader，GPU 1采样利用率21%，节点帧缓存39 GiB；正式步记录尚未到step100。另对固定32,000个训练窗口实际使用的未来目标帧做了源组成审计，计入样本重复曝光及CPP/R共同有效掩膜：Sili有效COT目标像元中CPP占32.04%、观测AGRI→R回退占67.96%；Zhujia为CPP 32.01%、R回退67.99%。逐时效CPP比例在Sili 31.84–32.24%、Zhujia 31.75–32.21%，不是少数时效造成。文件`results/seed42_cot_aux_v1/target_source_coverage_train.json`保存逐站逐时效精确计数。这为噪声假设提供了直接机制证据，但还不能证明其造成性能损失；当前mixed-target候选先继续，若其没有通过收益门槛，下一项单因素候选是仅用CPP有效掩膜训练COT辅助损失、移除68%的R回退监督，并用train-only梯度比重新校准权重。

00:50 BJT：step-0已确认完成，equal-season normalized AGRI RMSE=0.285743；正式训练loader仍在冷缓存选定训练行所需AGRI帧，缓存从39增至44 GiB，作业`214871`仍R、训练模型进程存活、无OOM。step100记录还未生成，当前是训练启动的冷数据阶段。训练辅助目标审计发现CPP监督仅占有效目标像元约32%，其余约68%来自观测AGRI→R伪标签；两站和16个时效比例接近一致。这个比例是精确固定训练行、按实际重复曝光加权的结果，已保存逐lead JSON。若mixed目标候选最终未过预设技能护栏，下一轮优先做CPP-only单因素对照。

00:51 BJT复核：step-0验证已落盘后，PBS`214871.tc6000`仍R。训练batch loader正在为32,000固定行冷缓存训练样本AGRI帧，节点缓存44→45 GiB；GPU1当前抽样利用率0%，模型进程和4个loader仍在运行，`recent_steps.json`尚未出现（按代码每100步才写一次）。没有失败或OOM证据。该阶段受随机训练行冷缓存影响，不把尚无step100误判成训练停滞；继续观察步数文件和GPU/缓存推进。

00:51 BJT复核：PBS`214871.tc6000`仍在运行，缓存从45增至46 GiB/19,266个文件，GPU1采样3%利用率，训练进程与4个worker存活；step100文件未出现。该随机训练批次冷缓存仍在持续增长，未见退出或OOM。队列2/3。继续当前训练，不重复提交。

00:51 BJT更新：当前训练作业`214871.tc6000`仍R；训练数据缓存约46 GiB且近一分钟增长趋缓，分配GPU1采样利用率升至81%，表明已进入计算密集更新阶段。step100日志尚未出现（训练器每100步落盘）；无退出/OOM。继续等待第一个正式训练记录，再据损失和验证确定下一动作。

00:52 BJT复核：同一PBS`214871.tc6000`仍R，队列2/3。训练输入缓存增至47 GiB，较上次再增加约1 GiB；GPU1利用率在短采样间歇性升高/回落，符合冷缓存与批次计算交替。step100记录还未出现，正式验证后训练尚处早期；过程存活、未见失败或OOM。

00:52 BJT复核：PBS `214871.tc6000`仍运行、队列2/3。节点缓存增至48 GiB；GPU1本次采样0%，四个训练数据worker存活，当前仍在随机训练行的冷帧读取/缓存。正式step100记录尚未生成；无退出/OOM。继续按同一训练合同运行并等待首次步记录。

00:54 BJT复核：`214871.tc6000`仍R，约44分钟；节点缓存增至52 GiB（20,113→22,084文件），4个DataLoader worker运行中。GPU1抽样利用率20%，缓存读写与计算交替。step100文件尚未出现；当前保留原作业，队列2/3，未观察到OOM/异常退出。

00:54 BJT：作业`214871.tc6000`仍R，节点缓存增至54 GiB/22,654文件；4个训练DataLoader worker存活，当前GPU1采样0%，主要等待本批冷帧读取。输出目录尚无`recent_steps.json`，step100未确认。缓存仍有增量，无退出/OOM，继续当前作业。

## 2026-10-19 01:18 BJT 持续推进更新

PBS `214871.tc6000`仍为运行态，运行约50分钟；账户全队列为`214221`排队、`214871`运行，共2/3。M1 step-100训练记录尚未落盘，step-0验证和profile已完成。计算节点node22/GPU1显示约34.5 GiB显存占用；4个数据加载worker仍存活。节点缓存从此前约54 GiB增至72 GiB、文件数约30,464，训练主进程仍在等待读取/缓存，未见退出或OOM。当前判断是大规模训练帧冷缓存耗时，不能把无step日志当作卡死；继续跟踪同一作业，不重复提交。

已为后续唯一因素“CPP-only辅助目标”建立隔离候选`refine-logs/cot_aux_cpp_only_20261019`，远端候选代码位于`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/cot_aux_cpp_only_20261019`。固定32,000行、同一SimVP初始化、seed42、batch8、4,000步、冻结R、相同CPP∩R有效支持及训练内梯度比标定；仅移除CPP无效处的AGRI→R回退目标。独立结果目录、lock、scratch和PBS脚本已准备但未提交。

候选远端SHA256清单、py_compile、CPP-only掩膜测试、辅助损失CPU测试、配对审计合成测试及真实冻结R CPU反传测试均通过；最后一项确认R参数无梯度、SimVP输入梯度有限非零。因本机没有PyTorch，目标张量单测已在服务器`swc`环境执行通过。提交条件仍是214871完成并独立复算其同掩膜COT/AGRI护栏；通过则转跨seed稳定性，不通过或收益不确定才按单因素方案提交CPP-only。test保持关闭。

01:21 BJT follow-up: live PBS still shows `214871.tc6000` R and `214221.tc6000` Q (2/3 unended). M1 output directory still has no step-100 record; node22 cache measured 74 GiB. A remote source check confirmed `RawCache` keys frames as `<relative_path>.npz` and uses per-frame file locks, so the already-populated M1 cache is safely reusable by the same node22 dataset contract. Updated CPP-only PBS launcher to point to this cache while keeping its own output, lock, and temporary scratch; it will populate any missing frames. Candidate hash validation still passes. No second job has been submitted.

## 2026-10-19 01:23 BJT：M1仍在训练数据预热

再次实时核验PBS `214871.tc6000`为R、运行约56分钟；`214221.tc6000`为Q，队列2/3。输出目录未出现`recent_steps.json`（step-100仍未确认）；node22训练主进程存活、GPU1显存约34.5 GiB但当前采样利用率0%。本地缓存从79 GiB增至84 GiB，说明仍有新的输入帧写入；因此保留当前运行，不复制提交。该训练缓存已按相同`RawCache`相对路径合同接入CPP-only候选，后续在node22可复用。

01:24 BJT：对CPP-only候选完成服务器真实数据合同见证。冻结月度loader确认35,320个train样本、3,140个validation样本，`test_payloads_read=0`，SimVP为CPU加载且合同通过。同期M1 PBS `214871.tc6000`仍R（约59分钟），账户队列2/3；输出没有step-100文件。node22 cache现为93 GiB，训练进程及4个worker存活、主进程在等待DataLoader，GPU1抽样仍0%。PBS stdout确认脚本已通过hash/py_compile与此前profile，当前python训练仍在取样/缓存阶段。未见失败/OOM，继续保持同一作业；CPP-only候选已具备真实train/val合同、冻结R反传与目标掩膜见证，待M1审计后按门槛启动。

## 2026-10-19 01:33 BJT：M1进入正式更新

远端首次出现`recent_steps.json`，确认seed42 COT辅助训练已完成100/4,000步。对已保存的100步记录复算：L_AGRI均值0.119399，L_COT均值0.030505，梯度范数最大3.0892，非有限值0；这些是训练损失健康检查，不是验证技能结论。随后一次独立轮询在60秒后仍读到step100，故不据此外推更新速度；node22缓存继续从98增至106 GiB，GPU1短采样为0–27%波动，符合训练与冷帧读取交替。PBS最新可见状态214871=R、214221=Q，账户2/3，尚无2000步验证。继续原作业，不提交重复任务。

CPP-only候选远程真实数据见证通过（train 35,320 / val 3,140 / test读取0），冻结R梯度与目标掩膜测试通过；其单卡启动脚本共享M1同一节点帧缓存。下一科学动作仍是M1完成后同掩膜独立审计，按COT/AGRI护栏决定跨seed或CPP-only单因素训练，之后必须落到匹配未来输入的GHI对照。

01:35 BJT短间隔复核：PBS `214871.tc6000`仍R（远端qstat显示运行1:07），step仍100，未到200；约45秒内缓存109 GiB、比前次107 GiB增加，GPU短采样为0%。因此训练仍在冷帧读取和更新间交替，不能据一次零利用率判定停死；同时将慢数据阶段作为本次实际吞吐风险继续追踪。候选CPP-only启动脚本已可复用该缓存，训练源/损失和split没有改动。账户仍2/3。

远端时钟 2026-10-19 01:18 CST 复核：`214871.tc6000`仍R，运行约1:09；`recent_steps.json`仍为100，cache已增至113 GiB / 23,991个NPZ帧。60秒观察内新增约674帧，GPU1采样在0–45%间变化，说明数据准备和计算交替，尚未出现step200或中期验证；没有终止/OOM证据。账户仍为214221(Q)+214871(R)，2/3。保持当前训练；不要把一个60秒窗口内未跨100步解读成整个作业停滞。

01:21 BJT持续推进复核：PBS `214871.tc6000`仍R，账户队列`214221`=Q、`214871`=R，共2/3。训练结果仍停在step100（recent_steps.json上次更新01:10:25），无validation_step_02000或完成标志；compute节点gpu2上的训练器PID 19003及两个活跃Python子进程仍在，CPU约11–16%，M1分配GPU1显存34.5 GiB，10秒采样利用率为7%/100%/0%，输入缓存约121 GiB。该快照显示作业在计算与数据加载间歇切换，无失败/OOM证据，但不能仅凭瞬时GPU利用率推断有效吞吐；继续跟踪同一作业，不重提。自动化`cot`仍为ACTIVE、每30分钟唤醒，目标是完成当前同掩膜COT/AGRI审计后推进跨seed或CPP-only和固定GHI头对照；test保持关闭。

01:23 BJT 数据通路诊断：对同一PBS句柄在gpu2做了进程级复核。M1分配GPU1显存34.5 GiB、当时利用率1%；主训练器与4个DataLoader worker仍存活，worker在文件系统/缓存锁等待状态（wchan包含`ofs_tj`、`lock_b`），节点缓存124 GiB、26,154个NPZ。训练step仍为100。该证据把当前慢点定位到冷帧读取/缓存而非GPU显存不足或异常退出；短时GPU空闲不能说明模型停止。当前合同设定4 worker、预取1批；不改活动作业、不再并发预热以免抢共享I/O。节点缓存由M1填充且CPP-only复用同一路径，避免下一候选再次冷启动；继续监控step200和缓存增量，若本次8小时作业未完成，则依据实际耗时修订后续同科学合同的恢复/缓存方案。

01:25 BJT同句柄复核：step仍100，PBS仍R（已运行约1:15），账户2/3；但node22缓存从124 GiB/26,154个NPZ增至126 GiB/26,698个NPZ，约100秒新增544帧；训练器与4个loader仍活跃，GPU1当前0%采样。数据缓存继续前进但较慢，证实此阶段是缓存吞吐受限，不能把无step更新视为作业停止。现有路径有足够PBS余时，先保留本次合同跑完；CPP-only候选复用这一节点缓存，减少其冷启动成本。没有提交新的PBS任务。

01:26 BJT复查同一运行句柄：`214871.tc6000`仍R（1:17），全账户2/3；step仍100，当前尚无新checkpoint或验证文件。gpu2缓存从126 GiB/26,698个NPZ增至129 GiB/27,406个NPZ，约100秒新增708帧；训练PID 19003存活，GPU1显存34.5 GiB、抽样利用率0%。按冻结行统计的51,236个train唯一帧加4,497个val唯一帧，缓存约到一半；按当前缓存速率粗估尚需约1小时完成冷数据覆盖，估算不等于完成保证。作业还有约6小时40分钟PBS预算，先让同一运行继续填充并训练；无失败/OOM，未重复提交。

01:28 BJT持续观察：重复核验同一PBS句柄`214871.tc6000`仍R，账户仍2/3；M1步数尚100。节点缓存由01:26的27,406帧/129 GiB增至01:27:47的28,143帧/133 GiB，约72秒增加737帧；训练器仍存活，GPU1短采样0%、显存34.5 GiB。冷缓存继续前进，按当前速率覆盖剩余唯一帧约需50分钟量级，仍在8小时作业预算内。继续等待冷缓存补齐和step200，不重复提交；目标COT/GHI结果尚未产生。

01:30 BJT复查：`214871.tc6000`仍为R，1:20左右，账户队列2/3。step仍100；最近约88秒缓存由28,143增至28,649个NPZ（+506），约135 GiB；训练器PID 19003存活，GPU1此刻0%利用、34.5 GiB显存。说明冷数据I/O继续推进但训练步写盘尚未跨100；无可安全在线更改的模型参数或并发预热动作，避免在同一节点加重I/O。当前科学结果未产生，持续跟踪同一作业；下一事件为step200/中期验证或PBS状态变化。

01:31 BJT复核：PBS `214871.tc6000`仍R，账户2/3；M1 step仍100，缓存由28,733升至29,129个NPZ（约+396帧/72秒），到约138 GiB，训练器仍存活。冷缓存增长速率本次约5.5帧/秒，步数写盘仍未跨100。这个变化与按需缓存机制一致，尚不能推出训练效率或技能效果；不启动同节点预热或第二作业，继续使用当前运行填充共享缓存，待缓存完善后复核step/验证。

01:33 BJT：新增“标签来源差异”只读诊断准备完成。`audit_label_source_disagreement.py`在本机`py_compile`和合成数值见证通过，并已同步服务器实验目录，`swc` Python 3.10.18远端语法检查退出码0，SHA256=`f166051cd15c01ab86cb5e982005e681bba42f50f05901a1fbffb5a87371ac03`。脚本只读取固定验证集，按采样帧键比较CPP有效像元上的CPP与真实AGRI→R差异，并对比CPP支持区/R回退区的R目标分布；明确标注CPP非独立真值，不会调模型或打开test。为避免与M1共享存储竞争，尚未读取大数组，待冷缓存完成后运行。

同期M1句柄`214871.tc6000`仍R，账户2/3，step100；node22缓存由29,129增至29,804个NPZ（约141 GiB），训练进程存活、GPU1当前0%、34.5 GiB显存。缓存仍有增长，未发现失败/OOM；按当前合同继续，不重复提交。

01:37 BJT 阶段进展：单核诊断PBS `214874.tc6000`正常完成（C），输出 `results/label_source_audit_val_1024.json`；抽样1,024个val窗口，共1,643,712个共同有效像元。R−CPP共同支持RMSE：四里9.371、竺家8.527、合并8.960 COT；均值偏差分别+0.284、+0.245、+0.264。更明显的来源分布差异是R回退区厚云比例约四里22.42%、竺家25.07%，而CPP覆盖区对应R比例为11.90%、13.09%；R回退区中位COT也更高。CPP覆盖率在30.6%–34.7%之间。此为验证集描述性诊断，CPP非独立真值、相邻窗口重叠，不报独立像元显著性，也不作为模型技能结果；支持继续检验CPP-only，但尚不能因果断言R回退造成GHI退化。独立复核摘要和原始JSON已归档本地`results/`，远端输出未打开test载荷。

同期M1作业`214871.tc6000`跨至step200（`recent_steps.json` 01:36:44更新），1:27运行；账户未结束任务2/3（214221=Q、214871=R；214874已C）。缓存145 GiB/30,711帧，GPU1利用率抽样44%、显存34.5 GiB，训练PID仍存活。M1的正式验证仍只在step2000/4000执行，step200训练日志不是质量证据；当前保持作业，不重复提交。

01:39 BJT同句柄进展：PBS `214871.tc6000`仍R，step200（`recent_steps.json`01:36:44更新），作业运行约1:30；节点缓存由31,341帧/148 GiB增至32,017帧/151 GiB，训练器存活、GPU1本次采样0%、显存34.5 GiB。相对55,733个train/val唯一帧约完成57.5%冷缓存。M1正式技能验证尚未到step2000；不以训练loss或缓存比例声称收益。来源诊断PBS `214874.tc6000`已C并释放名额，qstat未结束作业仍2/3。保持现作业运行，CPP-only仍等M1独立审计门。

01:42 BJT 完成第二轮标签噪声分层审计：PBS `214875.tc6000`正常完成，输出SHA256与本地一致（`676aa021f20d3bccf22942e47f0a9e18f6d4bfdf5a3953c7efac66dc8ee54ba4`）。基于同一1,024个val窗口的共同支持，按CPP参考COT分层：CPP<1时R均值3.524、R落入COT<1仅81.87%；CPP 1–10时R RMSE 2.905；CPP≥30时R均值33.617 vs CPP 42.636，偏差−9.020、RMSE 13.703，R仍≥30比例63.42%（四里56.38%、竺家70.28%）。支持R有近零虚警和厚云压低的条件性差异；不把CPP当真值、不以重叠像元作显著性推断。文件`results/label_source_disagreement_summary.md`与CPP-only计划补充已记录，不改论文。

复核同一M1作业`214871.tc6000`仍R，step200，缓存158 GiB/33,505帧，GPU1瞬时0%、显存34.5 GiB，训练PID存活；全账户未结束任务2/3，214874/214875诊断作业均C，214221仍Q。M1技能验证未到step2000，继续原运行。

## 2026-10-19 01:46 BJT：未来GHI验证门准备

本轮将工作重心明确到真正的预报链收益。M1 `214871.tc6000`实时仍为R，服务器时间01:43:40时运行01:34:24，正式训练仍step200/4000，step-2000验证尚未出现；输出日志最后仍在启动段，因此继续保留同一作业，暂不重复提交或并发预热。全账户未结束作业为214221(Q)+214871(R)，2/3。注意：共享登录节点看不到计算节点本地`/tmp`缓存，`du`返回0不能当作缓存为空；缓存只在PBS分配节点核验。

复核发现旧`forecast_ghi_window_20261004_v1`为旧日期合同探索性结果，`r_geometry_downstream_ghi_20261016`使用真实AGRI历史帧，二者都不回答未来预测COT对未来GHI的边际作用。新工程门`FUTURE_GHI_GATE.md`已建立：固定同一SimVP预测AGRI，配对A/AC比较“不输入COT”与“输入冻结R(预测AGRI)的COT”；严格控制同站、同init/target、15分钟标签、白天掩膜、3×3局地头、种子及归一化。训练输入要求上游预测按日期两折交叉拟合，避免在SimVP训练日期上直接用in-sample外推结果训练GHI头。固定月度验证为2024-04至2026-06跨26个月分散留日，不称连续一年。只有M1审计和现有作业/队列门允许后才启动训练；test/buffer仍关闭。

用户已要求睡眠期间继续自动推进。heartbeat `cot` 已更新为每30分钟唤醒：以未来GHI边际收益为最终目标；追踪当前作业；过门后继续COT候选或固定输入A/AC GHI对照；先处理I/O，再按队列限制提交；不以oracle或旧合同成绩代替部署预报结果。论文正文不改。

01:46 BJT计算节点级复核补记：从tc6000经SSH到PBS分配的gpu2后，确认训练进程PID 19003仍活、step仍200；4个worker中至少一个在`ofs_tjobq_timedwait`等待共享存储，缓存为162 GiB/34,364帧（相对01:42的33,505帧继续增长），GPU1短采样21%、显存34.5 GiB。登录节点读不到该节点`/tmp`，因此后续缓存检查必须在分配节点执行。现象支持冷帧I/O瓶颈而非进程退出；保留当前训练，不在同节点并行预热。M1离4000步仍很远，后续30分钟heartbeat继续检查缓存增长、更新步数和PBS剩余墙钟；若缓存停止增长或worker状态异常，再按有证据的故障处理方案行动。

## 2026-10-19 01:50 BJT：I/O瓶颈复核与下一轮预热改进

对同一PBS `214871.tc6000` 做计算节点gpu2实时复核：作业仍R（运行01:41:12），全队列为214221(Q)+214871(R)，2/3；step仍200/4000，最近写盘01:36:44。node-local cache从01:46的34,466帧/163 GiB增至35,120帧/166 GiB；GPU1瞬时0%、显存34.5 GiB。进程仍活但主进程等待本地锁、DataLoader worker等待共享存储；确认当前速度受冷帧I/O限制，不应通过申请更多显存解决。

为避免候选CPP-only作业重复经历逐窗口冷读，新增`code/prewarm_hunan_frames.py`：按训练/验证manifest去重预加载AGRI帧，沿用同一个`hunan_data.load_frame`和node-local `.npz`格式，按帧锁安全复用已缓存文件，test/buffer仅保留元数据、不读取载荷。同步加入PBS启动流程，在GPU更新前完成缓存完整性检查/预热。合成测试覆盖重复帧、分割过滤和test/buffer不触碰；远端`swc`下所有SHA256、py_compile及合成见证通过。该改变只作用于下一候选的数据加载，不改活动M1或训练科学合同；候选仍等214871最终独立审计门，不提前提交。

未来GHI方向保持不变：待M1/CPP-only或稳定性门完成后，使用同一SimVP预测AGRI严格配对A/AC，未来COT仅由冻结R处理预测AGRI生成；GHI头训练预测输入采用按日期交叉拟合，test关闭。
## 2026-10-19 未来GHI精确配对数据门

- 建立 `audit_future_ghi_crosswalk.py`，将 `hunan_8to16.csv` 中训练/验证初始化与 `ghi_head_pack_new_r_v1/rows.csv` 按 canonical BJT→UTC、init+15…+240 min、站点键逐项对齐；只读元数据，不读取AGRI/CPP/GHI数组、test或buffer。
- 匹配率：781,504 / 1,230,720 个候选站点×初始化×时效键；训练714,914、验证66,590，split mismatch=0。四个站点×split计数见 `results/future_ghi_crosswalk.json`。
- 写出 `results/future_ghi_crosswalk_keys.npz`，含split、split内序列索引、站点、lead和原始GHI行索引。新增独立语义复核 `validate_future_ghi_crosswalk.py`，完整重建每个时间/标签行键后通过，见 `results/future_ghi_crosswalk_independent_audit.json`；781,504个forecast instance键均唯一。
- 解释边界：同一目标标签可能对应多个不同初始化/lead预报实例（每个目标最多16个），这属于不同预报问题；置信区间按目标日期块聚类，不能把实例数当独立天气样本数。此产物仅是键对齐门，不是技能结果。
- 同时复核PBS：`214871.tc6000`仍R；全账户2/3未结束任务（另一个`214221.tc6000`排队）；在gpu2上M1 step推进到300/4000，训练进程存活，node-local cache约37,677帧。正式COT验证尚未到step2000，不基于训练loss作结论、不重复提交。
- 下一动作：继续等待M1到正式验证/终止节点；通过当前资产和资源门后生成预测AGRI→冻结R COT序列，并先做A/AC未来GHI筛选pilot。先前额外层面的in-sample训练输入只允许标筛选pilot；论文主结果要求日期交叉拟合的train预测。

## 2026-10-19 未来GHI链路工程推进

- 新增 `build_forecast_agri_r_bank.py`：按固定train/validation序列清单生成8→16 SimVP预测AGRI及冻结R预测COT；历史最后两帧的AGRI/COT只由初始化前观测帧得到。未来时刻仅读取太阳几何侧车，脚本从未对未来帧调用AGRI读取器；不读CPP/test/buffer。训练日期上的预测来自已训练SimVP，故明确限定为筛选pilot。
- 新增/完善 `build_future_ghi_head_pack.py` 的流式键写出，避免把约78万行键累计为大量Python对象；同一固定头、三帧因果上下文、训练集归一化共享规则保持不变。
- 本地及服务器 `swc` 的语法检查通过；服务器合成因果打包测试通过（`FUTURE_GHI_CAUSAL_PACK_WITNESS_PASS`）。源代码已同步到项目远端工程目录。
- 服务器预检只核对冻结checkpoint/归一化哈希、序列索引、站点行列元数据和train/val目标帧键，不读取卫星数组或CPP、不占GPU。首次预检发现文件名stem与R帧表同用UTC键、只有太阳侧车查询才转BJT，已据此修正；最终预检结果待写盘。
- 当时PBS队列为 `214871=R`、`214221=Q`，共2/3未结束任务；不在M1正式验证与独立审计之前启动占卡预测/训练。预检通过后下一步是等门完成，在单卡上生成bank，再生成GHI pack并做A/AC筛选；收益仍由同样本的配对验证预测和按目标日期块bootstrap判断。

02:14 CST补记：两次远端元数据预检均显示进程落在共享文件系统等待；第二次轻量边界版仍需读取并哈希126 MB SimVP权重，和活动M1共用慢存储。为避免继续竞争M1冷数据I/O，已停止我启动的两个预检进程；PBS训练未受干预，当前无未来GHI bank结果。代码/测试已同步远端。待M1到正式评估/独立审计门后，在获分配GPU节点上一次完成资产哈希、bank构建与运行时逐帧元数据校验，不再在登录节点重复扫权重。

02:15 CST实时状态：`214871.tc6000=R`、`214221.tc6000=Q`，全账户2/3；M1训练进程存活并已写到step 400/4000，尚无step-2000验证或完成标记。step400只是训练运行记录，不是技能结论。未来GHI bank仍未启动；代码和因果打包见证准备完成，继续等当前门，同时不再做共享存储预检。

02:17 CST工程补充：新增 `audit_future_ghi_a_ac.py`，验证A/AC及三种子是否严格同键同真值；以BJT目标日期为重采样单元计算两站等权RMSE差及95%区间，并列出逐站、逐种子和时效分组指标。合成正向样例、日期块bootstrap及因果打包见证均在本地和服务器 `swc` 通过。为避免重复共享盘读取，预检哈希只计算一次并在后续manifest复用。此为分析/执行准备，不是GHI结果；M1实时仍step400，正式step2000审计未出现。

02:22 CST执行准备补充：metadata crosswalk覆盖的实际唯一序列为train 30,436、val 2,817，共33,253；预测银行原始数组约690 MB，配对GHI头输入约1.36 GB，工作量和存储已据此估算。预测银行增加了原子进度记录与严格合同哈希断点续跑，单卡默认8序列批量、R批量1024，并记录吞吐和峰值显存；PBS分成“预测银行”和“pack+三种子A/AC+日期块分析”两阶段，均已同步远端，未qsub。独立未来GHI分析器要求跨臂/种子逐键完全一致并按BJT目标日bootstrap。M1实时`214871=R`、step500/4000，`214221=Q`，队列2/3；无step2000正式验证或独立审计，故继续保留门禁。

02:24 CST分析器补充：结果报告同时给出两站等权RMSE的绝对差、相对变化百分比、站点/种子/时效分组，以及按目标日期配对bootstrap区间；合成实验确认正向收益会得到负的AC−A与相应负相对变化。代码、测试及独立SHA清单再次同步服务器并通过校验。活动M1仍运行至step500；新GHI作业尚未提交。

02:25 CST I/O优化：预测银行将重叠未来帧的太阳几何 sidecar 做有界内存缓存，只存 `cosSOZ/cosRAA/day_mask`，未来AGRI读取路径仍仅限 `paths[:8]`。本地因果见证及服务器语法/哈希复核通过，更新脚本已同步。实时节点本地帧缓存达46,304个文件、较前次约38k继续增加；M1仍step500、进程活、GPU短采样18%，无step2000验证。继续保留当前作业，等待冷缓存完成和正式验证。

02:27 CST实时复核：PBS 214871.tc6000 仍为R，账户全队列仍为2/3（214221排队、214871运行）。M1 recent_steps.json已到step 600/4000（最后记录02:25:50），训练主进程仍在，缓存目录按年分层存在；进程累计写入约30.4 GB，说明缓存/数据写入仍在推进。GPU 1单次采样约34.5 GiB、利用率0%，属于瞬时值，不能据此判断停止；step-2000独立验证尚未产生。继续等待当前作业，不并发占用其节点或共享数据链。

02:32 CST 噪声来源排查与下一单因素候选：已读当前 M1 的目标来源覆盖和独立的 1,024 窗口验证来源差异诊断。当前混合训练标签中两站、16个时效的 CPP覆盖约31.8%–32.2%，其余约67.8%–68.2%由观测AGRI经R形成伪目标。验证诊断仅在CPP/R共同有效支持上：合并 R−CPP RMSE=8.96 COT；四里CPP<1子集R均值4.93、其中6.13%达到COT≥30；该站CPP≥30子集CPP均值41.78、R均值32.89，R达到≥30的比例56.38%。竺家亦有近零区偏厚（R均值2.65）及厚云低估（均值差约−10.7）的信号。CPP是检索参考，非独立真值；这组诊断提出标签噪声假设，不证明其造成GHI退化。

已准备隔离候选目录 cpp_only_noise_ablation_20261019：同初始化、日期划分、固定32,000训练行、seed42、batch8、AdamW、4,000步及冻结R，只在CPP与R有效共同支持上使用CPP监督，缺失处不以R补标签；COT损失权重固定为当前混合M1训练期校准值0.005891630984842778，避免同时调权。测试目标掩膜的CPU见证代码已写，训练脚本、唯一输出目录PBS及协议已同步到服务器，SHA-256复核、py_compile与bash -n通过。服务器当前conda环境导入PyTorch时等待共享文件系统，目标掩膜运行见证未完成；已终止仅由本轮启动的两个挂起CPU见证进程，未影响PBS训练。未提交新训练：当前M1继续运行，02:32进度step700/4000；账户队列2/3，step-2000验证和同掩膜审计尚未出现。下一步仍以当前M1正式审计结果决定是否启动该单因素候选，随后验证预测COT的GHI增益。

02:35 CST 实时推进：PBS 214871仍R，账户队列2/3；M1仍step700，step-2000验证未出现。当前节点本地缓存增至48,828个AGR​​I帧npz（相对预期train+val唯一帧约55,733已约87.6%），228 GiB量级；GPU 1采样利用率100%，说明训练仍能计算，低采样主要受数据加载/本地缓存节奏影响而非GPU故障。CPP-only后续作业已加入复用同一node22上已验证的M1原始帧缓存（仅当缓存存在且超过1,000帧，否则自动回退私有缓存），避免重复冷读；同时移除PBS对pytest的依赖，改为standalone合成掩膜见证。更新文件的SHA-256和bash -n再次通过。未提交新训练，继续等M1正式验证门。
02:37 CST live poll of the same PBS handle: 214871 remains R and 214221 remains Q (2/3 unfinished); M1 recent_steps still step700, no step2000 validation. On gpu2 the training PID 19003 is alive, has switched between filesystem wait and compute; cache rose from 49,170 to 49,316 files in about 50 seconds and GPU sample was 59%. This confirms ongoing cache/data work, not a terminal job. No rerun or competing bank job submitted.

02:40 CST future-GHI bank启动记录：第一次PBS 214878在仅完成环境/prologue后以exit_status=1退出，原因为bash set -u下空数组展开；没有运行模型、没有打开卫星/CPP/GHI载荷、没有生成bank。已把恢复参数改为显式字符串条件展开，增加standalone empty/nonempty flag witness；远端sha256sum、bash -n和flag witness全部通过。检查全账户队列为2/3、确认无部分bank后重新提交214879，当前R、node22 GPU0；214871 M1仍R、node22 GPU1，214221 Q，现3/3。重提后日志通过flag witness及GPU/磁盘检查，bank Python进程存活但处在共享文件系统等待，尚未写出progress或manifest；无重复任务。M1同步推进到step800/4000，无step2000验证。继续保留两个运行作业，暂无新的qsub名额。
02:41 CST future-GHI bank has crossed startup: PBS 214879 is R on GPU0 and has written BUILDING progress for 8/33,253 train/validation sequences (256 forecast instances); first batch 22.4 s, peak allocation 4.97 GiB, contract hash recorded, test_payload_opened=false and future_observed_agri_read=false. Arrays are being initialized/written; this is bank construction progress, not a GHI score. M1 is still step800/4000 on GPU1. Queue is full at 3/3; no further jobs until a slot clears.

02:44 CST live check: 全账户PBS为214221(Q)、214871(R, cot_aux_svp, walltime 02:35:28)、214879(R, forecast bank, walltime 00:05:13)，共3/3。214879 train/validation bank 已写到520/33,253序列（16,640个forecast instances），最近一个batch约2.72序列/s，峰值CUDA分配3.68 GiB；尚在BUILDING，没有终态manifest。日志和progress均确认test_payload_opened=false、future_observed_agri_read=false。当前batch虽占用显存不高，但不中断扩批：检查恢复路径发现resume模式目前以写入模式重开index.csv，重启会丢失已完成序列索引；现作业持续运行更安全，等完成后再修复恢复实现并做吞吐profile。214871仍R；本次仅确认其PBS存活，未发现新正式验证文件。队列满，不提交新作业。下一关为bank完整性/因果/键审计，之后在资源允许时运行未来GHI配对A/AC筛选pilot；COT/AGRI结果本身不替代GHI收益。

02:47 CST未来GHI打包门修正：复核固定历史窗口索引后确认bank保存的两张末帧是输入历史的 `init-15` 与 `init`；打包器实际已按 `target-30/target-15/target` 正确选择样本，但旧manifest把历史帧相对init标成`[-30,-15]`，合成测试又用两个相同常数，无法发现时间标签错误。已将metadata改为`[-15,0]`，完善context注释，并让合成历史两帧取不同值；另将打包脚本SHA写入pack manifest。局部`test_future_ghi_context.py`通过`FUTURE_GHI_CAUSAL_PACK_WITNESS_PASS`，服务器swc下py_compile及同一见证也通过，更新后的`FUTURE_GHI_SHA256SUMS`全项校验通过。该修正仅改变未来pack的可审计时间说明/见证，不改在运行的bank生产器或A/AC科学输入定义。

02:47 CST实时PBS复核：214221=Q、214871=R、214879=R，队列3/3。214879 bank当前1096/33,253序列、35,072 forecast instances，最近batch 3.35序列/s、峰值CUDA分配3.68 GiB，仍BUILDING；test与future observed AGRI标记均false。214871的recent_steps.json推进至step1100/4000，仍未到step2000正式验证；不以训练损失代替验证结果。继续保持现有两项运行，不qsub，不开test。bank完成后以修正的pack代码生成并核对时间窗，然后运行未来GHI A/AC配对pilot。

02:53 CST未来GHI处理臂增强：原A/AC总对比同时改变历史COT和未来预测COT，不能单独归因未来预报。已保留A(无COT)与AC(历史+未来预测COT)，增加AH(仅初始化时已观测历史COT；所有未来COT置零)，并把AC−AH设为未来预测COT增量、AC−A设为无COT部署基线上的实际增益、AH−A作为历史COT诊断。三臂共享样本/模型容量/初始化/行序/优化器/标签/归一化/配对seed。更新了头训练、按BJT目标日bootstrap审计、PBS运行器及`FUTURE_GHI_GATE.md`；另将训练输入上下文掩膜作为纯函数并加入合成校验。A/AC上下文见证、三臂日期块bootstrap合成审计均在本地通过；远端swc的py_compile、基础见证、旧全部哈希及runner `bash -n`通过。

三臂专用Torch CPU见证在登录节点等待共享文件系统锁（进程40020约95秒无I/O进展），已只终止本轮启动的这个合成见证，没有动PBS训练/推理。该见证已加入未来GHI PBS runner，由下一次正式分配的计算节点预检运行；不单独提交作业。更新的远端`FUTURE_GHI_GATE.md`明确：可用收益需AC−A改善；只有再见到AC−AH改善，才归因于未来预测COT。test/buffer继续关闭。

02:54 CST live check：214221=Q、214871=R、214879=R，3/3。214879 bank进度1992/33,253（63,744 forecast instances），最近batch 1.64序列/s，峰值CUDA 3.68 GiB；进程继续写数组，test/future observed AGRI仍false。214871 recent_steps为1300/4000，step-2000验证未产出。当前不qsub，保留有效作业，未来COT训练收益尚未得到验证。

02:55 CST follow-up: future bank advanced to 2,120/33,253 sequences (67,840 forecast instances), latest batch 2.39 seq/s, peak CUDA 3.68 GiB; PBS214879 remains R and `pt_main_thread` is alive. M1 PBS214871 remains R, recent_steps=1,300/4,000; GPU1 sampled 71% utilization and 34.5 GiB, bank GPU0 sampled 25% and 5.2 GiB; do not infer from one GPU sample, but both assigned devices are actively used. Unallocated/shared GPU2 was not used. Full queue remains 214221=Q, 214871=R, 214879=R (3/3). Three-arm COT-source design and gate are synced; final remote SHA256 list and PBS shell syntax pass. Existing NumPy-only context/bootstrap witnesses pass locally; Torch-specific arm witness is intentionally deferred to the future GHI PBS compute-node preflight because login-node import blocked on shared filesystem. No test/buffer opened, no GHI score yet.

02:57 CST live continuation: PBS 214871 and 214879 remain R; 214221 remains Q, full account 3/3. Forecast-COT bank is now 2376/33,253 sequences (76,032 forecast instances), BUILDING, latest batch 2.75 seq/s; no test/future observed AGRI access flags. M1 remains at the last persisted step 1300/4000; process PID19003 remains alive on node gpu2 and assigned GPU1 sampled 97% utilization with 34.5 GiB allocated. The PBS output log `experiments/cot_aware_simvp_20261017/cot_aux_svp.o214871` confirms correct GPU assignment and all M1 source hashes; step 2000 validation/audit output does not exist yet. Continue current jobs; no new submission and no test read.

03:01 CST 未来GHI诊断准备完成：新增预测站点中心COT×时效分层审计，输出各层A无COT、AH仅历史COT、AC历史+预测COT的RMSE/MAE/偏差与配对差；只读validation pack/逐样本预测，不读CPP、未来真实AGRI、test或buffer，也不用于挑模型。本地py_compile与合成分层见证通过；新脚本、测试、PBS runner、门控文档与SHA清单已同步服务器，远端全SHA清单通过、bash -n通过，swc环境合成见证输出FUTURE_GHI_COT_STRATA_WITNESS_PASS。PBS实时复核03:00:46：214221=Q、214871=R、214879=R，队列3/3。214871 recent_steps到1500/4000，尚无step2000验证；214879 bank到2952/33253序列（94,464个forecast instances），BUILDING，约3.50序列/秒，peak CUDA分配3.68 GiB，test_payload_opened=false、future_observed_agri_read=false。暂不提交新作业；现有bank按当前速率仍需较长时间，优先等待其完成，不中断。
03:03 CST后续在线核验：PBS全队列仍3/3（214221=Q、214871=R、214879=R）。214871训练进程PID 19003在gpu2存活，recent_steps已到1500/4000；step记录短时间未变但分配GPU1采样100%利用、34.5 GiB显存，判为仍在计算而非已完成/已挂死，step-2000审计仍未出现。214879 bank进度3144/33253，100,608 instances，最后批次1.98序列/秒；COT/AGRI数组持续增长，因果门禁仍false/false。更新后的cot heartbeat已设为每30分钟持续推进并要求每轮完成可复核事项，直到达成GHI收益门或真实阻塞；本轮不改论文、不解封test、不提交超限任务。
03:04 CST续查同一PBS handle（不是仅凭状态文件判活）：214871的训练进程PID19003在gpu2为R，GPU1采样98%利用、34.5 GiB，recent_steps已推进至1600/4000；214879的bank进程PID43865在gpu2为R，进程CPU约62%，GPU0分配约5.2 GiB，写入/读计数持续增加，progress推进至3272/33253、最近批速约2.0 seq/s。PBS仍214221=Q、214871=R、214879=R，3/3；没到step2000或bank终态，不作模型结论，不提交新作业。
03:05 CST对同一作业再次验证：214871仍R、M1 step仍1600（约2分钟未跨100步日志点），实际训练PID19003存活，内核短时处于lock_blkgrp_sync，GPU1取样17%；这是共享文件系统等待的瞬时状态，不能据此重启或判死。214879仍R，bank进程43865存活，progress一分内从3336推进至3464/33253，最近写入时间03:04:50；该推进证明bank作业未卡死。保持两任务，不干扰运行中的输出。
03:06 CST 数据缓存资源审计：node22本地frame cache现有107,089个文件（53,549个npz及对应lock），占252 GiB；`/tmp`所在盘共1.5 TiB、剩余约509 GiB、使用64%。该缓存由M1与forecast bank共享；未见空间将立即耗尽的迹象，但继续监控容量，不清理正在使用的数据。该轮仅记录资源风险，没有暂停或重启进程。
03:06 CST阶段进度复核：PBS仍214221=Q、214871=R、214879=R（3/3）；M1 recent_steps推进到1700/4000，bank推进到3656/33253。两项都已跨过上一条记录的步数且bank progress更新时间为03:06:10，现阶段继续运行比改参数/重启更能接近未来GHI配对结果。预估仅基于当前吞吐，实际以完成标记和正式验证审计为准。
03:08 CST实时续查：214221=Q、214871=R、214879=R，PBS 3/3。214871仍最近步1700/4000（记录03:05:52），但训练PID19003在gpu2存在、GPU1取样99%利用、进程线程由共享盘锁等待转为futex等待；未见step2000产物，继续观测而不重启。214879 bank进程43865存活，progress在约1分钟内从3720推进至3848/33253，GPU0分配约5.2 GiB，`test_payload_opened=false`及`future_observed_agri_read=false`仍保持。node22缓存npz数53699，约接近预计train/val唯一帧量，/tmp剩508 GiB。
03:09 CST Luna独立只读监控复核：214221=Q、214871=R/GPU1、214879=R/GPU0，均有存活主进程；两卡采样约100%/76%利用。M1为1700/4000，step-2000验证未产生；bank推进至3912/33253（125,184 instances），状态BUILDING，test_payload_opened=false、future_observed_agri_read=false。node22缓存约253 GiB，本地盘剩约508 GiB；共享/public盘剩余约104 TiB。无OOM/失败，先保留作业。自动化唤醒频率已从30分钟调整为15分钟，以减少bank完成后启动后续GHI头的等待；无实质变化继续静默。
03:09 CST阶段门复核：全账户PBS仍3/3，214221=Q、214871=R、214879=R。M1 recent_steps推进到1800/4000，step-2000验证尚未写出；forecast SimVP→frozen-R bank推进到4040/33253，仍BUILDING，未见bank_manifest.json。当前仅验证任务在推进，不存在GHI结果；后续按三臂门执行，不用COT RMSE代替GHI收益结论。

03:10 CST continuous-work check: Luna independently confirmed the live PBS state (214221=Q, 214871=R, 214879=R; account queue 3/3). M1 advanced to 1800/4000; step-2000 formal validation is not yet present. The frozen SimVP→R forecast bank advanced to 4,232/33,253 sequences (135,424 instances), remains BUILDING, with both test/future-observed-AGRI flags false and no terminal manifest. GPU1 (M1) sampled 85% utilization/34.5 GiB; GPU0 (bank) sampled 31%/5.2 GiB. Cache ~254 GiB; /tmp has ~508 GiB free. Local causal-context and COT-strata synthetic witnesses pass; paired A/AH/AC bootstrap witness passes. Python compilation passes. Two local Torch-specific witnesses cannot run because this workstation has no PyTorch; they are intentionally deferred to the allocated PBS compute-node preflight. No new training result or GHI score exists, no test/buffer payload was accessed, and no job was submitted because the account limit is full.

03:15 CST live continuation: PBS remains 214221=Q, 214871=R, 214879=R (3/3). M1 has reached 2000/4000, but `validation_step_2000.json` has not yet been atomically written; the process is alive and GPU1 allocation remains active, so the step-2000 validation is still pending and no audit/result is inferred. Forecast bank advanced to 4,936/33,253 sequences, remains BUILDING, and both causal exclusion flags remain false. Prepared the next one-factor CPP-only auxiliary-loss ablation for the mixed-label-noise hypothesis; verified its local SHA manifest, Python syntax, PBS shell syntax, synchronized exact source files to the server, and verified remote SHA256SUMS plus `bash -n`. It is not submitted: wait for the current mixed-M1 validation/audit and an actual PBS slot; do not run alongside M1. No test/buffer payloads read.
03:16 CST Luna follow-up (read-only): M1 step remains 2000; validation file/log not yet emitted. Main PID and all four validation workers are alive; workers are using CPU while the main process waits, consistent with validation data processing rather than a dead process. GPU1 sampled 0% at that instant, with 34.5 GiB allocated. Bank reached 5,000/33,253, active; causal flags remain false. Keep both current jobs running; queue remains full, so no new submission.
03:17 CST validation liveness confirmation: Luna sampled M1 worker I/O over 19 seconds; a validation worker read counter advanced by ~6.0 GB while the step-2000 validation file was still absent. Main PID 19003 waits for the active workers; this is slow data-consuming validation, not evidence of a deadlock. M1 GPU1 had 34.5 GiB allocated but 0% utilization at the sample. PBS remains full (214221 Q, 214871/214879 R); bank reached 5,192/33,253 and remains causal-gated. Keep jobs undisturbed.
03:18 CST live validation check: step-2000 artifact is still not emitted, but the four newly spawned validation workers (PIDs 78761–78764) remain active at ~80% CPU each; Luna measured roughly 5 GB additional reads over 18 seconds, including filesystem reads. This is positive evidence that full validation is still consuming data, not a deadlock. Bank advanced to 5,384/33,253 sequences; causal exclusion markers remain false. PBS 3/3 (214221 Q, 214871/214879 R). No restart or new submission.

03:20 CST live check: 全账户PBS仍3/3（214221 Q cot_interval_pac；214871 R cot_aux_svp；214879 R 
ghi_forecast_ba）。M1 step-2000 checkpoint与正式验证文件已落盘：候选val_equal_season_rmse=0.2856219593，同合同MSE控制step-2000=0.2856157556，候选暂高0.0022%；DJF/MAM/JJA/SON均仅有极小差异，AGRI +1%护栏当前通过，但完整COT/厚云同掩膜审计未完成，且没有GHI结果。PBS 214871仍R；recent_steps文件尚停在step2000快照，不能据此断言已到下一百步。forecast-COT bank进度5704/33253（182,528个instances），BUILDING；test_payload_opened=false，future_observed_agri_read=false。队列满，不提交CPP-only新训练；等M1到4000并通过独立审计、bank完成且PBS槽释放后再按协议决策。

03:22 CST verified progress: 同一PBS handle仍为214871=R、214879=R，队列214221=Q/214871=R/214879=R（3/3）。M1训练最近记录已从step2000推进至step2100；checkpoint/validation尚未到step4000。forecast bank由5704推进至5896/33253（188,672 instances），数组、index.csv和progress.json均于03:21更新；状态BUILDING，test_payload_opened=false、future_observed_agri_read=false。当前R状态与新写入时间证明两项作业仍在实际推进；不重启、不提交新作业。

03:36 CST吞吐优化记录：为利用分配A800，先只读采样原batch8 GPU0 12秒，绝大多数样本SM 0%，峰值CUDA分配约3.68 GiB。重启自有bank作业并保留数组/键，按完整边界恢复；batch64首个测得批次64序列/24.70秒，2.59 seq/s，峰值CUDA allocated 28.94 GiB。batch128实测GPU0显存70.2/81.9 GiB，低于15%余量规则，已停止；未提交额外任务。当前PBS214882=R、batch96、GPU0，03:36瞬时采样47.1/81.9 GiB及100%SM；bank progress文件最近完整边界6624/33253，训练作业214871仍R且recent_steps到2700/4000，step-2000 AGRI验证仅为中期护栏，完整COT审计和GHI比较仍未出。队列214221=Q、214871=R、214882=R，共3/3；无test或真实未来AGRI读取。

03:39 CST批处理profile结束：batch=96首个checkpoint覆盖96序列用时48.40秒，1.98 seq/s，峰值CUDA allocated43.38 GiB；低于batch64首批64序列/24.70秒（2.59 seq/s）且显存更多，故停止96。batch128此前显存70.2/81.9 GiB，不满足15%余量。最终固定回batch64，不再反复profile；在已写盘7008基础上回退至6976重算32序列。PBS214882已结束，214883已提交并排队；提交前核对全账户队列214221Q、214871R两项未结束，提交后未结束共3/3。M1上次实证step2700/4000，step2000 AGRI检查仅护栏；COT审计/GHI收益均未完成。

03:40 CST live confirmation: PBS214883已由Q转R，分配node22 GPU0；bank恢复边界6976/33253，test与future-observed-AGRI排除标记均false。启动约十余秒，进程尚在初始化，尚无新的完整batch进度；不据此判停滞。PBS全队列214221Q、214871R、214883R，共3/3。

03:41 CST live check: M1 recent_steps从step2700推进到step2800/4000；PBS214871仍R。未来bank的PBS214883在node22 GPU0运行，batch64，process PID107603存活且短间隔内rchar增加约361 MB，说明仍在读取/准备数据；当前最新完整bank进度6976/33253，尚未到下一flush点。PBS队列214221Q、214871R、214883R为3/3；test_payload_opened=false、future_observed_agri_read=false。

03:46 CST live continuation: 全账户队列仍为214221=Q、214871=R、214883=R，共3/3；214882已结束，其作业日志仅保留PROGRESS 7008/33253且无traceback，当前bank从安全回滚边界6976恢复后已推进至7232/33253。PBS进程107603仍运行，GPU0分配约37.3 GiB（抽样时SM 0%，但进程和progress均有更新）；bank的test_payload_opened=false、future_observed_agri_read=false。M1 PID19003仍运行，recent_steps到2900/4000，step-2000验证文件在、step-4000尚未完成；GPU1在03:42曾抽样95%利用率。当前不重启、不提交新作业；214882退出原因不能从保留的PBS记录中确认，现有证据仅说明恢复边界之前的进度可继续使用。

03:46 CST fresh progress check: PBS 214221=Q、214871=R、214883=R（3/3）。M1从step2900推进到step3000/4000；进程仍活跃，step-4000验证和COT/厚云独立护栏未出。Forecast-COT bank仍7232/33253，最近完整batch耗时30.10 s（2.13序列/s），累计已记录223,232 forecast instances；进程107603活跃，GPU0显存约37.3 GiB，采样SM 0%，但CPU与数据读取持续增加。两项进程均未结束；test_payloads_read=0，bank test_payload_opened=false、future_observed_agri_read=false。PBS槽位满，不新开/重启任务；按既有门禁等待训练评估及bank继续推进。

03:48 CST noise-path recheck: the existing validation source audit is still `DIAGNOSTIC_ONLY` (1,024 sampled validation sequences; `test_payloads_opened=false`); on 1,643,712 common CPP/R pixels, R-vs-CPP reference RMSE is 8.960 COT units, with pooled thick fraction 12.50% for R versus 12.97% for CPP. Current mixed-M1 training coverage is separately confirmed as ~32% CPP and ~68% R fallback. This makes the already prepared CPP-only fixed-lambda ablation a justified next upstream noise test, but does not establish that fallback labels cause GHI harm or that CPP is physical truth. A code-path recheck also confirms `load_station()` expands the causal history mask to `[N,3,1,1,1]` before AH multiplication; no dimensionality patch is warranted. Training remains at step 3000/4000 and bank at 7232/33253; both processes remain live, queue 3/3, test/buffer closed.

03:49 CST verified wait/update: M1训练从step3000前进到step3100/4000；forecast-COT bank从7232前进到7744/33253（247,808 forecast instances）。PID19003与107603仍存活，短间隔rchar分别继续增加；PBS仍214221=Q、214871=R、214883=R（3/3）。step-4000审计和bank completion manifest尚未出现，test/buffer/未来真实AGRI仍关闭。继续保留当前运行，不抢占资源；释放槽后依序完成独立COT审计、pack完整性核验和A/AH/AC未来GHI对照。

03:51 CST partial-bank integrity clarification: a live index snapshot had 7,971 unique train keys while progress.json remained at 7,744. Source inspection confirms this is expected during writes: index rows are appended batchwise, while arrays/index/progress are flushed together every 8 batches (512 sequences); mid-batch index can therefore be up to 511 rows ahead. Current lead=227 is within that bound, keys are unique, and current prefix is train-only. Resume uses sequences_complete as rollback boundary and rewrites index from selected[:start_index], so partial tail is discarded/recomputed. No corruption indicated; do not audit this partial bank as complete. At terminal manifest, require index rows exactly 33,253, unique keys, full arrays, and exclusion flags false.

03:52 CST current process check: M1 progressed to step 3200/4000; bank progress remains 7744/33253 since its last 512-sequence flush, with PBS 214883 still R and process present. PBS remains 214221=Q, 214871=R, 214883=R (3/3). No completion result yet; no new submission is allowed while full. The partial index lead described above remains expected within the flush interval, not a bank-integrity failure.

03:52 CST refreshed after the previous 30-sample: PBS live state remains 214221=Q, 214871=R, 214883=R. M1 is step 3200/4000; bank has reached 8256/33253 (264,192 forecast instances), with last measured batch 25.12 s (2.55 seq/s). Both exclusions remain false for bank; training metadata reports zero test payload reads. No completion artifact yet; keep current jobs running and do not submit while 3/3.

03:53 CST Luna read-only live verification: PBS 214221=Q、214871=R、214883=R（3/3）；M1 step3300/4000，PID19003活跃、GPU1本次抽样95%；forecast bank 8256/33253且无完成manifest，PID107603存活、8秒读取计数增加约1.72 GB。短时间未刷新progress与每512条写一次的机制一致，当前无停滞证据。test_payloads_read=0，bank test_payload_opened=false、future_observed_agri_read=false。保持作业运行，等M1审计与bank完成后立即接GHI A/AH/AC筛选。

03:55 CST fresh remote check: authoritative server clock is 03:55 CST (the workstation clock was about 17 minutes ahead during this check). Full PBS queue remains 214221=Q, 214871=R, 214883=R (3/3), so no new submission is allowed. M1 durable `recent_steps.json` is still step 3400/4000; no step-4000 checkpoint, `training_complete.json`, or independent COT audit is present yet. Forecast-AGRI→frozen-R bank advanced to 8768/33253 sequences (280,576 forecast instances); last measured throughput is 2.124 sequences/s, peak CUDA allocation 28.94 GiB. Its test and future-observed-AGRI flags remain false. Continue both jobs unchanged; next gates remain M1 completion/audit and full train/validation bank integrity before packing or GHI-head training.

03:57 CST follow-up and next-ablation readiness: M1 has advanced to durable step 3500/4000 (recent_steps.json mtime 03:56); its final checkpoint, completion marker and independent COT audit are still absent. Bank remains at its last 512-sequence flush, 8768/33253, and its job log confirms continued batch progress through 8768; queue remains 3/3. The prepared CPP-only ablation bundle is already on the server and its SHA256SUMS pass; source files also pass `py_compile` under the actual `swc` Python 3.10.18 environment. The login-node default Python 3.6.8 is incompatible and is not the job runtime; local workstation has no torch, so the synthetic tensor witness remains correctly inside the PBS runner rather than being run on a login node. No ablation training has been submitted: first wait for M1's independent same-mask audit, then recheck queue and decide against that evidence.

03:58 CST verified progress delta: within roughly one minute of the prior remote snapshot, M1 advanced from step 3500 to 3600/4000, and the forecast-COT bank advanced from 8768 to 9280/33253 (296,960 forecast instances). Latest full bank batch took 23.45 s (2.73 sequences/s; 28.94 GiB peak allocation). PBS queue remains 214221=Q, 214871=R, 214883=R (3/3); no final M1 artifact or bank manifest yet, and bank test/future-observed-AGRI exclusions remain false. This confirms forward progress; retain both jobs and do not submit a fourth task. CPP-only ablation code is syntax-checked in `swc` and ready for its existing gate after M1 audit and a free PBS slot.

04:01 CST node22 Luna resource check (read-only): queue remains 214221=Q, 214871=R, 214883=R (3/3). M1 advanced to 3700/4000; step-4000 validation/checkpoint and terminal audit are not yet present. Bank advanced to 9792/33253 (~29.4%); no manifest. M1 GPU1 sampled 84% aggregate use, with pmon varying 11% to 0% across brief samples. Bank GPU0 holds ~37.3 GiB but sampled at 0% SM and its process waits in OFS (`ofs_tjobq_timedwait_locked`); logical read count rose ~508 MB over 8 s while physical read count did not, consistent with shared/cache I/O wait rather than GPU saturation. It continues to advance, so do not interrupt or reprofile the live job. Node NVMe reports ~502 GiB free. Exclusion markers remain closed (M1 test reads=0; bank test_payload_opened=false, future_observed_agri_read=false). Revisit local cache/prefetch only as a separately measured next-run optimization after these jobs finish.

04:02 CST Luna独立计算节点核验：PBS全账户仍为214221=Q、214871=R、214883=R（3/3）。M1 durable recent_steps仍3700/4000，PID19003存活；GPU1聚合约98%，该进程pmon约81% SM、52% memory，step-4000产物/独立审计未出。Forecast-COT bank仍9792/33253（约29.4%），PID107603存活，短样本约8秒rchar增加233 MB而read_bytes未变，GPU0占用约37.3 GiB但SM采样0%，进程处于Rl；属共享存储/页缓存I/O等待特征，未见停滞证据。约500 GiB NVMe可用。bank test_payload_opened=false、future_observed_agri_read=false；M1 test_payloads_read=0。队列满，不提交新作业、不重启当前运行。

04:03 CST direct server refresh: PBS remains 214221=Q、214871=R、214883=R（3/3）。M1 durable recent_steps.json advanced to step 3800/4000, latest records through step 3800; terminal checkpoint/audit not yet confirmed. Bank's durable progress.json remains 9792/33253, but the data arrays' mtimes refreshed at 04:03 and index at 04:02, consistent with writes occurring before the next 512-sequence progress flush; no manifest yet. Full queue and test/buffer exclusions unchanged. No restart or new submission.

04:04 CST direct server refresh: PBS全队列仍214221=Q、214871=R、214883=R（3/3）。M1 durable snapshot仍step 3800/4000（文件最近04:02写入，PBS作业保持R；尚无step-4000/终态产物）。Forecast-COT bank已由9792/33253推进到10304/33253（329,728 forecast instances），progress文件时间04:04，state=BUILDING，test_payload_opened=false、future_observed_agri_read=false。bank仍在正常构建，M1保存频率为百步级，当前状态无失败或需重启证据。队列满，不提交新任务；继续等两个门禁产物。


04:07 CST Luna节点核查：PBS全队列仍214221=Q、214871=R、214883=R（3/3）。M1=3900/4000，PID19003在GPU1持续运行，短采样SM约45%→75%；目前仅有step0/2000 checkpoint与验证文件，未见step4000、training_complete或最终审计；stdout /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/cot_aware_simvp_20261017/cot_aux_svp.o214871仍未显示完成。Bank仍10304/33253，progress mtime 04:04:35，>2分钟未过下个512条checkpoint；PID107603仍存活，位于共享FS等待，8秒rchar约+253MB、物理read_bytes不增，GPU0 0–3% SM并占约37.3GiB，last batch 30.29s/2.11 seq/s。该时间低于预计512序列刷新间隔，按活进程与读活动保留，不重启。两项test/future-observed-AGRI排除标志仍关闭，节点临时盘约500GiB可用。

04:09 CST Luna增量核验：M1 recent_steps从3900推进到4000/4000（mtime 04:07），PBS214871仍R、主进程PID19003存活，GPU1短采样约27% SM；新增4个高CPU DataLoader进程，符合可能进入验证的表现，但尚未出现validation_step_4000.json、training_complete或最终审计，不能认定训练/验证完成。Forecast-COT bank从10304推进到10816/33253（+512），仍BUILDING、无manifest；test_payload_opened=false、future_observed_agri_read=false。全账户PBS214221=Q、214871=R、214883=R，3/3无槽位；不提交、不重启，等待M1验证收据和bank继续构建。

04:10 CST验证等待复核：PBS仍214221=Q、214871=R、214883=R（3/3）。M1 recent_steps维持4000（mtime04:07），PID19003和4个高CPU DataLoader子进程仍活跃，GPU1约40% SM；结果目录仍无validation_step_4000.json、training_complete或final audit，判断为验证/后处理活动中而非已完成。Bank维持10816/33253，BUILDING，尚未新增512条持久化进度；PID107603仍活跃、GPU0采样0% SM，test_payload_opened=false、future_observed_agri_read=false。未见槽位释放或终态失败，暂不干预。

04:11 CST Luna核验：PBS全队列214221=Q、214871=R、214883=R（3/3）。M1仍step4000后处理中，PID19003与4个DataLoader子进程活动、GPU1约36% SM；stdout未检出error/traceback/failed/nan，但validation_step_4000、independent audit和training_complete均未落盘。Bank durable progress仍10816/33253，无新512条flush；然而forecast数组mtime约04:10:50，forecast_agri_raw.npy约498MB，PID107603活跃、GPU0采样约26% SM且rchar持续增加；当前batch仍在处理，不判停滞。test_payload_opened=false、future_observed_agri_read=false。继续保留运行作业；队列满，不提交或重启。

04:13 CST关键结果复核：M1 `validation_step_4000.json` 已落盘，step=4000、val_windows=3140、val_equal_season_rmse=0.2856077859；同新split、同3140窗口的MSE step4000对照为0.2856004358，候选差+0.0000073502（+0.00257%），AGRI全局护栏通过且数值几乎持平。该结果不等于COT改善，更不等于GHI收益；M1仍在保存三套COT预测分数并等待PBS后置独立audit，training_complete与independent_cot_audit.json尚未出现。Forecast-COT bank已到11328/33253，last batch 19.12s、3.35 seq/s，仍BUILDING，无manifest；test_payload_opened=false、future_observed_agri_read=false。全队列214221=Q、214871=R、214883=R（3/3），不提交新作业。

04:17 CST live checkpoint: PBS full-account queue is 214221=Q, 214871=R, 214883=R (3/3). M1 has saved `step_4000.pt` and `validation_step_4000.json` (3,140 validation windows; equal-season AGRI RMSE 0.2856077859 versus matched MSE 0.2856004358, +0.00257%). The candidate COT scoring directory `cot_score_aux_4000/` exists but contains no score files yet; original/MSE COT scores, independent audit, and `training_complete.json` are absent, so M1 remains in postprocessing and is not complete. Node22 shows PID19003 on GPU1 using 34.5 GiB; GPU1 sampled 6% utilization at 04:16, consistent with scoring/postprocessing rather than a failed job. Forecast-COT bank is durable at 11840/33253 (378,880 forecast instances), BUILDING; progress last flushed 04:14 and job PID107603 remains alive on GPU0 using 37.3 GiB. Its test/future-observed-AGRI exclusion flags remain false. Queue is full; do not submit or restart. Next gates remain final same-mask COT scores plus independent audit, and complete bank manifest/integrity before building packs and submitting the paired A/AH/AC GHI pilot.

04:19 CST new artifact: M1 produced `cot_score_aux_4000/cot_aux_validation_score.json`, status `PASS_CPP_COMMON_SUPPORT_FP64`, on the monthly-block validation split (3,140 sequences; 46 CPP-supported dates; zero test reads). Candidate center-3×3 station/lead-macro COT RMSE is 10.74063 and full-patch RMSE is 11.02875; pooled thick-cloud CSI is 0.42517. These are candidate-only figures: original-SimVP and MSE-step-4000 scores on the exact same support plus the independent audit are still pending, so no COT gain/loss conclusion is made. The forecast-COT bank advanced from 11,840 to 12,352/33,253 (395,264 instances; 2.686 sequences/s latest batch), remains BUILDING, and both exclusion flags are false. PBS remains full (214221=Q, 214871=R, 214883=R); do not submit/restart. Continue to the same-mask comparisons and audit, then complete bank validation before the matched-input GHI pilot.

04:20 CST Luna cross-check: bank progressed 11,840→12,352 during 04:16:47–04:18:03; by 04:19:33 durable progress was still 12,352, but forecast arrays were modified at 04:19 and PID107603 logical reads increased ~9.4 GB (last 8 s +41 MB), with shared-filesystem lock wait and no physical read increment. Treat as active batch before the next 512-sequence flush, not a confirmed stall. M1 candidate score/forecast arrays are present; `cot_score_original_simvp/` exists but is empty, MSE score directory is absent, and independent audit/`training_complete.json` are still absent. PID19003 plus four workers remain active (workers ~79–83% CPU; GPU1 sampled ~8%). Queue at 04:20:01 remains 214221=Q, 214871=R, 214883=R (3/3); all exclusion flags remain closed. Keep current jobs running; next work is still same-mask original/MSE comparison and audit, then bank completion/integrity and paired GHI screening when a PBS slot frees.

04:23 CST downstream-alignment check: reran `test_future_ghi_context.py` locally; `FUTURE_GHI_CAUSAL_PACK_WITNESS_PASS`. The synthetic witness verifies the three-frame T−30/T−15/T context, observed-history versus forecast source indices at +15/+30/+240 min, and the COT history-only mask boundary; it reads no project payload and does not replace the future pack's key-level audit. This removes a code-level time-context ambiguity before interpreting the planned A/AH/AC comparison. Remote 04:22:59 check still shows queue 3/3; only the candidate COT score is present, bank durable progress is 12,864/33,253, and the forecast arrays' mtime is 04:22, so the bank process remains actively writing ahead of its next durable flush. Do not restart either live job.

04:24 CST exact same-mask preliminary comparison: `cot_score_original_simvp` is now present and its `validation_dataset_indices.npy` is byte-identical to the COT-aux candidate indices. On the common 3,140-sequence / 46-CPP-date support, original SimVP center-3×3 station-lead macro COT RMSE is 10.6641613 versus 10.7406269 for COT auxiliary (candidate is +0.0764656 worse); full-patch RMSE is 10.9526937 versus 11.0287499 (+0.0760561 worse). Pooled thick-cloud CSI is 0.4139968 versus 0.4251665 (+0.0111697 better). These are point comparisons, not yet the paired date-block interval; MSE-step-4000 score and independent FP64 audit are still missing. The mixed-label COT auxiliary candidate therefore does not yet improve overall COT RMSE, despite a small CSI increase. Keep the planned CPP-only single-factor noise ablation as the next diagnostic after M1 audit; keep the independent original-SimVP forecast bank running because it is required for the direct downstream A/AH/AC GHI test.

04:26 CST follow-up: original-SimVP COT score is present and byte-identical validation indices to the aux candidate were confirmed; center-3×3 RMSE 10.6641613 (original) vs 10.7406269 (aux), pooled thick CSI 0.4139968 vs 0.4251665. MSE-step-4000 score remains in the same live M1 postprocessing sequence; no independent audit or completion marker yet. Forecast bank advanced another 512 to 13,376/33,253 (428,032 forecast instances; latest batch 2.62 sequences/s), still BUILDING with test/future-observed-AGRI flags false. PBS remains 3/3. Once MSE score lands, run the prewritten FP64 date-block audit; retain the direct forecast-COT bank for the later A/AH/AC GHI comparison.

04:28 CST verified progress: forecast-COT bank durable count advanced 13,376→13,888/33,253 (444,416 forecast instances; latest batch 27.85 s, 2.30 sequences/s), so the earlier ~2-minute durable gap was the normal 512-sequence flush cycle. M1 original-SimVP score remains complete; MSE-step-4000 score is still pending while job 214871 remains R and GPU1/process are present. Bank job 214883 remains R; test/future-observed-AGRI exclusions remain false. Queue remains 3/3 (214221 Q, 214871 R, 214883 R). Keep both jobs running; execute the paired date-block audit immediately after MSE scoring and `training_complete.json` appear.

04:30 CST three-arm score completion: MSE-step-4000 `cot_aux_validation_score.json` and `training_complete.json` are present; the training PBS script has launched `audit_cot_aux_validation.py` in the same allocation (live PID29195 at 04:29:27; independent audit JSON not yet written). Same-support center-3×3 RMSE: original 10.6641613, equal-budget MSE 10.7501119, mixed-label COT auxiliary 10.7406269; full-patch: 10.9526937, 11.0389034, 11.0287499. Pooled thick CSI: 0.4139968, 0.4238940, 0.4251665. Thus both fine-tunes are worse than original on point RMSE; COT auxiliary has a small thick-CSI gain, whose date-block interval and low-COT/AGRI guards remain pending. Do not select or launch CPP-only from validation point estimates alone; use the completed independent audit to decide. Forecast bank remains at 13,888/33,253, process alive, with exclusions false; PBS remains 3/3.

04:35 CST CPP-only noise ablation submitted as PBS `214884.tc6000` after M1's independent audit completed and the full account queue showed two unfinished jobs (214221=Q, 214883=R); post-submit queue is 3/3 including 214884=R. The candidate changes only the COT target source to CPP-only on common valid support; it holds seed 42, 4,000 updates, batch 8, AGRI objective, frozen R and the calibrated mixed-M1 lambda fixed. Server checksum verification and `CPP_ONLY_TARGET_WITNESS_PASS` completed in the job prologue; unique output path was clean. No test/buffer or future observed AGRI input is used. M1 gate outcome: overall fail; candidate COT-RMSE was +0.07647 versus original in center 3×3 (paired 95% CI [-0.09532,+0.24643]) and near-zero false alarm increased versus MSE (paired 95% CI [+0.00054,+0.00195]); thick CSI improved versus MSE but not enough to pass all gates. The separate original-SimVP forecast-COT bank remains at 13,888/33,253 and is required for the direct GHI A/AH/AC pilot.

04:36 CST startup update: PBS 214884 remains R on node22 with exactly one allocated A800 80 GB GPU. The checksum and CPP-only synthetic target witness passed; the training script is active and has allocated about 3.8 GiB, but no step/progress artifact has appeared yet. This is startup/preparation evidence, not training progress. PBS queue is 214221=Q, 214883=R, 214884=R (3/3). The forecast-COT bank is 14,912/33,253 sequences (477,184 forecast instances), BUILDING; its test-payload and future-observed-AGRI flags remain false. Keep both active jobs; no new submission or restart.

04:37 CST CPP-only preflight completed with `status=PASS`, fixed mixed-M1 lambda 0.005891630984842778, CPP-only targets, frozen R, and `test_payloads_read=0`. Training process on node22 is actively preparing/reading data; it has not emitted its first step record. GPU memory rose from 3.8 to about 14 GiB while utilization remained near zero, and process I/O counters increased, so this is active input preparation rather than confirmed training progress or a stall. Bank 214883 remains 14,912/33,253 with test and future-observed-AGRI exclusion flags false. Queue remains 214221=Q, 214883=R, 214884=R (3/3).

04:39 CST single-GPU profile passed on the allocated A800: four discarded warm-up/profile steps, peak allocation 30.42 GiB of 79.33 GiB, mean throughput 9.72 samples/s after warm-up; the formal contract remains batch 8 and 4,000 updates. `preflight.json`, `profile.json` and `run_metadata.json` are present with fixed rows/hash and `test_payloads_read=0`; step-0 validation/checkpoint has not yet appeared, so the model is still in pre-training evaluation/setup. Bank advanced to 15,424/33,253 sequences (493,568 forecasts), still BUILDING and with both future/test exclusion flags false. Full queue remains 214221=Q, 214883=R, 214884=R.

04:40 CST live runtime witness: candidate 214884 is still R; the allocated GPU1 rose to about 34.5 GiB and sampled 54% utilization, consistent with the initial validation/setup using the model rather than a dead process. Profile remains PASS (30.42 GiB peak, 9.72 samples/s); no formal step record/checkpoint yet. Bank 214883 has not advanced its durable count since 15,424, but its process remains Rl and logical reads increased about 2.0 GB over 22 seconds; treat it as active I/O before a durable batch flush, not as stopped. Test/future-observed-AGRI exclusions remain false. Queue still 3/3; no restart or submission.

04:42 CST follow-up: bank 214883 advanced another 512 sequences to 15,936/33,253 (509,952 forecasts); both exclusion flags remain false. Candidate 214884 still has no step-0 checkpoint: its parent is waiting on four active DataLoader workers, each using roughly 80% CPU, while the allocated GPU1 remains around 9% utilization at this sample. That is consistent with the full train-only validation pass preparing the first checkpoint; keep the job running and recheck rather than interrupting it. Queue remains 214221=Q, 214883=R, 214884=R (3/3).

04:43 CST CPP-only run reached its initial checkpoint: `validation_step_0000.json` (3,140 validation windows; equal-season AGRI RMSE 0.2857435) and atomic `step_0000.pt` are now present. The run has therefore passed preflight, profile and baseline validation and entered the formal 4,000-update stage; this is still not a treatment result. Forecast-COT bank remains 15,936/33,253 at the latest durable check, with the process active and no test/future-observed-AGRI reads. Queue remains full at 214221=Q, 214883=R, 214884=R.

04:45 CST CPP-only candidate reached formal step 100/4,000. Recent finite losses show no numerical failure; across the first versus last 20 of the saved 100 steps, mean AGRI loss was 0.1098→0.1164 and mean CPP COT loss 0.0380→0.0456, which is too early and batch-variable to judge efficacy. Both allocated GPUs were sampled at 100% utilization, with approximately 37.3 GiB on bank GPU0 and 34.5 GiB on candidate GPU1. Forecast bank advanced to 16,448/33,253; test/future-AGRI flags remain false. PBS stays at 3/3.

04:46 CST CPP-only candidate reached step 200/4,000; the 100-step progress record is finite and the process remains active on GPU1 (about 34.5 GiB, sampled 100% utilization). From step 100 to 200 took about 85 seconds, consistent with a healthy training rate. Forecast bank remains 16,448/33,253 at the latest durable flush; its process is still present, with shared-filesystem wait observed between flushes. No test/future-AGRI reads; PBS queue remains 3/3.

04:49 CST fresh server check: CPP-only candidate advanced to step 300/4,000; the last 100 batches remained finite (mean AGRI loss 0.1216, COT loss 0.0401), which is training-health evidence only, not a validation gain. Forecast bank advanced to 16,960/33,253 (about 51.0%; latest complete batch 2.54 sequences/s); `test_payload_opened=false` and `future_observed_agri_read=false`. PBS jobs 214221=Q, 214883=R, 214884=R; no slot available.

04:50 CST downstream readiness audit: the metadata-only future-GHI crosswalk audit reports 781,504 unique train/validation station×init×lead keys, zero split mismatches, and zero AGRI/CPP/GHI-array/test/buffer payload reads. Source inspection confirms the planned GHI screen uses a shared head initialization, date/sample order, labels and optimization schedule for A/AH/AC; AC−AH remains the future-COT-specific contrast. Pack generation correctly refuses incomplete bank manifests, so the full bank remains a hard dependency before GHI training.

04:50 CST formal training progress: CPP-only candidate is at step 400/4,000; the 100-step snapshot is finite (mean AGRI loss 0.1121, COT loss 0.0383), and GPU1 remains allocated with roughly 34.5 GiB. This is still optimization telemetry, not validation evidence. Forecast-COT bank remains 16,960/33,253 at its latest durable snapshot; the process is live. PBS account is 3/3 (214221 Q, 214883 R, 214884 R), so no additional job can be submitted until a slot clears.

04:51 CST verified continuation: CPP-only candidate reached step 500/4,000; last-100 mean losses are finite (AGRI 0.1146, CPP COT 0.0402). Forecast-COT bank advanced to 17,472/33,253 (52.5%), last completed batch 3.27 sequences/s; both test/future-AGRI exclusion flags remain false. The jobs 214883 and 214884 are still R, with 214221 Q, so PBS remains 3/3. No model selection is made from training losses.

04:52 CST CPP-only candidate advanced to step 600/4,000; bank's last durable count remains 17,472/33,253 while PID107603 continues running between 512-sequence flushes. No test or future observed AGRI reads are recorded. Queue remains full at 214221=Q, 214883=R, 214884=R.

04:56 CST夜间持续推进复核：全账户PBS为214221=Q、214883=R（forecast AGRI→冻结R bank）、214884=R（CPP-only COT辅助候选），共3/3。214884的`recent_steps.json`已于04:56更新至step 900/4000，最近100步训练记录有限且无数值异常；GPU1约34.5 GiB、采样利用率99%，仅作为训练健康信息，不作效果结论。预测COT bank的最近完整progress为17,984/33,253（575,488 forecast instances，BUILDING；test_payload_opened=false、future_observed_agri_read=false）；progress文件时间04:55，但底层AGRI数组在04:56继续写，PID107603仍运行并使用GPU0约37.3 GiB。视为512序列批量flush间隔，不判停滞、不重启。队列满，不提交新任务。下一步等待候选2000步验证节点及bank完整manifest；完成后执行预注册验证审计，再以同键同样本构建A/AH/AC GHI小头对照。正文不改，test保持关闭。

04:59 CST夜间监控节点：CPP-only候选已推进至step 1000/4000，最近100步平均 AGRI loss=0.11433、CPP COT loss=0.03799，训练正常但尚未到预定正式验证节点。预测COT bank仍完整落盘17,984/33,253；progress文件未刷新，但原始AGRI数组04:58:53仍有写入，进程活跃，故继续等待下一512条批次flush。全账户PBS仍214221=Q、214883=R、214884=R，共3/3；test及未来真实AGRI读取标记保持false。没有提前选模或提交新训练。等待bank完整收据和候选后续验证，随后按门禁进入预测COT→GHI同键配对验证。

05:01 CST继续监控确认有新增数据：预测COT bank从17,984推进至18,496/33,253（591,872条forecast实例），状态BUILDING；最近512序列批次耗时39.38秒、1.63序列/秒，test_payload_opened=false且future_observed_agri_read=false。CPP-only候选推进到step 1100/4000，checkpoint/训练仍进行中，验证点尚未到。PBS仍为214221=Q、214883=R、214884=R（3/3）；当前不提交新作业，保持两个实验连续运行。

05:04 CST继续推进复核：CPP-only候选达到step 1300/4000，仍在训练；未来COT bank从18,496推进到19,008/33,253（608,256预测实例），最近512序列用时23.83秒、2.69序列/秒。test和未来真实AGRI读取标记仍为false。PBS维持214221=Q、214883=R、214884=R，共3/3；资源释放前不加作业。当前还没有COT正式对比或GHI收益结果，继续跑到验证/完整性收据后立刻按既定门禁推进下游配对实验。

05:05 CST监控：CPP-only模型step 1400/4000；bank完整计数仍19,008/33,253，但AGRI/COT/geometry数组的mtime更新到05:05，构建进程仍活跃，等待下一批512序列提交progress。抽查候选目标实现：只在CPP mask、R输入有效mask、有限且非负CPP值的交集上监督`log1p(CPP)`，未用R伪标签补空缺；这符合本次“隔离混合监督噪声”的单因素定义。队列214221=Q、214883=R、214884=R（3/3）。候选验证和bank manifest仍未完成，不提前启动GHI评估。

05:07 CST下游门提前复核：在服务器`swc`环境重新运行`test_future_ghi_cot_arms.py`与`test_future_ghi_context.py`，分别通过`FUTURE_GHI_COT_SOURCE_ARMS_WITNESS_PASS`及`FUTURE_GHI_CAUSAL_PACK_WITNESS_PASS`；仅验证合成对照臂和时序上下文，不读取项目载荷。CPP-only候选到step 1500/4000。预测COT bank最后完整计数19,008/33,253，数组写入时间更新到05:07，进程仍运行，等待下一批progress flush；test/future-observation排除标记均false。PBS仍214221=Q、214883=R、214884=R（3/3）。

05:09 CST有效推进：CPP-only候选达step 1600/4000；SimVP→冻结R预测COT bank从19,008升到19,520/33,253（624,640预测实例），最近512序列26.02秒、2.46序列/秒，状态BUILDING。GPU0/1分别约37.3/34.5 GiB；抽样利用率变化，进程仍运行。test_payload_opened=false、future_observed_agri_read=false。PBS仍3/3；下游未来GHI见证已通过，等bank完成后可立即执行预设配对评估。

05:10 CST新的有效进度：CPP-only候选已到step 1700/4000；预测COT bank完整计数仍19,520/33,253，底层预测AGRI数组在05:09:48更新，作业仍活跃，预计后续512序列批次flush后更新progress。队列214221=Q、214883=R、214884=R（3/3）；不重启、不增加作业。此刻尚无验证分数或GHI收益结论，后续以step-2000/4000验证、完整bank manifest和配对GHI审计推进。

05:13 CST监控获得新进度：CPP-only候选step 1900/4000；forecast-AGRI→R COT bank推进至20,032/33,253（641,024个预测实例），最近512条26.83秒、2.39序列/秒，仍BUILDING。两个核心作业仍R，账户队列3/3。测试及未来真实AGRI读取仍false。候选step-2000验证预计即将写出；之后会拿到更有意义的中期质量证据，并继续等4000步完整COT护栏和GHI配对门。

05:17 CST：CPP-only候选已到step 2000，正在执行这一里程碑的验证/保存；`validation_step_2000.json`尚未落盘，GPU1利用率约80%、PBS仍R，故继续等待而不重启。预测COT bank从20,032推进到20,544/33,253（657,408实例；最近批次43.54秒、1.47序列/秒），测试/未来真实AGRI读取标志仍false。队列214221=Q、214883=R、214884=R，3/3。验证完成后先比较候选AGRI护栏，再等step4000同掩膜COT审计；bank完整后推进A/AH/AC未来GHI。

05:20 CST CPP-only step-2000 validation已落盘。候选equal-season AGRI RMSE=0.2856169836；同split、同窗口混合标签M1 step-2000为0.2856219593，候选仅低0.0000049757（约0.0017%），可视为AGRI预测持平。JJA候选0.4236914、M1为0.4236892，也基本持平。该节点没有CPP COT/厚云终评，所以不能据此判定降噪成功，更不能判定GHI获益。Forecast bank仍20,544/33,253、BUILDING；PBS 3/3；测试和未来观测输入仍关闭。继续到step4000同掩膜COT评分及GHI配对审计。

05:22 CST新复核：CPP-only候选推进至step 2100/4000，训练仍正常；预测AGRI→COT bank为21,056/33,253（BUILDING，test_payload_opened=false、future_observed_agri_read=false），最近数组仍有更新。PBS队列214221=Q、214883=R、214884=R，3/3，尚无新槽位。step-2000 AGRI护栏已核实接近持平；COT厚云质量与下游GHI效应仍待step-4000评分和bank完成后的配对训练。

05:25 CST进度复核：CPP-only候选到step 2300/4000；预测COT bank推进至21,568/33,253（690,176实例），最近512序列17.65秒、3.63序列/秒。两项任务均仍R；队列214221=Q、214883=R、214884=R（3/3）。下游A/AH/AC合成因果见证通过；当前无正式COT最终分数或GHI收益，继续按门禁等待step4000与bank完成。

05:26 CST实时检查：CPP-only训练到step 2400/4000；预测COT bank最近完整progress为21,568/33,253，底层文件05:25:33仍更新，当前批次尚未flush。两作业存活，测试及未来真实AGRI排除标志保持false，PBS全队列3/3。继续等待后续更新，不启动冲突作业。

05:29 CST两项任务均有新进度：CPP-only候选step 2600/4000；预测COT bank推进至22,080/33,253（706,560 forecast instances），最近512批次29.62秒、2.16序列/秒。候选已通过step2000的AGRI护栏但COT终评未出；bank状态BUILDING且因果排除标志为false/false。全PBS 3/3，不提交新任务，继续现有训练与银行构建。

05:32 CST继续运行检查：CPP-only候选到step 2800/4000；SimVP→冻结R COT bank到22,592/33,253，最近完整批次已刷新，当前批次进行中。两个PBS GPU任务仍R，另有214221排队，账户3/3。test_payload_opened=false、future_observed_agri_read=false。现有AGRI中期指标近乎持平；COT厚云审计和GHI对照仍未完成。

05:35 CST监控：CPP-only候选已到step 3000/4000；预测COT bank进展到23,104/33,253（739,328预测实例），最近512序列20.61秒、3.10序列/秒。账户PBS仍214221=Q、214883=R、214884=R，共3/3。所有因果排除标志保持关闭，COT/GHI收益仍待后续评估。下一个正式节点为step4000验证及三模型共同CPP掩膜审计。

05:38 CST运行态复核：CPP-only候选到step 3200/4000，主进程仍活跃且GPU持续计算；step-2000已通过AGRI护栏。预测COT bank最近完整progress 23,104/33,253，底层数据仍在写，等待下一个512序列flush；无完成manifest。队列214221=Q、214883=R、214884=R（3/3），测试/未来真实AGRI输入均未读取。

05:39 CST新进度：CPP-only候选step 3300/4000；预测COT bank推进至23,616/33,253（755,712实例），最新512批次26.47秒、2.42序列/秒。因果排除标志仍为test=false、future-observed-AGRI=false。PBS 3/3，两个任务均R。下一节点为step4000验证并随后完成同支持三模型COT审计，继续运行。

05:41 CST检查：CPP-only候选达到step 3400/4000；forecast COT bank完整计数仍23,616/33,253，但底层数组更新至05:40:35，处于下一flush周期。两个任务PBS状态R，当前账户总数3/3；test与未来真实AGRI仍未读取。继续运行，不作提前效果判断。

05:43 CST持续推进：CPP-only候选step 3600/4000；forecast-COT bank完整进度24,128/33,253，当前批次仍在写入（原始预测数组更新到05:42:58），无manifest，因果排除标记仍安全。队列214221=Q、214883=R、214884=R（3/3），正式COT审计和GHI收益均未完成。

05:50 CST continuation after the user asked for unattended forward progress: fresh PBS snapshot remains 214221=Q, 214883=R (forecast-AGRI→R bank), 214884=R (CPP-only COT auxiliary), 3/3. Candidate recent_steps reached 4000/4000 at 05:48, but final validation/COT audit/checkpoint artifacts are not yet present; GPU1 is active (~82% sampled) and the training process remains alive, so treat it as final validation/postprocessing, not completed. Forecast bank advanced to 25,664/33,253 (821,248 forecast instances), still BUILDING; test_payload_opened=false and future_observed_agri_read=false. Current safe action: keep both live jobs running, wait for candidate same-mask COT audit and bank manifest/integrity; do not queue another PBS job while full. GHI A/AH/AC screening remains the next downstream stage after both gates and a fresh slot check.

05:55 CST new gate artifact: CPP-only candidate `validation_step_4000.json` and `step_4000.pt` are now present. Equal-season AGRI RMSE is 0.2856031372 versus the equal-budget MSE control 0.2856004358 (candidate +0.00000270, effectively tied; not a COT/GHI claim). Candidate process remains in post-training evaluation/scoring; same-mask COT outputs and independent FP64 audit are not yet present. Forecast bank advanced to 26,688/33,253 (854,016 forecast instances), last batch 21.26 s, still BUILDING, with test/future-observed-AGRI exclusions false. PBS remains 214221=Q, 214883=R, 214884=R (3/3); keep both active jobs and wait for the COT audit/bank completion before the downstream GHI screen.

05:59 CST CPP-only preliminary COT score landed at `seed42_cot_aux_cpp_only_v1/cot_score_aux_4000/cot_aux_validation_score.json`, status `PASS_CPP_COMMON_SUPPORT_FP64`, 3,140 validation windows and zero test reads. Center-3×3 station/lead-macro RMSE=10.743569; full-patch macro RMSE=11.031193; pooled thick CSI=0.426772. This is candidate-only and cannot establish gain. It is directionally similar to prior same-split M1 comparisons (original SimVP center RMSE 10.664161 / CSI 0.413997; mixed COT auxiliary 10.740627 / CSI 0.425167; equal-budget MSE 10.750112 / CSI 0.423894): CPP-only has small thick-CSI improvement but no RMSE win. Await exact support/hash pairing, date-block FP64 audit, and low-COT/AGRI guards before any decision. Forecast bank remains live and test/future-AGRI exclusions are false.

06:02 CST post-score execution clarified from the live run script: after saving the CPP-only candidate score, the same PBS job scores original SimVP and equal-budget MSE checkpoints on the identical CPP support, then writes `training_complete.json`; the PBS wrapper next invokes `audit_cot_aux_validation.py`. At 06:02 those two control score files and audit were not yet present, while the candidate parent/workers remained active. Forecast-COT bank advanced to 28,224/33,253; PBS remains full. This confirms the delay is the planned sequential scoring/audit stage, not an absent next action.

06:05 CST continued scoring: candidate and original-SimVP COT score files now both exist under the CPP-only run; the script is proceeding through its last equal-budget MSE score before setting the completion receipt and launching the independent audit. The bank advanced to 28,736/33,253 (919,552 forecast instances), with both exclusion flags false. PBS is still 3/3. No new training can be submitted until a slot is released; the next planned downstream task remains the paired future-GHI screen once the forecast bank is complete.

06:07 CST progress: the COT auxiliary job remains alive in its ordered scoring stage; candidate and original scores are present, and the final equal-budget MSE score/audit receipts have not appeared yet. Forecast bank advanced to 29,248/33,253 (935,936 forecasts), still train/validation only, with test/future-AGRI exclusions false. PBS remains 3/3. No manual intervention is warranted while worker activity and durable bank progress continue.

06:10 CST final CPP-only ablation gate: `independent_cot_audit.json` reports `PASS_INDEPENDENT_FP64_PAIRED_COT_AUDIT`, 3,140 validation windows, 46 BJT target-date blocks, 5,224,460 common valid CPP pixels over both sites and no test reads. CPP-only candidate failed the prespecified screening gate at both 16×16 and station-local 3×3: center-3×3 RMSE 10.743569 vs original 10.664161 (delta +0.079409; paired 95% CI [-0.105936,+0.260568]) and vs equal-budget MSE 10.750111 (delta -0.006542; CI [-0.030278,+0.012902], crossing zero). Thick CSI improved vs MSE by +0.003068 (CI [+0.000969,+0.005101]); near-zero false-alarm rate also worsened significantly by +0.001726 (CI [+0.000819,+0.002897]). AGRI guard passes (+0.00095% vs MSE), but COT RMSE/no-false-alarm gates fail. CPP is retrieval reference, not independent physical truth. Decision: do not promote or retrain this CPP-only candidate; finish the direct paired future-COT→GHI A/AH/AC screening using the already running base-SimVP forecast bank once its full manifest/integrity receipt lands. PBS 214884 completed; queue must be refreshed before any next submission.

06:15 CST bank progress: 30,272/33,253 forecast sequences complete (~91.0%; 968,704 station×lead instances). Last durable batch 19.96 s (3.21 seq/s), no test/future-observed AGRI inputs. PBS queue has the unrelated `cot_interval_pac` Q and bank R (2 unfinished); the CPP-only candidate job has completed. Wait for `bank_manifest.json`, then validate exact rows/unique keys/exclusion markers before submitting the already checksummed GHI-source screen.

06:19 CST forecast bank update: 31,296/33,253 sequences complete (1,001,472 forecast instances, ~94.1%). Last batch completed in 20.13 s; process is currently waiting on shared-filesystem write/lock (`lock_blkgrp_sync`), GPU SM sampled 0%, but PBS job remains R. This is a storage flush after a completed batch, not evidence of a failed job. Keep monitoring until manifest/integrity is present; queue remains 2/3 unfinished.

06:23 CST bank advanced to 31,808/33,253 (1,017,856 forecast instances; ~95.7%). Latest 512-sequence batch took 19.54 s (3.27 sequences/s). Test and future-observed-AGRI flags remain false; no manifest yet. The remaining 1,445 sequences are being generated by the live 214883 job; continue until full bank integrity passes, then start the reserved GHI pilot slot.

06:30 CST forecast bank reached 33,253/33,253; manifest reports 1,064,096 instances, exact expected array layouts, frozen Hunan monthly SimVP→frozen-R predicted COT provenance, and all test/buffer/future-AGRI/CPP read flags false. Independent index check: 33,253 rows, train 30,436, val 2,817, duplicate seq_id=0; array file sizes match manifest shapes. The full-account queue before submission was 214221=Q plus completed 214883=C (one unfinished). After confirming code checksums and clean unique output paths, submitted `run_future_ghi_a_ac.pbs` as 214888; post-submit queue is 214221=Q, 214888=R (2/3 unfinished). Initial job log passed checksums for all GHI pipeline scripts. Next monitor pack build and paired A/AH/AC head progress; keep pilot-vs-paper claim boundary explicit.

06:32 CST first GHI job 214888 failed in pack construction because `run_future_ghi_a_ac.pbs` pointed to `$RROOT/results/ghi_head_pack_new_r_v1`; live source inspection showed the authoritative complete label pack is under `$RROOT/data/ghi_head_pack_new_r_v1`. The actual pack is state COMPLETE, train=50,458, val=4,479, test_payload_opened=false; its `rows.csv` SHA matches the prebuilt crosswalk report (`a3cac8...`) and its manifest SHA matches crosswalk provenance (`27b1a1...`). The pack builder failed before creating output directories, so no partial artifact needs cleanup. Corrected only LABEL_PACK to the verified `data/` path and updated the PBS script checksum; next rerun its syntax/checksum/source gates and fresh PBS queue before retry.

06:34 CST retry preflight: corrected the PBS `LABEL_PACK` path to the verified RROOT/data pack, confirmed its pack-manifest SHA256 and rows.csv SHA256 exactly match the existing crosswalk provenance, and confirmed both future-pack/head-run outputs remain absent (the failed pack builder exited before mkdir). First remote checksum attempt exposed Windows CRLF line endings in the locally rewritten checksum list; normalized the list to LF, resynced it, and the complete remote `sha256sum -c FUTURE_GHI_SHA256SUMS` now passes, as does `bash -n run_future_ghi_a_ac.pbs`. Recheck PBS queue immediately before resubmitting.

06:40 CST GHI input-pack progress: job 214889 remains R. `build_future_ghi_head_pack.py` is actively writing train arrays/keys on CPU (single process ~100% of one core; GPU allocated but unused during this preparation stage). `train/keys.csv` grew 21,759→26,066 rows over ~85 s; pack is incomplete and has no manifest, so head training has not started. Keep the live pack build intact; monitor its durable key-row count and transition to GPU head training once train+val keys, arrays, audits, and manifest complete. This pilot allocation is 8 h; do not overclaim until the full run finishes.

06:46 CST GHI pack builder progress remains monotonic: train/keys.csv now has 47,739 lines (header + 47,738 of 714,914 expected train rows), about 6.7%. The builder is at ~100% one CPU core, stable RSS ~338 MiB; GPU is idle as expected for this CPU-side materialization. No validation split files/pack manifest yet. At the observed ~65–75 rows/s, the remaining pack stage is several hours; 214889 has an 8 h PBS allocation and remains R. Preserve the in-progress directory; do not resubmit or alter its contents.

06:49 CST续跑：GHI三臂pilot 214889仍为R，PBS全账户队列214221=Q、214889=R（2/3）。修正LABEL_PACK路径后的脚本校验和、COT arm witness及forecast bank gate均通过。CPU打包进程在gpu2单核约100%运行15分44秒；train/keys.csv到58,350行（含表头；58,349 rows of 714,914 expected; pack manifest not yet present），本作业分配的GPU0约2 MiB、0%利用率；同节点GPU2采样为619 MiB、86%，归属未确认，不计作本作业占用。当前阶段仍为pack materialization，不能将数组预分配误认为完成。test保持关闭；pilot仍受训练上游SimVP in-sample限制。保留作业，待pack完成及key/array/manifest审计后进入三臂头部训练。

06:51 CST：214889持续运行，PBS全账户仍为214221=Q、214889=R（2/3）。未来GHI pack的train/keys.csv增至66,832行（含表头，66,831/714,914，约9.35%）；构建PID29468已运行17分32秒，CPU约99.8%，keys单调增长，无pack manifest/训练启动标记。PBS预留8小时，目前约17.6分钟已用，按当前速度pack仍需数小时；不重启。脚本使用作业分配的GPU0，当前2 MiB/0%；同节点GPU2的619 MiB/91%属于未确认进程，不能计成本作业负载。队列2/3，保持运行。


06:57 CST状态：214889仍为R；全账户仅214221=Q与214889=R（2/3）。未来GHI head pack train/keys.csv为90,074行（含表头，90,073/714,914，12.6%），PID29468持续运行23:35、CPU约99.8%；pack manifest未出，head训练尚未启动。当前任务分配GPU0为2MiB/0%，处于CPU打包阶段。

06:57 CST下一假设预备：为直接检验预测未来COT偏差过大或噪声抵消信号，新增独立AC50候选：历史COT不动，只把未来train-standardized COT异常乘0.5；同包重训A/AH/AC/AC50，同3 seeds、相同IntervalHead、配对样本与30epoch预算，使用预先日期块bootstrap审计。独立方案与脚本仅为候选准备，不改活动214889，不提交新作业；需等当前pilot终态结果后决定AC50还是转cross-fit确认。新审计器的numpy合成日期块见证通过，本机Python无torch，Torch arm见证安排在将来的PBS环境执行。test/buffer仍关闭。

07:00 CST：在当前服务器重新核对，214889=R、214221=Q（2/3）。train/keys.csv增至93,526行（含表头，93,525/714,914；13.1%），pack manifest及训练阶段标记仍未出现；当前job CPU打包进程持续运行。为不等单次pilot后才开始准备，已隔离实现未来COT异常收缩0.5的AC50四臂候选、日期块审计及PBS脚本；本地审计器的合成配对/日期bootstrap见证通过，目标服务器SHA256清单全通过、PBS脚本bash -n通过。Torch输入见证将在正式PBS执行时运行。未提交该候选；等当前A/AH/AC配对GHI结果再按预设分支决定：若AC−AH稳定改善则交叉拟合确认，否则运行AC50单因素诊断。

07:00:05 CST复核：214889仍R，队列仍214221=Q、214889=R（2/3）。train keys增至100,487行（100,486/714,914，14.1%），PID29468已运行26:18且CPU99.7%；manifest未出、GHI head训练未开始。此为已证实持续推进，继续保留作业。

07:02 CST 审计加固：补充跨seed严格真值相同断言，并在合成审计中注入跨seed标签偏差验证会fail-closed；Python编译及配对日期bootstrap合成见证通过。此修改只作用于尚未提交的AC50候选脚本，不触碰运行中的214889。服务器副本与checksum同步需在下一候选启用前完成。

07:03 CST AC50审计护栏已同步：在原有同臂跨seed配对键检查上新增所有臂、所有seed真值严格一致校验，并用故意污染seed43标签的合成用例验证审计会拒绝。Python合成日期块审计在本地与服务器swc环境均通过；7项服务器SHA256校验全部OK。当前214889不受此修改影响，AC50仍等主pilot终态后按结果门控。

07:05 CST实时复核：214889=R（已运行32:02），214221=Q，账户2/3。未来GHI train pack的keys.csv为120,736条数据键（文件121,737行含表头的情况待下次核验；当前命令显示120,737行，故数据键120,736，约16.9%）；无val目录/pack manifest，训练头未启动。作业分配GPU0为A800 80GB，当前2 MiB/0%；仍处单核CPU pack生成阶段。工作节点最新日志已通过SHA、COT arms witness与forecast-bank gate。

07:06 CST更正并复核：上一条pack键计数是前一观察点，且误写了行数说明；本次终端实际输出wc -l为123,487行，含表头即123,486条数据键（17.3%）。保持以当前远端wc -l为准；val和manifest尚未生成。

07:08:58 CST同作业复核：214889仍R、214221仍Q（2/3；作业已运行35:15）。train keys文件134,703行，扣表头为134,702/714,914（18.84%）；相对07:06:52的126,675数据键，在126秒增加8,027，约63.7键/秒。pack manifest尚未生成，head训练未开始；此为持续写入而非停滞，保留当前作业。

07:11:55 CST任务监控：214889仍R（运行38:12），214221仍Q，全账户2/3。pack train/keys.csv为146,143行，含表头，故数据键146,142/714,914（20.3%）。与07:09:28的136,902条相比，147秒增加9,240条，约62.9条/秒；无pack manifest，仍在CPU构建。作业活跃且进度单调，不重启、不并发重复打包。

07:13 CST Luna只读核验：214221=Q、214889=R（2/3）；PID29468持续构建pack，train keys 150,432/714,914（21.0%），12秒约+770行；CPU约99%，I/O递增，无错误，manifest与head训练未开始。服务器日志的SHA/source/forecast-bank gate均通过。另在swc环境执行AC50 Torch输入见证，输出FUTURE_GHI_FUTURE_COT_SHRINKAGE_WITNESS_PASS；测试仅CPU，不触碰运行作业或test/buffer。

07:15:45 CST远端直连复核：214889=R（资源8h，用时约40min），214221=Q，2/3；node22分配1张A800 80GB及8核。GHI pack train/keys.csv为161,323行，扣表头161,322/714,914（22.6%）；目录仅有train数据，manifest/audit尚未出现，三臂GHI训练未开始。PBS日志此前通过脚本SHA、COT-source witness、forecast-bank gate；当前显存2MiB/0%符合CPU pack阶段。保留活作业，不并行重复构建；完成完整pack与配对审计后再训练。

07:16:50 CST串行构建性能审计：Luna只读核验214889仍R/node22，PID29468单核约99.7%；pack train keys 164,953条（23.1%），manifest尚无。build_future_ghi_head_pack.py逐行做Python索引/3帧组装/memmap与CSV写入；PBS脚本与builder均无断点续写，输出目录已存在时会拒绝重启，数组用w+创建。故当前不能安全并行第二builder、原位提速或重启续用部分产物；保留活作业是无数据损坏风险的路径。队列214221=Q、214889=R（2/3）；test/buffer未读。

07:17:40 CST远端PBS复核：214889=R（已用42:03/8h），214221=Q，账户2/3；node22仍由同一pack builder运行。train keys.csv 168,143条数据（168,144行含表头，23.5%/714,914），较07:16:50的164,953增加3,190条，进度单调；目录仍只有train、manifest尚未生成，GPU0 2MiB/0%符合打包阶段。作业活跃且预计仍有充足墙钟时间，继续保留，后续manifest完整性通过后再进入GHI头部训练。

07:19:02 CST live monitor：214889=R，214221=Q（2/3）；未来GHI pack train keys为173,424条数据/714,914（24.3%），较07:17:40的168,143增加5,281条，约78条/秒。输出仍在train打包，manifest尚无，尚未启动GHI头训练；当前进展单调，保留运行任务等待完整审计门。

07:20:31 CST监控复核：PBS 214889=R、214221=Q（2/3），作业已运行46:48/8h。train keys.csv为179,143条数据（25.1%/714,914），较07:19:02增加5,719条，约64条/秒；manifest未出，训练阶段未启动，队列与日志无错误迹象。保持原作业继续构建，待完整性审计后立即进入配对GHI训练与复算。

07:22:07 CST远端复核：214889=R（48:24/8h），214221=Q（账户2/3）。train keys.csv为185,194条数据（25.9%/714,914），较07:20:31的179,143增长6,051条，约62条/秒；仍只有train目录、无pack manifest，GHI头训练尚未开始。进程持续写入且作业仍有约7小时墙钟；继续当前链路，manifest落盘后立即核验并训练。

07:23:27 CST进度复核：214889=R、214221=Q（账户2/3），214889运行49:44/8h。train keys已至190,365条数据（26.6%/714,914），较07:22:07增加5,171条，约65条/秒；日志仍无异常，pack manifest与validation目录尚未出现，GHI训练仍处待启动状态。

07:24:57 CST远端复核：214889=R（50:18/8h），214221=Q，账户2/3。train keys为196,086条数据（27.4%/714,914），较07:23:27增长5,721条，约63条/秒；包仍在连续写入，manifest/val尚未出现，GHI头训练未启动，无报错。按近期速度估计pack仍需约2.3小时，当前墙钟余量约7.2小时；保持任务运行，完成后先验收再进入训练。

07:26:14 CST远端监控：214889=R（52:19/8h），214221=Q，全账户2/3。pack train keys 200,926/714,914（28.1%），较07:24:57增长4,840条/77秒，约63条/秒；无manifest，仍在CPU打包，日志无错误。项目后续将由同一PBS脚本在包生成及脚本完整性检查后进入GHI头训练。

07:27:33 CST远端监控：214889=R（qstat显示约54min已用），214221=Q，账户2/3。train keys为206,097条数据（28.8%/714,914），较07:26:14增加5,171条/79秒，约65条/秒。仍处单核pack生成，尚无val目录/manifest；按照当前速度剩余train键约2.2小时，PBS约7小时剩余，后续头训练预算尚充足。

07:28:53 CST远端监控：214889=R，214221=Q，队列2/3；运行55:10/8h。train keys为211,161条数据（29.5%/714,914），较07:27:33增加5,064条/80秒，约63条/秒；目录仍只有train、manifest未生成，GHI训练未启动，日志未见异常。继续当前作业，剩余打包约2.2小时、作业约6.9小时。

07:30:10 CST远端监控：214889=R（56:27/8h），214221=Q，账户2/3。train keys.csv为216,099条数据（30.2%/714,914），较07:28:53增加4,938条/77秒，约64条/秒；manifest仍未生成，GHI头训练未开始，当前工作节点及日志无异常。

07:31:27 CST远端监控：214889=R（57:44/8h），214221=Q，账户2/3。train keys为220,939条数据（30.9%/714,914），较07:30:10增加4,840条/77秒，约63条/秒；manifest和val目录未生成，训练未启动，当前无异常。继续推进当前打包，完成后进入GHI三臂训练与独立配对审计。

07:32:43 CST远端监控：214889=R（59min/8h），214221=Q，账户2/3。train keys 225,852条（31.6%/714,914），较07:31:27增加4,913条/76秒，约65条/秒；仍在train pack生成，manifest/val未出现，GHI头未启动。按当前速率约2小时完成余下train行，时限余量充足。

07:34:04 CST远端监控：214889=R（1:00:21/8h），214221=Q，PBS账户2/3。train keys已至231,050条数据（32.3%/714,914），较07:32:43增加5,198条/81秒，约64条/秒；日志无错误，pack尚未生成val/manifest，GHI头未启动。按当前斜率剩余train打包约2.1h，墙钟仍足够，继续原实验。

07:35:21 CST远端监控：214889=R（1:01:38/8h），214221=Q，账户2/3。train keys到235,951条（33.0%/714,914），较07:34:04增加4,901条/77秒，约64条/秒；无pack manifest/val目录、训练未启动，任务活动且日志无错误。预计余下train打包约2.1小时，后续处理继续依脚本执行。

07:36:40 CST远端监控：214889=R（1:02:18/8h），214221=Q，队列2/3。train keys=240,965/714,914（33.7%），较07:34:04增长9,915条/156秒，约63.6条/秒；仍在单核CPU打包，日志无错误，manifest和GHI训练未出现。当前余墙钟约6:57，预计pack余时约2小时，保留运行并等待同脚本续训。

07:37:59 CST远端监控：214889=R（1:04:16/8h），214221=Q，账户2/3。train keys为245,978条数据（34.4%/714,914），较07:36:40增加5,013条/79秒，约63.5条/秒；输出仍仅train，manifest未生成，GHI头尚未启动，日志无错误。预计剩余打包约1.9小时，队列和墙钟允许当前任务完成训练与审计。

07:39:20 CST远端监控：214889=R（1:05:37/8h），214221=Q，账户2/3。train keys为251,171条（35.1%/714,914），较07:37:59增加5,193条/81秒，约64条/秒；pack仍只有train文件、无manifest，GHI头训练未启动；脚本日志无错误。按趋势预计剩余train打包约2小时，当前仍留有约6:54墙钟。

07:39:33 CST远端复核：214889=R（1:05:50/8h），214221=Q（2/3）。train keys为252,020条数据（35.3%/714,914），较07:37:59增加6,042条/94秒，约64条/秒；输出仍只有train文件，manifest/val和GHI训练尚无，作业日志无错误。预测COT→GHI三臂链保留运行，训练与评估将在pack审计后自动接续。

07:41:07 CST远端复核：214889=R（1:07:24/8h），214221=Q，全账户2/3。train keys=257,962条数据（36.1%/714,914），较07:39:33的252,020增加5,942条/94秒，约63条/秒；目录仅train，manifest/val尚无，GHI训练未启动，作业无错误。当前约剩6:53墙钟、train包预计约1.9h，延续运行。

07:42:30 CST远端复核：214889=R（约1:08/8h），214221=Q，队列2/3。train keys为263,375条数据（36.8%/714,914），较07:41:07增加5,413条/83秒，约65条/秒；仍只有train文件，pack manifest、val和GHI训练未生成，日志无错误。按当前速度余train包约1.9h，保留任务继续。

07:43:54 CST远端复核：214889=R（约1:10/8h），214221=Q，队列2/3。train keys 268,575/714,914（37.6%），较07:42:30增加5,200条/84秒，约62条/秒；日志无错误，仍处打包阶段，manifest和GHI头训练尚未出现。余train键约446k，按当前速率约2小时；继续运行。

07:46 CST远端复核：214889=R（1:13/8h），214221=Q，账户2/3。train keys为280,250条数据（39.2%/714,914），较07:43:54增长11,675条/180秒，约65条/秒；目录仍只有train，无manifest/val，GHI头未开始；A800处于pack阶段，2MiB显存、0%利用符合当前单核CPU打包阶段。按当前速率余train约1.9h，作业余墙钟约6:47，继续执行原计划。

07:49 CST local代码审阅：核对build_future_ghi_head_pack.py与train_future_ghi_heads.py的时间拼接和三臂掩码。+15 min使用init-15、init历史帧及+15预测帧；AH只屏蔽初始化后预测COT，AC保留历史与预测COT，归一化只拟合train并三臂共享；未发现时间索引或臂间口径错误。当前作业继续原样运行，待pack和实测GHI对照结果完成后再按单因素门槛调整。
07:50:05 CST 远端复核：214889=R（node22，单核pack builder约99.8% CPU），214221=Q，账户2/3。train keys为292,251条数据（约40.9%/714,914），15秒增加762条、约51条/秒；日志无错误，来源检查通过，pack manifest/val及训练产物尚未生成。
07:51:35 CST AC50备选修正复核：服务器swc环境运行test_future_ghi_cot_shrink.py通过；本地bootstrap审计测试test_audit_future_ghi_cot_shrink.py通过。该对照只将初始化后的未来标准化COT异常量乘0.5，历史COT保持原样；作为主pilot未通过时的下一单因素候选，当前未提交训练。同期214889仍R，train pack 298,012/714,914数据行，持续增长。
07:52:51 CST 远端复核：214889=R（node22，1:19/8h），214221=Q，账户2/3。train keys为302,904条数据（42.4%/714,914），较07:51:35增加4,892条/76秒，约64条/秒；manifest/val尚未出现，GHI头未启动，GPU仍是pack阶段预期的近空闲状态，日志无错误。
07:54:08 CST Luna资源复核：214889仍R（walltime已用约1:18:34/8h、余6:41），214221=Q，队列2/3。PID29468约99.8%单核CPU；keys从07:53:52的307,036增至307,798数据键，16秒约+762、47.6键/秒，rchar同步增长；manifest未生成、日志无错误。估计剩余train约1.8–2.4h，加val约17–23min及收尾核验后pack共约2.1–2.8h，当前墙钟余量足够。
07:55:20 CST Luna复核：214889=R（已用约1:20:18/8h、余6:39:42），214221=Q，账户2/3。PID29468仍约99.8% CPU；train keys从07:55:06的311,497增至312,476数据键，14秒+979、约70条/秒，rchar持续增长。目录仍只有train、无manifest/val/head输出，日志无错，来源门通过，墙钟余量充足。
07:56:48 CST 远端复核：214889仍R（1:23/8h），214221=Q，队列2/3。train keys 318,017条数据（44.5%/714,914），较07:55:20增长5,541条/88秒，约63条/秒；无manifest/val/head产物，日志无错误，pack继续稳定推进。

07:59 CST远端复核：214889=R（1:25/8h），214221=Q，账户2/3。node22 PID29468持续以约99.8%单核CPU执行pack；train keys为325,541条数据（45.5%/714,914），较07:56:48增加7,524条/145秒，约52条/秒；train数组持续写入，manifest/val及GHI头尚未生成，日志与来源门无异常。GPU0保持2MiB/0%符合CPU打包阶段；不并发重启，按现合同继续运行。

08:00 CST远端复核：214889=R（约1:27/8h），214221=Q，账户2/3；node22 PID29468仍以99.8%单核CPU执行train pack。keys.csv在08:00:13为331,154条数据、08:00:40增至333,635条（约92条/秒），约46.7%/714,914；训练数组仍在写入，完整manifest/val/head结果未生成。PBS分配GPU0当前2MiB/0%，属于CPU打包阶段；其他卡有他人任务，未计入本任务资源。/public剩余104TB，作业余墙钟约6:33，日志无错误，继续当前任务、不重启。

08:03 CST 远端复核：214889=R（已用约1:30/8h，余约6:30），214221=Q，队列2/3。node22 PID29468持续约99.8% CPU；train keys.csv为343,889条数据（48.1%/714,914），较08:00:40增加10,254条/174秒，约59条/秒；数组仍在写入，manifest/val/head结果尚未生成，日志无错误。继续当前打包阶段，不重启。

08:05 CST复核：214889仍R（1:31/8h），214221=Q，队列2/3。PID29468仍以约99.8% CPU运行；train keys为350,475条（49.0%/714,914），较08:04:55增加1,296条/22秒（约59条/秒），数组继续增长，manifest/验证包/GHI头训练日志尚未出现，来源检查通过。

08:07 CST审计代码复核：audit_future_ghi_a_ac.py会检查A/AH/AC各臂及不同seed之间站点/init/target/lead/GHI行号完全配对，并要求真值数组一致；按北京时间目标日期对两站共同做区块bootstrap。AC−AH未来COT主对比和AC−A总对比与注册门槛一致，未发现实现不匹配。当前214889仍处于训练包生成阶段。

08:07 CST Luna实查：214889=R、214221=Q，队列2/3。node22 PID29468以约99.8% CPU运行；train keys=359,434（50.3%/714,914），比08:06:35增加3,670条/62秒（约59条/秒），keys更新时间08:07:37。manifest及head输出尚未出现，仍在CPU打包；来源检查与见证通过，无错误。

08:08 CST Luna实查：214889=R、214221=Q。PID29468仍以约99.8% CPU运行pack；train keys=362,347条（50.7%/714,914），比08:07:37增加2,913条/47秒（约62条/秒），文件更新时间08:08:24。manifest和head输出尚未出现，日志无异常，仍在CPU打包。

08:09 CST Luna实查：214889=R（已用1:34，余约6:25），214221=Q，队列2/3。node22 PID29468约99.8% CPU；train keys=364,938条（51.0%/714,914），比08:08:24增加2,591条/40秒（约65条/秒），mtime 08:09:04。manifest/head输出未出现，日志无错，witness和bank gate通过。按当前速度剩余训练键约需90分钟，墙钟余量足够。

08:09:55 CST Luna实查：214889=R、214221=Q。PID29468约99.7% CPU构建pack；train keys=368,177条（51.5%/714,914），比08:09:05增加3,239条/50秒（约64.8条/秒），mtime为当前。manifest/head结果未出现，日志无异常；已运行约1:36/8h，剩余约6:24。按当前速度剩余训练键约需89分钟，墙钟余量足够。

08:10:41 CST Luna复核：214889=R、214221=Q。node22 PID29468约99.8% CPU；train keys=371,093条（51.9%/714,914），比08:09:55增加2,916条/46秒（约63条/秒），mtime 08:10:41。验证包、manifest和head结果未出现，日志无异常、门禁通过。墙钟已用约1:37/8h，余约6:23，pack继续运行。

08:11 CST下一单因素备选复核：在服务器swc环境重新运行test_future_ghi_cot_shrink.py及test_audit_future_ghi_cot_shrink.py，两个见证均PASS。AC50仍受主pilot结果门控，尚未提交。测试后的实时状态：214889=R、214221=Q（2/3）；PID29468仍约99.8% CPU，train keys=374,874条（52.4%/714,914），manifest/head产物尚未出现。

08:12 CST Luna复核：214889=R、214221=Q（队列2/3）。node22 PID29468约99.8% CPU；train keys=377,139条（52.8%/714,914），比08:10:41增加6,046条/95秒（约63.6条/秒），mtime 08:12:15。manifest/head产物未出现，日志无错误，witness和bank门通过。已运行约1:38/8h，余约6:21，仍在pack阶段。

08:12:59 CST Luna实查：PBS 214889=R、214221=Q。node22 PID29468约99.8% CPU；train keys=379,838条，比08:12:16增加2,699条/43秒（约62.8条/秒），mtime 08:12:58。manifest/head产物未出现，日志无错，witness和bank门通过。已运行约1:39/8h，pack继续。

08:14 CST链路复核：检查train_future_ghi_heads.py与build_future_ghi_head_pack.py。AH只保留初始化时刻及之前的COT帧（+15 min保留两个历史槽，+30 min仅保留init槽，+45 min起无历史槽）；AGRI/COT归一化只由已物化训练行拟合并在各臂共享；GHI由kt乘匹配的15分钟晴空辐照度标签还原。未发现时序掩膜或归一化错配。

08:14:34 CST Luna实查：214889=R、214221=Q（队列2/3）。node22 PID29468约99.8% CPU；train keys=385,992条（54.0%/714,914），mtime 08:14:34。pack manifest/head训练输出尚未生成，来源/见证门通过、日志无错。已运行约1:41/8h，余约6:19，仍在CPU打包。

08:15:03 CST Luna实查：PBS 214889=R、214221=Q；PID29468约99.8% CPU。train keys=387,828条（54.2%/714,914），比08:14:34增加1,836条/29秒（约63.3条/秒），mtime 08:15:03。val/manifest/head产物未出现，日志正常、门禁通过；已用约1:41/8h，仍在pack。

08:17:30 CST 复核：远端时间08:17:29；PBS仍为214889=R、214221=Q（2/3），214889 elapsed约1:43/8h。训练pack keys.csv增至397,116条数据，mtime 08:17:28；agri_raw/cot_log1p/geometry数组同步写入，val、manifest、head产物仍未出现。source与forecast-bank门此前通过，作业仍在CPU打包阶段；assigned GPU0为2MiB/0%，节点其他卡属其他作业。按约63–65键/秒估算，余317,798条训练键约需82–84分钟，之后还需生成val和做完整包审计。未见失败或需干预信号。

08:20:17 CST 进度复核：PBS 214889=R（1:46:34/8h）、214221=Q，账户2/3。训练keys.csv为407,908条数据（header共407,909行），mtime 08:20:17；较08:17:28增加10,792条/169秒，约63.9条/秒。仍仅train目录，val/manifest/head训练未启动，作业正常推进。剩余约307,006条，按当前速度估计约80分钟，之后需生成验证包并完成manifest核验；walltime余6小时13分。

08:22:02 CST Luna只读复核：PBS 214889=R、214221=Q（2/3），运行约1:48/8h。node22 PID29468约99.8% CPU；train keys 414,599，mtime08:21:46，较08:16:43增加20,400条，平均约64条/秒；train各数组更新时间至08:21。val/manifest/head仍未生成；指定GPU0为0%/2MiB，日志与来源门无错误。监控中的一次find格式错误未影响任务，无需干预。

2026-10-19 08:43 CST（本机时间；集群机时约08:23）分布错配诊断准备：复核bank_manifest确认in_sample_train_forecasts=true，且forecast bank声明只用冻结SimVP预测AGRI经冻结R生成未来COT；当前GHI pilot为screening，不足以作论文结果。新增只读脚本audit_future_cot_input_shift.py，按站点/月/lead/cosSOZ宽分层统计预测COT分位数、厚云比例、3x3空间纹理及15分钟变化，严格标注为输入分布诊断、不代表反演精度或GHI收益；合成见证test_audit_future_cot_input_shift.py通过。脚本当前在本地，待输入pack完整后随下一轮自动化通过受控方式同步并运行；未碰当前PBS作业、未改正文、未提交训练。


2026-10-19 08:45 CST（本机）远端实时复核：集群机时08:26:44，PBS 214889=R（1:53:01/8h）、214221=Q，账户2/3。train keys.csv为432,736条数据（总714,914），mtime08:26:44；较08:23:00增加14,357条/224秒，约64.1条/秒。仍在CPU打包，val/manifest/head未出现；暂无异常。集群机时比本机时约慢18分钟，walltime按PBS elapsed统计，后续ETA以PBS elapsed及同一集群机时快照为准。


2026-10-19 08:47 CST（本机）复核集群进度：集群机时08:27:37，PBS 214889=R、214221=Q（2/3），pack builder仍在node22 CPU运行；train keys=436,190，较08:23:00增加17,811条（约64.3/s），占714,914训练目标约61%。val/manifest/head未出现；witness与forecast-bank gate通过，GPU0空闲符合当前CPU打包阶段。输入COT分布诊断脚本已本地编译并通过合成见证；仅待pack完整后使用，不读未来AGRI/CPP作为推理输入，也不影响当前作业。


2026-10-19 08:47 CST（本机）复核：集群机时08:28:15，PBS 214889=R，elapsed约1:54:31/8h；214221=Q（2/3）。node22 PID29468仍执行输入pack生成，train keys 438,564条，mtime08:28:14，val/manifest/head尚无。近4分37秒增17,811条，约64条/秒；来源witness及forecast-bank gate通过，无错误。当前已准备本地训练/验证预测COT输入分布诊断脚本，合成见证通过；等pack complete后再同步并运行。


2026-10-19 08:51 CST（本机）未来COT bank分布诊断已准备并在服务器swc环境验证：新增audit_forecast_cot_bank_distribution.py，直接读取已完成的train/val预测COT bank和target-frame几何，按站点、目标月份、lead、宽太阳角带计算预测COT分布、3x3纹理及15分钟变化；不读GHI标签、未来观测AGRI或CPP，结果只用于训练/验证输入分布诊断，不代表准确率或收益。远端脚本SHA256=9346193dd500ecba587c6374ca86b4b5abf016c0b41777925053c89f2ae56bdf，服务器合成见证FORECAST_COT_BANK_INPUT_SHIFT_WITNESS_PASS；已备好normal队列单核30分钟审计脚本且bash -n通过。为避免与当前pack builder争用共享文件系统，审计作业暂不提交，待pack stage结束后按PBS队列门禁执行。当前远端08:32:38，214889仍R（1:58:55/8h），train keys 455,510条；约63.8条/秒，val/manifest尚未出现。


2026-10-19 08:54 CST（本机）继续推进复核：远端集群08:34:11快照中PBS 214889=R（2:00:28/8h，node22），214221=Q，账户2/3；214889仍在因果输入pack构建段。上一份Luna现场证据截至08:33:12为train keys 457,669/714,914，增长约64/s，val/manifest/head尚未出现；本次没有伪报更新后的键数。GPU在CPU打包阶段闲置属于当前阶段行为。复核训练脚本确认18个配对头（两站×3 seed×A/AH/AC）使用同初始化/样本顺序与30 epoch；评估仍是screening pilot，因为训练期SimVP预测不是日期外推。当前自动心跳`cot`保持ACTIVE，每15分钟唤醒并按门禁推进。包完成后先验manifest/键/数组/隔离，再完成预制分布诊断并读取pilot；若未过门则依据结果执行单因素AC50，不会只反复轮询。

2026-10-19 08:56 CST：新增`audit_future_ghi_horizon_bootstrap.py`，从同一批A/AH/AC逐样本验证预测中按+15–60、+75–120、+135–240分钟及全时效计算两站等权、seed平均的RMSE与目标BJT日期块配对95%区间；所有时段共用相同日期重采样，重叠初始化不当成独立样本。合成配对见证通过（`FUTURE_GHI_HORIZON_BOOTSTRAP_WITNESS_PASS`），py_compile通过。该审计仅读保存预测，测试集/未来真实AGRI/CPP均不需要。待214889产出验证预测后用于补充判读，不干扰当前运行。

2026-10-19 08:37 CST（Luna远端只读复核）：全队列214221=Q、214889=R（2/3），214889 elapsed约2:03:21/8h。node22 PID29468以99.8% CPU运行pack builder；train keys 472,551/714,914，较08:33:12的457,669新增14,882条，约64.1条/s。val keys/manifest/head未出现，日志和source/bank见证均PASS，GPU0 0%/2MiB符合当前CPU阶段。train剩余约242k行，按当前速度约63分钟；墙钟余约5小时57分，现阶段不更改正在运行的输入合同或提交竞争任务。

2026-10-19 08:37:45 CST（Luna远端只读复核）：PBS全队列214221=Q、214889=R（2/3），214889 elapsed 2:04:02/8h；node22 PID29468持续99.8%运行pack builder。train keys=475,465，较08:37:03增长2,914/42秒（约69.4条/s）；val、manifest、GHI头输出未出现。日志检查、见证和bank gate仍PASS，无错误或完成标记。当前仍是CPU构建阶段，不提交与其争用共享存储的分布审计任务；主作业保持运行。

2026-10-19 08:38:34 CST（Luna远端只读复核）：214889仍R（elapsed约2:04:50/8h），214221=Q，PBS 2/3。node22 PID29468持续运行builder；train keys=478,271，较08:37:45增长2,806条/49秒，约57条/s；val/manifest/head输出仍未出现，检查/witness/bank gate PASS且无错误。没有重启或并发提交，等待完整train/val包后立即做manifest验收并接训练与结果审计。

2026-10-19 08:57 CST（本机）分时效审计增强：新增分析脚本现支持A/AH/AC及A/AH/AC/AC50两种运行臂；各时效共用同一组BJT日期bootstrap重采样，便于直接比较短/中/远时效区间，且下一轮AC50无需另写统计逻辑。复核时发现最初版本虽写了“共用日期重采样”，实现上时效间会推进随机数状态；已修正为预生成同一resample矩阵。3臂和4臂合成见证及py_compile均通过。此更正只影响新诊断代码，当前PBS作业代码/输入未修改。

2026-10-19 08:40:39–08:40 CST现场复核：Luna见到214889=R（约2:06:56/8h）、214221=Q；train keys至少486,801条，val/manifest/head无，继续pack。随后经当前登录会话在node22直接wc核实keys.csv共487,450行（含表头，即487,449个数据键），PID29468仍运行builder。检查期间任务持续增长，不能把Luna稍早计数误当最新；无失败。train还约227k键，按此前约64/s约59分钟，另有val与完整性验收；账户2/3。

2026-10-19 08:41:25 CST（Luna远端只读复核）：PBS 214889=R（2:07:42/8h）、214221=Q，2/3。train pack 489,716/714,914条键；较08:40约+2,915条，仍约99.8%单核构建。val、manifest、head训练无；日志PASS，无失败或结束标记，GPU 0%/2MiB。按64条/s剩余约58.6分钟train键，仍有约5小时52分walltime；不打断当前生产包。

2026-10-19 08:45 CST（集群现场复核）：PBS 214889=R、214221=Q（2/3）；214889 elapsed 02:11:37/08:00:00。node22 PID29468仍以99.8% CPU运行未来GHI输入pack builder。train keys为504,289条数据（keys.csv共504,290行含表头），较08:44快照502,779条增加1,510条，约63条/s；train目标714,914条，val/manifest/头训练尚未出现。作业绑定GPU0当前闲置（2MiB/0%），属CPU pack阶段；同节点其他卡有其他进程占用，不改变GPU分配。刚同步的时效分层bootstrap审计脚本远端SHA256核验通过，合成配对见证`FUTURE_GHI_HORIZON_BOOTSTRAP_WITNESS_PASS`及py_compile均通过。当前不提交争用共享存储的诊断作业；pack阶段完成后验完整性，继续A/AH/AC筛选并按门禁决策。

2026-10-19 08:46:16 CST（集群实时复核）：PBS 214889=R（elapsed 02:11:04/08:00:00），214221=Q，2/3。node22 PID29468继续运行pack builder；训练keys共507,960行（含表头，即507,959条数据），相对08:45快照504,289条增加3,670条，约63.3条/s；完成率约71.1%。仅train数组/keys正在更新，val与manifest未生成，训练尚未开始。指定GPU0为0%/2MiB，符合当前CPU打包阶段；没有检测或启动竞争性作业。当前继续等待pack完整后在同一作业内训练并保存逐行验证预测，再运行总效应、AC−AH未来COT增量和逐时效日期块审计。

2026-10-19 08:48:22 CST（集群实时复核）：PBS 214889=R（elapsed约02:14:40/08:00:00），214221=Q，账户2/3。node22 PID29468仍在运行未来GHI因果输入pack builder；train keys=516,060条数据（文件共516,061行含表头），距08:46:16增加8,101条，约64.8条/s，训练键完成率约72.2%。val keys、pack manifest及GHI训练结果均未产生。过程活跃、无错误；GPU0仍闲置符合当前CPU打包阶段。保持pack不变，等待构建完成后让原作业继续训练和保存全部配对验证预测。

2026-10-19 08:49:10 CST（集群实时复核）：PBS 214889=R（elapsed约02:15:03/08:00:00），214221=Q，2/3。node22 PID29468以99.8% CPU继续构建causal GHI pack；train keys为519,083条数据（519,084行含表头），完成率约72.6%；较08:48:22增加3,023条，约63条/s。val keys、pack manifest和训练输出仍未出现；作业日志暂无错误，进程存活。输入隔离仍由forecast bank gate和packer代码保证，只接受train/val forecast bank索引，不使用test/buffer/未来实测AGRI/CPP。保持运行，不重启、不抢占。

2026-10-19 08:51:02 CST（集群实时复核）：PBS 214889=R（elapsed约02:17:19/08:00:00），214221=Q，2/3。node22 PID29468仍以99.8% CPU运行pack builder；train keys为526,313条数据（526,314行含表头），相比08:49:10增加7,230条，约64.2条/s，进度73.6%。val pack与manifest尚无，训练未启动。作业仍活跃、未见错误；指定GPU0空闲是CPU pack阶段预期。由于当前单核已饱和且写共享存储，不启动并发审计或抢占；继续等待其完成后同一作业自动进入A/AH/AC头训练。

2026-10-19 08:51:41 CST（集群复核）：同一PBS 214889仍R（elapsed约02:17:04/08:00:00），214221仍Q；node22 PID29468持续运行。train keys=528,796条数据（528,797行含表头），较08:51:02增加2,483条/39秒，约63.7条/s；pack build仍在正常推进，无val键、manifest或训练日志新阶段。

2026-10-19 08:52:37 CST（Luna只读复核）：PBS 214221=Q、214889=R，账户2/3；214889 elapsed 02:18:51/08:00:00。node22 PID29468仍以99.8% CPU运行pack builder。train keys=532,251条数据（共532,252行含表头），mtime 08:52:37，较约08:52快照530,956条增加约1,295条，pack持续推进。val keys、manifest、GHI头训练输出未出现；日志尾部PASS且无异常。指定GPU0为0%/2MiB，符合当前CPU打包阶段。未做额外并发读写。

2026-10-19 08:53:23 CST（集群实时复核）：PBS 214889=R（elapsed约02:18:48/08:00:00），214221=Q，2/3。node22 PID29468仍以99.8% CPU执行pack builder。train keys=535,273条数据（文件535,274行含表头），较08:51:41增加6,477条，约63.4条/s，完成率约74.9%。val keys、pack manifest及head训练目录未出现；作业日志无异常。无安全并发加速动作，保留现有pack和作业。

2026-10-19 08:55:14 CST（集群实时复核）：PBS 214889=R（elapsed约02:21:31/08:00:00），214221=Q，2/3。node22 PID29468继续运行pack builder；train keys为542,398条（542,399行含表头），已完成约75.9%。较08:53:23快照增加7,125条，平均约63.8条/s。未出现val keys、manifest或训练进程；进程仍活动，日志无错误。当前阶段不额外并行读取共享bank。

2026-10-19 08:56:07 CST（集群实时复核）：PBS 214889=R（elapsed约02:21:03/08:00:00），214221=Q，2/3。node22 PID29468仍以99.8% CPU运行pack builder；train keys=545,852条数据（545,853行含表头），较08:55:14增加3,454条，约63.9条/s，完成率76.4%。val keys、manifest和训练输出均未出现，日志无异常。当前保持作业不变；按当前速度train余下约44分钟，之后还要构建val及完成manifest，walltime约余5小时39分。

2026-10-19 08:56:40 CST（Luna只读复核）：PBS 214221=Q、214889=R；214889 elapsed约02:22:54/08:00:00，余约5小时37分。node22 PID29468仍以99.8% CPU构建pack。keys为547,794条数据（547,795行含表头），较08:52:37增加15,543条/4:03，约64键/s，mtime 08:56:40。manifest未生成，尚未进入训练；日志门禁PASS且无错，GPU0为0%/2MiB。未对当前共享数据做扫描或干预。

2026-10-19 08:57:20 CST（集群实时复核）：PBS 214889=R（elapsed约02:23:04/08:00:00），214221=Q，2/3。node22 PID29468持续以99.8% CPU运行pack builder；train keys=550,493条数据（550,494行含表头），约占714,914条目标的77.0%。较08:56:07增加4,640条/73秒，约63.6条/s。val keys、manifest、head训练尚未出现，日志PASS无错误。按当前速度train部分约还需43分钟，再处理验证子集与manifest；作业墙钟余约5小时37分。

2026-10-19 08:59:05 CST（集群实时复核）：完整PBS队列仍为214221=Q、214889=R（2/3）。node22 PID29468以99.8% CPU运行future GHI pack builder；train keys.csv为557,186行（557,185条数据键），较08:58附近快照继续增长，目标714,914，约78.0%，按约64键/s仍需约41分钟完成train阶段。val、manifest及GHI头训练尚未生成；当前日志无错误，source witness与forecast-bank gate通过。指定GPU0仍空闲，属CPU构包阶段；不并行扫描共享输入。作业elapsed约2:25/8h，时限有余量。
PBS脚本衔接复核（2026-10-19 08:59 CST）：214889的同一allocation会在train/val pack与manifest完成后顺序训练A/AH/AC三臂、seed 42/43/44、30 epochs，再执行配对GHI审计和COT分层审计；当前活动作业不需要人工重提或改变输入合同。

2026-10-19 09:02 CST（停滞复核后确认仍在推进）：全账户PBS仍214221=Q、214889=R（2/3）；node22 PID29468状态R，99.8% CPU，CPU时间持续增长。train keys.csv共569,815行（569,814个数据键），较09:00左右的561,504行增约8,310行；文件mtime为09:02:15，train agri_raw.npy仍写入且为1,003,739,384字节。pack manifest/val键尚无，训练未开始。09:00附近短间隔wc接近不变是观察间隔与输出缓冲造成的假象；新一轮精确wc、mtime和CPU现场证明没有停滞。按近1-2分钟吞吐粗估train部分还需约35分钟；继续保留当前作业。
2026-10-19 09:03 CST（Luna实时复核）：PBS全账户214221=Q、214889=R（2/3），214889 elapsed 02:29:45/08:00:00。node22 PID29468以99.8% CPU继续运行build_future_ghi_head_pack.py；train keys.csv为574,132条数据键（574,133行含表头），mtime 09:03:32，较08:59:05的557,185条增加16,947条/4:27，约63.5条/s。val keys、pack_manifest、head训练目录均未出现，仍处train pack阶段；指定GPU0为0%/2MiB。日志source arms witness与forecast bank gate通过，无错误/终态标记。pack builder门禁限定train/val并检查bank的test_payload_opened=false、future_observed_agri_read=false；本次未读取test/buffer/未来观测载荷。继续等待；下一触发点为train键达到714,914后检查val/manifest，pack完整后再验完整性并跟踪同一作业训练。
2026-10-19 09:04 CST（Luna现场复核）：PBS全账户214221=Q、214889=R（2/3），214889 elapsed 02:30:38/08:00:00。node22 PID29468以99.8% CPU继续运行build_future_ghi_head_pack.py；train keys.csv为577,479条数据键（577,480行含表头），mtime 09:04:24，较09:03:32增加3,347条/52秒，约64.4条/秒。val keys、pack_manifest、head run及两份validation audit均未出现，仍为train pack阶段；GPU0=0%/2MiB。日志检查/witness/forecast-bank gate PASS，无错误或终态标记。未读test/buffer载荷；继续等待pack完整，后续先验manifest/键数/数组元信息与隔离字段，再跟踪同作业训练。
2026-10-19 09:05 CST（Luna增量复核）：PBS 214221=Q、214889=R（2/3），214889 elapsed 02:31:26/08:00:00。node22 PID29468仍99.8% CPU执行pack builder；train keys.csv=580,609条数据键（580,610行含表头），较09:04:24的577,479条增加3,130条/45秒，约69.6条/s。val keys、pack_manifest、GHI head run与validation audit均未出现，阶段未变，故暂无pack完整性审计对象。指定GPU0为0%/2MiB；日志gate/witness PASS，无错误。未读取test/buffer载荷；继续等待，至train目标714,914后观察val/manifest及同作业训练转换。
AC50降噪候选代码/协议复核（2026-10-19 09:05 CST）：FUTURE_COT_ATTENUATION_PLAN与train_future_ghi_heads_cot_shrink.py一致，按train-only log1p(COT)均值/标准差标准化，仅把未来预测COT的标准化偏差乘0.5，历史COT原值、AGRI/geometry、三臂A/AH/AC、标签、种子/样本顺序和30-epoch预算保持配对；AC50独立重训其余三臂避免跨运行预算差。当前A/AH/AC pack未完成，按门禁不提交AC50。
2026-10-19 09:06 CST（Luna实时复核）：PBS全账户214221=Q、214889=R（2/3），214889 elapsed 02:32:31/08:00:00。node22 PID29468仍以99.8% CPU运行build_future_ghi_head_pack.py；train keys.csv为584,711条数据键（584,712行含表头），较09:05:09增加4,102条/约65秒，约63.1条/秒。val keys、pack_manifest、GHI head run和validation audit均未出现，尚未完成pack/未转训练。指定GPU0为0%/2MiB；日志静态检查、source witness及forecast-bank gate PASS，无错误。未读test/buffer。下一步继续等待train目标714,914及val/manifest出现，再运行只读完整性与隔离验收。
2026-10-19 09:09 CST（Luna实测）：PBS队列214221=Q、214889=R（2/3），214889 elapsed约02:36:13/08:00:00。node22 PID29468以99.8% CPU继续运行pack builder；train keys.csv=598,961条数据键（598,962行含表头），mtime09:09:58；较09:06:14的584,711条增加14,250条/3:44，约63.6条/s。pack_manifest、val keys和head训练输出仍未出现，作业未转训练；日志witness/bank gate PASS，无错误。指定GPU0为0%/2MiB。未读test/buffer；保持运行，待train目标714,914后检查val/manifest与同一PBS阶段转换。
2026-10-19 09:10 CST（Luna实测增量）：PBS 214221=Q、214889=R（2/3），214889 elapsed约02:37:12/08:00:00。node22 PID29468仍以99.8% CPU运行pack builder。train keys.csv为602,739条数据键（602,740行含表头），mtime09:10:57；较09:09:58的598,961条增加3,778条/59秒，约64.0条/s；root于09:10:24与09:10:40的独立复核分别报告600,689及601,660条，方向一致。pack_manifest仍不存在，仍未进入val/训练；指定GPU0空闲。
2026-10-19 09:11 CST（Luna实测增量）：PBS 214221=Q、214889=R；214889 elapsed约02:37:26/08:00:00。node22 PID29468继续99.8% CPU执行pack builder。train keys.csv为603,603条数据键（603,604行含表头），约11秒内较root 09:10:59快照602,847条增加756条；val目录当前尚不存在，pack manifest/训练阶段未到。构建持续推进、无重启。
2026-10-19 09:11 CST（连续增量复核）：214889仍R，node22 PID29468约99.8% CPU，PBS队列214221=Q、214889=R。键数在09:11:10为603,604行、09:11:27为604,683行、09:11:36为605,223行；本机只读复核605,223行并确认val keys/pack_manifest不存在。builder持续推进，仍未切换val或训练阶段。
2026-10-19 09:12 CST（Luna增量）：PBS 214221=Q、214889=R；214889 elapsed约02:38:15/08:00:00。node22 PID29468仍99.8% CPU运行pack builder。train keys.csv在09:11:48为605,978行，09:11:58实测606,734行（606,733数据键）；train agri_raw.npy与keys.csv mtime已到09:12，确认写入继续。pack manifest仍无，未有阶段切换；墙钟仍有充分余量。
2026-10-19 09:12 CST（Luna增量复核）：09:12:11实测PBS 214221=Q、214889=R；node22 PID29468为R、99.8% CPU。train keys.csv为607,598行（含表头，即607,597条数据键），较09:11:58的606,734行增加864行/13秒；complete.json与pack_manifest.json均未出现，仍在pack。根线程09:12:14复核记录键数继续增加至607,598。
2026-10-19 09:12 CST（持续增量）：PBS 214889仍R、214221=Q；node22 PID29468约99.8% CPU。train keys.csv于09:12:45为609,756条数据键（609,757行含表头），较09:12:11的607,597条增加2,159/34秒，约63.5条/s。val目录与pack_manifest仍不存在，仍处train pack阶段。
2026-10-19 09:13 CST（pack进度实测）：PBS 214221=Q、214889=R；node22 PID29468约99.8% CPU持续运行。09:13:08 train keys.csv=611,266行（含表头，611,265条数据键），较09:12:45的609,757行增加1,509/23秒，约65.6条/s；agri_raw.npy mtime已更新至09:13。val与manifest尚未出现，仍在train pack。
2026-10-19 09:14 CST（Luna连续实测）：PBS 214221=Q、214889=R，214889 elapsed约02:40:34/08:00:00。node22 PID29468以99.8% CPU继续pack；train keys.csv于09:14:17为615,693行（含表头），较09:13:08的611,266行增加4,427/69秒，约64.2条/s。root在09:14:08独立实测614,938行，结果同向。manifest仍无，仍train pack；当前墙钟余量充足。
2026-10-19 09:14 CST（增量复核）：09:14:32 PBS仍214889=R、214221=Q；train keys.csv为616,448行（616,447条数据键），较09:14:17的615,693行增加755/15秒。root 09:14:21实测615,801行，持续增长；仍在pack阶段，未见manifest或阶段切换。
2026-10-19 09:15 CST（持续监控进展）：PBS 214221=Q、214889=R；214889 elapsed约02:42:01/08:00:00。node22 PID29468仍99.8% CPU运行pack builder。root 09:15:29记录train keys 620,118行；Luna 09:15:44实测621,197行（含表头），较09:14:21的615,801行增加5,396/83秒，约65条/s。val/manifest仍未出现，阶段保持train pack。
2026-10-19 09:16 CST（持续pack进展）：PBS 214221=Q、214889=R；elapsed约02:43:14/08:00:00。node22 PID29468仍99.8% CPU。root 09:16:51记录train keys 625,407；Luna 09:16:57实测625,947行（含表头、625,946数据键），相较09:15:44的621,197行增加4,750/73秒，约65.1条/s。pack_manifest和val尚无，训练阶段未启动；继续等待。
2026-10-19 09:18 CST（持续pack进展）：PBS 214221=Q、214889=R；214889 elapsed约02:44:33/08:00:00，node22 PID29468仍99.8% CPU。train keys.csv于09:18:16为631,019行（含表头，631,018数据键），较09:16:57的625,947行增加5,072/79秒，约64.2条/s；root 09:18:08独立快照630,264行，趋势一致。manifest尚无，val/训练未开始。
2026-10-19 09:18 CST（即时增量）：root 09:18:21记录train keys=631,235；Luna 09:18:30实测631,775行（含表头），node22 PID29468仍99.8% CPU，214889=R、214221=Q。pack_manifest仍未生成，构建持续推进。
2026-10-19 09:19 CST（Luna实测）：PBS 214221=Q、214889=R；elapsed约02:45:57/08:00:00，PID29468在node22以99.8% CPU运行。09:19:40 train keys.csv=636,416行（636,415数据键），较09:18:30的631,775行增加4,641/70秒，约66.3条/s；root 09:19:32报告635,660行，方向一致。manifest仍无，pack未完成。
2026-10-19 09:21 CST（pack进展）：PBS 214221=Q、214889=R，elapsed约02:47:17/08:00:00。node22 PID29468仍99.8% CPU。09:21:00 train keys.csv=641,382行（641,381数据键），较09:19:40的636,416行增加4,966/80秒，约62.1条/s；root 09:20:52报告640,842行，趋势一致。val keys和manifest未出现，train目标714,914尚余73,533条。
2026-10-19 09:22 CST（持续监控）：PBS 214221=Q、214889=R，214889 elapsed约02:48:30/08:00:00；node22 PID29468仍99.8% CPU。09:22:13 train keys.csv=646,133行（646,132数据键），较09:21:00增加4,751/73秒，约65.1条/s；root 09:22:07记录645,593，持续推进。manifest未出现，仍train pack。
2026-10-19 09:23 CST（train pack进度）：PBS 214221=Q、214889=R；elapsed约02:49:48/08:00:00，node22 PID29468约99.8% CPU。09:23:31 keys.csv=651,099行（651,098数据键），较09:22:13的646,133行增加4,966/78秒，约63.7条/s；root 09:23:24记录650,560行。val目录和manifest仍未出现，尚未转阶段。
2026-10-19 09:24 CST（持续进展）：PBS 214221=Q、214889=R；elapsed约02:51:03/08:00:00。node22 PID29468仍99.8% CPU。train keys.csv在09:24:46为655,849行（655,848数据键），较09:23:31的651,099行增加4,750/75秒，约63.3条/s；root 09:24:39报告655,417行。manifest仍未生成，pack继续train段。
2026-10-19 09:25 CST（实时复核）：PBS 214221=Q、214889=R，elapsed约02:51:33/08:00:00；node22 PID29468约99.8% CPU。root 09:24:49实测train keys.csv=655,957行（655,956 keys）；Luna 09:25:16实测657,792行（657,791 keys），较09:24:49增加1,835/27秒，约68.0条/s。val目录和pack_manifest仍无，pack仍在train阶段。
2026-10-19 09:28 CST（阶段监控）：PBS 214221=Q、214889=R，elapsed约02:54:45/08:00:00。node22 PID29468仍99.8% CPU执行train pack。09:28:28 keys.csv=670,097行（670,096数据键），较09:25:16的657,792行增加12,305/192秒，约64.1条/s；root 09:28:11报告668,910行。train目标714,914尚余44,818键；val目录、manifest未出现。日志无错，source witness/bank gate PASS，GPU0 0%/2MiB。
2026-10-19 09:31 CST（train pack进展）：PBS 214221=Q、214889=R，elapsed约02:57:24/08:00:00；node22 PID29468约99.8% CPU。09:31:07 keys.csv=680,356行（680,355数据键），较09:28:28的670,097行增加10,259/159秒，约64.5条/s；root 09:30:50报告679,169行。train目标714,914尚余34,559键，val/manifest仍无，正常推进。
2026-10-19 09:36:28 CST（只读增量）：PBS全队列为214221=Q、214889=R（elapsed 03:02:45/08:00:00）；node22 PID29468仍以99.8% CPU执行build_future_ghi_head_pack.py。train/keys.csv为700,868行（约700,867数据键），较09:35:47的698,169行增加2,699；pack仍处于train构建阶段，pack_manifest.json尚不存在。未观察到失败或阶段转换。
2026-10-19 09:40:48 CST（train→val阶段转换）：PBS 214221=Q、214889=R，elapsed约03:07:05/08:00:00；PID29468仍运行pack builder。train/keys.csv=714,915行，等于crosswalk声明的714,914个train匹配键；val/keys.csv初见3,153行（3,152数据键），val数组已创建，pack_manifest.json尚无。crosswalk声明val匹配键66,590、val序列3,140、GHI标签4,479，因此val仍在写入中，当前计数不作完整性判定。
2026-10-19 09:43:34 CST（val pack增量）：214889仍R，elapsed约03:09:51/08:00:00，PID29468维持99.7% CPU；val/keys.csv=12,582行（12,581数据键），较09:42:25的8,238行增4,344/69秒，约63键/秒。相对crosswalk目标66,590仍在构建，粗估余约14分钟，当前walltime余约4小时50分；manifest尚未生成，GPU仍处于pack期空闲。
2026-10-19 09:57–09:59 CST（pack完成与训练启动）：manifest为COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK，train/val行数分别714,914/66,590，与crosswalk一致；test_payload_opened=false、buffer_payload_opened=false、future_observed_agri_used_as_input=false、cpp_used_as_input=false。独立只读审计通过：train/val sample keys各自全唯一且交集为空；全部数组float32、shape分别为agri_raw[N,3,13,3,3]、geometry[N,3,2,3,3]、cot_log1p[N,3,1,3,3]、ghi/clear[N]，所有值有限；manifest中10个npy哈希及2个keys.csv哈希均匹配。PBS 214889仍R，同一作业已创建future_ghi_cot_source_heads_v1/run_manifest.json（RUNNING，seeds 42/43/44、30 epochs、batch 1792、A/AH/AC定义清楚；test=false），进入头部pilot训练阶段。
2026-10-19 10:00 CST（首批训练资源快照）：同一214889已完成Sili seed42 A/AH/AC三臂30 epochs（每臂5,760 updates，耗时28.3/21.8/17.8秒），seed43-A又完成30 epochs/5,760 updates（19.3秒），当前继续seed43-AH。GPU0 A800抽样显存1,255–1,291 MiB、利用率14–36%；单arm很快，按18个station×seed×arm估算总训练耗时远低于8h walltime，无需在活跃作业上优化或争卡。当前仍为screening pilot（train_forecast_oof=false），不构成paper-grade估计。
2026-10-19 10:03:45 CST（pilot训练增量）：214889仍R，同一head训练日志已有14/18个station×seed×arm完成（Sili三seed全臂完成；Zhujia seed42全臂、seed43 A/AH完成，当前Zhujia seed43-AC在训），最近ARM_COMPLETE均为30 epochs/完整updates并带checkpoint/prediction SHA256、test_payload_opened=false。GPU0抽样显存约1,335 MiB、利用率26–36%；日志连续推进、无错误。
2026-10-19 10:05 CST（pilot与双审计完成）：PBS 214889= C，资源walltime 03:32:01；日志末行为FUTURE_GHI_COT_SOURCE_PILOT_COMPLETE test_payload_opened=0。run_manifest为COMPLETE_FUTURE_GHI_COT_SOURCE_SCREENING_PILOT；18/18臂均完成。paired validation audit为COMPLETE_PAIRED_FUTURE_GHI_COT_SOURCE_VALIDATION_AUDIT，52个目标BJT日期块、2,000次bootstrap；等站点均值RMSE A=129.321、AH=129.391、AC=129.163 W/m²，AC−A=-0.158、AC−AH=-0.228 W/m²，95%日期块CI分别[-1.951,1.530]、[-1.811,1.380]，均跨0；AH−A=+0.070，CI[-0.584,0.714]。strata审计COMPLETE_DESCRIPTIVE_FUTURE_GHI_COT_STRATA_AUDIT，72行；test/buffer均false。此为validation screening pilot，train_forecast_oof=false，不是论文级结果，未提交任何新作业。
2026-10-19 10:10:09 CST（AC50作业启动监控）：完整PBS队列为214221=Q、214889=C、214906=R（仅2个未结束任务）；214906在node22申请1张A800、8 CPU、8h，elapsed 00:00:51。日志通过source hashes、未来COT收缩合成见证、配对bootstrap见证、pack/base pilot gates，并确认baseline A/AH/AC pilot complete；run目录future_ghi_cot_shrink_v1/run_manifest.json为RUNNING，AC50定义为history COT不变、未来normalized COT anomaly乘0.5；test_payload_opened=false。启动初期GPU0仅2MiB/0%，训练仍在初始化，无错误。
2026-10-19 10:11:37 CST（AC50训练阶段）：214906仍R、214221=Q、214889=C。node22 PID129718运行train_future_ghi_heads_cot_shrink.py；Sili seed42的A/AH/AC已完成，AC50正到epoch20/30（3,840/5,760 updates），前3臂各自已落checkpoint/metrics/validation_predictions；run manifest中AC50仅将未来normalized COT anomaly乘0.5、历史COT保持不变。GPU0抽样1,327MiB/27%，无错误；输出仍为验证预测，test flag false。
2026-10-19 10:12:54 CST（AC50 pilot增量）：214906仍R（00:03:36），全队列214221=Q、214906=R；training log显示Sili seed42与seed43的A/AH/AC/AC50共8/24臂已完成，每臂30 epochs、5,760 updates，并带checkpoint/prediction哈希、test=false；当前Sili seed44-A在epoch1。GPU0约1,291MiB、24%，日志无错。
2026-10-19 10:14:42 CST（AC50 pilot增量）：214906仍R（00:05:24），全队列214221=Q、214906=R；training log已有Sili三seed×A/AH/AC/AC50共12/24臂完成，所有完成臂30 epochs、5,760 updates并带hash、test=false；当前进入Zhujia seed42-A epoch1。GPU0约1,339MiB/23%，无错。
2026-10-19 10:22 CST（AC50 pilot及配对审计完成）：PBS 214906=C；全队列214221=Q、214906=C，log末行为FUTURE_GHI_COT_SHRINKAGE_PILOT_COMPLETE test_payload_opened=0。run_manifest状态COMPLETE_FUTURE_GHI_COT_SHRINKAGE_SCREENING_PILOT，24/24臂、seed42/43/44，train_forecast_oof=false；paired audit COMPLETE_PAIRED_FUTURE_GHI_COT_SHRINKAGE_VALIDATION_AUDIT，要求每臂/seed精确匹配station/init/target/lead/GHI-row键及truth，52个目标BJT日期块、2,000 bootstrap，test=false。两站等权seed均值RMSE A/AH/AC/AC50=129.323/129.396/129.169/129.168 W/m²；AC50−AH=-0.229（95% CI[-1.985,+1.574]），AC50−AC=-0.001（[-0.633,+0.644]），AC50−A=-0.155（[-2.122,+1.744]），均跨0；AC−AH=-0.227（[-1.851,+1.427]）。无稳定未来COT/收缩收益，AC50也未优于完整AC。lead分段描述显示AC50−AH在Sili +15–60为+0.566 W/m²、Zhujia -0.598；Sili 75–120/-0.819、135–240/-0.933，但Zhujia相应为+0.495/+0.313，表现依站点而异。结果只属in-sample forecast-input screening，不是论文级估计；按计划不应盲目继续调收缩系数，后续需先结合分时效和既有COT分层定位。未提交后续任务。
2026-10-19 10:42 CST（forecast-COT train/val输入分布诊断完成）：PBS 214918=C；全队列仅214221=Q、214918=C。日志状态COMPLETE_FORECAST_COT_BANK_INPUT_SHIFT_DIAGNOSTIC并有FORECAST_COT_BANK_SHIFT_AUDIT_PASS。JSON记录bank 33,253序列，train/val分别973,952/90,144个站点×lead输入实例，5,086个分层CSV行；forecast来源为冻结Hunan SimVP预测AGRI→冻结R，无CPP。train forecasts为in-sample；test/buffer/future_observed_agri/CPP标志均false。CSV独立核对得2,494个train-val匹配分层，98个train-only分层（11,846样本，目标月仅2025-09/10），无val-only。全分层像素加权均值预测COT train/val=22.282/22.688；>=10比例55.43%/57.77%，>=30为27.53%/28.28%；3x3空间std 0.4743/0.4828、梯度0.3926/0.4002；target-to-prior变化2.9389/2.9397。仅共同分层按样本加权的val-train均值差+0.488、P90差-8.02，显示分布变化不单调；总体差异与月份/太阳高度/时效构成及train in-sample状态混杂。此为输入分布诊断，不是COT精度或GHI收益，也不能单独证明in-sample过拟合是原因。
2026-10-19 11:04 CST（214922反事实审计失败）：最新完整队列为214221=Q、214922=C（仅1个未结束）；214922 elapsed 00:06:24、exit_status=1，PBS记录峰值mem约654,962,000 kB。日志显示pack与AC/AC50 pilot元数据门禁通过（pack test/buffer/future_observed_agri/CPP均false，pilot test=false），但在 audit_future_ghi_cot_counterfactual.py:192 的 `torch.from_numpy(model_cot(...)).to(cuda)` 抛出CUDA OOM，尝试申请312.07 GiB；counterfactual JSON/CSV均未生成，配对比较与52目标BJT日期块2000次bootstrap未完成，因此不得报告AC/AC50的future_zero或future_permuted相对native差值。失败前节点PID49525约655GB RSS、CPU约97%、GPU约556MiB，内存与I/O状态显示异常增长后作业退出；现场现已恢复内存。脚本静态形状核对提示可能的广播维度错误：load_station返回history_cot_mask为[N,3,1,1,1]，而model_cot再次按[N,3]假设扩成[:, :, None,None,None]，产生多余维度并触发超大广播；需修复并用合成见证验证后另行授权重跑，本轮未修改/重提任务。该诊断仅针对验证集固定head的输入敏感度，不等于可部署收益。
2026-10-19 11:13 CST（214924复跑启动即SHA门禁失败）：PBS全队列214221=Q、214924=C（仅1个未结束）；214924 elapsed 00:00:03.6、exit_status=1，未使用GPU/CPU（PBS资源0）。新日志显示`sha256sum -c`从`$PROJECT`根目录校验`$EXP/FUTURE_GHI_COUNTERFACTUAL_SHA256SUMS`，但SUMS当前列出的三文件路径只有basename（audit_future_ghi_cot_counterfactual.py、test_audit_future_ghi_cot_counterfactual.py、run_future_ghi_cot_counterfactual.pbs），实际文件位于`refine-logs/cot_aware_simvp_20261017/`，故均报No such file；脚本在任何诊断载荷读取/计算前退出。JSON/CSV不存在，配对与bootstrap没有开始，无法报告敏感度RMSE差。此为复跑环境/校验清单路径不一致，不代表数据或科学审计失败；本轮未改脚本/提交任务。需另行修正SUMS相对路径或校验cwd并通过服务器见证后再授权重跑。test/buffer载荷未读取。
2026-10-19 11:16 CST（214926未来COT反事实敏感度审计通过）：PBS最新全队列214221=Q、214924=C、214926=C（1个未结束）；214926 exit_status=0、walltime 00:00:30、PBS资源mem 579,108 kB，node22 GPU0分配；完成后GPU回到2 MiB/0%。日志依次通过三文件SHA、反事实合成见证、pack/pilot门禁，并以FUTURE_COT_COUNTERFACTUAL_AUDIT_PASS / FUTURE_GHI_COT_COUNTERFACTUAL_COMPLETE结束。输出future_ghi_cot_counterfactual_v1.json/csv已生成（JSON 5,087 B、CSV 10,103 B，144条唯一arm×station×seed×scenario×lead-group聚合行）；JSON pack/run manifest SHA与实际文件独立比对一致。状态COMPLETE_VALIDATION_FUTURE_COT_COUNTERFACTUAL_SENSITIVITY_AUDIT，52个目标BJT日期块、2,000次配对日期块bootstrap；test_payload_opened、buffer_payload_opened、future_observed_agri_read、cpp_read全false。配对设计核验：每站点/seed/arm固定同一checkpoint和同一validation `data`/keys/GHI truth，native、future_zero、future_permuted只换未来COT输入；置换在station×target month×lead×solar-regime内进行；所有对比共享同一套52日期重采样。all-lead replacement-minus-native RMSE（W/m²，95%日期块CI）：AC future_zero +31.946 [20.844,43.052]，AC future_permuted +22.493 [14.583,32.100]；AC50 future_zero +15.185 [7.914,22.854]，AC50 future_permuted +12.975 [8.004,19.110]。均为正且区间不跨0，说明该固定screening head在此validation样本上对未来COT对齐敏感；不等于COT带来可部署GHI收益、独立因果效应或论文级估计（训练forecast输入为in-sample）。此前214924仅路径门禁失败；本次未重提其他任务、未读test/buffer载荷。

2026-10-19 12:08 CST（OOF M1正式fold manifests生成通过）：完整月度SimVP官方manifest与future forecast bank train index按seq_id联结。官方train 35,320条/587个BJT训练日；其中1,967条24帧窗口跨BJT日期，折分前整体purge；33,353条入五折。bank所需30,436条全部覆盖且唯一OOF。fold0–4上游train分别25,839/26,172/26,643/27,521/27,237，留出7,514/7,181/6,710/5,832/6,116；各折train-heldout帧交集均为0。fold按月hash平衡。生成器、固定epoch OOF训练器及合成见证已本地/服务器py_compile和运行通过。发现并堵住两处潜在泄漏：M0只看bank而低估正式上游训练池；原训练脚本逐轮用fold heldout验证选checkpoint。新runner只构造train dataset，固定12轮使用epoch12，不读取fold heldout载荷。M1输出`results/future_ghi_oof_manifests_v1/manifest_build_audit.json`。当前队列需提交前即时复查；M2计划一张A800完成fold0归一化、batch profile和固定12轮首折试跑；test/buffer仍未触碰。

2026-10-19 12:18 CST（fold0 M2运行中；OOF预测器准备完成）：PBS全账户队列214221=Q、214933=R。214933在node22只分配一张NVIDIA A800-SXM4-80GB，elapsed约00:06:18，完成部署哈希、M1 manifest及本地合成见证门禁；fold0 train-only归一化处理12,000/38,121唯一训练帧，近段约33帧/s。GPU尚处归一化前的空闲阶段；尚未profile或开始SimVP训练，不能报告模型效果。新增predict_future_ghi_oof_fold.py只对前8帧调用AGRI loader；未来16帧只读取太阳几何sidecar，然后运行SimVP和冻结R，输出fold-heldout train键的预测AGRI/COT。因果路径合成见证及服务器编译通过，防止推理阶段读到未来真实AGRI。后续等214933完成归一化/profile/首折，以实测耗时推进余折。

2026-10-19 12:22 CST（OOF下游预测/合并管线准备完）：新增OOF forecast predictor和bank merge工具。predictor按fold heldout ID推理，只调用AGRI读取函数处理8帧历史，未来16帧只读取确定性太阳几何sidecar，之后使用epoch12 SimVP及固定R生成3×3 AGRI/COT。merge器要求每个原train bank seq_id恰好来自唯一折，并保留已有validation array；拒绝任何未来AGRI/CPP/test/buffer读取标志或不完整key覆盖。两者在本地与服务器py_compile和合成测试通过；validation保持原全train SimVP预测输入。当前214933仍R、214221仍Q；fold0归一化到19,500/38,121帧，约32帧/s，尚未进入GPU profile/SimVP训练。

2026-10-19 12:28 CST（fold0归一化近完成）：PBS全队列214221=Q、214933=R，214933 elapsed约00:16:29，仍在node22单A800作业内；日志进度32,500/38,121个fold-train唯一帧。CPU预处理速率约32帧/s，CUDA profile/训练尚未开始，未生成实验指标。代码、计划、M1 audit、OOF预测/合并脚本及其合成见证均已同步到服务器并核对SHA；测试与buffer仍未读。
2026-10-19 12:14 CST（214933 fold0 归一化完成并进入单卡profile）：新鲜qstat为214221=Q、214933=R，elapsed约00:20:26/36h，node22单张A800。日志`NORMALIZATION_DONE`（最后进度38000/38121）；fold_0/hunan_train_norm.json核验`train_unique_frames=38121`、`validation_test_used=false`，manifest/train-frame SHA字段已写入。当前PID125827执行`--mode profile`；此快照GPU 2MiB/0%，profile.json尚未生成，尚无batch吞吐/峰显存结果。未开始固定epoch训练。
2026-10-19 12:16 CST（214933 fold0单卡profile通过并进入固定训练）：新鲜队列214221=Q、214933=R；profile PASS，A800实测batch4/8/12/16/20/24 samples/s分别20.96/22.60/23.48/23.98/24.23/22.77，峰显存GiB分别12.26/24.37/36.48/48.59/60.70/72.81；batch24超过85%门限，按最大可用且近峰吞吐规则选batch20（≤85%），profile checkpoint reload最大输出差0，test_payloads_read=0。训练PID127887开始fold0固定seed42/12 epochs，profile.json与training/run_metadata.json已落盘；该快照GPU约65,545MiB/81,920MiB（瞬时利用率0%），node22本地cache盘约320GB可用。尚无epoch12或training_complete产物。

2026-10-19 12:16 CST（fold0 profile通过并进入固定训练）：PBS 214221.tc6000=Q、214933.tc6000=R。fold-train唯一帧归一化完成（38,121帧）。同一张node22 A800真实profile选batch20，吞吐24.23 samples/s、峰值显存60.70 GiB；batch16为23.98 samples/s/48.59 GiB，batch24为22.77 samples/s/72.81 GiB并超过85%显存门槛。profile checkpoint保存重载输出最大差0；test_payloads_read=0。固定12轮随机初始化训练已启动；目前尚无epoch完成、预测或GHI指标。根据实测吞吐，先等fold0完整训练收据再估算余下四折和OOF预测时长。
2026-10-19 12:20 CST（余折调度器预备）：为fold1–4制作通用单折PBS执行器`run_future_ghi_oof_fold.pbs`，要求显式折号0–4，seed=42+fold，逐折检查无覆盖、norm新鲜、测试关闭及完整M1覆盖；远端`bash -n`通过。当前fold0训练仍运行，尚未提交其它折，保持单卡优先和队列门禁。
2026-10-19 12:24 CST（实训吞吐复核）：step100于12:22:09、step200于12:24:35，100步用146秒，batch20实际约13.70 samples/s；Luna节点采样A800 GPU约100%利用率、显存约65.5GB，CPU主进程及4个worker繁忙。Profile的24.23 samples/s来自直接取样后GPU计算，不含完整DataLoader，故仅作显存/计算profile，不作为整轮耗时。按每折约1,292步/轮外推约31分钟/轮、12轮约6.3小时；以epoch记录实测修正。当前仍不需要改workers/批量，避免无证据更改折间训练合同。
2026-10-19 12:28 CST（214933 fold0训练吞吐复测）：PBS仍214221=Q、214933=R；epoch1 step400。step100→200为146秒（batch20约13.70 samples/s），step200→400为214秒（约18.69 samples/s，profile直采24.23的77.1%）。第二段吞吐提升且GPU dmon样本为100%利用、显存65.5GB；主进程约102% CPU、4个DataLoader worker约82–84%各自CPU。本地cache约180GB、/tmp可用约320GB。当前没有证据判为纯数据加载瓶颈；按近段吞吐估计约23分钟/epoch、约4.6小时完成12轮，walltime充足。运行中未改参数；后续余折如需调整，可在fold0完成后统一评估最多6 workers（8核配额内），当前先保持合同。
2026-10-19 12:33 CST（fold0首轮数据加载吞吐复核）：qstat仍214221=Q、214933=R；step400→500耗323秒，batch20约6.19 samples/s（profile直接GPU batch20=24.23 samples/s的25.5%）。A800 8秒dmon抽样为3秒SM=100%、5秒SM=0，显存固定约65.5GB；主进程约103% CPU，4个DataLoader workers约82–84% CPU。缓存位于节点本地/tmp约180GB、可用磁盘约320GB。该窗口提示训练DataLoader/CPU样本准备可能使GPU间歇等待，尚需后续窗口验证；不干预当前作业。余折如需执行可在8核配额内将workers由4增至6（给主进程约1核并留系统余量），以实测利用率/吞吐决定是否保留。
2026-10-19 12:39 CST（fold0训练进展复核）：qstat为214221=Q、214933=R；epoch1 step800。step500→800耗366秒，batch20折算约16.39 samples/s（profile 24.23的67.7%），较400→500窗口6.19 samples/s恢复；8秒A800 dmon约6秒SM=100%、2秒低/空，显存约65.5GB。4 workers各约83–85% CPU，节点cache盘仍余约320GB。早期吞吐波动暂未显示持续瓶颈，按近段估计约26分钟/epoch、12轮约5.2小时；作业walltime充足，继续原样运行。
2026-10-19 12:43 CST（fold0训练吞吐/输入链路核查）：PBS 214221=Q、214933=R，epoch1 step900；step800→900耗223秒、batch20约8.97 samples/s（profile 24.23的37.0%），显著低于此前500→800的16.39 samples/s。A800 8秒dmon约2秒SM=100%、6秒SM=0；主进程约102% CPU、4个worker各约82–84%。静态核对上游MonthlyDataset：每个样本顺序访问24帧，每帧从节点本地NPZ cache解包；每个batch20涉及约480帧文件读取/解码，可能解释间歇GPU空等（cache约180GB，节点本地盘余约320GB）。建议只用于fold0完成后的余折：在每作业8核内将workers 4提高到6，保留主进程和系统余核，再用step区间吞吐及GPU duty复测；当前作业保持不变。墙钟36h充足。
2026-10-19 13:03 CST（OOF首折继续运行，后续流水线加固）：PBS新查为214221=Q、214933=R（2/3）；214933 node22单A800，最新Luna现场12:43记录epoch1 step900，step800→900耗223秒，batch20约8.97 samples/s；GPU 8秒采样中约2秒SM满载、其余低/空闲，数据loader读取/解包是候选瓶颈，但时段吞吐波动大，不停止当前训练。累计训练checkpoint仍未完成第1轮。

后续单折PBS允许workers 4–6，下一折拟以6 worker测实际吞吐；样本顺序生成仍按seed+epoch固定，模型合同不变。准备完成预测折作业及GHI OOF头作业。另加固来源证明：pack要求明确5折OOF来源、完整bank训练键覆盖、五个fold checkpoint SHA及OOF银行数组/索引SHA；合并时逐折复核预测数组与index哈希；GHI audit可要求OOF并检查所有station×seed×arm产物齐全。相关本地py_compile、OOF来源见证、成对bootstrap见证、merge见证、未来输入因果见证及3个PBS bash -n均通过。变更尚未同步服务器；未提交新作业，test/buffer仍未读取。

2026-10-19 12:49 CST（fold0实际DataLoader读量与吞吐）：PBS仍214221=Q、214933=R，fold0 epoch1 step1000；step900→1000耗361秒，batch20约5.54 samples/s。A800 8秒dmon有5秒SM非零、3秒为0，显存约65.5GB。四个DataLoader worker `/proc/PID/io` 10秒两次采样合计rchar增加约13.6GB（1.36GB/s），read_bytes近零，表示热本地/页缓存读取量大而非远端物理盘等待；结合每样本24个NPZ缓存帧读取与worker CPU约78–83%，支持npz读取/解码和样本组装使GPU间歇空等的判断。仍不改当前作业；fold0结束后余折可在8核内将workers由4调至6并实测吞吐/GPU duty。

2026-10-19 13:08 CST（OOF流水线就绪）：远端队列214221=Q、214933=R（2/3）；fold0训练日志已到epoch1 step1000，作业继续运行。通用折PBS支持4–6 workers，下一折以6 workers测同合同吞吐。OOF预测折runner和GHI OOF头runner已写好；来源审计增加五折checkpoint hash及数组/index hash核验，成对GHI审计要求OOF且检查请求的两站×种子×三臂全部完成。远端文件散列通过，远端py_compile、四项合成见证、三份PBS bash -n通过。改动同步至服务器但未qsub，test/buffer仍关闭。

2026-10-19 13:11 CST（fold0训练增量）：PBS仍214221=Q、214933=R（2/3）；214933 elapsed约00:59:14，fold0日志到epoch1 step1100，尚未完成首轮。step900→1000实测5.54 samples/s；随后日志有继续增长，近段约6 samples/s。作业健康、未中断；当前不提交折1，待首折完成并按单卡优先及qsub前队列门禁续接。
2026-10-19 12:55 CST（fold0首epoch末段）：qstat 214221=Q、214933=R；日志到epoch1 step1200，1000→1200用392秒，约10.20 samples/s（profile 24.23的42.1%）；6秒A800 dmon中4秒SM=92–100%、2秒0，显存约65.5GB。尚无epoch_end/checkpoint，按1292 steps/epoch约93%完成。
2026-10-19 12:57 CST（214933 fold0首轮完成）：PBS全队列214221=Q、214933=R；epoch1固定训练结束step1292，train_loss_mean=0.339361，elapsed 2440.33s（40.7min），`epoch_001.json`和`last.pt`已落盘。按该轮25,839样本计约10.59 samples/s，为profile24.23的43.7%；线性估算12轮约8.1h，PBS剩余约34.9h，未见资源故障。训练metadata/epoch记录heldout_payload_read=false，test_payloads_read=0；尚无epoch12 checkpoint与training_complete。


2026-10-19 13:20 CST（OOF fold0持续运行，运行中合同未改动）：新鲜qstat全账户为214221=Q、214933=R（共2个未结束作业）；214933 node22单A800，elapsed约1:08:25，日志推进至epoch2 step1400。epoch1固定12轮训练中的首轮记录为step1292、loss均值0.339361、耗时2440.33秒（约10.59样本/秒）；尚无验证预测或GHI结果，test/buffer未读。登录节点单独执行persistent-worker顺序见证超过3分钟无输出，已终止本次自启检查进程；未触碰PBS训练作业。该见证状态为未完成，不作为PASS/FAIL；折间6-worker/persistent优化继续门控，先确保折间训练语义一致。当前折继续原样跑，不重启、不改参。


2026-10-19 13:11 CST（服务器时钟；折间并行启动）：qsub前全账户队列214221=Q、214933=R（2/3），完成fold1 manifest与输出目录唯一性检查后提交214941.tc6000，参数fold=1、workers=6；提交后队列214221=Q、214933=R、214941=R（3/3）。214941已开始但日志仍为0字节，需待其PBS前置门禁写出结果；脚本先运行训练源/manifest检查及persistent-loader顺序见证（120秒超时），只有通过才进入固定12轮训练。fold0最新日志epoch2 step1600；node22 A800瞬时GPU util100%、显存70865/81920MiB。仍未打开test/buffer，未改论文正文；队列到3/3后不再提交任何作业。


2026-10-19 13:19 CST（服务器时钟；顺序问题已修复并重试）：214941在训练前持久worker顺序见证失败，原因是DataLoader复用iterator时不会像每轮新建loader那样重新抽取worker base_seed，导致该epoch的RandomSampler排列偏移；该PBS已C，未读取训练样本、未生成fold输出。训练脚本在epoch>0的persistent-loader路径显式消耗同一int64 base_seed；本地py_compile、远端swc py_compile及全套代码SHA检查通过。折1于全队列未结束2/3时重提为214944（fold=1、6 workers，队列提交后214221=Q、214933=R、214944=Q；13:19已R）。214944重复执行顺序见证，通过前不会normalize/train。fold0已epoch2 step1800。测试与buffer保持关闭。

2026-10-19 13:24 CST（训练语义修复；等待计算节点见证）：214941和214944两次均仅在persistent DataLoader顺序见证失败后结束，未进入normalize/train且未创建fold1目录。额外消耗base_seed的修复无效，已撤销；训练器恢复每轮按seed+epoch新建非持久DataLoader，保持与fold0一致的样本shuffle语义，fold1拟使用6 workers、batch20测实际吞吐。新的登录节点合成见证20秒内未返回，不能算通过；PBS前置脚本将在node22上以120秒限时运行低开销（num_workers=0）的顺序见证，失败仍自动停在数据载荷读取前。远端代码已py_compile，剩余计算节点验证门禁未通过前不宣称fold1开始训练。队列fresh snapshot 214221=Q、214933=R、214941/214944=C，未结束2/3；test/buffer保持关闭。


2026-10-19 13:27 CST（fold1通过前置见证）：计算节点PBS 214945已通过manifest、train-only和逐epoch shuffle顺序三项见证，fold=1、seed=43、GPU_ID=1（gpu2）、workers=6；fold1为train 26,172、heldout 7,181、OOF bank 6,430序列，帧交集0。13:27:42日志到NORMALIZATION 500/38,454，仍在fold-only归一化，未进入profile/train。PBS队列214221=Q、214933=R、214945=R（3/3）。fold0日志step2000，较13:19 step1800新增200步；未动首折。测试/缓冲载荷未读。

2026-10-19 13:29 CST（fold1归一化进度）：214945 normalization日志由13:27:42的500推进至13:28:40的1,000/38,454帧，workers=6；此阶段GPU仍空闲，尚未profile或训练。214933 fold0仍step2000；队列214221=Q、214933=R、214945=R（3/3）。未读取heldout/test/buffer payload。

2026-10-19 13:45 CST（首折挂起诊断与checkpoint恢复提交）：214933从epoch2 step2000起约22分钟无CPU/I/O/GPU进展，主进程/4 workers计数静止，GPU0仍占约70.8GiB但SM=0%，3秒只读strace见poll/futex等待；epoch1原子checkpoint/optimizer mtime 12:56:51、step1292。仅结束本项目挂起作业214933（PBS转C），未动其他任务；epoch1 checkpoint仍在。恢复runner、metadata witness与shell/Python语法已本地及目标计算节点检查，训练预算仍12 epochs，resume从epoch2继续，dropout RNG以seed+1,000,003×resume_epoch确定性重启，epoch sampler继续seed+epoch。本次worker=0绕开疑似多进程等待。全账户fresh queue：214221=Q、214945=R、214946=Q（3/3）；214946尚未开始，214945 fold1归一化13:43记录14,000/38,454。未读取heldout/test/buffer payload。

2026-10-19 13:58 CST（服务器现场复核）：全账户PBS为214221=Q、214945=R、214946=R（3/3）。恢复作业214946已通过OOF_RESUME_METADATA_WITNESS_PASS与OOF_FOLD0_RESUME_PREFLIGHT_PASS，核验fold0 epoch001/step1292 checkpoint及train-only来源；日志尚未出现恢复副本哈希或训练step，暂不算续训已开始。fold1 214945的fold-only规范化推进至18,500/38,454帧，进程仍在共享文件系统读取；未触碰heldout/test/buffer。队列满，不提交新作业。

2026-10-19 14:06 CST（恢复任务越过前置门禁）：PBS 214946已通过fold0恢复检查；esume_from_epoch001.pt与原last.pt均131,910,310 bytes，SHA-256相同11bbf9cbc089199c6933c8e93a51cafa1218c89d3d2f316e4e79c6a8a3e44524，来源epoch1/step1292，heldout/test未读。启动日志已解析GPU_ID=0、NODE=gpu2，但实际resume Python/训练step尚未出现；PBS内
vidia-smi进程在节点共享文件系统等待（wchan=lock_blkgrp_sync），不作GPU训练已启动的表述。fold1归一化推进到23,000/38,454。全账户214221=Q、214945=R、214946=R（3/3），暂不提交新作业。

2026-10-19 14:07 CST（恢复Python进程已启动）：实时qstat仍214221=Q、214945=R、214946=R（3/3）。fold0恢复进程esume_future_ghi_oof_fold.py PID112660已出现，约28秒、WCHAN=lock_blkgrp_sync，说明已越过nvidia-smi检查并进入Python恢复程序；截至14:07:21尚无RESUME_TRAIN_STEP，GPU0空闲，不能算训练step已开始。fold1 normalization日志推进至23,500/38,454。checkpoint恢复副本与原last.pt SHA一致；heldout/test/buffer关闭。

2026-10-19 14:14 CST（恢复训练进入checkpoint载入阶段）：214946 fold0恢复Python PID112660已越过脚本门禁、建立CUDA上下文；/proc读取计数约380,304,152 bytes，20秒内增加约1.7 MB，当前WCHAN=ofs_tjobq_timedwait_lock；GPU0使用约473 MiB、利用率0%，尚未出现RESUME_TRAIN_STEP。进程fd曾从torch模块文件推进到torch/依赖及checkpoint读取链，表明初始化仍在缓慢读取共享存储，暂不判为失败。fold1最近可核实记录14:11为25,500/38,454，继续train-only normalization。全账户PBS 214221=Q、214945=R、214946=R（3/3）；test/buffer未触碰。

2026-10-19 14:23 CST（持续推进复核）：fresh PBS全账户仍为214221=Q、214945=R、214946=R（3/3），不提交新作业。fold1的train-only归一化日志已推进到约34,000/38,454帧。fold0恢复进程PID112660已通过恢复元数据门并加载checkpoint/模型，正按训练样本顺序读取AGRI并写入node-local缓存；缓存约2.3 GiB，I/O和缓存文件有增长，但仍未出现RESUME_TRAIN_STEP，因此不报告恢复训练已开始。GPU0显存约10.5 GiB、采样利用率0%，该显存分配可能处于首批/首轮初始化，仍需step日志确认。heldout/test/buffer载荷未读。下一步继续观察缓存与step日志；若初始化/I/O出现持续无变化，再作短时只读strace判断，先不重提、不改活动任务。

2026-10-19 14:32 CST（训练前置阶段推进）：fresh PBS全账户为214221=Q、214945=R、214946=R（3/3）。fold1（214945）已完成38,454个train-only归一化帧，进入单卡profile，PID 9090已启动；尚未写出profile吞吐记录。fold0恢复（214946）仍无step日志；对PID112660的8秒只读strace观察到它持续读取下一批AGRI原始帧及太阳几何辅助文件，且本地缓存文件此前已有增长，故当前主要是首批顺序数据装载/共享存储等待，不是已确认的死锁或终态。GPU0显存约35 GiB、抽样SM 0%，尚未确认训练步；继续监控，不重启活动作业。heldout/test/buffer载荷未读、未提交新PBS任务。

2026-10-19 14:38 CST（OOF折间阶段更新）：全账户PBS仍214221=Q、214945=R、214946=R（3/3）。fold1 214945 train-only归一化已完成38,454/38,454帧，进入单卡profile；batch4/8/12暂测吞吐20.41/22.29/7.86 samples/s，batch8当前最快，profile尚未完成reload witness及训练前门。fold0恢复214946继续按训练顺序从源AGRI/辅助几何读取并填充自己的node-local缓存，缓存从约480增至约1,140个NPZ（约5.4 GiB）；短时strace证实源帧读取活动，GPU0约65.5 GiB分配但采样SM=0%，截至本记录仍无RESUME_TRAIN_STEP。没有判定作业失败，也没有新GHI结果；不跨折共享缓存、不重启活动作业、不提交新PBS任务，test/buffer保持关闭。

2026-10-19 14:41 CST（fold1进入固定训练）：全账户PBS仍214221=Q、214945=R、214946=R（3/3）。fold1（214945）train-only归一化、单卡profile及reload witness通过；batch20为选定配置，吞吐23.09 samples/s、峰值60.70 GiB；batch24峰值72.81 GiB超过85%显存门未选。fold1已进入固定12轮训练（PID 18884），首个TRAIN_STEP尚未出现。fold0恢复（214946）继续读取真实train序列并填充独立本地缓存，约1,599帧/7.6 GiB；短时跟踪确认源AGRI和几何文件访问，仍无RESUME_TRAIN_STEP，GPU0显存约65.5 GiB但瞬时SM=0%。两个作业保持运行；不跨折共享缓存、不提交新PBS任务、不读test/buffer。

2026-10-19 14:46 CST（两折实际训练/读取进展）：PBS全账户214221=Q、214945=R、214946=R（3/3）。fold1（214945）已完成train-only归一化、profile及reload witness，batch20为选定档（23.09 samples/s、60.70 GiB）；batch24超85%显存门。固定12轮训练PID18884已启动，GPU1的pmon采样SM 44/77/95%、显存约65.5GiB，显示训练计算活动，但尚未写出每100步的首条TRAIN_STEP。fold0恢复（214946）仍无RESUME_TRAIN_STEP；PID112660已读入并缓存约2,257个训练帧，strace证实源AGRI/几何辅助数据访问，GPU0约65.5GiB、短采样SM=0%，需继续观察缓存增量。未判失败、不重启、不跨折共享缓存。当前队列满，不提交新任务；test/buffer未读。


2026-10-19 14:52 CST（fold1训练步日志与fold0缓存复核）：fresh PBS全账户214221=Q、214945=R、214946=R（3/3），不提交新任务。fold1 214945固定12轮训练已记录epoch1 step100（loss 0.56643）与step200（loss 0.39958），GPU1近期pmon抽样SM=98%、显存约65.5 GiB，确认训练正在进行；尚无验证或GHI结果。fold0恢复214946的checkpoint/hash门已过，但PID112660仍无RESUME_TRAIN_STEP；node-local专属缓存约15.7 GB、6,231个文件（含锁文件），缓存最新文件时间推进至14:52，说明仍在顺序缓存训练输入；GPU0显存约65.5 GiB、短采样SM=0%。不判死、不重启、不跨折共享缓存。test/buffer关闭。

2026-10-19 14:55 CST（OOF两折继续推进）：fresh PBS全账户214221=Q、214945=R、214946=R（3/3），无新提交。fold1固定12轮训练已到epoch1 step300；step100/200/300 loss为0.56643/0.39958/0.33128，确认在迭代，尚无验证或GHI结论。fold0恢复作业PID112660仍未出现RESUME_TRAIN_STEP，但专属node-local缓存由14:52约15.7 GB/6,231文件增至14:55约17 GB/6,957文件，进程仍运行（CPU约7.5%），因此继续等待其完成输入缓存，不判死、不重启、不共享缓存。GPU0未显示训练计算。test/buffer关闭。

2026-10-19 15:04 CST（fold0恢复训练启动）：全账户PBS实时为214221=Q、214945=R、214946=R（3/3），无新提交。fold0恢复作业214946首次出现RESUME_TRAIN_STEP，epoch2 step1300 loss=0.24420，确认从epoch1/step1292检查点续训已实际开始；node-local专属缓存约23 GB、9,661文件（含锁文件）。fold1 214945已到epoch1 step700，step500/600/700 loss=0.32522/0.32444/0.30447。尚无验证评分、OOF预测或GHI结果；瞬时GPU采样为0%不推翻训练步日志。test/buffer关闭。

2026-10-19 15:07 CST（折1训练继续）：PBS全账户214221=Q、214945=R、214946=R（3/3）。fold1 214945已到epoch1 step800，step500/600/700/800 loss=0.32522/0.32444/0.30447/0.20211，PID18884仍运行。fold0恢复214946最新仍为epoch2 step1300 loss=0.24420，PID112660仍运行，恢复专属缓存约23 GB/9,661文件。两折尚无验证评分/OOF预测/GHI结果。瞬时GPU util为0%不能覆盖训练步日志。未提交新任务，test/buffer关闭。

2026-10-19 15:39 CST（训练推进与噪声路径核查）：fresh PBS为214221=Q、214945=R、214946=R（3/3）。fold1已完成epoch1并到epoch2 step2200；最新step1800/1900/2000/2100/2200 loss=0.16865/0.27619/0.21912/0.20949/0.17981，epoch1 checkpoint已保存。fold0恢复仍最新step1300，但PID112660运行中；node-local cache约65GB/27,516文件，短时strace观察到持续读取AGRI与太阳几何NPZ并写缓存，CPU约14.8%，故按慢速I/O继续观察，不重启。噪声假设代码审计：本地当前OOF trainer SHA与远端及run_metadata一致；AGRI数据、SimVP和训练代码SHA均与服务器一致。当前上游SimVP训练无显式加性噪声或几何扰动，模型配置drop=0；输入用train-only均值/方差标准化，QA无效像元置零，未来COT由预测AGRI经过冻结R产生。故可疑噪声更可能来自预测AGRI→R的误差传播或COT幅度偏差，而不是人为噪声增强。已有AC50单因素方案仅在当前OOF GHI筛选完成后、依据AC−AH门禁决定是否提交。没有修改论文或正在运行的实验合同；test/buffer关闭。
2026-10-19 15:54 CST（现场复核）：PBS全账户仍为214221=Q、214945=R、214946=R（3/3），未提交新任务。fold1固定12轮训练已完成epoch2（step2618，train_loss_mean=0.20984948，耗时1744.63秒），当前进入epoch3，日志到step2700；epoch1与epoch2 checkpoint均已保存，GPU1监测显存约66.9 GiB、瞬时利用率82%。fold0恢复作业214946仍未出现高于epoch2 step1300的新训练步，GPU0短采样利用率0%；但node22上的PID112660仍运行（CPU约19%、WCHAN为共享存储等待），专属本地缓存继续从约65 GB/27,516文件增至约94 GB/约39,056文件，故仍有输入缓存读取活动，不重启、不判死。两折尚无验证评分、OOF预测或GHI收益结果。test/buffer未读取，未改样本、掩膜、阈值或合同超参。

2026-10-19 16:05 CST（现场继续复核）：PBS全账户214221=Q、214945=R、214946=R，3/3。fold1 214945 epoch2完整结束于step2618（train_loss_mean=0.20984948，checkpoint已保存）；epoch3推进至step3300，step2700–3300日志持续增长，Luna与服务器日志一致。fold0恢复214946仍无step1300之后的新训练步；Luna在node22复核PID112660运行、CPU约21.7%，fold0节点本地缓存增至约111 GB，GPU0采样利用率0%，表明输入缓存阶段仍在推进但未开始可观察的后续训练步。无新OOF预测/验证/GHI文件，当前仍不能回答GHI是否获益。未提交作业、未改合同；test/buffer与未来真实AGRI/CPP推理输入继续关闭。

2026-10-19 16:09 CST（fold0本地盘保护门触发，已核实终态）：PBS 214946已为C，`qstat -xf`显示exit_status=1、walltime 2:15:23、执行节点node22；日志明确报错`node-local disk reserve below 25 GiB; refusing cache growth`。失败发生在训练输入读取/缓存阶段，未产生新checkpoint或续训完成标记，最近训练checkpoint仍为epoch1完成（续跑日志到epoch2 step1300）。node22临时盘实测1.5 TiB、99%已用、约25 GiB可用；fold0失败缓存约114 GiB，fold1缓存约181 GiB。fold1 214945仍正常训练，16:09日志epoch3 step3500；不得在其活动期间删除fold1缓存。已检查`RawCache.frame`：命中现有`.npz`会直接读取，resume路径不跑归一化，因此fold0部分缓存可保留用于续跑。下一步待fold1结束并保存完整checkpoint后，先核对checkpoint，再只清理已完成fold1的临时缓存以释放空间；之后fold0在node22复用部分缓存续训，避免同节点缓存叠加。其余fold待相同盘余量门禁通过后串行进入，未修改模型/数据/优化超参，也未提交新任务。当前PBS 214221=Q、214945=R，2/3；test/buffer及未来真实AGRI/CPP推理输入未读。

2026-10-19 16:12 CST（PBS队列更新）：fold0失败已核为PBS C，原因是node22本地盘可用空间低于保护阈值；全账户随后新增同账户作业214986（`hr2_rescore`，normal队列、node15），故当前未结束数为3/3：214221=Q、214945=R、214986=R。214945仍在node22推进，16:12日志到epoch3 step3600，epoch3尚未完成，最近checkpoint仍epoch2。node22 `/tmp`近满，fold0残余缓存约114 GiB、fold1缓存约181 GiB；维持缓存隔离，不提交任何新作业、不清理活动fold1缓存。下一步仍是待fold1完成后保存检查点并清理其自身缓存，再利用fold0现存缓存恢复，逐折串行释放本地盘空间。暂时没有OOF或GHI结果，训练中间损失不代表预报增益；test/buffer保持关闭。

2026-10-19 16:14 CST（折1增量）：Luna及服务器日志再次复核fold1从step3600推进至step3700，214945仍R；step/epoch3尚无结束记录或新checkpoint，当前last.pt仍是epoch2。fold0 214946已终态C（exit_status=1），失败原因为本地盘余量门触发。全账户未结束作业为214221=Q、214945=R、214986=R（3/3）；node22本地盘仍约99%/25 GiB可用。保持活动fold1和两份专属缓存不动，不提交新job。当前仅有训练进度，没有OOF预测、配对GHI评分或收益结论。

2026-10-19 16:20 CST（fold1完成epoch3）：PBS全账户214221=Q、214945=R、214986=R，3/3。fold1在16:19完成epoch3，step=3927，train_loss_mean=0.18465268，heldout_payload_read=false；epoch_003.json及last.pt均已原子落盘。训练继续固定12轮。fold0仍因node22本地盘门限退出，checkpoint未前进；缓存盘复核仍为1.5 TiB盘99%、约25 GiB可用，fold0/1缓存分别约114/181 GiB。维持fold1活动缓存，等待全程checkpoint后清理其缓存，再恢复fold0并顺序安排后续折。当前没有OOF预测、GHI头或收益结果；未提交新作业、未改变训练合同，test/buffer保持关闭。

2026-10-19 16:32 CST（折0恢复脚本加固，未提交）：fold1 214945日志实测epoch4 step4400，epoch4尚未结束；last.pt为已完成的epoch3 checkpoint。fold0旧PBS保持C，node22盘满是已验证原因。为避免恢复时重新写完整缓存并覆盖已有resume副本，新增`run_future_ghi_oof_fold0_resume_cache_reuse.pbs`：复用失败作业留下的warm cache，强制校验缓存存在、节点可用空间≥100 GiB、`resume_from_epoch001.pt`与原`last.pt` SHA一致；不改模型/数据/训练预算。服务器`bash -n`通过，脚本哈希83b310b6421791569d8b98343b0e5ec13a63507120ae4f78426db1e47771421b。全队列214221=Q、214945=R、214986=R（3/3），新脚本只准备、未qsub；仍须待fold1固定12轮完成并释放其缓存后再运行。当前无OOF/GHI结果。

2026-10-19 16:36 CST（折1持续训练）：现场队列复核仍为214221=Q、214945=R、214986=R（3/3），不提交新作业。fold1 214945推进到epoch4 step4600，GPU1采样约100%利用率、显存约66.9 GiB；最近完整checkpoint仍为epoch3，epoch4尚未完成。fold0因node22本地盘门限失败仍待fold1结束后复用缓存恢复；无OOF预测或GHI配对结果。保持固定12轮与既定验证合同，test/buffer及未来真实AGRI/CPP推理输入关闭。

2026-10-19 16:40 CST（折1继续训练）：服务器日志实测fold1 214945到epoch4 step4800，GPU1采样约100%、显存约66.9 GiB；PBS仍214221=Q、214945=R、214986=R（3/3）。epoch4未结束，最近完整checkpoint仍为epoch3。折0缓存复用脚本已审阅：要求既有warm cache及节点空闲空间≥100 GiB，且校验恢复权重SHA；保持未提交，待fold1完成、核实checkpoint并清理其自有缓存后执行。尚无OOF/GHI结果，不据训练损失判断收益；数据与测试隔离合同不变。

2026-10-19 16:50 CST（折1完成第4轮）：现场日志及epoch_004.json相互核验，fold1 214945到step5236，train_loss_mean=0.17108291796799194，heldout_payload_read=false；last.pt及epoch_004.json于16:50落盘，训练继续固定12轮。PBS全账户仍214221=Q、214945=R、214986=R（3/3）。fold1缓存仍被活动作业使用，不能清理；折0恢复继续等待该作业结束。尚无OOF预测或GHI配对结果。

2026-10-19 16:53 CST（折1进入第5轮）：现场训练日志到epoch5 step5300，epoch4 checkpoint仍有效；PBS为214221=Q、214945=R、214986=R（3/3）。暂无OOF/GHI评分。已将折1结束后的安全顺序交由Luna现场监控：验证12轮完成收据和终态、只清理折1专属缓存、复查全账户队列/磁盘/脚本哈希后再提交折0 warm-cache恢复；任一门禁不满足即等待。合同与test/buffer隔离不变。

2026-10-19 16:57 CST（折1训练进展）：服务器日志核实fold1 214945推进到epoch5 step5500，GPU1约100%采样、显存66.9GiB，作业仍R；last.pt仍是已完成的epoch4 checkpoint。全账户PBS 214221=Q、214945=R、214986=R（3/3），不提交新任务。暂无OOF预测和GHI评估；继续等待固定12轮训练完成后再进入折0恢复。

2026-10-19 17:00 CST（折1持续推进）：Luna现场监测并由PBS作业状态复核，fold1从step5500推进至epoch5 step5600（约83秒/100步），PID仍运行、CPU约137%、GPU1约100%/66.9GiB；数据读取计数持续增长。PBS队列214221=Q、214945=R、214986=R（3/3）。训练正常，不重启；node22临时盘仅25GiB余量，保持折0缓存与折1活动缓存不动。尚无OOF/GHI成绩，固定合同与测试隔离继续生效。

2026-10-19 17:04 CST（折1继续推进）：现场折1训练日志推进至epoch5 step5800，作业214945仍R；此前约1分半未出现100步标记，但随后step5800正常落盘，确认没有卡死。全账户214221=Q、214945=R、214986=R（3/3）。尚无OOF/GHI结果；固定训练合同、现有缓存和活动作业保持不变。

2026-10-19 17:07 CST（折1训练确认继续）：Luna对214945作短间隔检查，日志由step5800推进到5900；PID 138% CPU，数据读取计数增长，GPU1采样99%/66.9GiB。全队列仍214221=Q、214945=R、214986=R（3/3），临时盘99%使用。确认作业未停滞；不重启、不提交新任务。无OOF/GHI评估，保持合同。

2026-10-19 17:09 CST（折1训练推进）：Luna现场复核fold1从step5900推进至6000，PID仍运行、CPU约138%、GPU1采样100%/66.9GiB，磁盘余25GiB。PBS 214221=Q、214945=R、214986=R（3/3）。此轮无停滞或异常；尚无Oof预测和GHI头结果，保持训练不变。


2026-10-19 17:17 CST（现场训练推进）：PBS全账户为214221=Q、214945=R、214986=R（3/3），未提交新任务。fold1 214945训练日志由step6000推进到epoch5 step6300；step6200/6300 loss分别为0.13517/0.08485，qstat显示作业运行中且累计CPU时间继续增长。最新完整checkpoint仍为epoch4（step5236），故12轮尚未完成。fold0仍待warm-cache恢复；不得在fold1使用缓存时清理或重启。当前无OOF预测及配对GHI结果，未来COT增益尚无结论；test/buffer和未来真实AGRI/CPP推理输入关闭。

2026-10-19 17:20 CST（折1训练现场核验）：服务器时间17:20:15，PBS全账户214221=Q、214945=R、214986=R（3/3）。fold1 214945由epoch5 step6300推进至step6400（step6400 loss=0.12834）；qstat elapsed 03:53:39，节点GPU1采样99%/66883 MiB，训练继续。完整checkpoint仍为epoch4（step5236），未见epoch5完成收据。node22本地盘仍99%使用、可用约25 GiB；fold0缓存复用恢复暂不可启动。无OOF预测/验证/GHI头结果，不能判断未来COT收益；不提交新作业，数据隔离合同不变。

2026-10-19 17:42 CST（折1训练＋OOF噪声消融准备）：现场PBS仍为214221=Q、214945=R、214986=R（3/3）；fold1日志推进至epoch6 step7300，完整epoch checkpoint为epoch5（step6545），无训练终态。node22 /tmp仅约25GiB可用，fold0缓存恢复仍未满足启动门。趁训练运行，修正后续AC50收缩代码来源标记：训练manifest和审计现在读取pack的train_forecast_oof，并新增独立OOF入口与fail-closed门禁，要求同一五折OOF pack SHA、完成的A/AH/AC审计和测试排除标记；只有预注册AC−AH门未通过时允许执行，若已通过则拒绝AC50并进入独立确认。服务器swc环境合成见证（未来COT仅缩放、配对日期块审计、OOF门禁）全部通过，py_compile、bash -n及两套SHA校验通过；未提交消融作业，没有读取项目验证/测试载荷。当前仍无OOF GHI指标或增益结论。

2026-10-19 17:45 CST（折1训练复核）：服务器实测PBS仍为214221=Q、214945=R、214986=R（3/3）。fold1由step7300推进至epoch6 step7400，step7400 loss=0.13228；qstat elapsed 04:18:42、累计CPU继续增长。完整checkpoint仍为epoch5（step6545），没有fold1终态。node22本地盘约25GiB可用，fold0恢复仍等待fold1释放缓存；未提交作业。OOF预测与GHI结果尚未产生，COT收益仍待A/AH/AC日期块审计。
2026-10-19 17:52 CST（fold0节点盘故障诊断与缓存复用修复）：直接查PBS历史确认214946于16:07以exit_status=1结束；日志定位为fold0 train cache新增文件时触发
ode-local disk reserve below 25 GiB，并非模型/数据或GPU错误。epoch1/step1292恢复检查点与SHA不变。214933前序fold0作业日志确认同一M1 manifest（fold0 train_unique_frames=38,121），节点上对应缓存现存38,121个NPZ；已修改fold0恢复PBS入口使用该完整缓存，并新增对train_frames.txt逐路径集合、非零文件大小及无未完成.tmp的fail-closed检查，避免再扩张缓存。该门禁尚待远端语法及实际清单见证；未删除任何缓存、未提交训练。当前PBS 214221=Q、214945=R、214986=R（3/3）；fold1日志到epoch6 step7700、GPU1约97%/66.9GiB，训练仍进行，fold1缓存不能清理。没有OOF/GHI结果，未来COT收益尚无结论。

2026-10-19 17:58 CST（fold0缓存恢复门禁通过）：服务器swc环境通过恢复metadata见证；节点22直接用train_frames清单核验前序fold0缓存，38,121个预期训练帧NPZ逐路径完全匹配、全部非空、无.tmp，输出OOF_FOLD0_COMPLETE_TRAIN_CACHE_INVENTORY_PASS 38121。恢复PBS入口和helper已同步服务器，bash -n及Python编译通过。因为完整缓存不再增长，把本地临时盘运行/检查点保留线设为20 GiB（cache loader仍保留25 GiB防增长门，但复用缓存时不会写新NPZ）；当前约25 GiB可用，正式续跑前再次检查。PBS仍214221=Q、214945=R、214986=R（3/3），未提交fold0。fold1 214945已完成epoch6 step7854并启动epoch7，step8000 loss=0.13806；epoch6 checkpoint在17:53保存，GPU1 100%/约66.9GiB。fold0恢复点继续使用epoch1/step1292。无OOF或GHI收益结果，用户目标仍未完成。

2026-10-19 18:03 CST（折1进度确认）：PBS现场仍214221=Q、214945=R、214986=R（3/3）。fold1完成epoch6 step7854，epoch7从step7900推进至8200；15秒后复核时GPU1仍99%/66.9GiB，训练进程有活动。最新完整checkpoint为epoch6。fold0仍保留epoch1/step1292恢复点；复用前序同合同缓存的逐路径见证38,121/38,121通过，恢复PBS和helper已同步，OOF SHA清单、bash语法、metadata测试及缓存见证全通过。队列满，未提交作业、未清理活动缓存。尚无五折OOF预测或GHI结果。

2026-10-19 18:05 CST（fold1现场续查）：PBS仍214221=Q、214945=R、214986=R（3/3），fold1从step8200推进到step8300；GPU1采样100%/66.9GiB，job elapsed 04:38:31。epoch7尚未结束，最新完整checkpoint仍epoch6；不提交恢复fold0，不清理活动缓存。没有OOF预测或GHI头结果。

2026-10-19 18:08 CST（活动作业短间隔见证）：PBS 214945仍R（elapsed 04:41:41），日志从step8300推进至step8400；GPU1 pmon SM 79%、显存占用仍约66.9GiB，训练PID 18884存活、CPU 141%。全账户214221=Q、214945=R、214986=R（3/3）。fold1继续固定12轮，不判停、不重启；fold0待队列空位。

2026-10-19 18:10 CST（fold1短间隔复核）：作业214945仍R（elapsed 04:43:33），epoch7由step8400推进至8500；PID18884 CPU时间增长、仍存活。单次pmon采样SM为0%，但与前后100%采样和日志步进一致，属于瞬时波动，不判停。队列仍214221=Q、214945=R、214986=R（3/3），未提交fold0。

2026-10-19 18:11 CST（OOF调度并行化）：队列仍214221=Q、214945=R、214986=R（3/3）；fold1到epoch7 step8500、GPU1采样100%。fold0完整train cache已通过38,121逐路径清单审计且恢复只读命中缓存，故不再等待fold1结束或清缓存；将M3执行安排改为PBS有空位时在独立A800并行续跑fold0，保持每折随机种子、manifest、norm、output、固定12轮预算不变。当前无可用PBS名额，未提交；node22可用空间约25GiB，恢复脚本保留20GiB运行/检查点门槛。

2026-10-19 18:14 CST（折1进程确认）：作业214945 elapsed 04:47:28，epoch7由step8500推进到8600；30秒观察后PID18884仍存活且CPU时间继续增长，GPU1 pmon SM88%、显存约66.9GiB。PBS全账户仍214221=Q、214945=R、214986=R（3/3）；214221自10/15提交后仍排队，214986运行约2小时且尚有约6小时walltime。队列未释放，fold0不能提交；未触碰任何其他作业。

2026-10-19 18:18 CST（节点盘清理与训练续查）：fold1 214945已由step8600推进到step8800，GPU1 100%/66.9GiB；epoch7仍进行。PBS全账户214221=Q、214945=R、214986=R（3/3），未提交新作业。通过erify_redundant_oof_cache.py证明失败作业214946的部分缓存24,159帧为fold0完整缓存38,121帧逐路径匹配且文件尺寸相同的子集；确认没有进程引用后，仅删除该失败作业冗余缓存，并复核完整cache清单38,121通过。node22可用盘空间从25GiB升至139GiB（90%已用）；活动fold1缓存及完整fold0缓存未动。恢复/清理脚本均已记录并加入校验清单；无OOF/GHI结论。

2026-10-19 18:20 CST（训练短间隔确认）：fold1 214945由step8800推进至step8900；30秒后复核PID18884仍运行、CPU约142%，GPU1 pmon SM94%、显存约66.9GiB。PBS仍214221=Q、214945=R、214986=R（3/3），未获得fold0提交名额。冗余缓存已安全清理，node22可用139GiB；fold0完整cache逐路径见证仍通过。

2026-10-19 18:22 CST（折1短间隔现场见证）：fold1 214945由step8900推进至step9000；30秒复核PID18884仍运行、CPU约142%，GPU1 pmon SM36%/66.9GiB。PBS仍214221=Q、214945=R、214986=R（3/3）。node22可用139GiB，冗余部分fold0缓存已清理，完整缓存逐路径核验通过。fold0待PBS名额释放后并行续跑；无OOF/GHI结果。

2026-10-19 18:24 CST（折1训练推进）：fold1从step9000推进至9100；30秒后复核PID18884仍存活、CPU时间增长，GPU1 SM98%/66.9GiB。PBS仍214221=Q、214945=R、214986=R（3/3），未提交fold0。node22可用139GiB，fold0完整train cache校验通过；OOF和GHI结果仍未产出。

2026-10-19 18:26 CST（fold1 epoch完成）：fold1 214945完成epoch7、step9163，epoch_007 checkpoint于18:25保存，EPOCH_END报告train_loss_mean=0.145741、heldout_payload_read=false。短现场复核PID18884仍运行，GPU1 pmon SM84%/66.9GiB。PBS仍214221=Q、214945=R、214986=R（3/3），无fold0提交槽位。fold0完整cache与139GiB临时盘余量保持。

2026-10-19 18:28 CST（fold1进入epoch8）：PBS 214945仍R（elapsed 05:01:27），epoch8已开始并到step9200；35秒后PID18884仍运行，CPU约142%，GPU1 SM98%/66.9GiB。全账户214221=Q、214945=R、214986=R（3/3），无fold0名额。

2026-10-19 18:32 CST（fold1 epoch8推进）：fold1由step9300推进至step9400，PID18884仍运行，CPU约142%，GPU1采样SM59%/66.9GiB。再次查PBS全队列仍214221=Q、214945=R、214986=R（3/3）；node22可用139GiB。epoch8尚未完成，fold0没有PBS名额。

2026-10-19 18:33 CST（fold1短间隔推进）：fold1 epoch8从step9400推进至9500；PID18884仍活跃，CPU约143%，GPU1 pmon SM98%/66.9GiB。PBS全账户仍214221=Q、214945=R、214986=R（3/3），fold0无名额。

2026-10-19 18:35 CST（fold1训练步进）：epoch8 step9500→9600；30秒现场复核PID18884仍运行、CPU约143%，GPU1 pmon SM90%/66.9GiB。PBS 214221=Q、214945=R、214986=R（3/3），fold0继续等待名额；node22剩余139GiB。

2026-10-19 18:41 CST（现场复核）：直连服务器核对PBS仍为214221=Q、214945=R、214986=R（3/3），禁止提交fold0。fold1 `214945` 日志已从epoch8 step9600推进到step9800；epoch8尚未结束，最新完整checkpoint仍为epoch7（18:25保存），训练作业保持R。恢复fold0所需的38,121帧完整train cache及独立A800计划不变。当前五折OOF预测、COT bank和GHI A/AH/AC结果均未生成；没有任何未来COT收益结论。服务器登录节点无法直接观测计算节点GPU进程，本次只依据PBS、作业日志及共享盘checkpoint判断，不将其误写成实时GPU利用率。

2026-10-19 18:45 CST（fold0恢复提交并开始）：队列复核显示`214221=Q, 214945=R, 214986=C`，未结束任务2个；恢复脚本及清单SHA通过，epoch1/step1292恢复checkpoint文件齐全后提交fold0缓存复用作业`215005`（WORKERS=0）。约49秒后PBS为`214221=Q, 214945=R, 215005=R`（3/3），fold0日志已出现且前置依赖SHA检查通过；恢复训练前的缓存逐路径核验、checkpoint SHA门和首个resume step尚待日志确认。fold1同期epoch8已推进至step10000。此后不再qsub，直到空出PBS名额；无OOF/GHI收益结果，test/buffer输入保持关闭。

2026-10-19 18:48 CST（fold0恢复现场核验）：PBS仍`214221=Q, 214945=R, 215005=R`（3/3）。`215005`日志确认`OOF_FOLD0_COMPLETE_TRAIN_CACHE_INVENTORY_PASS 38121`、恢复预检及checkpoint SHA均通过，作业运行于node22 GPU0；计算节点PID5248已分配约10.7 GiB显存，20秒间隔内`/proc/5248/io`的rchar由约0.39 GB增至2.83 GB，说明仍在读取/装载数据，但尚未打印首个resume TRAIN_STEP，GPU采样0%。不据此判停或重启，继续观察首个训练步。fold1 GPU1同一时刻SM 0%为瞬时采样且日志尚在step10000，不构成停滞证据；两项作业保持运行。无OOF预测或GHI收益结论。

2026-10-19 18:51 CST（折0恢复训练已步进）：fold0 `215005`日志首次出现`RESUME_TRAIN_STEP epoch=2, step=1300, loss=0.24420`，证明已从已核验的epoch1/step1292 checkpoint继续优化；PID5248仍运行，GPU0分配约65.8 GiB。fold1 `214945`已推进至epoch8 step10200。PBS全账户`214221=Q, 214945=R, 215005=R`，3个未结束作业，无新提交名额。仍只是上游训练进度，五折OOF预测、COT输出及GHI A/AH/AC对照尚未产生，不能宣称未来COT有效。test/buffer及未来观测AGRI/CPP输入仍关闭。

2026-10-19 18:53 CST（当前作业用途核对）：PBS仍为`214221=Q, 214945=R, 215005=R`。`214945`是fold1/seed43的固定12轮SimVP训练；日志到epoch8/step10300。`215005`是fold0/seed42从epoch1/step1292 checkpoint恢复的固定12轮训练，已到epoch2/step1300；分配GPU0。两者当前都只生产OOF上游预测模型，还没生成五折OOF影像/COT，也没训练GHI头。`214221`并非本OOF流程的新提交，qstat显示它是10月15日提交的`monthly_R_station16_20261013_v1/interval_alignment_20261015/run_interval_tensor_pack_20261015.pbs`，用途为R区间对齐tensor pack。已结束的`214986`不计入当前队列。测试/缓冲载荷保持关闭。

2026-10-19 18:57 CST（GPU卡数核实）：PBS全账户为`214221=Q, 214945=R, 215005=R`。OOF作业`214945`请求并使用node22 GPU1（SimVP fold1），`215005`请求并使用node22 GPU0（恢复SimVP fold0）；计算节点`nvidia-smi`同时见PID18884与5248，分别占约66.9与65.8 GiB。GPU2另有用户`qliang`的`gmx mdrun`进程占用约0.6 GiB，不属于本项目；`214221`排队中，没有占GPU。因此本项目当前使用2张A800，node22上可见3张GPU有进程；PBS未结束作业3/3，暂不能再提交。

2026-10-19 19:01 CST（运行时间复核）：fold1 `214945`在epoch8耗时1,780.5秒后进入epoch9，日志到step10700；按最近完整epoch估算，剩余约3轮/1.5小时。fold0 `215005`仍在epoch2/step1300；从日志首个step1300至本次约10分钟尚未出现step1400，但计算节点PID5248仍运行、CPU时间及rchar持续增长、GPU0有约65.8 GiB显存分配，单次GPU利用率采样为0%。当前恢复使用WORKERS=0，存在数据读取成为速度瓶颈的可能，但尚未据此判死或重启；折0剩余时长无法可靠外推。全流程还需fold2–4、OOF预测/冻结R转换及三类GHI头比较，远未到GHI收益结论。队列仍214221=Q、214945=R、215005=R（3/3）。

2026-10-19 19:12 CST（OOF筛选耗时复估）：服务器队列仍`214221=Q, 214945=R, 215005=R`（3/3）。fold1完成epoch8（单轮1780.5秒），epoch9推进至step11100；按约30分钟/epoch，训练余量约1.5小时。fold0恢复任务从18:51的step1300推进至19:11的step1400，约20分钟/100步；PID5248仍在运行且CPU/数据读取计数增加，GPU0约65.8GiB已分配，单次GPU利用率采样为0%。若该速率持续，fold0剩余约14,000步可能需要约45–50小时；这是粗略线性外推，串行输入速度会波动。五折余下训练、OOF预测与冻结R转换、GHI头训练和配对bootstrap仍未完成，当前不能给出实验收益。需要评估并在不改模型/样本/seed的条件下解决fold0输入吞吐瓶颈；test/buffer关闭。
# 2026-10-19 19:29 CST（现场续查）

服务器直连复核：slfu全队列为`214221=Q, 214945=R, 215005=R`（3/3），没有新提交槽。`214945`折1已完成epoch9（step 11781，`train_loss_mean=0.137400`，`heldout_payload_read=false`），进入epoch10/step11800；完整轮次约31.6分钟。`215005`折0恢复仍到epoch2/step1500，恢复检查点epoch1/step1292的SHA与缓存清单门已通过，尚未完成epoch2检查点。node22两张本项目卡分别为GPU0折0、GPU1折1；现场GPU0显存约65.8GiB，GPU利用率短采样在0–30%间变化；GPU1约66.9GiB且采样91–100%。折0主进程单核CPU约92%，作业配置`workers=0`，显示输入读取可能限制吞吐，但25秒短间隔内没有新增百步记录，不能仅凭此判停。因恢复点仅在epoch1，贸然改worker并重启会丢弃本轮尚未落盘的epoch2进度；现保留作业，待下一个安全epoch检查点和完整吞吐窗口再决定是否调整运行时worker数。显式样本、seed、12轮、优化器及模型合同不变。折2–4、OOF预测、冻结R转COT和GHI A/AH/AC日期块对照均未完成；当前没有任何COT→GHI收益结论。test/buffer及未来真实AGRI/CPP推理输入保持关闭。

2026-10-19 19:41 CST（训练进度续查）：直连服务器确认全账户PBS仍为`214221=Q, 214945=R, 215005=R`（3/3）。折1 `214945`完成epoch9（step11781，train_loss_mean=0.137400，elapsed 1897.06秒），进入epoch10并到step12300；GPU1采样98%/约66.9GiB。折0恢复`215005`从step1500推进到step1600，elapsed约56分钟，PID5248 CPU约94%（接近单核），GPU0约65.8GiB已分配但此刻采样0%；workers=0，输入侧吞吐仍可能是瓶颈。恢复门此前已通过，最新完整检查点仍为epoch1/step1292；无step1700或epoch2完成文件。没有足够证据安全热切换worker，也不丢弃当前epoch；保留现有作业，到epoch边界再复核速度和检查点。全队列3/3，不提交新任务。五折OOF、冻结R预测COT及GHI头A/AH/AC配对评估尚未完成，暂无COT增益结论；test/buffer及未来真实AGRI/CPP推理输入关闭。

2026-10-19 19:56 CST（折0与折1现场续查）：直连服务器复核PBS仍为`214221=Q, 214945=R, 215005=R`（3/3）。折1 `214945`完成epoch9并到epoch10/step12900，最新完整checkpoint为epoch9（step11781）。折0 `215005`由step1600推进到step1800，仍在epoch2；最新完整恢复checkpoint为epoch1/step1292，epoch2尚未写入完整检查点。折0elapsed约71分钟，主进程CPU约95%、workers=0，GPU0约65.8GiB但本次SM为0%；折1GPU1约66.9GiB但本次SM也为0%，此前样本较高，故仅记瞬时状态。折0step1500至1800约27分钟，粗估剩余训练20–30小时，受共享I/O影响较大，不作承诺。继续等待epoch2原子检查点；到边界后再考虑是否在保持随机种子和样本序列不变的前提下调整DataLoader workers。没有新OOF/冻结R-COT/GHI配对产物或收益结果；test、buffer及未来真实AGRI/CPP推理输入关闭。

2026-10-19 20:11 CST（折1新检查点与折0续训）：PBS直查为`214221=Q, 214945=R, 215005=R`（3/3）。折1 `214945`已完整保存epoch10/step13090，train_loss_mean=0.134929，历时1901.94秒；当前epoch11到step13600，checkpoint已含epoch10。折0 `215005`由step1800推进至1900，仍在epoch2；epoch2未结束，最近完整恢复检查点仍为epoch1/step1292。node22 GPU0/1分别分配折0/折1，显存约65.8/66.9GiB；本次SM采样GPU0 100%、GPU1 0%，显示瞬时使用波动，不能据单次采样认定GPU空转。折0主进程约96%单核CPU，workers=0；该设置可能影响数据供给，但瓶颈尚未定论。按最近step区间粗估折0余时约20–30小时，波动较大。队列满，不提交新任务；到epoch2原子检查点再核验是否可安全优化加载。暂无折0完整12轮、其余三折、OOF AGRI/COT或GHI对照结果；test/buffer和未来真实AGRI/CPP推理输入保持关闭。

2026-10-19 20:26 CST（折0/折1续查）：PBS全账户仍为`214221=Q, 214945=R, 215005=R`（3/3）；未提交任何新任务。折1 `214945`正跑epoch11，到step14200；最新完整checkpoint为epoch10/step13090，heldout载荷标记false。折0 `215005`推进至epoch2/step2100，最新完整checkpoint仍epoch1/step1292；训练进程PID5248存活、CPU约97%，GPU0约65.8GiB，本次SM采样0%（较早样本曾100%），暂不归因为单一I/O问题。step1900→2100约15分钟，至完整12轮仍需较长时间。等epoch2原子检查点后再核验worker调整的顺序等价性；当前模型、样本、seed与固定12轮不变。仍无fold完整OOF预测、未来COT bank或GHI头A/AH/AC比较，不能得出COT增益；test/buffer与未来真实AGRI/CPP推理输入继续关闭。

2026-10-19 20:41 CST（折1进入最后一轮，折0续训）：PBS全账户`214221=Q, 214945=R, 215005=R`仍为3/3。折1 `214945`已进入固定12轮的epoch12/step14800；epoch11 checkpoint于20:30保存，step14399、train_loss_mean=0.133413、elapsed=1940.74秒。折0 `215005`推进到epoch2/step2200，恢复以来尚未生成epoch2完整checkpoint；最新安全点仍是epoch1/step1292。PID5248 CPU约97%，GPU0约65.8GiB、本次SM采样0%；瞬时SM采样此前在0–100%间波动，不据单点诊断根因。step2100→2200约15分钟。无新OOF/COT/GHI产物；队列满不提交。等epoch2 checkpoint后再做运行时worker调整的顺序等价性与风险判断，训练样本、seed和固定12轮不变；test/buffer及未来真实AGRI/CPP输入关闭。

2026-10-19 20:58 CST（折0疑似停顿核验）：全账户PBS仍`214221=Q, 214945=R, 215005=R`（3/3）。折1在最终epoch12推进到step15500；无终态收据，尚未完成。折0仍在epoch2/step2300，epoch2检查点尚未生成；短间隔检查时PID5248仍运行，CPU约97%，状态从Sl/futex等待转为Rl。30秒进程计数中rchar增加约8.16GB、syscr增加约88k，显示其持续读取本地缓存帧；GPU0显存约65.8GiB，单次SM为0%但不能代表完整利用率。没有证据表明进程死锁，继续保留，不重启、不清理缓存。待epoch2原子checkpoint后再评估worker调整；无OOF/GHI可复核结果，test/buffer及未来真实AGRI/CPP推理输入关闭。

2026-10-19 21:16 CST（折1完成、折2启动）：折1 PBS `214945`已结束。经swc环境重算，epoch12 checkpoint SHA与`training_complete.json`匹配（`13e120c5cbe2e937ceba6334ffee160c4df6d193c0256eb3ca339094b4c02616`），固定12轮/15,708步完成，`heldout_payload_read=false`、`test_payloads_read=0`。确认折1训练进程已退出后，清理了唯一对应的node-local临时缓存`/tmp/slfu_future_ghi_oof_fold1_214945_tc6000`，节点剩余空间从139GiB升至320GiB；折0完整缓存未动。
+检查PBS全账户2/3、折2输出目录为空、折2固定manifest存在及节点盘余量后，提交fold2/seed44/workers=6作业`215021.tc6000`。提交后队列为`214221=Q, 215005=R, 215021=R`（3/3）。至21:16日志已通过部署源码SHA、manifest-builder witness和train-only trainer witness，仍在折前语法/清单门，尚未开始profile/训练，GPU尚未分配给该进程。折0 `215005`仍step2400、epoch2中，epoch2 checkpoint尚未落盘；主进程保持活动。没有OOF预测或GHI收益结果；test/buffer及未来真实AGRI/CPP推理输入保持关闭。
2026-10-19 21:29 CST（折0/折2现场复核）：slfu全账户PBS为`214221=Q, 215005=R, 215021=R`（3/3）；214221是既有区间对齐打包作业，不属于本OOF训练，不提交新任务。折1 `214945`已完成12轮/15,708步，`training_complete.json`所载epoch12 checkpoint SHA复核一致，`heldout_payload_read=false`、`test_payloads_read=0`；仅在作业结束后删除其专属缓存。折0恢复 `215005`日志由step2400推进至step2500（epoch2），最新完整checkpoint仍epoch1/step1292；PID5248活跃，GPU0约65.8GiB，本次GPU利用率短采样0%，不能据单点判停。折2 `215021`已过源代码SHA、manifest、train-only与loader-order见证；归一化日志从9,000推进至12,000/39,270帧，仍未进入profile或训练，GPU1虽已分配但本次仍近空闲。node22 `/tmp`可用261GiB、使用率82%。保持折0当前进程至安全epoch checkpoint，再评估吞吐；不清理活动缓存、不改训练合同。暂无完整五折OOF AGRI/COT银行或GHI A/AH/AC评价结果，不宣称收益；test/buffer及未来真实AGRI/CPP推理输入保持关闭。
2026-10-19 21:56 CST（折0提速并行复核）：用户指出折0等待过久，故在不改变折/seed/样本/全局batch/12轮目标下，停止workers=0作业`215005`并从完整epoch2/step2584 checkpoint恢复。恢复点SHA=`1c8b0a94620587ae1d05cf9601c0e36933a020a6f8a88b176e078e1779175b6d`，payload元数据确认fold0/seed42且未读heldout/test。合成采样器比较workers0与6、固定`seed+epoch`的3轮顺序完全一致；训练dataset的`__getitem__`为确定性读取/归一化，无随机增强。新作业`215028`使用workers6，经源码SHA、完整cache inventory（38,121帧）、resume metadata及checkpoint SHA门后通过；step2600→2700约90秒，约22.2样本/秒，GPU0短采样SM约94%，显著快于workers=0整epoch约2.65样本/秒。折2 `215021`已完成归一化和profile（batch20，峰值60.70GiB，24.19 samples/s），固定训练已到epoch1/step600，GPU1短采样有活动。当前为两张A800并行跑两个独立折（非单折DDP），全账户PBS `214221=Q, 215021=R, 215028=R`，3/3；node22 `/tmp`可用135GiB，两个活动缓存禁止清理。暂无五折OOF/COT bank或GHI A/AH/AC结果，不能判断COT增益；test/buffer及未来真实AGRI/CPP推理输入保持关闭。
2026-10-19 21:59 CST（双卡训练复核）：全账户PBS为`214221=Q, 215021=R, 215028=R`（3/3）。折0恢复作业`215028`由step2700推进至step2800；折2作业`215021`由step600推进至step700。node22 GPU0/GPU1分别使用约65.7/65.6GiB，SM短采样97%/96%；两折都在独立A800上训练。`/tmp`可用135GiB、91%已用，较21:42检查稳定；两个活动缓存仍不可清理。折0worker6实测首个百步区间约90秒（约22.2样本/秒），相对workers0 epoch2的约2.65样本/秒显著加速。没有OOF预测/COT银行/GHI对照结果，也没有COT增益结论；test/buffer和未来真实AGRI/CPP推理载荷保持关闭。

## 2026-10-19 22:17 CST：OOF训练与噪声路径复核

Luna现场核验全账户PBS仅有`215021=R`（fold2）和`215028=R`（fold0），共2个未结束作业。fold0恢复训练到epoch3/step3500，最近完整检查点仍为epoch2/step2584；fold2完成epoch1/step1333（train_loss_mean=0.336207，heldout_payload_read=false）并进入epoch2/step1400，已保存epoch1检查点。两卡各约65.7–66.1GiB；15秒间隔的短采样显示GPU0 89→56%、GPU1 0→100%，即时利用率有波动但两作业均有日志推进。node22 `/tmp`剩余135GiB，活动缓存保持不动。fold1固定12轮已完成；fold0/2、fold3/4以及完整OOF AGRI预测、冻结R→未来COT bank和GHI A/AH/AC日期块评价尚未完成，因此目前没有未来COT下游收益结论。

同轮静态源码核查：fold训练路径没有对AGRI输入或目标显式注入加性噪声，也没有翻转/裁剪式数据增强；QA无效像元置零，AGRI使用train-only均值/标准差归一化。普通dropout为0，但SimVP训练态有DropPath随机正则，概率0.01递增至0.1；所以准确表述是“没有显式噪声注入”，不是“训练完全没有随机性”。原始AGRI→R的预测COT构造会将预测AGRI反归一化，再直接按R训练统计量标准化；该路径没有物理范围裁剪。它可能形成R输入分布偏移，但目前缺少本轮OOF样本的R输入范围/超训练分布比例，不能据此断言噪声已定位。fold0恢复记录说明模型随机流按`seed+resume_epoch`重置、样本顺序仍按`seed+epoch`；该折恢复后的DropPath轨迹与不中断训练不完全相同，作为恢复元数据限制记录，不改变样本、12轮和评价门。服务器源SHA及折2元数据核对结果由Luna于22:17回报；test/buffer载荷均未读取。

## 2026-10-19 22:31 CST（OOF bank依赖与备用节点容量核查）

只读核验 `qstat -u slfu`：当前队列为`215021=R`（fold2，elapsed 1:16:10）和`215028=R`（fold0恢复，elapsed 0:41:01），共2/3。fold0日志已进入epoch4/step4000；最近完整检查点为epoch3/step3876（约22:26写入）。fold2在epoch2/step1800，最近完整检查点为epoch1/step1333（约22:14写入）。node22 `/tmp`剩余约135GiB，两个活动缓存保留。GPU0/1分别分配给fold0/fold2，显存约70.8/66.1GiB；短采样利用率约96%/99%。GPU2/3分配给其他PBS任务；GPU4–7虽在nvidia-smi中近空闲，但没有本项目PBS分配，不视为可直接使用资源。

R输入偏移诊断仍被OOF依赖阻塞：现有`results/future_simvp_r_bank_v1/bank_manifest.json`虽标为COMPLETE且含预测AGRI train/val样本，但训练预测是in-sample（`in_sample_train_forecasts=true`）；其`progress.json`仍显示BUILDING，不能作为完整、独立OOF银行。`results/future_ghi_head_pack_v1/pack_manifest.json`是pilot head pack，同样不能替代五折OOF。fold0与fold2仍训练，fold3/4未训练，当前不存在完整`future_ghi_oof_bank`。冻结R的`normalization.json`只有train-only均值/标准差和计数，没有逐通道经验分位边界；因此无法按要求报告“相对R训练支持的超界率”。待五折OOF预测银行与训练R输入经验分布统计产物齐备后再做诊断；本次未读取真实未来AGRI、test或buffer载荷。

备用节点只查PBS状态、共享路径可见性和本地盘元数据：node1/5/15无GPU查询输出；node21/23有A100且`/tmp`各约1.3TiB可用，且共享`/public/home/.../SolarCOTIFS_20260907`路径可见，但两节点均有其他作业占用CPU slots，现场存在其他GPU进程或利用率，当前不属于可直接借用的空闲分配。若后续调度器批准新GPU allocation，node21/23的scratch容量可容纳单折缓存；必须通过PBS正式分配并复核资源，不能在未分配GPU上运行。未提交或干预任何作业。
## 2026-10-19 22:42 CST（fold3 node23启动）

按用户授权并通过提交前门禁后，提交fold3/seed45/workers=6单卡作业`215030.tc6000`，显式申请`node23:ppn=8:gpus=1`。提交前全账户队列仅`215021=R`与`215028=R`（2/3）；fold3 `manifest_audit.json`验证frame-disjoint、train/val帧交集0、test条数0、`validation_test_used=false`、跨BJT日期purge=1,967，fold3为27,521 train / 5,832 heldout / 5,271 OOF；训练输出目录及fold norm原先均不存在。node23 `/tmp`可用1,373,007,769,600字节，数据源共享路径可见。

PBS脚本仅做调度设备识别适配：fold3要求`node23-gpu/<id>`且写明assignment，其余fold仍要求node22；模型、数据、loss、seed规则、batch20、12轮预算、manifest均未改。更新后的脚本SHA-256为`e4ad590b19923694ff9132f3c3cc5627e226fce84910ac35b1e7f7cd3928e047`。远端源码`deployment.sha256`、`bash -n`、py_compile、manifest-builder witness、train-only witness和per-epoch loader-order witness均通过；日志复核`OOF_FOLD_PREFLIGHT_PASS 3`。提交后全账户为`215021=R, 215028=R, 215030=R`（3/3）；`qstat -f`确认新作业`exec_gpus=node23-gpu/2`且`Resource_List.nodes=node23:ppn=8:gpus=1`。

22:42:18日志已进入fold3 normalize阶段，worker数6、seed45，node23 GPU2此刻0MiB/0%（归一化尚未进入GPU profile，属阶段预期）。显存峰值和profile吞吐尚未产生，待profile后记录；fold3独立缓存为`/tmp/slfu_future_ghi_oof_fold3_215030_tc6000`，不触及fold0/2缓存。该PBS仅访问train数据与清单；heldout/test/buffer/未来实况AGRI仍不读取。
2026-10-19 22:44 CST增量：全账户仍`215021=R, 215028=R, 215030=R`（3/3）。fold0日志到epoch4/step4500（最近完整checkpoint仍epoch3）；fold2到epoch2/step2400（最近完整checkpoint仍epoch1）。fold3 `215030`在node23 GPU2上完成启动门并进入6-worker train-only normalization，日志从1,000推进到3,500/40,286 unique train frames；约85秒增加2,500帧（约29帧/s，仅短窗口估速，非训练吞吐）。GPU2仍0MiB/0%，此阶段尚未进入profile；node23 `/tmp`可用约1.3TiB。profile吞吐和峰值显存尚待实测。
2026-10-19 22:46 CST：PBS仍为`215021=R, 215028=R, 215030=R`（3/3），无fold4名额。fold3 node23 normalization推进至8,000/40,286帧（约20%）；从22:45:17的5,500到22:46:45的8,000，短窗口约27.8帧/s。GPU2仍0MiB/0%，尚未进入profile；node23 `/tmp`约1.3TiB可用。fold0到epoch4/step4600，fold2到epoch2/step2500。归一化仍在按预期推进，profile吞吐/显存待实测。
2026-10-19 22:50 CST：全账户队列继续为`215021=R, 215028=R, 215030=R`（3/3）。fold3 train-only归一化达到16,000/40,286帧（约40%），比22:49:30时的13,500增加2,500，短窗约33帧/s；GPU2仍0MiB/0%，尚未进入profile。node23 `/tmp`约1.2TiB可用。折2日志到epoch2/step2600，fold0最近日志为epoch4/step4700。没有fold4提交名额，继续等归一化和现有作业阶段变化。
2026-10-19 22:52 CST：队列`215021=R, 215028=R, 215030=R`仍3/3。fold3 normalization达到20,000/40,286帧（约50%），GPU2仍未进入profile，node23 `/tmp`约1.2TiB可用。fold0推进到epoch4/step4800。fold2完成epoch2/step2666，`train_loss_mean=0.2067571868484215`且`heldout_payload_read=false`，随后进入epoch3/step2700；仍无队列空位可启动fold4。
2026-10-19 23:02 CST阶段变化：fold3 `215030`完成train-only归一化`40,286/40,286`，日志标记`NORMALIZATION_DONE`，正在进入profile；归一化约20分钟，未读取heldout/test载荷。阶段交接采样GPU2仍0MiB/0%，profile数值尚无。node23 `/tmp`约1.1TiB可用。fold0 `215028`完成epoch4/step5168（`train_loss_mean=0.17035663505041562`，`heldout_payload_read=false`），进入下一训练阶段；fold2 `215021`到epoch3/step3000。全账户队列仍3/3，无fold4名额。
## 2026-10-19 23:04 CST：fold3 profile显示batch合同偏离

`215030.tc6000`已在node23获得PBS分配`node23-gpu/2`；卡为NVIDIA A100-SXM4-40GB（40,536MiB）。23:03:48 profile报告：batch4吞吐20.196 samples/s、峰值12.261GiB；batch8 21.522 samples/s、峰值24.370GiB；batch12 OOM；reload最大差0；`test_payloads_read=0`。profile自动选择batch8。源码确认train读取`profile.json.selected_batch`，23:04:52的`training/run_metadata.json`也明确记录`batch=8`、seed45、12轮；因此该作业实际合同不符合请求的batch20，必须隔离为硬件profile/非正式fold，不能纳入正式OOF科学对照。heldout/test/buffer/未来实况载荷标记仍false/未读取。训练进程已进入`--mode train`，截至23:04:52尚无TRAIN_STEP或epoch checkpoint记录。全队列仍`215021=R,215028=R,215030=R`（3/3）；fold0完成epoch4/step5168，fold2在epoch3/step3100附近推进。

不停止当前作业：项目指令禁止停止现有PBS任务；同时fold3 run目录及run_metadata已创建，执行器会拒绝原路径覆盖，终止会留下缓存与部分产物，不能安全原路径重提。首选保持A800 80GB资源并在正式调度分配后复跑batch20（已有A800 profile约60.70GiB峰值）；替代方案需另行批准、实现和验证双卡/梯度累积的global batch20及optimizer-update、sampler和随机正则等价性，不能直接采用本次batch8。node23单张40GB A100无法满足batch20，batch12已OOM。
2026-10-19 23:06 CST续查：fold3 `215030`已开始`epoch=1, step=100`；`run_metadata.json`记录batch8、seed45、12轮、heldout/test flags false。node23 GPU2为A100 40GB，采样显存27,764/40,536MiB、SM 100%。该fold只作batch8硬件/profile训练记录，不并入正式fold3 OOF对照。全账户PBS仍3/3；fold0 epoch5/step5300，fold2 epoch3/step3200。

可用GPU调度主机限PBS GPU queue ACL所列node21/22/23：node21与node23为A100 40GB，`/tmp`分别约1.3TiB/1.1TiB；node22为A800 80GB，当前仅135GiB可用。node22 fold0、fold2活动缓存约180GiB和185GiB，不能清理。待其中一折PBS终态且缓存确认专属后，按授权的终态缓存治理流程释放该缓存，预计node22可用空间增加至约315GiB以上，达到>260GiB预留门；之后在全账户未结束作业<3时申请node22一张A800、batch20的正式折。node21/23单卡40GB不能满足batch20（node23 batch12已OOM）；双A100保持全局batch20需新DDP/梯度累积实现及optimizer-update、采样器、DropPath等价性验证，不作当前批量替代。

未撤销215030：当前用户明确禁止停止已有作业，且该作业已开始step100、run_metadata已写入；不应把运行中的checkpoint/缓存/路径作为可无损覆盖状态。保持完成运行并隔离为非正式batch8实验；其fold3输出不得覆盖或进入正式OOF银行。队列3/3，无fold4新提交。
## 2026-10-19 23:11 CST：batch20等效性与现场资源复核

只读代码审查确认：正式OOF trainer每个DataLoader batch执行一次`zero_grad → forward/loss → backward → clip_grad_norm_(1.0) → optimizer.step`（`train_monthly_simvp_oof_20261019.py`）；其shuffle由`seed+epoch`控制。SimVP每个MetaBlock含两个BatchNorm2d；训练态DropPath按样本生成随机mask，概率从0.01递增至0.1。故单卡micro-batch累积即使总样本数为20，也会改变BatchNorm统计、每批DropPath随机路径及optimizer更新/梯度裁剪位置，不能视为当前batch20合同的严格等价替代。DDP需SyncBatchNorm并额外验证全局样本顺序/尾批、有效像元归一化loss、裁剪/更新数、学习率步数及按样本随机正则；现有`train_triad_ddp.py`的通用DDP结构并非此OOF trainer的直接替代。建议优先等待node22单卡A800与scratch门槛，不在当前合同下自行切换梯度累积或DDP。

23:10 CST现场：全账户`qstat -u slfu`为`215021=R`(fold2,node22 GPU1)、`215028=R`(fold0,node22 GPU0)、`215030=R`(fold3,node23 GPU2)，3/3。node22 `/tmp`可用135GiB；活动缓存保持不动。node23 `/tmp`可用约1.1TiB。215030的运行元数据仍明确`batch=8, seed=45, epochs=12, heldout=false, test_payloads_read=0`；GPU2约27.8GiB/40.5GiB、100%利用率。此attempt继续隔离为非正式profile，不并入fold3 OOF。test/heldout/buffer载荷未读取，未干预作业。

## 2026-10-19 23:17 CST：输入随机增广与DropPath静态审计（仅候选，不改合同）

只读检查当前冻结源快照：`hunan_data.py`从manifest读取帧，载入AGRI和几何辅助量，按train-only均值/标准差标准化，并把无效QA/非有限像元置零；代码中未见随机噪声、翻转或裁剪等随机增广。`triad_models.py`构造SimVP时显式`drop=0.0`、`drop_path_prob=0.1`；`simvp.py`每个MetaBlock含两个BatchNorm2d，DropPath按样本抽取随机mask，模型各层概率由0.01递增到0.1。因此可以提出下一步单因素DropPath消融候选，但这只是静态实现事实，不能证明噪声或DropPath造成现有收益损失，也不能改变当前固定OOF基线。只有正式固定OOF对照完成后，才可另立同seed、同fold、同manifest、同训练预算的单因素DropPath对照；不在本轮改动或重训。

源码SHA-256：`hunan_data.py`=`f4b4aab55101bde575f474d5daa4a184466c16399b401c2ac67bf666c5f28b91`；`triad_models.py`=`9273447d357bad92bcf3cd976b65fb4c440187da456fb56d5f36ded349c681eb`；`simvp.py`=`2e0d280f7962a77f0c9b7e4cc817fed21d142c1436a12eb858e3234485313d16`。未读取test/heldout/buffer载荷。

## 2026-10-19 23:27 CST：fold2检查点推进

新鲜`qstat -u slfu`仍为215021=R（fold2、node22 GPU1）、215028=R（fold0恢复、node22 GPU0）、215030=R（node23 batch8隔离attempt），共3/3。fold2新增`epoch_003.json`与`last.pt`（23:26写入），epoch=3、step=3999、train_loss_mean=0.1823841327911736、heldout_payload_read=false。fold0最近完整结果仍epoch4/step5168（23:00）；作业与GPU进程仍在运行。node22 `/tmp`可用135GiB，专属缓存fold0约180GiB、fold2约185GiB，均保留；formal batch20重跑的>=260GiB scratch及队列空位门槛均未通过。fold3 batch8保持隔离，不计入正式OOF。未读取test/heldout/buffer载荷。

## 2026-10-19 23:29 CST：慢epoch活动复核与fold3 profile进度

经23:28:07与23:29:22间隔约75秒的只读复核，fold0 `215028`最近完整checkpoint仍为epoch4/step5168（23:00写入），但该折主PID61574及6个worker持续存在；worker CPU约77–94%，A800 GPU0两次采样显存约70.8GiB且末次利用率100%。因此当前证据指向epoch仍在执行/计算，不能只凭checkpoint未更新判作挂起；不干预作业。fold2 `215021`已于23:26完成epoch3/step3999（loss=0.1823841327911736，heldout_payload_read=false），主进程和workers持续活动，GPU1分配显存约66.1GiB。非正式fold3 `215030`于23:28新增epoch1/step3441（train_loss_mean=0.2925220619658681），run metadata仍为batch8、seed45、12轮、heldout=false、test_payloads_read=0；node23 GPU2显存约27.8GiB、利用率100%。队列仍3/3。node22 `/tmp`仍135GiB，fold0/fold2活动缓存约180/185GiB，保留不动；未满足新正式batch20 attempt的scratch/slot门禁。测试、heldout、buffer载荷未读取。

## 2026-10-19 23:35 CST：fold0检查点推进

现场`qstat -u slfu`仍为215021=R、215028=R、215030=R（3/3）。fold0恢复作业215028新增epoch5/step6460（23:35写入，train_loss_mean=0.15993890381957354，heldout_payload_read=false；resume from epoch2）。fold2仍epoch3/step3999（loss=0.1823841327911736，heldout=false）；隔离的fold3 215030仍epoch1/step3441（loss=0.2925220619658681，metadata batch8、test_payloads_read=0）。node22 `/tmp`可用135GiB，活动缓存fold0约180GiB、fold2约185GiB，均保留；无队列空位，正式fold3 batch20门槛不满足。无test/heldout/buffer载荷读取。

## 2026-10-19 23:39 CST：fold2 epoch边界只读诊断

针对fold2 `215021`的step3999从23:27起未更新，23:36:19至23:38:51采样显示job仍R，exec GPU为node22-gpu/1；主PID50548持续存在，6个worker连续高CPU（约74–100%）。两名worker在23:37:52至23:38:51间`/proc/<pid>/io`的rchar分别增加约14.6GB、15.3GB；GPU1显存约66.1GiB，利用率在不同采样间0–100%波动。由epoch3完整checkpoint后作业/worker仍活跃、数据读取持续增长，当前证据更符合epoch4训练中的DataLoader/CPU读取和批次计算，不支持认定挂起。最新持久化epoch仍为3/step3999；PBS `Output_Path=/dev/null`，`TRAIN_STEP` print没有持久日志可查，故无法直接报告epoch4首个100-step标记；不推断未观测步数。未终止或修改任务，未读test/heldout/buffer载荷。

## 2026-10-19 23:44 CST：fold2 epoch4持续活动复查

延续fold2 epoch边界诊断：23:44:37时`215021=R`，epoch3/step3999仍是最后durable marker（23:26）；主PID50548和六个worker持续存在，worker CPU约69–100%。两名跟踪worker自23:37:52起rchar约增加100.8GB和104.0GB；allocated GPU1显存约66.1GiB，最新利用率采样66%。读数与计算资源持续活动，支持epoch4仍在进行，不构成停滞证据。未能读取首个step日志：PBS stdout路径为`/dev/null`，训练step打印只写stdout；因此不报告未观测的step。队列仍3/3，node22 `/tmp`可用135GiB，缓存均保留。无干预及test/heldout/buffer载荷访问。

## 2026-10-19 23:52 CST：fold0 epoch6活动复核

针对fold0恢复作业`215028`最近durable epoch5/step6460（23:35）未变的问题，23:51:14至23:52:36复查显示作业仍R、allocated node22 GPU0；主PID61574与6个worker持续存在，worker CPU约70–100%。三个采样worker的rchar约增加18–19GB/worker；GPU0显存约70.8GiB，利用率从0%波动至98%。一名worker首采处于rwsem D-state，后续恢复R；其余worker保持活动。因此证据符合epoch6输入读取/训练计算仍在推进，未确认挂起。PBS stdout=/dev/null，无可读持久step日志；最近保存epoch仍为5。全队列3/3，node22 `/tmp`135GiB；fold0/fold2缓存保留。未干预及test/heldout/buffer载荷访问。

## 2026-10-19 23:54 CST：隔离batch8 fold3至epoch2

`215030`于23:52写出epoch2/step6882（train_loss_mean=0.18869523024059284、heldout_payload_read=false）。运行元数据仍为seed45、batch8、12轮、test_payloads_read=0；PBS仍R，分配node23-gpu/2；GPU2采样约27.8/40.5GiB、100%，node23 `/tmp`约1.1TiB。此run继续只作非正式batch8硬件/训练profile，不能纳入正式fold3 OOF。全账户队列仍215021/215028/215030均R（3/3）；node22 `/tmp`135GiB、正式fold0/fold2活动缓存保留。fold0最近epoch5/step6460，fold2最近epoch3/step3999；无终态/队列空位/scratch门槛，不清理、不重提。未访问test/heldout/buffer载荷。

## 2026-10-19 23:56 CST：跨日队列与scratch复核

集群时间仍为2026-10-19 23:56 CST（本地日期已跨至10月20日）。全账户队列仍215021=R（fold2,node22 GPU1）、215028=R（fold0恢复,node22 GPU0）、215030=R（隔离batch8 fold3,node23 GPU2），3/3；三作业walltime分别约2:39、2:04、1:14。最新durable产物不变：fold0 epoch5/step6460（23:35），fold2 epoch3/step3999（23:26），batch8 fold3 epoch2/step6882（23:52）。node22 `/tmp`135GiB，fold0/fold2专属活动缓存约180/185GiB；node23 `/tmp`约1.1TiB、隔离fold3缓存约190GiB。无PBS槽位、node22 scratch未达到260GiB；所有正式作业仍运行，故不清理任何缓存、不提交正式batch20 attempt。无test/heldout/buffer载荷读取。

## 2026-10-19 23:57 CST：PBS日志路径核实与step更正

更正此前“stdout=/dev/null，因此没有持久step日志”的监控表述。两个PBS脚本虽设置`#PBS -o /dev/null`，随后均执行`exec > >(tee -a "$LOG") 2>&1`，因此存在持久日志：`logs/future_ghi_oof_fold2_215021.tc6000.log`和`logs/future_ghi_oof_fold0_resume_215028.tc6000.log`。fold2脚本SHA-256=`e4ad590b19923694ff9132f3c3cc5627e226fce84910ac35b1e7f7cd3928e047`；fold0恢复脚本SHA-256=`8045ef6976af525c8f75cb33242b7453ec3c6dc94652369671a2ac0d14e79d4c`。因此可直接读取日志尾部，而不应称step不可观测。

23:57:17 CST新鲜全账户PBS仍215021=R、215028=R、215030=R（3/3）。日志：fold2最新`TRAIN_STEP`为epoch4/step5100（23:56:44日志mtime）；fold0最新`RESUME_TRAIN_STEP`为epoch6/step7200（23:55:02日志mtime）；隔离batch8 fold3最新epoch3/step7500（23:57:03日志mtime）。fold3日志仍对应seed45、batch8非正式run，不并入正式OOF。node22 `/tmp`135GiB，fold0/fold2活动缓存保留；node23约1.1TiB。无终态、队列空位或正式A800 scratch门槛，不清理、不提交。未访问test/heldout/buffer载荷。

## 2026-10-19 23:59 CST：五折正式训练状态与剩余时间估计

只读检查fold0–4训练目录、training metadata/receipts、checkpoint文件名与活动PBS日志尾部（未加载权重、未读取heldout/test数据）。fold1是已完成的正式batch20模型：metadata为seed43、12 epochs、batch20；`training_complete.json` state=`COMPLETE_OOF_SIMVP_FIXED_EPOCH_TRAINING`、steps=15,708、selected_checkpoint=`epoch_012.pt`、heldout=false/test_payloads_read=0；epoch_012.pt存在且SHA-256与receipt一致=`13e120c5cbe2e937ceba6334ffee160c4df6d193c0256eb3ca339094b4c02616`；日志含`OOF_FOLD_FIXED_EPOCH_COMPLETE_PASS 1`。这使fold1 checkpoint可用于其固定manifest的OOF预测生成；全五折银行仍未完成。

当前活动step来自tee日志而非推测：fold0 `215028=R`，epoch6/step7300（最新resume log mtime 23:57:39），正式batch20、seed42；最近已完成epoch5耗时2057.8s≈34.3min，按已观测1292 updates/epoch估算，epoch6约剩12min，随后6轮约206min，剩余总时长约3h38m。fold2 `215021=R`，epoch4/step5100（log mtime 23:56:44），正式batch20、seed44；最近epoch3耗时2153.8s≈35.9min，按1333 updates/epoch估算，当前轮约剩6min，后续8轮约287min，总余约4h53m。两项仅按最近完整epoch耗时线性估算，实际数据加载/调度会波动。隔离的`215030=R`fold3 seed45 batch8（非正式），日志epoch3/step7800、mtime 23:57:03；其最近完整epoch2耗时1434.1s≈23.9min，当前epoch约剩18min、其余9轮约215min，但该attempt不得并入OOF。

fold0、fold2仍无终态；fold3正式batch20重跑未启动，fold4目录尚不存在。全队列仍3/3。node22 `/tmp`可用135GiB，fold0/fold2专属活动缓存约180/185GiB，保持不动；待正式作业之一终态并经receipt/checkpoint门核验后，只清理该终态作业的专属缓存，复核scratch达到>=260GiB且qstat<3，方可在node22 A800启动唯一、独立输出路径的batch20 fold3 attempt。fold1验证资产可支持OOF预测，但不等同于完整OOF银行。未执行清理/qsub，未读test/heldout/buffer payload。

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

## 2026-10-20 01:15 CST：fold1 OOF 预测完成并通过独立核验

PBS `215032.tc6000` 已完成。fold1 OOF 预测银行覆盖 6,430/6,430 条序列；独立检查通过：输出键与预先冻结的 `fold_oof_ids.txt` 顺序完全一致且无重复；两数组 shape/dtype 为 AGRI `[6430,2,16,13,3,3]`、COT `[6430,2,16,1,3,3]`，均为有限 FP32；预测数组、index、epoch-12 checkpoint、fold-train normalization 的 SHA-256 均与 manifest 一致。fold 模型来源为固定12轮日期留出 SimVP；COT由冻结R作用于预测AGRI和未来几何得到。`future_observed_agri_read=false`、`cpp_read=false`、`validation_test_payload_opened=false`、`buffer_payload_opened=false`。bank manifest SHA-256：`9544e786eb8ef7f56fd1e12747c21a47345f45e7f391cb5fc4484a5232be929b`。

已清理该已完成预测作业自己的 node22 临时缓存（约35 GiB）；未触碰fold0/2活动训练缓存。当前 PBS 为 `215021` fold2 与 `215028` fold0 恢复，2/3；node22 `/tmp` 约135 GiB可用，仍低于下一正式折要求的260 GiB，故暂不提交新折。fold0日志到epoch8/step10200，fold2到epoch6/step7900，二者均未完成12轮。现有单折完整性通过不构成GHI收益结果；待五折OOF预测银行齐全后，继续同键A/AH/AC配对与日期块bootstrap，若预设AC−AH门未通过再运行OOF版AC50单因素收缩。

## 2026-10-20 01:38 CST：两卡DDP启动正式fold3

为在不改全局batch=20、12轮、seed45、fold3日期清单和每轮样本顺序的前提下用node23两张A100训练，新增独立runner `train_monthly_simvp_oof_ddp_20261020.py` 与PBS `run_future_ghi_oof_fold3_ddp2g.pbs`，输出写入新目录 `results/future_ghi_oof_training_v1/fold_3_ddp2g_v1`，不覆盖此前隔离的batch8试跑和正在运行的fold0/2。rank各取全局排列中连续10个样本；全局最后不足两样本时复制尾样本使两rank共同参与，仍以全局有效像元数归一化损失；DDP内使用SyncBatchNorm以保持原batch20的BN统计。

本地与服务器swc环境的语法和CPU语义见证通过：分布式全局样本顺序与既有单卡DataLoader完全一致，pooled-valid-pixel梯度与单批梯度一致。PBS `215033`曾因runner缺少底层dataset所需`--root`参数，在读样本前失败，exit status=1；没有创建训练输出，立即补参后以新作业`215034`重跑。`215034.tc6000`提交时全账户为2/3，当前分配node23 GPU1/2、1个PBS任务；`/tmp`约1.1 TiB可用，复用已完成fold3归一化生成的node-local原始帧cache。

`215034`已通过fold3 train-only manifest/norm门（27,521 train、5,832 heldout、test=0）、两卡SyncBatchNorm全局batch20输出/梯度/运行统计对照见证。5步真实batch20 profile稳态吞吐41.18样本/s、单卡峰值显存30.46 GiB，满足≥20样本/s和<34 GiB门；随后开始正式固定12轮训练。01:38现场日志到epoch1 step200，两个rank在node23 A100上工作；node22 GPU0/1分别继续正式fold0/2。尚无fold3完整checkpoint、OOF预测或GHI结果；test/buffer和未来真实AGRI/CPP读取仍关闭。下一步以首个完整epoch的实测耗时及checkpoint复核DDP稳定性，再续跑至epoch12并生成fold3 OOF预测。

01:40 CST增量核验：`215034`继续到epoch1 step300，PBS仍R；node23 GPU1/2两卡各约36 GiB显存、SM约97%，node22 fold0/2也各自有新step日志。训练脚本SHA-256=`df4b51274cdb4b5eedd59af5bf7a7ff73b3a6b9aa60b74224eac3a7d821d593b`，PBS SHA-256=`2bc0b2adb61ebe205f5708c4b4dc2ddc9f62677da129c8f48deb595ef66ffa72`；本地和服务器文件哈希一致。Profile吞吐是短窗口测值，等完整epoch后再用实际速度修订剩余时间，不把profile冒充正式吞吐。

## 2026-10-20 01:56 CST：多卡随机流审计与后续fold4准备

现场 `qstat -u slfu` 为3/3：`215028` fold0恢复、`215021` fold2、`215034` fold3双卡DDP。最新持久训练日志分别为 fold0 epoch10/step11800、fold2 epoch8/step9500、fold3 epoch1/step1300；fold3还没有首个完整epoch checkpoint。现阶段只证明训练仍推进，不代表未来COT的GHI增益。

实现审计发现 fold3当前DDP两个rank在模型构建前后使用相同CUDA seed，模型含最大drop_path=0.1的逐样本随机掩码；由相同调用和局部batch形状推断对应rank位置可能有重复掩码。该风险不改global batch、样本顺序、loss或SyncBN，但使随机正则不能称为与单卡逐样本随机流完全等价。静态证据及边界见 `DDP_DROPPATH_RNG_AUDIT_20261020.md`。不中断当前正式fold3。

为后续折准备的rank-separated CUDA RNG runner、fold4双卡PBS和两项测试已同步服务器；fold4 split审计为27,237 train、6,116 heldout、0 test，train-only norm尚不存在，新PBS会在节点本地创建独立cache及norm。服务器swc已通过Python编译、PBS shell语法和CPU样本顺序/pooled-loss/RNG-seed语义测试；跨GPU随机掩码见证待双卡PBS执行。新作业因队列仍3/3未提交，等槽位自然释放后再依次核验scratch和分配卡。

当前唯一完整OOF bank仍为fold1的6,430条。fold0/2/3模型完成后继续生成各自OOF预测，随后才运行全五折同键GHI头A/AH/AC日期块配对；若AC−AH未过预注册门，按既定计划执行OOF版AC50未来COT幅度收缩。未读取test、buffer、未来真实AGRI或CPP载荷。

02:00 CST增量：`215034`完成首轮并保存epoch1/step1377，训练loss均值0.3542621，耗时1317.35秒；随后进入epoch2并到step1500。按首轮耗时外推，余下11轮约4小时，但只作粗略预算，后续轮时与缓存/调度有关。fold0继续到epoch10/step11900，fold2到epoch8/step9600；队列仍3/3。首轮checkpoint证明DDP训练可持续推进，尚不是COT/GHI效果证据。

## 2026-10-20 02:07 CST：多卡训练与活动折现场复核

实时PBS为3/3：fold0 `215028`（node22，gpus=1）训练到epoch10/step12200，最近完整checkpoint为epoch9；fold2 `215021`（node22，gpus=1）到epoch8/step9900，最近完整checkpoint为epoch7；fold3 `215034`（node23，gpus=2）已完成epoch1/step1377（耗时1317.35秒）并推进epoch2/step1900。故目前本轮共分配4张GPU。两卡fold3正在实质推进，未出现训练错误；全量OOF预测和A/AH/AC GHI配对结果仍未完成。

只读评估确认当前DDP runner不支持resume：从单卡fold0/2检查点切换需要先实现并验证optimizer/SyncBatchNorm恢复、rank独立DropPath随机流及checkpoint语义；直接qdel重跑会损失当前未完成epoch，并有活动输出/本地缓存冲突风险。因此保持已推进折继续运行。后续fold4双卡rank-separated RNG方案已准备，但PBS槽满时不提交；空出槽后重新核对队列、双卡分配和node23临时盘门禁再启动。测试、buffer、未来真实AGRI/CPP推理载荷继续关闭。

02:10 CST增量轮询：指定PBS进程仍为R，三项任务保持3/3，无槽位可提交后续作业。fold3 DDP日志从step2000推进到step2100；fold0仍在epoch10/step12300附近，fold2仍在epoch8/step10000附近，未产生新完整epoch checkpoint。此间隔内fold0/2没有步数刷新，但作业仍在运行；不以短时无日志判停或重启。OOF预测/GHI配对阶段尚未开始，测试与未来真实AGRI/CPP载荷保持关闭。

02:12 CST进度复核：PBS仍3/3。fold0日志由step12300前进至epoch10/step12400；fold3双卡由step2100前进至epoch2/step2200；fold2最近日志仍为epoch8/step10100，约90秒无刷新。GPU瞬时采样node22 GPU0=100%、GPU1=0%（显存仍约66GiB），node23分配的GPU1/2分别约79%/97%。短采样不足以判断fold2停滞，继续只读检查其进程与I/O；不重启、不提交新作业，PBS队列满。

02:14 CST复核：fold0由step12400进一步到step12500（epoch10），fold2由step10200到step10300（epoch8），fold3双卡由step2200到step2300（epoch2）；三项PBS仍R且3/3。fold2之前约90秒没有百步日志，随后出现新step，确认是在正常批次推进间隔内，并非挂起。当前训练继续，未生成终态/新OOF bank；不提交新作业，待自然释放槽位后按已准备的双卡fold4方案和OOF队列顺序推进。

## 2026-10-20 02:28 CST：fold2从单卡安全检查点迁移至双卡

用户明确要求可用多卡并提升速度。本轮开始时，正式fold0/2各占node22一张A800，fold3使用node23两张A800；账户3/3，合计4卡。没有把新任务简单排队叠加，而是先对fold2启用已保存epoch检查点续训：fold2单卡训练在完成epoch8后保存 `last.pt`（step=10,664，SHA-256 `856f8ebb12a09b1c6cfa1cf6bd21d15ff36b72dec6c1a603c873c5c8259bda3e`，test/buffer读取标记为0），随后结束该单卡PBS 215021并以不同输出目录双卡恢复，旧输出和原检查点保留。

新增恢复runner严格校验fold=2/seed=44、相同manifest与fold-train归一化、模型/数据代码哈希、12轮计划和global batch=20；模型权重以strict模式载入SyncBatchNorm，AdamW状态载入。逐样本顺序仍按原全局batch20排列，双rank各处理10个样本，pooled有效像元损失与单卡batch梯度合同一致。恢复会重置CUDA随机流并采用rank独立DropPath随机数，故记录为不同随机实现轨迹；它不是逐bit复制原单卡训练轨迹。兼容审计通过，但需由实际两卡GPU parity/profile和后续训练确认恢复可运行。

CPU语义与manifest/训练边界测试通过；epoch8真实检查点只读审计显示212组模型参数及AdamW状态、SyncBN缓冲均兼容，0数据载荷读取。新的PBS 215035已在node22分配两卡（GPU1、GPU4），当前全账户未结束任务仍为3个：fold0 215028单卡、fold3 215034双卡、fold2 215035双卡。PBS prologue已通过DDP全局batch20梯度见证及rank RNG独立性测试；profile与epoch9首个训练步尚待日志确认。单卡同期并未停止：fold0已完成epoch10/step12920后继续固定12轮。

该资源调整只为缩短固定筛选任务时间，不改折分、样本、归一化、输入、损失、全局batch、12轮计划或评价门。它本身不构成AGRI、COT或GHI效果。仍无完整五折OOF bank和A/AH/AC GHI结论；测试/缓冲及未来真实AGRI/CPP推理载荷保持关闭。

02:29 CST双卡恢复实测：PBS `215035` 的预设两卡profile门通过，5步稳态实测 **40.51 samples/s**，每张卡最大显存分配 **30.46 GiB**，全局batch仍20（每rank 10），BF16与SyncBatchNorm；相同全局样本排列及pooled有效像元梯度见证通过。已从epoch8/step10664继续到epoch9/step10700；实时GPU采样node22 GPU1/GPU4显存各约34.0 GiB、利用率92%/97%。按双卡global batch20吞吐粗估，fold2剩余4轮约35分钟计算时间（首轮启动与epoch差异会使实际时间变化），尚未取得首个双卡完整轮次checkpoint。profile只用于资源速度评估，训练效果仍须跑完固定12轮及OOF/GHI门。

02:31 CST噪声收缩后续路线复核：OOF版AC50代码、PBS脚本和fail-closed gate在服务器`swc`重新校验SHA-256，并通过合成收缩、配对bootstrap、OOF门禁三项测试。它只把未来标准化COT异常缩至0.5倍、保留历史COT，其余GHI头/标签/样本/训练预算固定；仅当五折OOF完整AC−AH未通过预设收益门时才允许启动。当前尚无OOF A/AH/AC主审计，因此AC50仍未提交，不先验假设收缩会有效。该门通过后会直接针对“预测COT幅度噪声抵消有用信号”给出可检验对照；若也失败，则按提前写好的规则基于实际逐时效/分层结果定位下一单因素，而不盲调系数。

02:31现场队列：215028 fold0单卡R，215034 fold3双卡R，215035 fold2双卡R，3个未结束PBS。fold2双卡已到epoch9/step10800，profile门与GPU等价性见证通过；尚未完成epoch9。fold0日志近期为epoch11/step13100，fold3为epoch3/step3400。fold1 OOF bank完整，其余折训练/OOF银行仍待完成；当前没有可用于AC−AH判断的五折GHI结果。

02:33 CST增量推进：OOF AC50噪声收缩训练、独立审计和fail-closed门禁在服务器环境的SHA、编译、三项合成测试均通过；路径已准备，但须等待五折A/AH/AC OOF门失败才运行。fold4双卡DDP也完成提交前准备复核：SHA和PBS语法/CPU语义见证通过，split为27,237 train、6,116 validation、0 test，train/heldout帧交集0，norm/output路径均符合新建条件；node23 `/tmp`现场可用约1.1 TiB。由于账户仍有三个运行作业，当前不提交fold4；待槽释放后重新查全账户和节点，再进入队列（预计先等待node23 fold3释放双卡资源）。

02:32现场状态：215028 fold0单卡epoch11/step13100附近；215034 fold3双卡epoch3/step3400；215035 fold2双卡epoch9/step10900。fold2 2卡profile实测40.51样本/秒，GPU1/GPU4采样均有训练显存和活动。仅fold1 OOF银行完成，仍无五折汇总与GHI增益结果。接下来优先等各活动折完成固定12轮，并在槽位释放时提交准备好的fold4；随后按冻结manifest生成全部OOF AGRI→冻结R→COT预测，再按日期块比较A/AH/AC。测试/buffer及未来观测AGRI/CPP输入继续关闭。

02:34 CST活跃进度复核：PBS全账户仍3/3（215028 fold0、215034 fold3、215035 fold2）。fold0单卡前进至epoch11/step13200；fold3双卡前进至epoch3/step3600；fold2双卡前进至epoch9/step11000，DDP两卡保持运行。fold2/3尚未产出新的完整epoch checkpoint；fold0尚余约两轮。node23 `/tmp`可用1.1 TiB，fold4双卡脚本及manifest前置审计、source checksums和训练语义测试通过，但当前满队列时未提交。没有五折OOF预测或GHI结果；测试/缓冲和未来真实AGRI/CPP推理仍关闭。

## 2026-10-20 02:38 CST：推进状态与OOF预测器路径修复

实时全账户PBS仍为3/3：`215028` fold0单卡（node22），`215034` fold3双卡（node23），`215035` fold2双卡恢复（node22）；fold0至epoch11/step13400，fold3至epoch3/step3900，fold2恢复训练至epoch9/step11400。node22 `/tmp`约135 GiB、node23约1.1 TiB可用。未提交第四个任务，仍只有fold1的6,430条OOF bank完成，尚无五折GHI对照结果。

期间核实到OOF预测PBS原先把所有fold固定映射到标准训练目录；但fold2、fold3、fold4的双卡产物分别写入独立的`fold_2_ddp2g_resume_v1`、`fold_3_ddp2g_v1`、`fold_4_ddp2g_rankrng_v1`目录。已将预测PBS改成按fold显式读取对应run目录，输出bank仍写入规范fold目录，并增加训练fold与manifest fold一致性断言。变更同步服务器；PBS `bash -n`、全包`sha256sum -c`、预测脚本编译及因果路径见证均通过。该修复只让训练完成后的OOF流水线可正确找到双卡checkpoint，不代表预测或GHI收益已产生。队列释放后继续双卡fold4，再并行生成剩余OOF bank并进入A/AH/AC同键GHI评估；若主门失败，执行已预注册的未来COT 0.5幅度收缩对照。

02:43 CST增量：fold3 `215034` 已完成epoch3/step4131（train loss均值0.1828854，未读held-out载荷），保存epoch3 checkpoint；fold0 `215028` 到epoch11/step13500，fold2 `215035` 双卡恢复到epoch9/step11600。PBS仍3/3，当前不提交新任务。预测器目录映射的服务器副本及SHA已核验一致，整包checksum、bash语法、Python编译和因果输入测试通过。

02:44 CST复核：PBS仍3/3，fold0到epoch11/step13600（完整checkpoint仍epoch10），fold2双卡到epoch9/step11700，fold3双卡已进入epoch4/step4200。队列满时没有安全的第四个PBS槽位，fold4不排空队列、不终止现有任务；node23临时盘约1.1 TiB，折4双卡门禁已就绪。下一次检查槽位若释放，先重查完整账户队列和节点分配，再提交fold4。

## 2026-10-20 02:53 CST：fold1预测COT时序变化诊断

为定位“未来COT是否因预测噪声抵消有效信息”，对fold1的6,430个训练日期OOF键做同键时序描述性审计。预测COT bank与历史COT bank使用相同冻结R（SHA-256 `342596094d1a41aa8b141972a7b4adadece179b4b9f58170f4e7f409a894f001`）；只读取预测COT和两个历史观测时次的COT，不读取未来真实AGRI、CPP、GHI、test或buffer载荷。审计代码为`audit_fold1_cot_temporal_screen_20261020.py`（SHA-256 `5af13879389455a61962783be5d6971bc49b13271a4dcabc52ee69b6a535a6b9`），结果JSON为`results/fold1_cot_temporal_screen_20261020.json`（SHA-256 `1bc04da486828f651604c68f221186c01a7c44d60e1b6fe65de68158c904814a`）。

| 站点 | 预测未来相邻15分钟变化/历史末两帧变化（平均绝对值比） | 历史末两帧 `|ΔCOT|>20` | 预测首个15分钟 `|ΔCOT|>20` | 预测全时效相邻变化 `|ΔCOT|>20` |
|---|---:|---:|---:|---:|
| 四里 | 0.655 | 5.29% | 8.66% | 2.21% |
| 竺家 | 0.707 | 4.49% | 6.76% | 2.26% |

初步解释：未来预测并非单纯“每一步都噪声过大”。首个预测步的大幅变化比例高于历史末两帧，随后全时效的相邻变化比例明显降低，符合“起始跃变偏多、之后动态偏弱”的混合迹象。该诊断没有真值COT对照，窗口像元也不是独立样本，因此只能指出需检查的噪声形态，不能判定预测误差、GHI因果作用或站点收益。下一项主证据仍是固定五折、同键A/AH/AC日期块OOF GHI对照；仅当预设AC−AH主门未通过，才运行预注册的未来COT异常0.5倍收缩AC50，保留历史COT及其余训练合同。队列仍3/3：`215028` fold0单卡、`215034` fold3双卡、`215035` fold2双卡；fold4尚未提交，等待名额释放后按双卡方案推进。

02:57 CST只读进度复核：PBS仍3/3，fold0 `215028` 前进至epoch11/step14000（最近完整checkpoint仍epoch10）；fold2双卡恢复 `215035` 前进至epoch10/step12600（最近完整checkpoint epoch9）；fold3双卡 `215034` 前进至epoch4/step5000（最近完整checkpoint epoch3）。三个作业的步数均较前次增加，未见错误或完成标记。Luna对活动卡的瞬时采样为node22 GPU0/1及node23 GPU1/2高利用率，node22 GPU4一次采样为0%但仍保有任务显存；单次采样不作为空闲或停滞证据。当前实际分配为5卡，仍无PBS名额；fold4不提交，待任一任务退出后再查全账户队列与目标节点资源。

03:02 CST训练里程碑：fold0 `215028`完成epoch11/step14212，`train_loss_mean=0.1330124`，保存`epoch_011.json`和`last.pt`，随后进入固定第12轮；`heldout_payload_read=false`。fold2双卡恢复`215035`到epoch10/step12900，fold3双卡`215034`到epoch4/step5300，二者继续运行。全账户仍3/3，无错误或新槽位。训练loss和checkpoint只证明训练推进，不构成未来COT或GHI收益。

03:03 CST空间纹理补充审计：同一fold1、同一冻结R、6,430个相同序列键（222个BJT初始化日），计算站点3×3 COT patch的12条四邻边平均绝对差，并对初始化日期作10,000次成组bootstrap。相比历史末帧，预测首个时效的物理COT空间纹理比值为四里0.569（95% CI 0.540–0.601）、竺家0.582（0.551–0.615）；16个预测时效均值比值分别为0.358（0.329–0.390）、0.380（0.348–0.416）。在log1p(COT)尺度，首时效比值为0.724/0.705，16时效均值为0.512/0.505，日期块区间也均低于1。脚本`audit_fold1_cot_spatial_screen_20261020.py` SHA-256 `0feab5b05dc3fab6c718fc0b9b1ff62523a81487414cae3547841a0964f730b3`；日期块JSON `results/fold1_cot_spatial_screen_20261020_dateblock.json` SHA-256 `eaeb5b18804b623875dd2788e9fdb3101af42b3f1a054e8f1ae57694a0f1073d`。这表明预测链的3×3 COT局地空间对比显著低于历史输入，但参考是历史而非未来真值，天气演变也会改变纹理；不能据此宣称COT预测误差或GHI受损。它将后续诊断重点从“只降噪”拓宽为检查SimVP外推后的局地纹理衰减。仍先完成固定五折同键GHI A/AH/AC主筛选；若forecast-COT无增益，按`FUTURE_GHI_GATE.md`的隔离oracle条款单独比较真实未来AGRI→同一冻结R的COT信息上限，区分外推云图信息损失与GHI融合限制，绝不把oracle输入纳入可部署预测。

03:06 CST训练进度：fold3双卡作业`215034`完成epoch4/step5508，train_loss_mean=0.16906494，保存epoch4 checkpoint，heldout载荷读取标记false；fold0 `215028`进入最终epoch12、约step14300，fold2双卡恢复`215035`到epoch10、约step13200。PBS仍3/3，无错误或新槽位。接下来关注fold0固定12轮完成与队列实际释放，再提交预备好的fold4双卡任务。

03:09 CST增量：fold2双卡恢复`215035`完成epoch10/step13330（train_loss_mean=0.13332819，heldout_payload_read=false），保存epoch10 checkpoint并进入epoch11。fold3双卡`215034`进入epoch5、step5700；fold0单卡`215028`在epoch12推进至step14400。Luna复核fold0进程与I/O计数继续增长，10秒GPU采样利用率93%→98%，当前是活跃训练而非卡死。PBS仍3/3；没有任务完成/槽位释放，暂不提交fold4。

03:15 CST现场更新：全账户PBS仍3/3，`215028` fold0单卡到epoch12/step14600，`215035` fold2双卡到epoch11/step13800，`215034` fold3双卡到epoch5/step6100；日志持续前进，三作业仍运行，尚无epoch12/COMPLETE标记，也无GHI结果。当前分配总计5卡（1+2+2）；GPU查询显示node22 GPU5–7与node23 GPU3–7未分配，但节点空卡不能绕过PBS三作业上限直接接入已有进程。fold4双卡脚本已就绪；fold0结束释放名额后按同一固定12轮/global batch20合同提交。既有硬件profile显示单卡24.23、双卡40.51 samples/s；这是profile吞吐，不是对整轮训练时长的严格配对比较。

03:23 CST活性复核：PBS仍3/3且未有折完成；fold0从step14700前进到14900，fold2到step14400，fold3到step6600。fold0暂停约两分钟未打印新step后又恢复，现场进程仍在运行且CPU/I/O活动，判断为慢速数据读取而非作业终止；保留原作业，不重启。当前尚无epoch12完成标记、OOF汇总或GHI收益。fold4保持双卡计划，待全账户名额释放后按既有脚本提交。

03:29 CST复核：全账户仍3/3；fold0第12轮step15100，fold2进入第12轮step14800，fold3第6轮step7000，日志均有前进。fold0余约400步，checkpoint/终态尚未出现；所有结果仍是训练状态而非GHI收益。fold4脚本仍按双卡和global batch20固定合同运行，不在已有任务中途改world size。

03:42 CST阶段更新：fold0单卡任务`215028`已正常结束（exit_status=0），固定12轮/step15504；选定epoch12 checkpoint SHA-256 `2a85d63f660404631526d7300417dcd0ea316021bb47c33848ad23e2c38e5f90`，heldout/test载荷读取均为0。释放名额后提交fold4双卡任务`215036`，node23、2 GPUs、36 h，固定global_batch=20及12轮；队列现有fold2 `215035`、fold3 `215034`、fold4 `215036`，PBS为3/3。fold4已进入R态，启动日志刚创建尚未出现GPU/训练步证据；待确认双卡preflight和首个step后才记为正式训练启动。目前仍无OOF预测或GHI收益结论。

03:50 CST推进记录：fold2正式SimVP训练`215035`在固定12轮step15996后正常完成，epoch12 checkpoint SHA-256 `181e12c2fb3fb5e4d73628aaddb8436d19223505176e1fea67e8bedb7eb8c36b`，heldout/test载荷均未读取。验证fold2 OOF manifest、training_complete收据及唯一干净输出目录后，利用PBS名额并行提交fold2预测银行`215037`（node22单卡）；当前log已生成、作业R，预测启动门仍在执行。完整账户队列为fold3 `215034`、fold4双卡训练`215036`、fold2预测`215037`（3/3）。fold4仍在CPU train-only normalization，约5500/40301 unique frames，DDP首步尚未出现；fold3继续双卡训练。所有阶段仍无GHI增益结论，待五折OOF bank完整后进入同样本A/AH/AC头部评价。

03:55 CST增量复核：fold2 OOF预测银行任务`215037`已通过`OOF_PREDICT_PREFLIGHT_PASS`，只用fold2固定epoch12 checkpoint（SHA-256 `181e12c2fb3fb5e4d73628aaddb8436d19223505176e1fea67e8bedb7eb8c36b`）、其训练折外日期序列及历史AGRI/未来几何；进度256/6214，状态BUILDING，未来观测AGRI/CPP/test/buffer均关闭。fold0现已独立核验可预测：manifest帧交集为0、test=0、checkpoint SHA-256 `2a85d63f660404631526d7300417dcd0ea316021bb47c33848ad23e2c38e5f90`，唯一输出路径干净，待槽位释放后启动预测。fold4双卡任务`215036`的CPU归一化到14000/40301，尚未开始GPU训练；fold3双卡`215034`已到epoch7/step8400。PBS仍3/3。当前没有合并OOF银行或GHI头结果，不将训练/预测进度表述为收益。

03:54 CST只读复核：PBS仍3/3，fold3 `215034`双卡到epoch7/step8500（epoch6完成、heldout_payload_read=false）；fold4 `215036`双卡任务的CPU归一化到17000/40301，尚未进GPU profile/训练；fold2 OOF预测`215037`经完整preflight后到512/6214，bank仍BUILDING，future_observed_agri_read=false、cpp_read=false。fold0预测输入和干净输出目录已预审完毕，等待预测槽位；fold1 bank已完整。当前无GHI头评价结果，仍须完成五折bank后才能检验COT是否贡献GHI收益。

03:56 CST现场进度：fold2 OOF预测银行`215037`到816/6214（13.1%），仍BUILDING；完整账户队列为fold3训练`215034`、fold4训练`215036`、fold2预测`215037`，未结束3/3。fold3双卡到epoch7/step8700；fold4 train-only CPU归一化到20000/40301（约49.6%），GPU训练首步尚未出现。fold0预测所需epoch12权重、零帧交集manifest和清洁输出目录均已通过独立预检。下一次PBS释放后，若fold2预测完成优先提交fold0预测；不读取test/buffer及未来真实AGRI/CPP，当前仍无GHI收益结论。

04:00 CST现场核验：fold2 OOF bank `215037`到1328/6214，progress仍BUILDING且future_observed_agri_read/cpp_read均false；fold3双卡`215034`到epoch7/step8900；fold4双卡`215036`的CPU train-only normalization到26000/40301，尚未出现profile/训练step。全账户仍3个未结束PBS任务，无安全槽位可提交fold0预测；不重启/干预现存任务。fold0 manifest/checkpoint/clean output已预审，等待fold2预测完成后立即接续。没有OOF合并及GHI收益结论。

04:02 CST作业活性复核：fold3 `215034`训练step从8900推进至epoch7/step9000；两rank进程存活、workers约满CPU，12秒内两个worker分别增加约4.33/4.68 GB读量，GPU1/2分别约35.9/36.3 GiB且利用率97%/99%，判为正常进行中的epoch，而非卡死。fold2预测`215037`到1504/6214，仍BUILDING且future_observed_agri/cpp标记false；fold4 `215036`归一化到28000/40301，尚无首个DDP step，GPU3/4空闲符合CPU归一化阶段。PBS三任务全R、3/3，无可提交名额。当前无完整bank或GHI收益。

04:05 CST验证等待更新：fold2 OOF预测`215037`进度1856/6214；fold4归一化`215036`到32000/40301。fold3日志末步仍9100，但现场rank PID52289/52290仍存活、各约128–130% CPU，node23 GPU1/2显存35.9/36.3 GiB、利用99%/94%；按活动证据继续保留原任务，不因短时step日志间隔重启。PBS仍3/3，无折0预测槽位。没有OOF合并/GHI头评价结果。

04:12 CST fold4双卡profile放行：归一化40301/40301及SyncBN global-batch20 parity、独立DropPath rank RNG见证均通过；DDP profile测得world_size=2、global_batch=20、每rank10、warmup后单步约0.482–0.495 s、global throughput 41.06 samples/s、每rank峰值30.46 GiB，`DDP_PROFILE_GATE_PASS`。node23 GPU3/4随后采样约96%利用、显存35.9/36.0 GiB，说明进入实算；尚待首个正式训练step。fold3刚完成epoch7/step9639（train_loss_mean=0.1439315，heldout false）；fold2 bank到2832/6214、仍BUILDING。PBS仍3/3，尚无GHI评价或收益。

04:14 CST正式训练启动：fold4 `215036`完成归一化、DDP SyncBN/global-batch20与rank独立DropPath见证，并通过双卡profile（41.06 samples/s，peak 30.46 GiB/rank）；正式训练已记录epoch1/step100，global_batch=20。fold2预测`215037`到3072/6214，halfway且bank状态BUILDING；fold3 `215034`进入epoch8/step9700。PBS仍3/3，无slot可启动fold0预测。以上仅代表上游训练/OOF生成进度，不是GHI收益。

04:24 CST多卡预测路径预备：现有fold2预测器`215037`仍按原单卡配置运行，本轮不终止、不改变其输入或输出。为下一折新增两卡推理PBS脚本`run_future_ghi_oof_predict_2g.pbs`，SimVP与冻结R均采用推理态`DataParallel`，按相同manifest键顺序写入唯一fold输出；记录并验证PBS实际分配的两张卡，工作盘低于260 GiB即退出；输出合同记录`parallel_gpus=2`及每批吞吐。fold2当前进度4496/6214，状态BUILDING，未来AGRI/CPP读取均false；fold3 epoch8 step10300，fold4 epoch1 step700。多卡脚本已同步服务器，SHA256包校验、bash语法、Python编译及因果路径见证通过。全账户PBS仍3/3，尚未做目标节点双卡速度/一致性实测，因此不声称已提速；等待fold2结束释放名额后，用独立fold0目录做正式两卡预测，再核验输出与因果标记。

04:43 CST多卡预测进入实跑：fold2单卡预测`215037`正常完成，6,214条双站/16时效OOF输出的清单、形状、隔离标记及数组/index哈希均通过复核，未来观测AGRI、CPP、test与buffer均未读取。队列核验后将fold0预测`215038`提交到node23，PBS实分GPU5/6两张，checkpoint与fold输出预检通过，工作盘约900 GiB可用；当前fold0推进480/6,954，日志batch速率约2.5–3.2序列/秒，两卡均有Python进程但短时GPU采样偏低。fold2端到端约2.19序列/秒；因fold0/2样本规模和运行阶段不同，这只是早期粗比，不算可靠提速结论，继续以整折耗时和一致性验收判断。当前PBS 3/3，fold3/4双卡训练继续；fold0输出仍BUILDING、未来AGRI/CPP标记false，尚无GHI收益结果。

04:48 CST fold0多卡预测进度：`215038`从480推进至1,344/6,954，约2.5–3.2序列/秒；GPU5/6均有本作业Python进程，pmon短样本利用率约0–12%，说明数据读取/CPU或间歇推理可能限制加速，当前不能宣称多卡显著提速。bank仍BUILDING，future_observed_agri_read=false、cpp_read=false；全账户3/3（fold3/4训练、fold0预测），不提交其他任务。fold0剩余预计约半小时，待终态后核验manifest、数组哈希和配对键，再按完整五折计划接入后续评价。

## 2026-10-20 04:51 CST：多卡现场复核

用户确认有卡时应采用多卡。实时全账户PBS为3/3：fold3训练`215034`双卡，fold4训练`215036`双卡，fold0 OOF预测`215038`双卡（node23 GPU5/6）；当前项目实际分配6张GPU。`215038`进度从04:50:25的1,696/6,954推进到04:51:04的1,808/6,954，状态BUILDING，近期约2.4–3.0序列/秒；`future_observed_agri_read=false`、`cpp_read=false`。抽样时GPU5/6利用率为0%，但两卡各有该Python进程占用约8.0/6.8 GiB显存，且预测步持续推进；表明进程采用双卡分配但利用率有批间波动，现有证据不足以认定计算饱和或因空卡而未并行。fold3、fold4仍各自占据一张PBS作业槽且训练推进；账户最大3个未结束作业，因此没有可安全新增的PBS作业槽。节点物理空卡不属于当前PBS分配，不能直接接入运行中作业。后续保持固定合同，当前两折训练及OOF预测均按双卡运行；每个任务终态释放槽位后重查全账户，再并行接续已完成checkpoint对应的OOF bank。尚无五折GHI结果或收益结论。

## 2026-10-20 04:59 CST：fold0双卡OOF进度

只读现场核验（Luna，PBS作业日志与节点状态）：`215038` fold0双卡OOF预测到3,136/6,954（45.1%），累计约2.65序列/s，近两分钟约3.1序列/s；日志持续推进，预测数组及进度文件正常写入。future observed AGRI/CPP仍关闭。PBS仍3/3：fold3 `215034`已完成epoch9（step 12,393），fold4 `215036`在epoch2/step2,700，fold0预测运行约19分41秒。node23 `/tmp`约883 GiB可用。fold0双卡速度相较fold2单卡呈现约21%较高的当前平均吞吐，但折大小和端到端阶段不同，最终整折耗时未出，暂不作严格提速结论。当前没有可提交PBS槽位或五折GHI结果；待现有任务释放槽位后继续预测与同键A/AH/AC评价。

## 2026-10-20 05:02 CST：fold0双卡OOF过半

服务器只读复核：fold0预测作业`215038`仍运行，进度3,728/6,954（53.6%），elapsed 23:17；约2.5–3.1序列/s的批次日志持续出现，progress状态BUILDING，未来观测AGRI/CPP标记均false，数组继续更新，尚无manifest/COMPLETE收据。全账户PBS仍3/3：fold3训练`215034`、fold4训练`215036`、fold0预测`215038`；本轮无空槽。node23临时盘可用818 GiB。训练/预测任务均未中断，待fold0终态后独立校验bank，再按槽位与fold checkpoint状态接续后续步骤。无五折GHI评分或收益结论。

## 2026-10-20 05:06 CST：fold0 OOF继续推进

服务器实时核验：`215038`仍为R，fold0预测进度4,368/6,954（62.8%），elapsed 27:06；状态BUILDING，future observed AGRI/CPP读取标记均false。全账户PBS仍3/3：fold3 `215034`、fold4 `215036`及fold0预测`215038`均运行，没有可提交槽位。输出尚无完成manifest/receipt，因此不开始合并或评分。待终态校验forecast/COT数组、样本键、完整性哈希与因果标记后再放行后续GHI头评价。

## 2026-10-20 05:10 CST：fold0双卡OOF超过七成

服务器现场复核：`215038`在00:31:13 elapsed时完成5,024/6,954条（72.2%），状态BUILDING；未来观测AGRI和CPP读取标志为false。全账户PBS仍3/3（fold3训练`215034`、fold4训练`215036`、fold0预测`215038`），没有作业终态或空槽。输出只有进行中的AGRI/COT数组与progress文件，完成manifest/receipt尚未出现，故尚不合并、不评分。待预测完成后逐项核验完整序列数、键顺序、数组与索引哈希、冻结R来源和因果隔离标志，再进入同键GHI评价。

## 2026-10-20 05:29 CST：fold0双卡预测银行完成并通过逐样本审计

PBS作业`215038`已于05:22以`exit_status=0`结束，双卡预测用时43分01秒。fold0 bank含6,954条、两站、16个时效；`index.csv`的6,954个`seq_id`与冻结fold0 ID清单逐项、同序完全一致且无重复。独立重算的`forecast_agri_raw.npy`、`forecast_cot_log1p.npy`和`index.csv` SHA-256均与完成manifest一致；数组形状为`[6954,2,16,13,3,3]`与`[6954,2,16,1,3,3]`。manifest确认未来真实AGRI、CPP、validation/test和buffer payload均未读取。fold0现可进入五折合并，但需等待fold3/4预测银行。

05:28现场PBS为2/3：fold3 `215034`在node23 GPU1/2双卡训练，日志到epoch11/step14100；fold4 `215036`在node23 GPU3/4双卡训练，日志到epoch4/step4400。两作业日志均更新、训练进程和逐卡显存占用存在；GPU瞬时利用率有采样波动，未发现停滞证据。fold0/1/2银行现已齐全，fold3/4固定12轮训练尚未完成；因此尚无可提交的有效fold3/4预测，也不能提前训练五折GHI头。释放的PBS槽位待任一模型达到固定epoch12并通过checkpoint/manifest预检后，立即用于对应双卡OOF预测。当前没有GHI收益结果。

## 2026-10-20 05:38 CST：未来GHI头多卡实现profile

为避免最终A/AH/AC头部阶段再次盲目使用单卡，提交纯合成双卡profile `215040`（node23 GPU5/6，exit_status=0，运行41秒；不读取项目样本/标签）。第一次尝试`215039`因漏配正式头训练使用的cuDNN `LD_PRELOAD`，在CUDA库加载前退出；补齐相同环境预加载后重跑成功，未生成实验评分或改动模型资产。

现有`IntervalHead`使用实际合同batch=1792、原前向/损失/AdamW。24步合成测量中，单卡146.51 updates/s（262,548 samples/s），DataParallel双卡17.22 updates/s（30,857 samples/s），双卡仅为单卡0.118倍；单/双卡前向输出最大差为0，最后loss同为0.0166147。该短合成profile不是实际训练耗时或科学结果，但显示此小型头用DataParallel会显著变慢。因此不把头部盲目改成DataParallel；后续若头部成为耗时瓶颈，优先将彼此独立且使用同样本顺序/共同初始化的A、AH、AC方法臂分卡并行，再用逐样本配对审计确认结果。此profile只用于执行路径选择。

当前全账户PBS仍2/3；正式fold3 `215034`双卡到epoch11/step14700，fold4 `215036`双卡到epoch4/step5000；两作业日志仍增长。OOF银行fold0/1/2已完成，fold3/4待固定12轮训练后各自双卡预测。所有五折银行齐全之前不训练或评分GHI头，当前没有预测收益结论。

## 2026-10-20 05:42 CST：前三折OOF银行共同复核

逐折重算fold0/1/2的预测AGRI数组、预测COT数组、样本索引SHA-256，并逐项比较`index.csv`中的`seq_id`顺序与冻结`fold_oof_ids.txt`。三折全部通过，样本行数分别为6,954、6,430、6,214；均无重复键，数组哈希与各自完成manifest一致；未来真实AGRI、CPP、validation/test及buffer载荷标记均为false。五折总覆盖合同为30,436条训练期折外序列，当前已通过键/哈希审计的三折银行覆盖19,598条；fold3/4仍在固定12轮训练，未提前合并或评分。

05:43 CST增量轮询：fold3 `215034`由step14700推进到step15000，仍在epoch11；fold4 `215036`由step5100推进到step5300，仍在epoch4。两个两卡任务保持R且日志增长；账户队列2/3。双卡头部synthetic profile `215040`正常结束（exit_status=0），仅作为执行性能诊断，不是项目样本评分。下一步仍是等待fold3完成固定12轮并通过checkpoint门后启动其双卡OOF预测。

05:46 CST轮次里程碑：fold3 `215034`固定训练完成epoch11/step15147，`train_loss_mean=0.13175837`，heldout payload标记false；fold4 `215036`完成epoch4/step5448，`train_loss_mean=0.17229630`，heldout payload标记false。PBS仍2/3，两个训练作业继续运行。fold3第12轮尚未出现完成收据，故暂不提交其预测。模型训练日志仅是运行状态，不是未来COT或GHI收益证据。

## 2026-10-20 06:08 CST：多卡训练实时核验

用户再次确认可使用多卡。实时全账户PBS为2/3：fold3正式训练`215034`和fold4正式训练`215036`仍分别申请`gpus=2`，均使用DDP `world_size=2`，固定global batch=20，合计4张GPU参与两折训练；`qstat -n`显示两任务均运行在node23。日志最新分别为fold3 epoch12/step15400、fold4 epoch5/step5600，fold3尚未写出12轮完成收据，fold4继续训练。当前不是因忘记申请多卡而串行：两个独立折已并行；也不能给运行中的DDP作业热加卡或将其改为4卡而不重启并改变分布式执行路径。等fold3固定12轮及checkpoint/隔离收据通过后，立即用释放的队列槽位提交其两卡OOF预测，同时让fold4继续；之后完成fold4预测，再做五折合并与同键A/AH/AC GHI评价。GHI头的两卡DataParallel合成profile仅得单卡吞吐的0.118倍，故头部阶段将评估把独立方法臂分卡并行，而不使用低效DataParallel。当前仍无COT/GHI收益结论。

06:10 CST增量核验：两卡DPP训练持续有更新，fold3由step15400推进至15600（epoch12），fold4由step5700推进至5800（epoch5）。node23 `nvidia-smi pmon`在两个独立采样中显示fold3 rank进程分别占GPU1/2，fold4 rank进程分别占GPU3/4，采样SM利用率约79–95%；因此四卡均有实际训练计算。fold3目标12轮尚未结束；账户PBS 2/3。无checkpoint完成收据前不启动其OOF预测。

## 2026-10-20 06:23 CST：折3预测任务已挂接训练完成依赖

折3/4两卡训练继续推进；折3日志已到epoch12/step16300，尚未出现training_complete收据，折4日志到epoch5/step6500。为消除训练完成后的排队提交延迟，在完整PBS队列为2/3时提交折3两卡预测`215041`，设置`depend=afterok:215034.tc6000`；提交后全账户为3/3，PBS显示该预测作业H（依赖等待），申请node23两张GPU。它不会在折3训练失败或尚未正常结束时读取数据；启动前的脚本门会再次核验epoch_012训练收据、固定清单、无旧输出、future observed AGRI/CPP/test/buffer隔离和node23临时盘余量。当前fold3/4训练仍活跃；没有OOF完整银行或GHI收益结果。

## 2026-10-20 06:25 CST：fold3固定训练完成，双卡OOF预测已启动

fold3训练`215034`以exit_status=0完成固定12轮：epoch12/step16524，epoch12 `train_loss_mean=0.1309816151`；`training_complete.json`状态为`COMPLETE_OOF_SIMVP_FIXED_EPOCH_TRAINING`，选用`epoch_012.pt`，checkpoint SHA-256=`12977fd66aacc5b7645c47a7b2624168faf570c4186c0d4fb31b4b1b75fefb91`，`heldout_payload_read=false`、`test_payloads_read=0`，脚本打印`OOF_FOLD_FIXED_EPOCH_DDP2_COMPLETE_PASS`。

依赖任务`215041`已自动启动并通过`OOF_PREDICT_PREFLIGHT_PASS`，申请node23两张GPU（GPU1/2），输出目录此前不存在，`/tmp`可用863 GiB。fold3预测manifest总目标5,271条，当前`progress.json`为48/5,271、状态BUILDING；`future_observed_agri_read=false`、`cpp_read=false`。全账户未结束PBS为fold4训练`215036`与fold3预测`215041`（2/3）；fold4至epoch5/step6700。fold3预测未终态前不合并/评分，五折GHI对比仍未产生。

06:31 CST增量：fold3两卡OOF预测`215041`持续运行并推进到240/5,271（约3.15序列/s），manifest为BUILDING，`future_observed_agri_read=false`、`cpp_read=false`。全队列在fold4训练`215036`、fold3预测`215041`运行时为2个活动作业；故将fold4两卡预测`215042`设置为`afterok:215036.tc6000`，提交后总未结束任务3/3，PBS状态H，申请node23两卡。该任务只能在fold4训练正常结束后启动，运行时重做完成收据、输出目录、因果隔离及临时盘检查；队列已满，不再提交其他任务。所有五折OOF及同键GHI比较仍未完成。

06:33 CST增量：fold3 OOF预测`215041`持续推进至352/5,271，最新批次约2.3–3.3序列/s；完成比例6.7%，状态BUILDING，future observed AGRI/CPP标志仍false。fold4训练`215036`完成epoch5（step6810，train_loss_mean=0.16185241，heldout_payload_read=false）并进入epoch6/step6900。fold4预测`215042`仍为afterok依赖等待；全账户3/3（fold4运行、fold3预测运行、fold4预测挂起）。尚未合并或评分，当前仍无GHI收益结论。

06:32 CST live check: fold3 OOF bank 215041 is 704/5,271 (13.4%), BUILDING; latest batch rate ~2.5–3.2 seq/s and forecast-observed AGRI/CPP flags remain false. Fold4 215036 remains in epoch6, latest step7100. Fold4 prediction 215042 remains H on afterok dependency. Full queue remains 3/3; no bank merge or GHI scoring has started.

06:36 CST live recheck: full-account PBS remains 215036 fold4 fixed-12-epoch SimVP training (R), 215041 fold3 OOF prediction (R), and dependent fold4 OOF prediction 215042 (H), 3/3 unfinished. Fold4 training advanced to epoch6 step7300 with global batch20; fold3 forecast bank advanced to 1,328/5,271, still BUILDING. The bank still declares `future_observed_agri_read=false` and `cpp_read=false`. No complete bank, merged OOF output, GHI A/AH/AC score, or gain exists yet. Source audit remains: no explicit additive input/target noise or random flip/crop augmentation in the current AGRI data path; stochastic DropPath regularization is present (0.01–0.1). Fold3's identical per-rank RNG caveat is documented; fold4 uses the independently seeded rank-RNG runner. Do not alter either active baseline or launch another job while the queue is full. Once the fixed baseline resolves, use the registered single-factor COT attenuation/low-sun noise screens only if their preconditions are met; neither is yet evidence of a fix.

06:39 CST verified follow-up: PBS remains 215036 (fold4 training, R), 215041 (fold3 OOF forecast, R), 215042 (fold4 forecast, H), 3/3. Fold3 reached 1,552/5,271 completed sequences, still BUILDING with future-observed AGRI and CPP flags false; fold4 reached epoch6 step7400. The fold3 arrays are live memory maps and must not be treated as complete or scored mid-write. Predictor source audit confirms forecast AGRI is de-normalized before frozen R, and R outputs are checked finite/nonnegative; no physical-range clipping is applied before R. This is a distribution-shift diagnostic question, not yet proof of harmful noise. After the full OOF bank is assembled, audit predicted-AGRI values in the R-normalized station support and pair that with GHI/COT error strata before selecting the already registered one-factor attenuation or stability arm. No bank merge or score yet.

06:43 CST partial OOF-to-R input range audit (complete folds0–2 only): checked the saved station-centred 3×3 forecast-AGRI values (19,598 OOF sequences; 5,644,224 pixel-channel entries) after standardizing with the frozen R's train-only 13-AGRI-channel mean/std. Pooled standardized mean=-0.154, SD=0.970; 0.615% of entries exceed |z|>3 and 0.024% exceed |z|>5. Per-channel |z|>3 fractions range 0.51–0.68%. This does not show a gross station-patch range explosion in the first three completed folds, so clipping is not presently justified by these data. Scope is only the GHI head's saved central 3×3 support, not the full 16×16 R receptive field; it is also not a COT accuracy or downstream GHI result. Fold3 remains BUILDING and fold4 prediction has not started; repeat/extend after all five folds complete before selecting any noise intervention.

06:26 CST cluster-clock correction and live status: this session's preceding tracker notes stamped 06:36/06:39/06:43 used the workstation clock, which was about 17 minutes ahead of the PBS host; use PBS host time as authoritative. Current PBS host time is 2026-10-20 06:26:29 CST. Queue remains full: 215036 fold4 training R, 215041 fold3 OOF forecast R, and dependent 215042 fold4 forecast H (3/3 unfinished). Fold3 is 2,896/5,271 BUILDING, with both future-observed AGRI and CPP flags false; fold4 training is epoch6 step7900. Completed folds0–2 remain manifest-verified. The synthetic OOF merge witness `test_merge_future_ghi_oof_bank.py` passed locally (`OOF_BANK_MERGE_WITNESS_PASS`), so the merge gate path is ready when folds3–4 complete; no merge/score is run on incomplete banks.

06:27 CST static target/context alignment audit: traced the packer's row construction from `lead_index` and the interval-end `target_time_utc`. For lead index0 (+15 min), the three GHI-head contexts map to observed AGRI/COT at init−15, observed init, and forecast +15; index1 (+30 min) maps to observed init, forecast +15, and forecast +30. The stored `init_time_utc` is target minus canonical lead. No one-frame target/context shift is apparent in this code path. This is a source-level check, not a runtime bank audit or GHI result; the merged keys/indices still must pass the existing end-to-end gate.
# 2026-10-20：GHI 头多卡并行调度纠偏

在用户指出可用卡应用于提速后，重新区分了单模型DataParallel与独立随机种子任务并行。既有profile只表明两卡DataParallel对小型IntervalHead明显变慢；它没有否定把seed43/44分配到两张卡并发训练。已准备两卡PBS runner：seed42仍按原结构门单卡训练；seed43和seed44使用一张卡一个独立训练进程，数据包、三组配对、seed、epoch、batch和评价合同不变。完成后用新增合并器检查完整覆盖和兼容字段，再由原审计器评分。

合并器及合成见证通过本地和服务器Python运行；PBS shell语法检查通过。服务器文件哈希全部通过。07:19 CST全账户队列仍3/3：`215036=R`（fold4，node23 GPU3/4，已到epoch9/step11100）、`215042=H`（fold4 OOF，申请node23两卡）、`215044=H`（旧单卡头流程）。撤销从未运行的`215044`后核验队列≤2，并替换提交双卡版`215045=H`，依赖`afterok:215042`，资源node22两卡；提交后未结束任务仍3/3。`215044=C`为被替换的held作业，不占GPU。新头流程尚未运行/分配GPU；暂无五折完整GHI OOF指标或增益结果。

## 2026-10-19

## 2026-10-20 07:26 CST — AC50 支线同样改为双卡种子并行

继续优化待运行的 GHI 头脚本：若三种子主对照未通过预设AC−AH门而触发AC50，seed42先完成，seed43/44随后分别在两张GPU并行训练四个配对臂；输出用通用合并器核对完整覆盖、合同一致性后再跑原审计。主对照与AC50都只更改运行调度，数据、seed、30轮、batch1792及指标定义保持不变。合成测试已同时覆盖A/AH/AC及A/AH/AC/AC50两种run-manifest合并。当前排队的215045仍为上一版双卡脚本，等待本次同步、服务器测试后再替换；队列上限保持3个未结束任务。
07:32 CST复核与脚本再迭代：主A/AH/AC的seed43/44并行及条件AC50的seed43/44并行均已纳入runner；通用合并器通过主三臂和AC50四臂两套合成见证。同步服务器后，SHA清单、bash语法、py_compile、OOF provenance和原配对审计见证全通过。全账户队列核验后将未运行的旧held 215045撤销，并提交更新版双卡头作业215046（afterok:215042，node22:gpus=2）；队列保持3/3，215045=C。fold4仍在215036两卡上epoch9/step11800；后续两阶段尚未开始，当前没有GHI增益结果。
## 2026-10-20 07:39 CST — fold4 two-GPU run verified active

Live PBS/log check: `215036` remains R and has completed epoch 9 at step 12,258 (`heldout_payload_read=false`), with epoch 10 at step 12,300. Two training ranks (PIDs 11312/11313) are live on node23 GPUs 3/4; Luna's independent read-only sample reported both GPUs at 99%, approximately 36.4/37.3 GiB memory, and both ranks using CPU. Node-local cache exists and node23 `/tmp` has 834 GiB free, so the brief low-utilization sample was transient rather than a stall. `215042` remains H afterok `215036`; `215046` remains H afterok `215042`; account queue is 3/3. No complete five-fold GHI evaluation or verified gain yet. No test/buffer or future observed AGRI/CPP inference data were used. No additional submission is allowed while the queue is full.
## 2026-10-20 07:46 CST — fold4 DDP training active; downstream jobs gated

PBS host-time live check: the account queue is full at 3/3 (`215036=R`, `215042=H` afterok `215036`, `215046=H` afterok `215042`). Fold4 SimVP is at epoch 10, step 12,700 of the fixed 12 epochs; epoch 9 completed at step 12,258 with `train_loss_mean=0.13861887` and `heldout_payload_read=false`. Both training ranks are active on node23 GPUs 3/4 (about 36.4/37.3 GiB allocated); short pmon samples varied from 46/80% to Luna's 97/98% SM, consistent with snapshot timing rather than an idle rank. `training_complete.json` and epoch-12 checkpoint do not yet exist. Fold4's two-GPU OOF prediction remains gated on successful training; the two-GPU GHI-head job remains gated on that prediction. No fold4 OOF bank or five-fold GHI score exists yet, so no future-COT benefit claim is available. No test/buffer or future observed AGRI/CPP inference payload was opened. Based on the observed ~22-minute epoch duration, roughly 60 minutes remain for the unfinished part of fixed training; this is a rough runtime estimate, not an ETA guarantee.

## 2026-10-20 07:53 CST — fold4 training still advancing

A one-minute live recheck confirmed PBS `215036` remains R and advanced from epoch10/step13000 to step13100; two afterok jobs remain queued (`215042` fold4 OOF prediction after training, `215046` GHI heads after prediction), so the account remains 3/3. The training ranks remain assigned to node23 GPUs 3/4; utilization snapshots vary during batch synchronization, and the step counter advanced, so there is no stall evidence. Fold4 epoch-12 completion receipt, OOF bank and GHI results are still absent. No extra task was submitted because the queue is full and the dependent stages are already scheduled. Continue to the saved OOF/GHI evaluation before selecting any noise intervention; test/buffer and future observed AGRI/CPP inference inputs remain closed.

## 2026-10-20 08:03 CST — fold4 enters epoch 11; two-GPU pipeline confirmed

PBS-host-time recheck: `215036` remains R and entered epoch11/step13700 after epoch10 completed at step13620; epoch10 checkpoint and metadata are durable, `heldout_payload_read=false`. The SimVP run is already distributed over two A100 GPUs (PBS request `node23:ppn=8:gpus=2`; ranks assigned GPUs 3/4). Dependent OOF prediction `215042` (two GPUs) and paired GHI-head stage `215046` (two GPUs, independent seeds concurrent) remain H afterok, so the full-account queue is 3/3. Fold3 OOF bank is complete at 5,271/5,271 with test/buffer/future-observed AGRI/CPP flags false; fold4 training receipt and bank are still pending. No complete five-fold merge or GHI score exists, so no benefit claim is available. Do not change the active DDP world size; review whether the queued stages can use more than two GPUs only after checking their measured throughput and runner's device-count assumptions. No paper正文 was changed.

## 2026-10-20 08:20 CST — fold4 training continues; OOF/GHI stages remain gated

PBS-host live check: account queue is 3/3 (`215036=R`, `215042=H` afterok `215036`, `215046=H` afterok `215042`). Fold4's fixed 12-epoch, two-GPU DDP run advanced from step 14,500 to 14,800 in epoch 11; no epoch-12 checkpoint or `training_complete.json` exists yet. Both downstream jobs remain held, and fold4 OOF bank/manifest and the five-fold GHI A/AH/AC score are absent. Do not score incomplete arrays or infer benefit from training loss. The registered same-OOF-pack AC50 check remains conditional on the primary AC−AH GHI gate failing. Test/buffer and future observed AGRI/CPP inference inputs remain closed; no manuscript changes.

## 2026-10-20 08:24 CST — fold4 enters its final fixed epoch

Direct PBS/log verification: fold4 epoch 11 completed at step 14,982 (`train_loss_mean=0.1346970271`, elapsed 1,360.52 s, `heldout_payload_read=false`), with `epoch_011.json` and `last.pt` written; epoch 12 has started at step 15,000. PBS remains 3/3 (`215036=R`, `215042=H` afterok training, `215046=H` afterok prediction). Luna sampled both assigned GPUs at about 97% utilization and 36.4/37.3 GiB; node23 `/tmp` has 834 GiB free. Fixed epoch-12 checkpoint and training-complete receipt are not yet present; fold4 OOF bank and GHI scores remain absent. Continue the queued OOF→frozen-R COT→paired GHI gate; no test/buffer or future observed AGRI/CPP inference data used.

## 2026-10-20 08:32 CST — final fold4 epoch advancing

PBS/log recheck: fold4 fixed-12-epoch training `215036` remains R and advanced to epoch 12, step 15,500; the epoch-12 checkpoint and `training_complete.json` are not yet present. Dependent OOF prediction `215042` and paired GHI stage `215046` remain H afterok, leaving the account queue at 3/3. No fold4 forecast bank, complete five-fold GHI pack, or paired score exists yet. Maintain the fixed sample/split and causal-input boundary; test/buffer and future observed AGRI/CPP inference remain closed.

## 2026-10-20 08:36 CST — fold4 final epoch continues

Direct PBS-host check: `215036` remains R and advanced to epoch 12 / step 15,700; training receipt and epoch-12 checkpoint are not yet present. The two-GPU DDP process continues; dependent two-GPU OOF prediction `215042` and two-GPU paired GHI stage `215046` remain held, so the account is still 3/3. No complete five-fold GHI pack or AC−AH score exists. Continue the existing fixed-split pipeline; do not start a parallel job while the queue is full, and do not interpret training loss as GHI benefit. Test/buffer and future observed AGRI/CPP inference inputs remain closed.

## 2026-10-20 08:46 CST — fold4 training passed; OOF prediction started

PBS `215036` completed the fixed 12-epoch fold4 SimVP run at 16,344 steps. Its completion receipt records `world_size=2`, `global_batch=20`, `heldout_payload_read=false`, and `test_payloads_read=0`; the SHA-256 of `epoch_012.pt` independently matches the receipt (`117d007ca74f2844d7faa4bec0dcd9d31d056e506a662fae9a75b3dfc28de70c`). Dependent two-GPU OOF forecast job `215042` started automatically. Its source-hash and causal-path preflights passed, and it assigned two node23 GPUs; first log snapshot shows inference initialization only, not a completed bank. GHI head job `215046` remains held afterok prediction. Queue remains 3/3. No five-fold GHI score or verified benefit exists yet; test/buffer and future observed AGRI/CPP inference remain closed.

## 2026-10-20 08:47 CST — fold4 OOF forecast bank actively building

After training receipt/hash validation, PBS `215042` passed its forecast-causality and source-hash preflights and began the fold4 OOF prediction on two node23 GPUs. The live progress log reached 112/5,567 sequences, with recent batches around 3.1–3.3 sequences/s; forecast AGRI/COT memory-mapped arrays are being written. This is active inference progress, not a complete bank. The GHI-head stage `215046` remains held until prediction succeeds. Full account queue remains 3/3. Continue to require complete row/key coverage, array/index hashes, unique OOF IDs, and false future-observed AGRI/CPP/test/buffer flags before GHI training/scoring.

## 2026-10-20 08:49 CST — fold4 OOF inference progressing

The live fold4 forecast bank advanced to 464/5,567 sequences under the same two-GPU PBS job; recent flushed batches remain active (about 1.6–2.5 sequences/s in the latest samples), node-local disk has ample free space, and the progress manifest still marks `future_observed_agri_read=false` and `cpp_read=false`. No complete-bank manifest or GHI scores exist yet. The paired GHI stage remains dependency-held; maintain the fixed OOF inputs and wait for complete bank integrity checks before scoring.

## 2026-10-20 08:51 CST — fold4 bank passes early inference progress

Fold4's two-GPU OOF prediction `215042` is live and has advanced to 752/5,567 sequences; recent batches are about 2.5–3.4 sequences/s. `progress.json` remains BUILDING and confirms no future-observed AGRI or CPP inference reads. The fixed 12-epoch training receipt/checkpoint was verified before inference; no bank manifest or five-fold GHI scores exist yet. The paired GHI job `215046` remains held on the prediction dependency. PBS shows two unfinished tasks, preserving one account slot; no independent submission is needed while the causal pipeline is actively progressing.

## 2026-10-20 08:53 CST — fold4 OOF bank continues without causal violations

The same two-GPU fold4 forecast job `215042` advanced to 1,088/5,567 sequences; recent batches are around 2.6–3.4 sequences/s. The bank remains in BUILDING state with no completion manifest yet, and source flags still exclude future observed AGRI and CPP. The downstream three-seed GHI comparison is still held by its afterok dependency. Continue the full-bank integrity gate before any GHI scoring or noise-scale decision.

## 2026-10-20 08:54 CST — fold4 OOF bank remains active

A live poll of the same PBS job confirmed `215042` is still R and progressed to 1,344/5,567 fold4 sequences; recent batches are about 2.5–3.3 sequences/s. The OOF bank remains BUILDING, with future observed AGRI/CPP flags false and no bank completion manifest. The two-GPU GHI head task `215046` remains afterok-held. The main paired GHI decision and any AC50 noise attenuation remain pending complete OOF bank assembly.

## 2026-10-20 08:56 CST — fold4 OOF inference still progressing

PBS `215042` remains R and the fold4 forecast bank advanced to 1,616/5,567 sequences; recent batches are near 3.2 sequences/s. No bank manifest has appeared, so the GHI job `215046` correctly remains held. Progress metadata continues to exclude future observed AGRI and CPP from inference. No GHI gain is claimed until the complete OOF pack has passed key/hash checks and the paired date-block audit finishes.

## 2026-10-20 08:59 CST — fold4 two-GPU OOF prediction advances

Live PBS-host check: `215042` remains running on node23 with `gpus=2`; the fold4 OOF forecast bank advanced to 2,208/5,567 sequences (39.7%) at approximately 2.2–3.4 sequences/s in recent flushed batches. The bank remains BUILDING with no completion manifest; `future_observed_agri_read=false` and `cpp_read=false`. The paired GHI job `215046` remains held afterok `215042`, so the account has two unfinished jobs and one submission slot, but there is no need to interrupt/restart the active predictor to add GPUs. No OOF/GHI score or gain is available yet.

## 2026-10-20 09:01 CST — fold4 forecast remains active

A further live check of PBS job `215042` confirmed continued two-GPU inference: fold4 advanced from 2,208 to 2,464/5,567 sequences (44.3%) in about 86 seconds. Recent flushed rate is approximately 2.3–3.4 sequences/s. Forecast arrays are still being written; the progress manifest remains BUILDING and excludes future-observed AGRI/CPP. Paired GHI stage `215046` remains dependency-held. Do not restart or score the partial bank; no GHI effect is measurable yet.

## 2026-10-20 09:02 CST — fold4 OOF forecast advances under two GPUs

PBS-host live check: `215042` remains R with `nodes=node23:ppn=8:gpus=2`; progress advanced 2,544→2,736/5,567 in about 68 seconds. The bank is BUILDING; forecast-AGRI and COT arrays are partial, and the manifest still excludes future observed AGRI and CPP. `215046` remains afterok-held until prediction exits successfully. Do not score incomplete output; GHI benefit remains unmeasured.

## 2026-10-20 09:04 CST — fold4 OOF forecast continues

PBS job `215042` remains running with two GPUs on node23; verified progress advanced 2,736→2,992/5,567 sequences (53.8%). The bank is still BUILDING, with no completion manifest yet; future-observed AGRI and CPP remain excluded. The dependent GHI-head job `215046` remains held. This is execution progress only; no paired GHI gain is measured.

## 2026-10-20 09:05 CST — fold4 forecast OOF reaches 58.4%

Live PBS check confirms `215042` is still running with its two-GPU allocation. Fold4 has advanced 2,992→3,248/5,567 sequences (58.4%) in the latest interval. The forecast bank remains BUILDING and has no completion manifest; causal exclusions remain `future_observed_agri_read=false` and `cpp_read=false`. The two-GPU GHI stage `215046` is still held on `afterok:215042`; no GHI comparison result is available yet.

## 2026-10-20 09:07 CST — fold4 forecast OOF passes 3,500 sequences

Live PBS check: the two-GPU predictor `215042` remains running and advanced 3,296→3,504/5,567 sequences (63.0%). Fold4 remains in BUILDING state without a completion manifest; source flags continue to report no future observed AGRI or CPP reads. The dependent paired GHI job `215046` remains held. There are no new scored outputs; do not interpret this partial forecast as evidence of a benefit or failure.

## 2026-10-20 09:08 CST — fold4 OOF forecast reaches 67.8%

Live PBS check: the two-GPU fold4 predictor `215042` is still active and advanced 3,568→3,776/5,567 sequences. Its OOF bank remains BUILDING with no completion manifest; `future_observed_agri_read=false` and `cpp_read=false`. The dependent two-GPU GHI stage `215046` remains held. No score or gain claim is available until all arrays and keys pass the completion checks.

## 2026-10-20 09:09 CST — fold4 two-GPU forecast reaches 4,000 sequences

Live PBS check confirms `215042` remains running on two GPUs and has produced 4,000/5,567 OOF sequences (71.8%). The bank remains BUILDING; causal-path flags still exclude future observed AGRI and CPP. The dependent GHI job `215046` is held until the predictor exits. No complete manifest or paired GHI score exists yet.

## 2026-10-20 09:11 CST — fold4 OOF forecast passes 4,200 sequences

PBS-host verification: `215042` remains running with two GPUs and advanced 4,048→4,240/5,567 sequences (76.1%). Fold4 still has only partial forecast arrays; no completion manifest is present. Causal flags remain false for future-observed AGRI and CPP. GHI job `215046` remains held afterok; no GHI score is available.

## 2026-10-20 09:12 CST — fold4 OOF forecast exceeds 4,500 sequences

The live two-GPU predictor `215042` advanced 4,304→4,512/5,567 sequences (81.0%) and remains R. The fold4 bank is still BUILDING, with no completion manifest; future-observed AGRI/CPP flags remain false. `215046` is still dependency-held. The remaining prediction workload is roughly 1,055 sequences at the observed throughput; GHI outcome is still pending.

## 2026-10-20 09:14 CST — fold4 forecast reaches 4,752 sequences

Live PBS check shows `215042` still active with two GPUs; progress advanced 4,560→4,752/5,567 (85.3%). The bank remains BUILDING and has no final manifest; the held paired-head job `215046` has not started. Forecast input provenance continues to exclude future observed AGRI and CPP. No GHI scoring is possible until completion and integrity gates pass.

## 2026-10-20 09:16 CST — fold4 OOF forecast reaches 5,200 sequences

The two-GPU fold4 predictor `215042` remains active and advanced 5,008→5,200/5,567 sequences (93.4%). Only 367 sequences remain at the current observed rate, but the bank still has no completion manifest and remains BUILDING. `215046` is held on the predictor dependency. Causal flags continue to exclude future observed AGRI and CPP; no GHI metric is available until completion and scoring.

## 2026-10-20 09:18 CST — all five OOF forecast banks complete; GHI stage starts

Fold4 predictor `215042` completed successfully (`FUTURE_GHI_OOF_PREDICT_FOLD4_COMPLETE`, exit state C). Its manifest records 5,567 rows, both stations, 16 leads, fixed epoch-12 fold checkpoint SHA-256 `117d007ca74f2844d7faa4bec0dcd9d31d056e506a662fae9a75b3dfc28de70c`, frozen-R provenance, and false flags for future observed AGRI, CPP, validation/test payload, and buffer payload. Array and index hashes are recorded in the manifest. The GHI head job `215046` auto-started and passed source-file hashes, synthetic merge witnesses, OOF provenance witness, and `FIVE_FOLD_OOF_BANK_PREFLIGHT_PASS`. The five-fold manifest audit reports 30,436 OOF rows over five date folds, zero train/heldout frame intersection in every fold, zero test rows/read, and metadata-only split audit. Head training/scoring is now in progress; no GHI gain result yet.

## 2026-10-20 09:23 CST — five-fold OOF merge passes; causal GHI pack is building

The GHI stage `215046` completed its explicit five-fold preflight and merged forecast banks. The merge receipt reports `PASS_OOF_BANK_MERGE`, 30,436/30,436 unique training keys covered by five folds, 2,817 validation rows preserved from the source, and no test, future-observed AGRI, or CPP reads. The merged bank manifest is present with the corresponding completion state and all five fold checkpoint hashes. The subsequent GHI head-pack builder is actively running on node22 (observed Python process at ~50% CPU); initial arrays have been written, but the pack manifest is not yet present. GHI head fitting and paired scoring have not started.

## 2026-10-20 09:25 CST — OOF GHI head pack construction is active

After the merged five-fold OOF forecast bank passed (`30,436/30,436` OOF train keys), job `215046` is building the causal GHI input pack. The builder process is active on node22 at about 97% CPU and has created the pack's `train/` output directory; no final `pack_manifest.json` is present yet. Head model training has not started, so no paired GHI result is available. The account now has one unfinished job after prediction `215042` completed.

## 2026-10-20 09:27 CST — causal GHI pack build continues

The five-fold OOF merge receipt remains PASS. The active `build_future_ghi_head_pack.py` process is using about 98% of one CPU core on node22, and pack-array mtimes advanced during the latest minute; it has not emitted its final provenance manifest yet. The downstream three-seed GHI head fit has therefore not begun. Keep the job running; no evidence of a stall or scientific result yet.

## 2026-10-20 09:59 CST — CPU-only vectorized GHI pack rebuild queued; two-GPU head stage gated

The original GHI stage `215046` was stopped after only 44,054/714,914 train keys had been materialized; its partial `future_ghi_oof_head_pack_v1` directory is preserved and is not scored. A chunked builder was added as a separate versioned producer. Remote synthetic causal-window tests pass. On actual OOF bank data, the first 8,192 train rows match the original partial pack exactly across AGRI, geometry, log-COT, GHI, clear-sky labels, and CSV keys. A read-only 16,384-row end-to-end benchmark (8,192 per split) completed in 4.4 s; gather-only timing was ~76x faster than rowwise indexing. This is a bounded benchmark, not a full-pack timing or forecast result.

PBS `215073` now builds a separate v2 pack on node22 with 8 CPU cores and no GPU request. PBS `215074` is held `afterok:215073` and requests two GPUs for the unchanged three-seed paired GHI head screen plus the prespecified conditional AC50 branch. Current account queue is 2/3 unfinished (`215073=R`, `215074=H`); no other account task was canceled. The source OOF bank and causal sample contract are unchanged; no test, buffer, future observed AGRI, or CPP inference input was opened. The prior pack’s one-core CPU bottleneck will be replaced by chunked batch gathers; final inference is still pending.

## 2026-10-20 09:42 CST — vectorized pack complete; two-GPU GHI screen running

The CPU-only pack job `215073` completed successfully in 37 s (`exit_status=0`) with exact contracted coverage: 714,914 train rows and 66,590 validation rows. Its full manifest is `COMPLETE_CAUSAL_FUTURE_GHI_HEAD_PACK`, train inputs are five-fold SimVP OOF forecasts passed through frozen R, and test/buffer/future-observed AGRI/CPP flags are false. This replaces the 44,054-row partial v1 build, which remains preserved but unused.

The first dependent head launch `215074` exited at the output-guard preflight because that guard also rejected the intentionally prebuilt v2 pack; it created no head outputs. The guard was narrowed to the run/audit outputs, checksums and bash syntax were revalidated, and the two-GPU runner was resubmitted as `215075`. Live state: `215075=R`, node22 assigned GPUs 1 and 3; seed42 structural gate is running on GPU1 (latest sample 17% utilization, ~1.25 GiB), after which seeds43/44 run concurrently one per GPU. Account queue is 1/3 unfinished. No GHI comparison result exists yet; the fixed paired A/AH/AC audit and conditional AC50 branch remain pending.

## 2026-10-20 09:44 CST — seed42 shows a preliminary paired future-COT signal; seeds43/44 parallel

Seed42 completed its six paired GHI arms and the date-block audit on the complete v2 OOF pack. Across both stations and 52 target BJT date blocks, validation RMSE differences are AC−AH = −1.7637 W/m² (95% block-bootstrap CI [−3.3554, −0.1839]) and AC−A = −1.4192 W/m² (CI [−3.0511, +0.2985]); AH−A = +0.3445 W/m² (CI [+0.0707, +0.6672]). This is an encouraging one-seed validation screen for forecast COT versus history-only COT, while the net comparison against no COT remains uncertain. It does not pass the final claim gate by itself; station/short-lead three-seed criteria still apply, and this is not independent-test evidence.

The fixed seed42 structural gate completed, and runner `215075` launched seed43 and seed44 concurrently, one process per assigned A800 (node22 GPU1/GPU3). Both child PIDs are live. Their paired predictions, three-seed date-block audit, station/short-lead checks, and conditional AC50 gate are pending. PBS account remains one unfinished task; no test/buffer or future observed AGRI/CPP input was read.

## 2026-10-20 09:47 CST — three-seed paired GHI audit complete; AC50 noise-scale test started

The full three-seed validation audit (`42,43,44`) completed on 52 BJT target-date blocks using the exact paired station/init/target/lead/GHI-row keys. Equal-station mean RMSE (W/m²): no COT A=130.930, history COT AH=131.064, history plus predicted-future COT AC=130.047. Primary AC−AH is −1.017 W/m² (−0.776%), but its paired date-block 95% CI is [−2.300,+0.219]; AC−A is −0.882 W/m² (−0.674%), CI [−2.284,+0.529]. Neither overall interval excludes zero, so the three-seed screen does not establish a robust gain. By station, Sili improves (A/AH/AC 146.165/146.135/144.231), whereas Zhujia is effectively flat to slightly worse versus A (115.694/115.994/115.863); the +15-min lead also regresses at both stations for AC versus AH. This points to a real station/lead heterogeneity, not a confirmed universal COT benefit.

The pre-registered gate allowed AC50 because the primary pooled CI crossed zero. AC50 changes only the future standardized-COT anomaly scale to 0.5 on the same OOF pack; observed-history COT, AGRI, geometry, model, masks, samples, seeds, and budget stay fixed. `215075` is currently training AC50 seed42; if its structural check passes, seeds43/44 will run one per assigned GPU in parallel, followed by the paired date-block audit. Test/buffer and future observed AGRI/CPP remain unused. The independent test remains closed.

## 2026-10-20 10:15 CST — train-only COT reliability calibration screen submitted

The prior three-seed full forecast-COT and AC50 screens did not establish a reliable GHI gain. A fold-excluded, train-only reliability audit found that OOF predicted COT retains signal but is systematically miscalibrated by station and lead. A new single-factor follow-up applies the cross-fitted station/lead affine map in log1p-COT space to future slots only; historical COT, AGRI, geometry, GHI labels, masks, sample keys, GHI head, and training budget are unchanged. The five arms are A, AH, AC, AC50, and calibrated-future-COT ACREL.

PBS `215084.tc6000` is running the seed-42 structural stage on one node22 A800. This first stage verifies the calibration transform, exact paired predictions, file hashes, and date-block audit before spending GPU time on extra seeds. After it passes, seed43 and seed44 are scheduled concurrently on two separate A800 GPUs under a single two-GPU PBS job; the 3-seed audit will use the frozen 52 target BJT date blocks. Current account queue after submission is 1/3 unfinished. No result from this candidate exists yet; validation only, with test/buffer, future CPP, and future observed AGRI inference inputs closed.

## 2026-10-20 10:22 CST — seed42 structural gate passed; two seeds running on two GPUs

PBS `215084.tc6000` completed the five-arm seed42 stage and its paired audit across 52 BJT target-date blocks. ACREL−AH is −2.089 W/m² with a date-block 95% interval [−3.539, −0.611]; the station point differences are both favorable (Sili −2.538, Zhujia −1.639 W/m²). Treat this as a structural-gate result only, not a multi-seed claim or final gain.

PBS `215085.tc6000` is now running seed43 and seed44 concurrently on two distinct node22 A800s (GPU IDs 3 and 1), one isolated process per GPU. Both processes have started; current account queue is 1/3. Once complete, the runner merges only hash-verified exact arm/station/seed artifacts and performs the registered three-seed date-block audit. No independent test or buffer payload has been accessed.

## 2026-10-20 — oracle pack first attempt stopped at exact-context coverage gate

PBS `215086.tc6000` requested three node22 A800 GPUs in one job. Preflight hashes, Python import, shell syntax, and the synthetic context-time test passed; the job then stopped after 27.5 s before any GHI-head training because the compact GHI-labelled actual-R pack lacks some COT context frames (first missing key: train/Sili/2024-04-01 22:00 UTC). This is a source-coverage issue, not a station/time offset to be patched by nearest-neighbour matching. The assigned GPUs were IDs 1, 3, and 4; no seed process started and no test/buffer/CPP payload was opened. Preserve the incomplete v1 pack. The v2 builder now uses the complete 55,733-frame frozen-R train/validation bank and first checks that every compact-pack row's bank index and centered 3×3 crop are exactly reproduced. Only after the complete context-key coverage gate passes will the three-seed GHI jobs launch in parallel on three A800s.

## 2026-10-20 11:06 CST — three-GPU future-COT oracle screen completed

PBS `215089.tc6000` requested three node22 A800 GPUs in one job and completed successfully in 7m05s (`exit_status=0`); allocated GPU IDs were 1, 3, and 4. Seeds 42/43/44 trained concurrently, each on a separate GPU. A sampled live utilization snapshot during training was 15%, 21%, and 15%, respectively. The complete 55,733-frame actual-R train/validation bank passed exact timestamp and centered 3×3 crop checks against the compact labeled pack; the constructed oracle pack has 714,914 train and 66,590 validation rows. It uses future observed AGRI only to derive future COT for this oracle diagnostic, not future CPP; test/buffer reads remain false.

On 52 target BJT date blocks with exact paired keys, equal-station mean validation GHI RMSE (W/m²) was A=130.929, AH=131.040, and oracle AC=113.536. The primary oracle AC−AH difference was −17.504 W/m² (95% date-block bootstrap CI [−23.552, −11.970]); AC−A was −17.392 [−23.308, −12.025]. By station, AC−AH was −22.059 at Sili and −12.949 at Zhujia; the first-hour pooled guard also improved (AH=112.826, AC=108.960 W/m²). This is a strong validation-only information-ceiling result: correctly aligned future COT has substantial GHI value when derived from observed future AGRI. It is not a deployable forecast gain and does not prove the predicted-COT path attains that ceiling. The result directs the next diagnosis toward forecast-COT quality/calibration/alignment versus the oracle, while keeping the final claim tied to OOF forecast-COT evaluation. Full artifacts are archived under `results/future_cot_oracle_validation_v2_receipt/`; remote raw run outputs are under `results/future_cot_oracle_ghi_v2/` and `results/future_cot_oracle_validation_audit_v2.json`.

## 2026-10-20 11:43 CST — three-GPU frozen-head oracle-COT substitution completed

PBS `215111.tc6000` requested and received three node22 A800 GPUs (IDs 1, 3, 4); the frozen forecast-COT AC-head inference audit completed in 22.5 s (`exit_status=0`), with seeds 42/43/44 evaluated concurrently, one seed per GPU. It used the same forecast-pack validation keys and labels and substituted only future COT with observed-future-AGRI→frozen-R oracle COT; historical COT and all other features were unchanged. Exact row/key, label, checkpoint, and pack provenance checks passed. Baseline numeric replay had maximum row difference ≤0.108 W/m² and RMSE drift ≤0.00003 W/m². No model was retrained.

Across six station×seed runs and 52 target BJT date blocks, oracle substitution changed GHI RMSE by −9.092 W/m² (paired date-block bootstrap 95% CI [−12.342, −6.271]). Mean station deltas were Sili −12.624 and Zhujia −5.560 W/m². Mean changes were −1.541 W/m² at +15–60 min and −12.947 W/m² at +120–240 min. All six station×seed runs improved, with larger gains at Sili and longer leads. This is a validation-only diagnostic, not deployable skill: it shows that the frozen head can use future COT when supplied with the oracle field, and that imperfect predicted COT is a material bottleneck, especially at longer leads. It does not prove that forecast-COT accuracy alone explains the full oracle gap or establish a forecast-COT GHI gain. Test and buffer payloads remained closed. Full receipt and paired predictions are under `results/future_cot_oracle_inference_counterfactual_v1/`.
## 2026-10-20 — 空间对比校准候选准备，待训练折外 COT 门

用户确认有可用卡时可以多卡。本候选不重复训练 SimVP，而是在已有五折 OOF 预测 AGRI→冻结 R 的未来 COT 上，先用训练 OOF 数据交叉拟合一个保持站点中心 COT 不变、只修正 3×3 邻域偏差幅度的系数。它只作用于未来 COT，历史 COT、AGRI、几何、GHI 标签、头结构、样本和优化预算固定。目标 COT 参考是训练日期真实 AGRI 经冻结 R 得到的反演值；不是 CPP，也不是独立真值。

方案和脚本已同步本地与服务器。`audit_future_cot_spatial_contrast.py` 在 CPU 侧按日期折外拟合，并以目标北京时间日期为块计算配对区间；只有等站点等时效的 3×3 COT RMSE 改善区间整体低于零、且两站均不退化时，才启动 GHI 训练。若通过，则一个 PBS 作业申请三张 GPU，种子 42/43/44 各占一张卡；严格配对比较既有 A、AH、AC、ACREL 与新 ACRELSP 输出。新 GHI 候选仍是验证筛选，不代替独立测试。

服务器 `qstat -u slfu` 核验为空后，CPU折外COT审计PBS `215114.tc6000`（8 CPU，无GPU）已结束。脚本已写出完整交叉拟合产物后，在打印摘要时因审计结果字段名错误退出；该日志错误不影响已落盘的COT数组统计。读回JSON后，等站点等时效 patch COT RMSE 为 ACREL=0.632942、邻域对比候选=0.633489，差值+0.000547，日期块95%区间[+0.000433,+0.000662]；四里和竺家均略退化，故按预设门禁将该候选标为NO-GO，不启动其GHI头训练。错误字段已修复。

为继续直接回答“未来COT空间细节是否损害/帮助GHI”，另立无调参的下游中心值消融：仅把未来校准COT的3×3邻域像元替换成同帧中心值，保持中心和历史COT及其余输入不变。PBS `215116.tc6000` 单任务申请三张A800，分配GPU 1/3/4，种子42/43/44各占一张卡，三seed均完成四里、竺家各30轮与6,240次更新，最终配对审计完成；测试和缓冲区未打开，未来真实AGRI/CPP未作推理输入。52个目标北京时间日期块、exact common keys的等站点均值：A=130.932，AH=131.065，AC=130.071，ACREL=129.810，ACRELCENTER=129.820 W/m²。ACRELCENTER−ACREL仅+0.010 W/m²，95%区间[−0.241,+0.289]；ACRELCENTER−AH为−1.246，区间[−1.290,+0.028]；ACRELCENTER−A为−1.113，区间[−1.414,+0.250]。区间均跨零，中心值化几乎不改变总效能，故不支持“预测COT邻域纹理是主要噪声源”这一解释。

下一步据此改查预测COT的时间变化误差：用训练日期五折OOF预测及真实训练AGRI→冻结R参考，仅拟合目标帧相对前一15分钟COT变化量的折外校准；若训练折外目标COT误差稳健改善，再用三张GPU各跑一个GHI seed，保持相同模型、样本、标签和预算。该候选尚未提交，需完成计划、来源/合成门和新PBS核验后启动。

## 2026-10-20 — Multi-GPU use and temporal-increment follow-up

The user confirmed that multi-GPU execution is welcome when independent work can run concurrently. The previous center-only future-COT ablation (`215116.tc6000`) already used three node22 A800s, one per seed, and completed. The newly queued temporal-increment COT calibration (`215119.tc6000`) is deliberately CPU-only: it performs fold-excluded least-squares calibration and a date-block COT pre-gate, so GPUs would not accelerate that stage. The GHI follow-up is prepared as a separate three-A800 job, one seed per GPU, and will be submitted only if the predeclared COT gate passes.

A code review caught and fixed a missing `fold_by_sequence_index` map in the calibration artifact required by the downstream trainer. Remote `swc` Python compilation and the synthetic causal/invariance test pass. `215119.tc6000` is queued after live full-account `qstat -u slfu` showed no unfinished jobs; current queue is 1/3. The fixed forecast pack and actual-R reference are unchanged; test/buffer, CPP, and future observed AGRI inference inputs remain closed. No new GHI result exists yet.

### Audit rerun correction — 215119 / 215120

The first temporal-increment audit (`215119.tc6000`) completed the numeric calculations but exited before writing its result because the newly added sequence-index-to-fold mapping used the reverse dictionary direction. It produced no calibration artifact and no pre-gate result; no GPU job was launched. The mapping was corrected to emit the training sequence index for each fold, the remote Python compile passed, and the audit was resubmitted as `215120.tc6000` after a fresh full-account queue check. Its result remains pending.

## 2026-10-20 — Temporal-increment COT pre-gate passed; 3-GPU GHI screen queued

The corrected temporal-increment COT audit (`215120.tc6000`, 8 CPU cores, no GPU) completed successfully. On 699,419 train OOF rows with an exact adjacent-frame frozen-R reference, fold-excluded station/lead alpha calibration reduced equal-station/equal-lead center log1p-COT RMSE from 0.636427 to 0.630861 (delta −0.005567; target-BJT date-block 95% CI [−0.006818, −0.004259]). Station-pooled center RMSE improved at Sili (0.690809→0.685587) and Zhujia (0.611040→0.609723). This is a train-only cross-fitted retrieval-reference pre-gate, not GHI evidence or independent cloud truth. The remaining 15,495 GHI training rows without an exact previous 15-minute reference were omitted only from fitting/scoring the temporal calibration; the GHI head pack and downstream training cohort remain unchanged.

The predeclared gate passed. Three-seed GHI training was submitted as one PBS job `215121.tc6000` requesting three A800 GPUs, one seed per GPU, unchanged 30-epoch budget, and exact-key date-block paired validation against A/AH/AC/ACREL. At submission, `qstat -u slfu` showed no unfinished jobs; the new job is the sole active request (1/3). The independent test and buffer, future observed AGRI inference, and CPP inference remain closed. GHI outcome is pending.

## 2026-10-20 — temporal-increment calibration: COT improves slightly, GHI gain not established

PBS `215121.tc6000` completed successfully on three node22 A800s (GPU 1/3/4), with seeds 42/43/44 trained concurrently. Each seed trained both station heads for the fixed 30 epochs; exact-key validation predictions and the 52-target-BJT-date paired bootstrap completed. Equal-station mean GHI RMSE (W/m²): A=130.9323, AH=131.0653, AC=130.0711, ACREL=129.8097, ACRELTEMP=129.8178. The primary ACRELTEMP−AH point difference is −1.2475 W/m², but its 95% date-block interval [−1.3392,+0.0249] crosses zero. ACRELTEMP−A is −1.1145 [−1.3578,+0.1604]; ACRELTEMP−ACREL is +0.0081 [−0.0296,+0.0380]. Thus the temporal-increment adjustment did not improve GHI over existing ACREL and no robust future-COT gain is established.

Station results show the unresolved heterogeneity: at Sili A/ACREL/ACRELTEMP=146.166/143.840/143.855; at Zhujia=115.698/115.779/115.781. ACREL and temporal calibration both help Sili relative to A, especially at +135–240 min (ACRELTEMP−AH=−2.99 W/m²), but Zhujia has no net gain over A and slightly worsens at +15–60 min. The temporal adjustment itself changes GHI negligibly in either station/lead band. COT improvement is a small train-only, cross-fitted frozen-R reference improvement; it is not independent physical truth. Validation outputs remain screening evidence; test/buffer, future observed AGRI inference, and CPP inference are closed.

The validation audit and CSV are copied locally under `results/`; tracker and artifacts are synced to the server. Next, diagnose per-sample COT error and paired GHI benefit by station/lead/forecast-COT regime before selecting the next train-only conditional calibration; do not tune on validation strata or claim a gain from the pooled point estimate.

## 2026-10-20 — validation error-transfer diagnostic submitted

The temporal COT calibration passed the train-only COT pre-gate, but its three-seed GHI screen was statistically indistinguishable from ACREL and showed persistent Sili/Zhujia and short/far-lead heterogeneity. A validation-only exact-key diagnosis is now running as PBS `215124.tc6000` (8 CPU cores, no GPU): it pairs per-instance calibrated future-COT center/patch error against the frozen-R target-time COT reference and the saved A/AH/AC/ACREL/ACRELTEMP GHI predictions, using fixed `<1`, `1–10`, `10–30`, `>=30` predicted-COT regimes, the fixed three lead bands, and target-BJT-date bootstrap. This is descriptive only; validation strata will not fit a calibrator or select thresholds. CPP, test, and buffer remain closed; actual-R is used only as a clearly labeled validation comparator, not an inference input. At submission, full-account `qstat -u slfu` showed no unfinished jobs; this job is 1/3. The purpose is to decide whether the next candidate should be sample-adaptive COT reliability correction or a different temporal-support/GHI-head path.

### Conditional validation diagnostic launch correction — 215124 / 215125

The first diagnostic submission `215124.tc6000` exited during manifest preflight because the PBS runner pointed to the GHI prediction directory for the temporal calibration JSON. It stopped before loading validation arrays or writing an output. The path is corrected to `results/future_cot_oof_temporal_increment_v1/temporal_increment_calibration.json`; the actual prediction root remains `results/future_ghi_oof_cot_temporal_increment_v1`. The diagnostic was resubmitted as `215125.tc6000` after a fresh full-account queue check. No data, split, metric, or threshold changed.

## 2026-10-20 — COT-error/GHI-transfer diagnostic completed; target-window alignment screen queued

The corrected validation diagnostic `future_cot_ghi_conditional_diagnostic_v1.json` completed on all 66,590 exact validation keys, each matched to an actual-AGRI→frozen-R target-time COT reference; 52 target-BJT dates, three GHI seeds, no missing target-COT keys. The correlation between per-instance absolute center log1p-COT reference error and ACREL−AH paired GHI ΔSSE is only r=0.011, so global COT error magnitude does not explain which examples gain GHI skill. Only three of the 24 station×lead×predeclared-COT-regime ACREL−AH date-block intervals exclude zero (uncorrected descriptive strata): Sili +15–60 min with predicted COT<1, and Sili +135–240 min pooled / predicted COT 10–30. Most strata, especially Zhujia, remain uncertain. This argues against another global shrink factor and points to station/lead/regime heterogeneity; the diagnostic did not fit thresholds or calibrators and uses R only as a held-out retrieval comparator.

Next, a predeclared temporal-support experiment replaces only the target future COT slot by the trapezoidal mean physical COT between the previous frame and target frame, matching the `(T-15 min,T]` 15-minute GHI label window. The other inputs, head, seeds, masks, samples, budget and training protocol stay fixed. Three seeds are assigned concurrently to three A800s in one PBS job `215128.tc6000`; at submission the full-account queue was empty. This is a GHI-target alignment test, not a claim that interval-mean COT is an instantaneous retrieval. Independent test/buffer, future observed AGRI inference and CPP inference remain closed.

## 2026-10-20 — 三卡窗口平均未来COT审计完成；简单时间平均未解决收益问题

PBS `215128.tc6000` 已以 `exit_status=0` 完成，在 node22 GPU 1/3/4 上并行训练种子42/43/44，每张卡一个种子；每个种子分别训练四里和竺家 GHI 头各30轮。使用五折 OOF forecast-AGRI→冻结R未来COT、52个目标北京时间日期块和完全一致的站点/init/target/lead/GHI-row键完成配对审计。三种子等站点平均 RMSE（W/m²）：A=130.9323、AH=131.0653、AC=130.0711、ACREL=129.8097、窗口平均 ACRELWIN=129.8366。主对照 ACRELWIN−AH 为 −1.2287，日期块95%区间[−1.3291,+0.0363]；对 ACREL 的增量为 +0.0270，区间[−0.0278,+0.0515]。两站 ACRELWIN−ACREL 分别为四里 +0.0252、竺家 +0.0288 W/m²，均仅有极小变化。结论是“相邻两帧物理COT取均值”没有带来可复核的 GHI 收益；不能据此否定所有噪声问题。CPP、未来真实AGRI推理输入、test及buffer均未使用。

完整结果见 `results/FUTURE_COT_WINDOW_AVERAGE_RESULT_20261020.md` 与配对审计JSON。下一步不是继续盲目平滑，而是先做训练日期内部嵌套折外的目标时刻相位诊断：仅比较固定 −15/0/+15分钟的预测COT与同一冻结R在训练日期真实AGRI上的目标参考，按外层日期折评估、内层训练折选一个全局时移。若折外COT误差有稳健改善且两站不退化，再用三张A800各训练一个GHI种子；不通过则转查SimVP未来云场的到达时刻/位移误差。验证GHI结果不参与时移选择，test/buffer保持关闭。

### 2026-10-20 — 相邻帧平均无效后，启动训练折内相位诊断

窗口平均COT三种子筛选（`215128`）完成，未显示相对ACREL的可复核增益。现有训练日期交叉拟合诊断的增量标准差比也不支持简单地说“整条预测轨迹都噪声太大”：四里/竺家跨lead中位数约0.79/0.85，+240 min平均约0.75，而+15 min平均约1.19；它提示短时第一步可能过冲、远期变化幅度反而偏小，需检验时间相位而不能再盲目统一平滑。

已建立 `FUTURE_COT_LEAD_PHASE_PLAN_20261020.md` 与嵌套五折训练审计脚本 `audit_future_cot_lead_phase.py`。只在训练样本上使用五折OOF预测COT和oracle pack中同键的真实未来AGRI→冻结R COT作伪标签，选`−15/0/+15 min`全局相位偏移；oracle仅作训练参考，不进入未来推理。`+240 min`从相位门排除，按目标北京时间日期块bootstrap；只有中心COT误差区间整体改善、两站不退化且至少4/5外层折选到同一相位，才解锁三GPU GHI头筛选。CPP、验证拟合、test、buffer和未来真实AGRI推理输入保持关闭。

首次PBS `215130`因runner误指向不存在的bank-index路径而停在索引读取处，无COT数组处理和结果；真实文件位置已核对为 `future_simvp_r_bank_oof_v1/index.csv`，折ID目录为 `future_ghi_oof_manifests_v1`。脚本路径修正并通过远端 `bash -n`/Python预检后，PBS `215131.tc6000`已提交（8 CPU、无GPU），提交前全账户未完成队列为空；目前待审计结果。GPU在这个索引/统计阶段不会加速；若COT门通过，GHI对照再按三个独立种子并行使用三张A800。

### 2026-10-20 — 相位审计第二次实现修正

`215131.tc6000`在读取完整训练包并进入候选计算后，于`L=225`处因目标`L+15=240`的相邻GHI标签行缺失而退出；未写出相位审计结果，也未启动GPU。问题是把“下一帧预测COT”错误地通过下一条GHI样本行取值。全量五折OOF bank本身含同序列、同站点的16个预测COT时效，因此修订实现直接从bank读取`L+15`预测帧，按该帧时效应用既有训练折外校准，并在相邻GHI行存在时逐像元核对bank结果与head pack一致。目标GHI行集合、候选偏移、嵌套日期折、指标及门槛均不变。该修正只修复源帧索引，不以GHI标签缺失改变预报特征可用性；需远端编译/核对通过后再提交重跑。当前相位门仍无数值结果，不作收益判断。

重跑任务`215132.tc6000`已在全账户队列空闲时提交，资源为node4单节点8 CPU、无GPU；启动后实时状态`R`，已通过`LEAD_PHASE_SOURCE_PREFLIGHT_PASS`。审计直接取完整OOF bank的同序列/站点`L+15`帧，并对head pack中存在相邻目标行的记录做一致性复核。等待相位COT门和日期块区间；在门结果出来前不启动任何GHI训练。该CPU审计不是GPU工作负载，后续独立GHI种子按三卡并行。

## 2026-10-20 — 五折相位审计完成；时间偏移不是当前瓶颈

修正直接读取完整OOF COT bank的实现后，PBS `215132.tc6000`正常完成（8 CPU，1分40秒，exit_status=0）。训练集仅相位预门覆盖671,110条完全配对行和577个目标北京时间日期块；5个外层折均选择0 min，`-15/0/+15`选择频数为0/5/0。全训练下等站点等时效中心log1p-COT RMSE：`-15=0.676583`、`0=0.624131`、`+15=0.692603`；嵌套折外候选选择结果与基线完全相同，中心和3×3 patch误差差值均为0。四里中心RMSE均为0.677435，竺家均为0.598452。该结果是NO-GO时间相位调整，不是GHI收益判断，也不能单独证明COT预测没有其他误差。直接bank `L+15`源与存在的head-pack相邻行在656,410条记录上逐像元完全一致（最大差0），test/buffer/未来真实AGRI推理/CPP均未打开。

下一步按`FUTURE_AGRI_TO_COT_DOMAIN_PLAN_20261020.md`做验证诊断：在4,497个真实验证目标帧上把SimVP预测AGRI 3×3邻域与同一时刻实测AGRI逐通道对照，再看该误差是否与`SimVP AGRI→冻结R COT`相对同刻真实AGRI→R参考的误差同变。真实未来AGRI只作验证参照、不作推理输入或拟合；CPP、test和buffer仍关闭。该诊断为CPU/数据IO，不申请GPU；若显示可修的AGRI通道或空间信息损失，再用多卡并行训练COT-aware SimVP OOF折并检验GHI收益，而不是在这一CPU阶段空占卡。

`FUTURE_AGRI_TO_COT_DOMAIN_PLAN_20261020.md`及验证期诊断脚本已就绪；远端Python编译、bash语法、输出空路径和数据来源预检通过。全账户PBS空闲时提交`215134.tc6000`（node12，8 CPU、无GPU）；作业启动后通过`AGRI_TO_COT_DOMAIN_SOURCE_PREFLIGHT_PASS`，先对4个目标帧做真实AGRI裁剪smoke，再遍历全部4,497个验证时刻。此任务不训练模型，读取未来验证AGRI仅作误差参考，test/buffer/CPP关闭；待结果后据通道/时效误差传递决定下一项多卡训练。

## 2026-10-20 — DropPath 单因素 OOF 首次多卡启动被短 profile 门拦截

为检验唯一改变 `drop_path_prob: 0.1→0.0` 是否能保留更有用的预测细节，已将折0/1/2分别分配给两张A100，共申请六张卡（PBS `215139/215140/215141.tc6000`）。三项均通过日期折隔离、test=0、DDP全局batch=20一致性和rank RNG独立性检查；但都在正式训练前的性能profile处失败，未产出训练step、checkpoint或OOF结果。失败原因为旧脚本只测4个去除首轮后的模型更新，并硬性要求≥20 samples/s；三折读数10.50/4.98/11.28，峰值显存约30.46 GiB/rank。首个warmup step耗时18–22秒，steady-step又出现约1–12秒的抖动；旧profile没有计入DataLoader等待，样本过短，不能据此判断固定12轮是否能在36小时资源限额内完成。

因此只修正profile测量与可行性门，不改模型、数据、全局batch、优化器、12轮预算、样本或训练目标：新profile将测20个完整fetch-to-update步骤，排除前三个warmup，报告稳健中位数/p90，并以p90外推12轮训练时间须低于28小时作为36小时作业的留量门，同时保留每卡显存<34 GiB门。三个失败任务均已结束；最近全账户PBS仍由其他任务占满，当前不提交重跑。test/buffer及未来真实AGRI/CPP推理输入保持关闭。

### 第二次六卡profile与输入缓存诊断

第二次提交`215149/215151/215152.tc6000`（折0/1/2，各用两张A100）把profile改为20个完整取数到参数更新的步骤，发现每隔数步出现约8–24秒停顿，p90估计12轮分别需79.4、58.1、59.6小时，故仍在训练前被36小时资源可行性门挡住。显存峰值仍为30.46 GiB/rank，DDP与batch语义正常。GPU抽样时计算卡利用率接近0，DataLoader子进程持续读源帧并写本地NPZ缓存；缓存由69 GiB增至166 GiB，表明冷缓存/源数据读取才是当前速度瓶颈。三任务均已结束，未生成训练产物。

现运行CPU预热任务`215154.tc6000`，只读取折0/1/2各自`train_frames.txt`列出的训练帧，并对每折先验检查训练/留出隔离；不读取验证、测试、buffer或CPP。最近进度23,000/49,108唯一帧，缓存166 GiB、node23 `/tmp`余638 GiB。缓存完成后将以同一六卡、固定batch20的设置重新profile；不通过时先按profile定位并修输入读取，不降低训练时长门。当前全账户未完成作业为该CPU预热任务1/3。

CPU任务`215154.tc6000`现已正常完成（exit_status=0，12分22秒）：只基于折0/1/2训练帧清单预热49,108个唯一帧，49,108/49,108缓存文件核验通过；记录了各折train_frames和manifest SHA，验证/测试/buffer/CPP均未打开。缓存231 GiB，node23 `/tmp`还余573 GiB。现准备使用全新的`future_ghi_oof_droppath0_v3_prewarmed`结果目录重跑三折六卡profile；固定12轮训练仍须先通过28小时预测门与34 GiB显存门，未通过即不训练。

## 2026-10-20 — DropPath-0 OOF three-fold training started on six A100s

After train-only raw-frame cache prewarm (49,108/49,108 files verified; no validation/test/buffer/CPP payload), PBS jobs `215161/215162/215163.tc6000` were launched for folds 1/2/0 respectively, two A100s per fold. All three full input-to-update profiles passed the declared runtime (<28 h) and allocated-memory (<34 GiB/rank) gates. Fold 0 first encountered a rendezvous port collision before any training; the runner was patched to derive unique rendezvous ports per PBS job/stage, and retry `215163` passed preflight/profile.

At 15:01 CST all three jobs were R (account queue 3/3), with global batch 20. Latest observed steps: fold 0 epoch 1/step 400, fold 1 step 500, fold 2 step 600. Six GPUs are allocated and latest sampled utilization was mostly near 99–100%, with expected per-rank fluctuation. Shared cache is about 231 GiB; node23 `/tmp` has about 573 GiB free. Fixed 12-epoch training is incomplete; no checkpoint, OOF forecast, frozen-R COT bank, or GHI comparison is available yet. This is training progress only, not evidence of benefit. Continue these live jobs; after they finish, train folds 3–4 under fresh queue/scratch checks, then run candidate OOF AGRI→frozen R→COT and exact-key paired GHI validation. Test/buffer and future observed AGRI/CPP inference remain closed.


## 2026-10-20 15:16 CST — DropPath-0 live status and validation-bank gap

Fresh Luna read-only verification: full-account PBS is 3/3 (215161/215162/215163.tc6000, all R on node23), assigned two A100s each. Fold0 and fold1 are at epoch1/step1000; fold2 is at epoch1/step1300. No ERROR/traceback/COMPLETE/FAIL markers and no checkpoints yet. Node23 /tmp has approximately 573 GiB free. No OOF or GHI result exists. Preserve the runs; folds3/4 cannot be submitted while the queue is full. The existing merge_future_ghi_oof_bank.py copies baseline validation rows and replaces only train rows. Therefore the candidate chain also needs a separately trained full-train DropPath-0 model, candidate validation forecasts generated without loading future observed AGRI, and a candidate-specific audited bank merge before the unchanged GHI comparison. This is an execution gap, not a change to data, labels, selection rule, or decision gate. Test/buffer remain closed.


## 2026-10-20 15:28 CST — Candidate validation bank merge implementation

Fresh Luna check: PBS 215161/215162/215163 remain R, queue 3/3; folds0/1 epoch2 step1400, fold2 epoch2 step1700; no errors, checkpoints, or terminal markers. Six A100s remain allocated; node23 /tmp has about 573 GiB free. To prevent a post-training evaluation stall, added merge_future_ghi_candidate_oof_validation_bank.py and a synthetic test. Unlike the existing merge, it requires and substitutes a separately produced candidate full-train validation forecast bank in addition to the five candidate OOF train folds. It enforces exact ordered train/validation keys, fold ID coverage, checkpoint/index/array hashes, array shapes and finiteness, and false provenance flags for test, buffer, future observed AGRI and CPP. Validation payload is explicitly barred as a model input. Local Python compile and the synthetic merge test pass. Script SHA-256: 6A56A403751A73A4D5E602FD9208DADF58C893FD837D2A834F166898B1B901A5; test SHA-256: 37186106D0B936A63BD6BA4A5BD2A049D2C7E58EEBCD9D81004715DD8891B4A9. Remaining pipeline work is the separate full-train candidate plus observed-history-only validation forecast producer. No model-quality conclusion exists yet; test/buffer remain closed.
## 2026-10-20 16:15 CST — Candidate full-train/validation COT path prepared

To keep the six-GPU DropPath-0 OOF training from stalling after the folds, added a two-GPU full-train PBS wrapper and an observed-history-only candidate validation forecast→frozen-R builder. The builder filters the official monthly validation manifest to the exact 2,817 validation `seq_id`s used by the baseline forecast bank, preserving bank order; 323 of the manifest's 3,140 val sequences are outside the paired GHI cohort and remain excluded. Future AGRI and CPP payloads are never opened; future deterministic geometry is permitted. Candidate checkpoint, train-only normalization, station-coordinate invariance, source hashes, exact index order, output hashes, and closed test/buffer/CPP provenance are checked. The PBS wrapper preserves seed42, 12 epochs, global batch20, and requires the measured 20-step p90 runtime projection <28 h and peak allocated memory <34 GiB. It is prepared only and has not been submitted.

Local `py_compile`, `bash -n`, and five unit tests pass (full-train validation selector, candidate-bank merge, and exact validation-ID order/subset mapping). Last live queue observation at 15:47 CST: DropPath-0 jobs `215161/215162/215163` all R (3/3), folds 0/1 step2000 and fold2 step2300, no failure or checkpoint yet; node23 `/tmp` 573 GiB free. Full-train data audit: 35,320 train sequences, 3,140 val sequences, 51,236 unique train frames; existing warm cache 49,108/51,236, leaving 2,128 train frames. No test/buffer/CPP payload opened. This remains pipeline preparation, not a performance result.


### 2026-10-20 16:08 CST — live DropPath-0 training recheck

Luna read-only check of the full `slfu` PBS queue: `215161` fold1 R, `215162` fold2 R, `215163` fold0 R (3/3 unfinished). All are on node23 using two A100s each (six GPUs total), global batch 20. Latest flushed log steps: fold1 epoch3/step2800, fold2 epoch3/step3100, fold0 epoch3/step2800; no error, NaN, OOM, failure, or traceback markers. Latest complete checkpoints are epoch2 for all three folds. Assigned-GPU memory is about 35–37 GiB; sampled utilization varies by rank, but trainer logs and all six rank processes are active, so there is no evidence to stop or restart. node23 `/tmp` has 573 GiB free. These are training-progress facts only: folds 3/4, candidate full-train model, candidate OOF/validation forecast banks, frozen-R COT, and paired GHI scores remain outstanding. No new job can be submitted while the account queue is full; test/buffer and future observed AGRI/CPP inference remain closed.

### 2026-10-20 16:14 CST — live DropPath-0 training recheck

A fresh full-account PBS poll at 16:14:33 CST shows `215161` fold1, `215162` fold2, and `215163` fold0 all still R (3/3). Compared with the 16:08 snapshot, logs advanced 200 global-batch-20 steps each: fold1 epoch3/step3000 (log mtime 16:14:25), fold2 epoch3/step3300 (16:13:45), and fold0 epoch3/step3000 (16:13:56). No error/traceback/NaN/OOM/failure markers were found. Latest complete checkpoint remains epoch2 for each fold (fold1 16:03, fold2 15:55, fold0 16:01); active epoch3 is still progressing. All six rank processes remain live and CPU-active. Assigned GPUs are fold1 4/5, fold2 6/7, fold0 1/3, each with about 35–37 GiB allocated; a short sample showed GPU utilization 0–100% varying by rank. Node23 `/tmp` remains 573 GiB free. This check occurred about six minutes after the previous 16:08 snapshot, so the measured per-fold step increase is 200, not a 24-minute rate estimate. No job was changed or submitted; the account remains full.
### 2026-10-20 16:18 CST — live DropPath-0 training recheck

Fresh full-account PBS snapshot at 16:18:26 CST: jobs `215161` fold1, `215162` fold2, and `215163` fold0 remain R (3/3), with elapsed times about 1:35:30, 1:34:50, and 1:32:34. Since the 16:14 log flushes, all progressed: fold1 epoch3/step3100 (log mtime 16:17:26), fold2 epoch3/step3400 (16:16:38), fold0 epoch3/step3100 (16:16:58), global batch 20. No error/traceback/exception/NaN/OOM/failure markers were found. Latest complete checkpoints remain epoch2 (fold1 16:03, fold2 15:55, fold0 16:01). All six rank processes are live at roughly 85–129% CPU; four-second `/proc/<pid>/io` samples showed increasing `rchar` and `syscr` for each process. Node23 allocation is fold1 GPUs4/5, fold2 GPUs6/7, fold0 GPUs1/3; assigned-card memory is about 35–37 GiB. A short utilization sample ranged from 0% on some ranks to 99–100% on others. Node23 `/tmp` has 573 GiB free. The logs were advancing beyond 16:14; no intervention was warranted.
### 2026-10-20 16:21 CST — live DropPath-0 training recheck

Fresh full-account PBS poll at 16:21:25 CST shows `215161` fold1, `215162` fold2, and `215163` fold0 still R (3/3), elapsed about 1:38:29, 1:37:49, and 1:35:33. Logs advanced 100 global-batch-20 steps since 16:18: fold1 epoch3/step3200 (mtime 16:20:22), fold2 epoch3/step3600 (mtime 16:22:28; this remote file timestamp is later than the login-node clock), and fold0 epoch3/step3200 (mtime 16:19:54). No error/traceback/exception/NaN/OOM/failure markers were found. Latest complete checkpoints remain epoch2 (fold1 16:03, fold2 15:55, fold0 16:01); no terminal receipt exists because all jobs remain active. Node23 allocations remain fold1 GPUs4/5, fold2 GPUs6/7, fold0 GPUs1/3; memory use is about 35–37 GiB per assigned GPU. A short utilization sample ranged from 0% on GPU3 to 98–100% on GPUs1/4/5/6/7. Node23 `/tmp` has 573 GiB free. No PBS slot is available; no job was submitted or altered.
### 2026-10-20 — repaired full-train preflight before submission

A read-only audit found the prepared full-train PBS wrapper asserted `manifest_audit.json` contained `train_val_frame_intersection==0`, but the authoritative audit schema does not define that field. The wrapper would therefore have failed before training. Replaced that stale-field check with an independent metadata-only set intersection of `train_frames.txt` and `val_frames.txt`, while retaining `frame_disjoint_splits=true`, train-only normalization hashes, and all existing model/runtime gates. Against the local authoritative manifest, the check passed for 51,236 train frames and 4,497 validation frames with an empty intersection. `bash -n`, Python compilation, and the five full-train/bank/selector unit tests pass. Corrected wrapper SHA-256: `9A0DB305374986DCFFAC4B52EBF6FE24A85515DCD0BD4A9F1C2E382B08F036E6`. The corrected wrapper was synced to the server and verified at 16:32 CST: remote SHA-256 matches, remote bash -n passes, and the metadata-only preflight reports 51,236 train frames, 4,497 validation frames, and zero overlap. The job is not submitted. A prior read-only server audit found 2,128 full-train frames absent from the existing shared raw cache (49,108/51,236); profile/runtime gate and scratch are still required before submission. No training data was opened by this repair.

### 2026-10-20 16:35 CST — live DropPath-0 training recheck

Fresh full-account PBS snapshot at 16:35:12 CST: jobs `215161` fold1, `215162` fold2, and `215163` fold0 remain R (3/3). Fold1 advanced to epoch3/step3700, fold0 to epoch3/step3700; fold2 completed epoch3 at step3999 and entered epoch4 at step4000. The fold2 `training/epoch_003.json` was present with mtime 16:34; folds0/1 latest complete epoch records remain epoch2. No error, traceback, OOM, NaN, exception, or failure markers were found. `/tmp` has 818 GiB available. The login node does not provide `nvidia-smi`, so this poll did not measure live GPU utilization; queue assignments remain active. No terminal job or PBS slot is available, and no submission or job intervention occurred. Fold3/4 remain pending; use the generic DropPath-0 fold PBS with `FOLD_ID=3` then `4` after a fold completes, and only after fresh queue, two-GPU allocation, and ≥260 GiB `/tmp` checks.

### 2026-10-20 16:37 CST — live DropPath-0 training recheck

Fresh full-account PBS poll at 16:37:28 CST: `215161` fold1, `215162` fold2, and `215163` fold0 remain R (3/3). GPU allocation remains fold1 node23 GPUs 4/5, fold2 GPUs 6/7, and fold0 GPUs 3/1; node23 `/tmp` has 818 GiB available. Fold2 is epoch4/step4100 and fold0 epoch3/step3800, each 100 steps beyond the prior 16:35 snapshot; fold2 epoch3 completion JSON is present. Fold1 remains at epoch3/step3700 and its log mtime is 16:35:03, while its PBS state remains R; this short interval is insufficient to diagnose a stall. No error/traceback/OOM/NaN/exception/failure markers were found. Latest complete epoch files are fold0/1 epoch2 and fold2 epoch3; no completion receipt exists. The account is full, so fold3 was not submitted. No intervention occurred.

### 2026-10-20 16:39 CST — fold0 epoch checkpoint milestone

Fresh full-account PBS poll at 16:39:40 CST: jobs `215161` fold1, `215162` fold2, and `215163` fold0 remain R (3/3). Fold0 completed epoch3 at step3876 (`training/epoch_003.json`, mtime 16:39); fold1 is epoch3/step3800 and fold2 epoch4/step4100, with no new epoch completion for those two. No error/traceback/OOM/NaN/exception/failure markers were found. PBS allocations remain fold1 node23 GPUs4/5, fold2 GPUs6/7, fold0 GPUs3/1. A node23 GPU sample showed memory allocated on all six job GPUs (about 35–37 GiB each); utilization was variable across the short sample. `/tmp` has 818 GiB available. Although fold0 reached an epoch boundary, its PBS job remains running, so the account is still full and no fold3 submission is allowed. No job was changed.

### 2026-10-20 16:43 CST — fold1/fold0 epoch boundary

Fresh full-account PBS poll at 16:43:10 CST shows `215161` fold1, `215162` fold2, and `215163` fold0 all still R (3/3). Fold1 completed epoch3 at step3927 (`epoch_003.json`, mtime 16:41); fold0 remains at completed epoch3 step3876 (`epoch_003.json`, mtime 16:39) and has entered epoch4 at step3900. Fold2 is epoch4/step4300. No errors, traceback, OOM, NaN, exception, or failure markers were found; no training-complete receipt exists. GPU allocations remain fold1 node23 4/5, fold2 6/7, fold0 3/1, with all six cards holding roughly 35–37 GiB; one short sample showed variable utilization. Node23 `/tmp` has 818 GiB free. Despite two epoch boundaries, all three PBS jobs remain active and no slot is free; fold3 was not submitted.

### 2026-10-20 16:45 CST — all folds active; queue remains full

Fresh full-account PBS poll at 16:45:24 CST shows `215161` fold1, `215162` fold2, and `215163` fold0 all R (3/3). Fold1 entered epoch4 at step4000; fold0 reached epoch4 step4000; fold2 remains epoch4 step4300. Latest completed epoch records are fold0/1 epoch3 and fold2 epoch3; no `training_complete.json` or terminal state is present. No error/traceback/OOM/NaN/exception/failure markers were found. GPU allocations remain fold1 node23 GPUs4/5, fold2 GPUs6/7, fold0 GPUs3/1; all six allocated cards show about 35–37 GiB memory in use, with utilization varying in the sample. Node23 `/tmp` has 818 GiB available. The queue is still full, so fold3 was not submitted and no job was changed.

### 2026-10-20 16:47 CST — all three folds continue training

Fresh full-account PBS poll at 16:47:05 CST shows `215161` fold1, `215162` fold2, and `215163` fold0 all R (3/3). All three logs advanced 100 global-batch-20 steps since the 16:45 snapshot: fold1 epoch4/step4100, fold2 epoch4/step4400, fold0 epoch4/step4100. Latest completed epoch records remain fold0/1 epoch3 and fold2 epoch3; no training-complete receipt or terminal state exists. No error/traceback/OOM/NaN/exception/failure markers were found. GPU assignments remain fold1 node23 4/5, fold2 6/7, fold0 3/1; a short node23 sample showed allocated-card memory about 35–37 GiB and variable utilization. `/tmp` has 818 GiB free. No queue slot is available, so fold3 was not submitted.

### 2026-10-20 16:51 CST — three folds remain active

Fresh full-account PBS poll at 16:51:56 CST shows `215161` fold1, `215162` fold2, and `215163` fold0 all R (3/3). Since 16:47, each advanced 100 global-batch-20 steps: fold1 epoch4/step4200, fold2 epoch4/step4600, fold0 epoch4/step4200. No new full-epoch JSON/checkpoint or completion receipt appeared; latest completed records remain fold0/1 epoch3 and fold2 epoch3. No error/traceback/OOM/NaN/exception/failure markers were found. GPU allocations remain fold1 node23 4/5, fold2 6/7, fold0 3/1; short utilization varies while allocated memory is about 35–37 GiB/card. Node23 `/tmp` remains 818 GiB available. Queue remains full, so no fold3 submission was made.

### 2026-10-20 17:03 CST — fold1 log resumed

At 17:03:08 CST the full `slfu` queue remains `215161/215162/215163=R` (3/3), with node23 GPU pairs still assigned fold1 4/5, fold2 6/7, fold0 3/1; `/tmp` has 818 GiB available. After its 16:58:33 log flush, fold1 wrote a new log at 17:01:32 and advanced to epoch4/step4600. Fold2 is epoch4/step4900 and fold0 epoch4/step4600. No new complete epoch receipt, terminal state, or error marker is present. The apparent short pause did not persist; no process intervention or new submission was needed.

### 2026-10-20 17:09 CST — node-local scratch clarification and live progress

Fresh full-account PBS poll at 17:09:40 CST shows `215161/215162/215163` all R (3/3): fold1 epoch4/step4800, fold2 epoch4/step5200, fold0 epoch4/step4800. Fold2 log mtime is 17:09:13; fold1/fold0 last flushes are 17:07:24/17:06:57. No new complete epoch receipt, terminal state, or error marker is present. GPU allocations remain fold1 node23 GPUs4/5, fold2 GPUs6/7, fold0 GPUs3/1; sampled allocated memory is about 35–37 GiB/card. Login and node23 clocks agree within one second. **Scratch correction:** earlier `/tmp` reports of 818 GiB were from the tc6000 login filesystem, not node23. The direct node23 measurement is `/dev/nvme0n1p4`, 1.5 TiB total, 803 GiB used, 573 GiB available, which remains above the 260 GiB gate. The queue is full; no fold3 submission or job intervention occurred.

### 2026-10-20 17:11 CST — active fold progress and resource recheck

Fresh login-node and node23 clocks at 17:11 CST agree within one second. Full `qstat -u slfu` shows `215161` fold1, `215162` fold2, and `215163` fold0 all R (3/3). Latest log steps are fold1 epoch4/step4900 (mtime 17:10:19), fold2 epoch4/step5200 (17:09:13), and fold0 epoch4/step4900 (17:09:57); no new epoch completion or `training_complete.json` exists, and error scans are empty. PBS assigns fold1 node23 GPUs4/5, fold2 GPUs6/7, and fold0 GPUs3/1. A node23 GPU sample shows about 35–37 GiB memory allocated on those cards with utilization varying by sample. Node-local `/tmp` is `/dev/nvme0n1p4`, 1.5 TiB total / 803 GiB used / 573 GiB free (above the 260 GiB gate); earlier 818 GiB values were login-node storage. Queue is full, so no fold3 submission was made.

### 2026-10-20 17:15 CST — fold2 epoch4 checkpoint milestone

Fresh login/node23 clock check at 17:15 CST agrees within one second. Full `qstat -u slfu` remains `215161/215162/215163=R` (3/3); no slot has opened. Fold1 is epoch4/step5000 (log mtime 17:13:12), fold2 completed epoch4 at step5332 and entered epoch5/step5400 (log mtime 17:15:09; `training/epoch_004.json` mtime 17:13), and fold0 is epoch4/step5000 (mtime 17:12:52). No error/traceback/OOM/NaN/exception/failure markers or training-complete receipts were found. Node23 rank processes are present and CPU-active; GPU allocations remain fold1 4/5, fold2 6/7, fold0 3/1, with about 35–37 GiB on allocated cards and variable utilization. Correct node-local `/tmp` reading is 573 GiB free on `/dev/nvme0n1p4` (above the 260 GiB gate); login-node storage is not used for this gate. No fold3 submission or job intervention occurred.

### 2026-10-20 17:19 CST — fold0 enters epoch5; all slots occupied

Fresh remote clock at 17:19:24 CST and full `qstat -u slfu` show `215161` fold1, `215162` fold2, and `215163` fold0 all R (3/3). Latest durable log records: fold1 epoch4/step5200 (log mtime 17:19:02), fold2 epoch4 ended at step5332 and epoch5 reached step5500 (mtime 17:17:59), fold0 epoch4 ended at step5168 and epoch5 reached step5200 (mtime 17:18:47). The fold0 epoch4 record is present (17:17); no `training_complete.json` or terminal state is present. No traceback/OOM/NaN/failure markers were found in the three active logs. PBS allocations remain node23 GPUs4/5 (fold1), 6/7 (fold2), and 3/1 (fold0); a node23 `nvidia-smi` sample showed active processes on allocated GPUs with utilization varying across cards. The corresponding rank Python processes were present and CPU-active (about 86–132% in the sample). Node23 `/tmp` is `/dev/nvme0n1p4`, 1.5 TiB total / 803 GiB used / 573 GiB available, above the 260 GiB gate. No PBS slot or GPU pair is free, so fold3 was not submitted and no running job was changed.

### 2026-10-20 17:24 CST — fold1 epoch5 starts; all jobs active

Fresh remote clock at 17:24:13 CST and complete `qstat -u slfu` show `215161/215162/215163` all R (3/3), with walltimes about 2:39:23, 2:39:00, and 2:37:06. Fold1 completed epoch4 at step5236 and entered epoch5/step5300; `epoch_004.json` mtime is 17:20 and log mtime 17:22:06. Fold2 is epoch5/step5700 (log mtime 17:23:52); its latest completed epoch JSON remains epoch4 (17:13). Fold0 is epoch5/step5300 (log mtime 17:21:45), with epoch4 JSON at 17:17. No training-complete receipt or failure marker is present. The allocated rank Python processes remain present and CPU-active (about 85–132% in the sample); GPU sample shows activity on allocated GPUs with utilization varying, memory roughly 35–37 GiB/card. Node23 `/tmp` remains 573 GiB free, above the 260 GiB gate. Queue remains full; no fold3 submission or intervention occurred.

### 2026-10-20 17:28 CST — all three folds continue; no queue slot

Fresh remote poll at 17:28:24 CST shows `215161/215162/215163` all R (3/3), elapsed 2:45:28 / 2:44:48 / 2:42:32. Logs advanced since 17:24: fold1 epoch5/step5500 (mtime 17:28:00), fold2 epoch5/step5800 (mtime 17:26:47), fold0 epoch5/step5500 (mtime 17:27:38). GPU allocation remains node23 4/5, 6/7, 3/1; all six assigned GPUs report active utilization and ~35–37 GiB memory. Rank processes remain present with CPU use about 85–132%. Node23 `/tmp` remains 573 GiB free. No job is terminal, so the queue is still full and no fold3 or full-train candidate was launched; wait for a slot and then recheck candidate cache/profile/resource gates before any submission.

### 2026-10-20 17:31 CST — all three folds active, no slot released

Fresh remote clock at 17:31:39 CST and complete `qstat -u slfu` show `215161/215162/215163` all R (3/3), elapsed 2:48:43 / 2:48:03 / 2:45:47. Current logs advanced: fold1 epoch5/step5600 (mtime 17:30:59), fold2 epoch5/step5900 (mtime 17:29:30), fold0 epoch5/step5600 (mtime 17:30:35). The active fold rank PIDs are present at roughly 86–131% CPU; GPU samples show processes on allocated GPUs 1/3/4/5/6/7 with utilization varying from 0–100% in this instantaneous sample. Node23 `/tmp` remains 573 GiB free. No `training_complete.json`, terminal state, or error markers are present. Queue remains full, so neither fold3 nor the candidate full-train run was started; after a slot opens, verify the completed fold receipt and reassess both the prescribed fold3 path and full-train cache/profile gates before deciding the next submission.

### 2026-10-20 17:35 CST — epoch5 continues; queue remains full

Fresh remote clock at 17:35:04 CST and full `qstat -u slfu` show `215161/215162/215163` all R (3/3; elapsed 2:52:08 / 2:51:28 / 2:49:12). Current log steps are fold1 epoch5/step5700, fold2 epoch5/step6000, fold0 epoch5/step5700. PBS still assigns fold1 node23 GPUs5/4, fold2 GPUs7/6, fold0 GPUs3/1. Node23 `nvidia-smi pmon` shows Python processes on the allocated cards; instantaneous SM utilization varies, while `/tmp` remains 573 GiB free. No `training_complete.json`, terminal job, traceback/OOM/NaN/failure marker, OOF bank, or GHI result is present. The three jobs remain active, so no slot is available; no fold3/full-train submission or intervention occurred.

### 2026-10-20 17:37 CST — all three DropPath-0 folds still training

Fresh remote clock at 17:37:14 CST; full `qstat -u slfu` remains `215161/215162/215163=R` (3/3). Latest logs: fold1 epoch5/step5800, fold2 epoch5/step6100, fold0 epoch5/step5800. PBS GPU allocations remain node23 4/5, 6/7, and 3/1; sampled utilization is active on most allocated devices and memory remains about 35–37 GiB/card. Node23 `/tmp` has 573 GiB free. No fold-complete receipt, OOF/forecast-COT bank completion, GHI prediction/metric/bootstrap artifact, or new audit manifest appeared after 17:31. Training epoch-end records inspected in the existing logs state `heldout_payload_read=false`; no test/buffer payload was opened. Queue is full, so no next job was launched; reassess fold3 first and the full-train candidate's cache/profile gate only when a slot opens.

### 2026-10-20 17:42 CST — folds advance; no completion or slot

Fresh remote clock at 17:42:09 CST and full `qstat -u slfu` show `215161/215162/215163` all R (3/3), elapsed about 2:59 / 2:58 / 2:56. Latest logs: fold1 epoch5/step5900, fold2 epoch5/step6300, fold0 epoch5/step5900. GPU allocations remain fold1 node23 5/4, fold2 7/6, fold0 3/1; `nvidia-smi pmon` sees Python processes on assigned cards with sampled SM utilization varying widely. Node23 `/tmp` remains 573 GiB free. No new training-complete receipt, OOF/COT bank, GHI prediction/metric/bootstrap artifact, or audit manifest appeared since the prior poll; no error markers were found. No test/buffer payload was read. Queue remains full, so there is no safe slot for fold3 or full-train candidate; no job was changed.

### 2026-10-20 17:43 CST — no terminal or downstream artifacts

Fresh remote clock at 17:43:56 CST; `qstat -u slfu` still lists only the three active runs `215161/215162/215163` as R (3/3). Fold1 reached epoch5/step6000, fold2 remains epoch5/step6300, and fold0 reached epoch5/step6000. GPU assignments remain node23 5/4, 7/6, and 3/1; sampled GPU memory is about 35–37 GiB/card and instantaneous utilization varies. Node23 `/tmp` remains 573 GiB free. No training completion receipt, OOF/COT bank, GHI output, or new audit manifest has appeared since 17:42; no log error markers. No queue slot is open, so no next job can be submitted; no job was modified.

### 2026-10-20 17:48 CST — current poll and remaining-time estimate

Fresh remote poll at 17:48:08 CST shows all three jobs still R (3/3): fold1 epoch5/step6100, fold2 epoch5/step6500, fold0 epoch5/step6100. Allocations remain node23 GPUs5/4, 7/6, and 3/1; `/tmp` has 573 GiB free. No training-complete receipt, OOF/COT/GHI audit artifact, or error marker appeared; no fold is terminal and no slot is open. Fold1 and fold0 had no new 100-step log line since the immediate preceding sample, but their rank processes remained CPU-active (roughly 95–117% on the sampled ranks) and GPU pmon still showed processes on assigned cards, so this short interval is not evidence of a stall. For planning only, latest completed epoch durations are 2312.5 s (fold1), 2340.7 s (fold2), and 2286.6 s (fold0); with current epoch5 progress and seven full epochs still remaining, linear step-fraction extrapolation gives roughly 4 h 37–43 min to the 12-epoch endpoint if epoch duration stays similar. This is an estimate, not a completion guarantee. Queue remains full; no job was submitted or changed.

### 2026-10-20 17:53 CST — fold2 epoch5 checkpoint recorded

Fresh remote clock at 17:53:00 CST and full `qstat -u slfu` show `215161/215162/215163` all R (3/3); no PBS slot opened. Fold2 completed epoch5 at step6665: `results/future_ghi_oof_droppath0_v3_prewarmed/fold_2_ddp2g/training/epoch_005.json` was written at 17:52 and records `heldout_payload_read=false`, fixed-epoch checkpoint selection, and elapsed 2334.52 s. Its job remains active. Fold1 is epoch5/step6300 and fold0 epoch5/step6300. GPU allocations remain node23 5/4, 7/6, 3/1; pmon shows Python processes on assigned cards with variable utilization. Node23 `/tmp` remains 573 GiB free. No fold has reached epoch12 or a terminal receipt; no OOF/COT/GHI audit/metric artifact appeared. Queue remains full, so no next job was submitted.

### 2026-10-20 17:56 CST — fold0 epoch5 complete; fold2 enters epoch6

Fresh remote clock at 17:56:31 CST; full `qstat -u slfu` still shows `215161/215162/215163` R (3/3). Fold1 is epoch5/step6400; fold2 entered epoch6 and is at step6800; fold0 completed epoch5 at step6460. The fold0 metadata record `fold_0_ddp2g/training/epoch_005.json` records elapsed 2295.38 s and `heldout_payload_read=false`; its PBS job remains R. Node23 `/tmp` remains 573 GiB free. No terminal receipt, new OOF/bank, COT audit, GHI prediction, or GHI metric artifact was found since 17:53. No slot is open; no job was changed.

### 2026-10-20 17:59 CST — fold1/fold0 epoch5 checkpoints; no downstream result

Fresh remote clock at 17:59:55 CST shows full `qstat -u slfu` still has `215161/215162/215163` R (3/3). Fold1 completed epoch5 at step6545 (`fold_1_ddp2g/training/epoch_005.json`, mtime 17:58:50, `heldout_payload_read=false`); fold0 completed epoch5 at step6460 (its epoch5 JSON was verified at 17:56); fold2 is epoch6/step6900. All three PBS jobs remain active. Node23 GPU pairs remain 5/4, 7/6, and 3/1; `/tmp` remains 573 GiB free. No new completion receipt, candidate OOF/COT bank, GHI prediction/metric, or audit result has appeared; queue is full, so there is no next-job slot. The preregistered ablation tests stochastic DropPath regularization (`0.1 -> 0.0`), not additive input-noise augmentation; it can test whether that regularizer suppresses useful forecast detail but cannot establish a general input-noise effect by itself. No test/buffer payload was read and no job was changed.

### 2026-10-20 18:02 CST — all three remain active; no epoch12 result

Fresh remote poll at 18:02:39 CST shows `215161/215162/215163` all R (3/3; elapsed 3:19:43 / 3:19:03 / 3:16:47). Fold1 entered epoch6 at step6600; fold2 is epoch6/step7000; fold0 is epoch6/step6600. Latest complete checkpoint JSONs are epoch5 for folds0/1 (mtime 17:56/17:58) and fold2 (17:52). Node23 allocations remain two GPUs/job (fold1 5/4, fold2 7/6, fold0 3/1); pmon sees Python processes on allocated GPUs with variable utilization. `/tmp` remains 573 GiB free. No epoch12 checkpoint/training-complete receipt, OOF/COT/GHI audit, or metric artifact has appeared; no terminal job or PBS slot is present. Thus no fold cache cleanup or next-fold submission is eligible. No test/buffer data was read and no job was modified.

### 2026-10-20 18:11 CST — authorized pause of DropPath-0 folds 0–2

Per the user's pause instruction, first rechecked the full queue at 18:07:14 CST: only our three jobs were unfinished, all R. Captured stop-point logs: fold1 epoch6/step6800, fold2 epoch6/step7100, fold0 epoch6/step6800. Latest durable epoch JSONs are epoch5 for all three (fold1 step6545, fold2 step6665, fold0 step6460); none reached epoch12 or wrote `training_complete.json`. Issued `qdel` only for `215161/215162/215163`. At 18:08:35 and again 18:11:51, full `qstat -u slfu` showed all three C and no unfinished jobs; each `qstat -xf` has `exit_status=271`, consistent with cancellation. PBS GPU allocations 5/4, 7/6, and 3/1 were released. Cancellation left one orphan rank process, PID 55934, whose verified command was the fold1 seed43 DropPath-0 runner writing only to fold1's output/cache. Sent SIGTERM to that exact process after confirming its command; it exited. A subsequent node23 pmon showed no compute PID on GPUs 1/3/4/5/6/7; GPU0 and GPU2 still had unrelated PIDs 61818/15573. `nvidia-smi` continued to report 36,755 MiB used on GPU5 despite no compute PID reported there; record this as unexplained residual device memory, not as an active job. Node23 `/tmp` remains 573 GiB free. No test/buffer payload was opened and no new job was submitted. No OOF/COT/GHI result exists for these incomplete candidate folds.

### 2026-10-20 19:38 CST — efficacy estimate and checkpoint-resume preparation

The user resumed the future-forecast GHI objective and asked for an efficacy estimate before continuing. Assessment: the DropPath 0.1→0 ablation is a reasonable, falsifiable test of whether stochastic depth weakens useful cloud detail, but no reliable downstream GHI gain can be predicted yet. The source audit found no explicit additive or geometric image-noise augmentation; station-neighborhood forecast AGRI spread is about 4–5% low and local forecast-AGRI errors correlate with COT retrieval errors (r=0.306 Sili; r=0.443 Zhujia). These associations make upstream recovery plausible, while previous predicted-COT/GHI results were not stable; require the full two-station +15–60 min paired date-block gate before claiming benefit.

Fresh remote check at 19:37:53 CST: only our train-cache-prewarm job `215229.tc6000` is active (R; account usage 1/3). It is rebuilding the node23 train-only cache; log advanced to 10,500/49,108 unique frames. The three interrupted folds retain complete epoch-5 optimizer/model checkpoints: fold0 step6460 SHA `5cce3a318126f4ddb75f7b109d70080e2ab213db0bdd1a1dc71343903d6ea53a`, fold1 step6545 SHA `b169f43f909a3782fd159dfd089354a48add242ff10edb03f87e15ef3fc7839c`, fold2 step6665 SHA `e7ca86c1e54f0b040f6ab1936e127e612b23213a6ebc7fff8a5ab8db07baf69d`. Added a two-GPU DDP resume PBS path for folds0–2 and repaired its output-directory handling; local Python compilation/Bash syntax and remote checksum/Python compilation pass. The resume runner preserves each fold's seed, split/norm/source hashes, global batch20, SyncBatchNorm and fixed 12-epoch LR schedule; it recovers from epoch5. The remote cache receipt and train jobs have not completed; no OOF/GHI result exists. Test/buffer/future-observed AGRI/CPP remain unopened.
### 2026-10-20 19:50 CST — three epoch-5 resumes submitted

The train-only cache rebuild completed: 49,108/49,108 unique frames are present and its receipt reports `test_payload_opened=false`, `validation_payload_opened=false`, and `cpp_opened=false`. Cache job `215229` is terminal. After a fresh account-wide qstat before each submission, submitted resume jobs `215231` (fold0/seed42), `215232` (fold1/seed43), and `215233` (fold2/seed44); the account is now 3/3. All three are running on node23, each allocated two A100 GPUs. Fold0 passed the epoch-5 checkpoint SHA/metadata and DDP global-batch witness and has entered epoch6: step6500 logged with train loss 0.1199975 from checkpoint step6460. Folds1 and2 passed their epoch-5 checkpoint preflight; fold1 also passed the DDP SyncBatchNorm/global-batch and rank-RNG witnesses, and is entering its resume stage. Startup logs show no error/traceback/OOM. Fold0 resume metadata records old_checkpoint_loaded=true, resume_epoch_completed=5, global_batch=20, world_size=2, heldout_payload_read=false, test_payloads_read=0, and future-observed-AGRI/CPP/test/buffer flags false. No candidate OOF prediction/COT/GHI metric is available yet; this is training progress only. The three jobs are left running, no fourth task is submitted, and no manuscript files are changed.
### 2026-10-20 19:52 CST — all three folds resumed; six GPUs active

All three running jobs now have `run_metadata.json` confirming checkpoint continuation: folds0/1/2 loaded epoch5 checkpoints at steps6460/6545/6665 respectively, `old_checkpoint_loaded=true`, global batch20, world_size2, deterministic RNG reinitialization recorded, and heldout/test payload flags false. Fold0 has logged epoch6 step6500 (loss 0.1199975). Folds1/2 have not yet emitted their first 100-step training marker, but their assigned ranks are present and node23 pmon sampled high utilization (about 90–97% SM) on the four GPUs, so no stall/restart action is warranted. Full PBS queue remains the three authorized runs `215231/215232/215233=R`; all six A100s are allocated and active. No OOF/COT/GHI outcomes exist yet.
### 2026-10-20 19:55 CST — resumed folds remain active; efficacy not scored

Fresh live check: account-wide PBS queue remains full at 3/3 (`215231` fold0, `215232` fold1, `215233` fold2; all R on node23). Each job uses two A100 GPUs; pmon sampled active Python ranks on the allocated GPU pairs. Resume logs advanced through epoch6: fold0 step6700, fold1 step6700, fold2 step6800, global batch20. No traceback/OOM marker was observed. The epoch-5 checkpoint lineage and resume metadata remain the basis for continuation; no epoch-12 model, OOF AGRI→R→COT bank, or paired GHI output exists yet. Therefore the DropPath-0 candidate's efficacy remains unknown; training loss/progress is not GHI evidence. No test/buffer/future-observed AGRI/CPP payload was opened. Queue full: no new job submitted and no existing job changed.

### 2026-10-20 19:56 CST — verified resumed model still uses DropPath=0

A targeted source-to-runner audit confirmed the resume jobs load the frozen `droppath0_source_20261020` through `base.model`; its `SimVPBackbone` instantiates `drop_path_prob=0.0`, and the PBS preflight explicitly asserts that source line and matches its SHA to each run metadata. Thus the five-epoch checkpoints being resumed were trained under the DropPath-0 source, and resumed epochs do not silently restore the baseline 0.1 setting. Current full queue remains `215231/215232/215233=R` (3/3); latest available steps remain fold0/1 epoch6 step6700 and fold2 epoch6 step6800. This validates the ablation identity, not its efficacy; no OOF/COT/GHI score exists yet.

### 2026-10-20 19:59 CST — next-fold manifests preflighted

Read-only preflight of the already frozen fold3/4 manifests passed: fold3 has 27,521 train / 5,832 heldout sequences; fold4 has 27,237 / 6,116; both have test count 0, zero train-validation frame intersection, and train-normalization manifest SHA matches. Neither candidate fold output directory exists, so the registered fresh-fold runner will not overwrite prior results. The runner accepts fold IDs 0–4 and fixes the same DropPath-0 source, two-GPU DDP, global batch20, and 12 epochs. Current full PBS queue remains 3/3 with folds0–2; therefore fold3/4 are prepared but not submitted. No payloads were read in this metadata-only check; no model efficacy conclusion is available.

### 2026-10-20 20:01 CST — candidate bank merger regression test passes

Ran the synthetic-only unit test `test_merge_future_ghi_candidate_oof_validation_bank.py` in the remote `swc` environment. It passed (1 test): fold OOF training rows and candidate full-train validation rows replace the baseline forecast arrays, while history arrays remain unchanged and the merged manifest retains the no-test/no-future-observed-input flags. The test uses temporary synthetic arrays and does not access project data. This verifies a downstream assembly safeguard, not a performance result. At the same check, PBS remained 3/3 with folds0–2 running; no result score or new fold slot was available.

### 2026-10-20 20:05 CST — verified live continuation

After a 60-second live wait, the same PBS handles remain R (215231/215232/215233; 3/3) and their logs advanced: fold0/1 epoch6 step7000, fold2 epoch6 step7200. No job is terminal, so no slot is available yet. This confirms continued training rather than a stale running-state record; still no epoch12 checkpoint or downstream COT/GHI score. No intervention or new submission was made.

### 2026-10-20 20:06 CST — three folds advanced within epoch 6

Fresh full-account queue check remains 3/3: `215231/215232/215233=R`. Logs advanced since the prior check to fold0 step7100, fold1 step7100, and fold2 step7300 (epoch6, global batch20). All remain on their two-GPU allocations. None has reached an epoch-end checkpoint or completion receipt; no OOF/COT/GHI metric is available. Fold3/4 remain preflighted, with submission gated only by an available PBS slot. No data/contract change or job intervention.

### 2026-10-20 20:09 CST — slow step-log interval checked; ranks are active

Fresh PBS and node23 process/GPU inspection confirms all three jobs remain R (3/3). Fold1 advanced to epoch6 step7200; folds0/2 had not emitted another 100-step line in the short interval since their last markers, so checked rank processes directly: all six Python ranks remain present with increasing CPU time, and pmon shows active GPU computation on one rank per fold while peer ranks are in expected synchronization waits. This is consistent with unequal batch/collective timing, not evidence of a hang; no restart or job change. No fold-end checkpoint or COT/GHI score exists yet.

### 2026-10-20 20:29 CST — resumed DropPath-0 folds reach epoch 6

Fresh full-account `qstat -u slfu` shows only the three authorized resume jobs, all running (3/3): `215231` fold0 on node23 GPUs 3/1, `215232` fold1 on GPUs 5/4, and `215233` fold2 on GPUs 7/6. Fold0 and fold1 wrote epoch-6 checkpoints at steps 7,752 and 7,854 (train losses 0.14627 and 0.14582; `heldout_payload_read=false`); fold2 advanced to epoch6/step7,900 and its epoch record had not yet been written. Rank processes remain active; GPU utilization is uneven in snapshots but no stall or failure is indicated. Node23 `/tmp` has 573 GiB free. No error markers, epoch-12 completion receipt, OOF/COT bank, or GHI paired metric/bootstrap artifacts exist yet. No test/buffer/future-observed AGRI/CPP payload was read; no jobs were submitted, stopped, or changed.

### 2026-10-20 20:40 CST — resumed folds enter epoch 7

Fresh PBS-host check at 20:40 CST shows `215231` (fold0), `215232` (fold1), and `215233` (fold2) all running, filling the 3/3 account limit on node23 with two A100s each. Each has a durable epoch-6 checkpoint; live logs reached epoch7 steps 8,200 / 8,400 / 8,500, respectively. Node23 `/tmp` has 573 GiB free. No traceback/OOM/NaN/failure marker was found. Folds3/4 have not started; no epoch-12 receipts, candidate OOF/COT bank, or paired GHI/bootstrap result exists. Based on observed epoch durations of roughly 38–41 minutes, the active three folds need about 3.5–4 hours; all five training folds about 11–13 hours if folds3/4 can start as slots release. OOF inference, COT generation, GHI-head scoring, and date-block bootstrap remain afterward and have no current measured ETA. Test/buffer/future-observed AGRI/CPP inputs remain closed. No jobs or scientific contracts changed.

### 2026-10-20 20:43 CST — user-requested stop of DropPath-0 training

At the user's request, verified the full `qstat -u slfu` queue and ownership before stopping only `215231` (fold0), `215232` (fold1), and `215233` (fold2), all owned by `slfu` and running the DropPath-0 resume script. All three are now terminal `C`, `exit_status=271`; the account has no unfinished PBS jobs. PBS cancellation left fold0's orphan training process group alive on node23; its processes were matched to the exact fold0 runner/output path, then only that process group received `SIGTERM`. A follow-up process/GPU check found no remaining processes from this experiment; the three GPU pairs were released (unrelated `gmx` on GPU0 remains). Node23 `/tmp` has 573 GiB free. Stop-point logs reached epoch7 steps 8,400 / 8,500 / 8,600 for folds0/1/2, but durable checkpoints remain at epoch6 (`last.pt`; SHA-256 fold0 `1bd4f14098b01d06769ff4c5da7369919fddb499a0771928392499fa7ec24961`, fold1 `142cc6044c334be4277f368cd091247a3e532b4baaaaf96f99a7a04998bcc3df`, fold2 `dc24c2742dc6b1d0d9c955c9d42bda39829e7af60e2511291a2f5b1408afd1fc`). No fold reached epoch12; no candidate OOF/COT/GHI result was produced. Test, buffer, and future-observed AGRI/CPP inputs remained unopened. No jobs have been resubmitted.
