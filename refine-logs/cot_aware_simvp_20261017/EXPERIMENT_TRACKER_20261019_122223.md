# Experiment Tracker — COT-aware SimVP

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
