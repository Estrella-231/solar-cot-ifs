## 最新推进（2026-10-21 04:40集群时钟；几何审计通过，SimVP重训进行中）

- **训练集和验证集太阳几何已核验完成。** 官方NOAA JavaScript公式独立复算覆盖62,802个train/val点（train 57,334、val 5,468）；最终修复数据包独立审计 `DATA_AUDIT_PASS_attempt12.json` 为PASS，覆盖10,473帧S数据及3,479条R记录，键、标签、AGRI、几何和日期隔离通过，test/buffer读取均为0。六个修正R模型（全train及5折）已完成。
- **上游重训已启动，但最终有/无预测COT的GHI配对头还未启动。** PBS `215311.tc6000`（node22/GPU4，SimVP full）和 `215312.tc6000`（node22/GPU3，fold_0）均为R；集群04:40时最新日志分别完成第2轮（验证loss 0.3049911）和第2轮（0.3215278），无训练异常。其它4折SimVP、预测COT bank和GHI头仍待队列控制器按依赖顺序提交。slfu队列另有既有任务 `215308.tc6000`，当前共3个未结束作业，已达上限；不再新增提交，控制器等待空位。
- **补上GHI训练前缺失的两项审计。** 复核发现原`GHI()`在COT bank生成后会直接训练GHI头，没有执行计划要求的逐时效COT评价；此外旧pack把`DATA_READY.json`误记成OOF bank报告哈希。现已将`evaluate_forecast_cot.py`和`audit_oof_bank_lineage.py`接入GHI阶段：生成COT bank后、训练头前，先在精确target-time和共同CPP有效像元上评估+15…+240 min，报告站点像元、站点3×3、16×16场、历史COT持续性及仅离线可用的未来真实AGRI→R参考，并按目标日期配对bootstrap；随后核对训练行来自对应留出折、验证行来自full-train bank及每行init/target/lead键。CPP明确标为检索参考，不作独立真值。
- 预测头pack gate现在读取OOF报告文件并校验实际SHA和PASS状态，不再接受任意非空哈希。五个改动文件已同步到服务器并逐项SHA一致；本地及node22 Python 3.10编译通过，4项合成对齐/指标/门禁测试通过。**这两个新增诊断尚未跑实数据**，需等全部SimVP折及验证COT bank完成；GHI配对头仍未开始。

## 最新推进（2026-10-21 04:26 BJT，attempt12重训运行；修正预测COT输入预处理）

- **训练/验证几何与修复数据包已经审计通过。** NOAA独立公式覆盖62,802个train/val点；`DATA_AUDIT_PASS_attempt12.json` 覆盖10,473帧S数据、4,086个train序列/407个val序列和3,479条R样本，键、标签、AGRI、几何、日期隔离及train-only归一化均PASS。修复反射率来源与SHA核对通过。test/buffer读取为0。
- **正式重训已经开始。** 修正R全量加5折六个模型已完成。最新qstat：SimVP full PBS `215311.tc6000` 在node22/GPU4，第2轮刚开始；fold_0 PBS `215312.tc6000` 在node22/GPU3，第1轮163/306 batch。账户队列 `215308`（既有作业）、`215311`、`215312` 共3个未结束作业，达到上限；控制器会等空位再逐折提交其余S模型。当前还没有新的COT预测bank或GHI配对结果。
- **发现并修正了一个R推理输入不一致。** R训练/验证把16个输入通道（含`day_mask`）都按train-only均值/方差标准化；旧的未来AGRI→R推理在标准化后又将`day_mask`改成0/1。独立复算全部3,479条train/val R输入后，新推理变换与训练pack完全一致（最大差0）；旧覆盖相对训练输入最大差12.4238个标准差单位。修正版在COT bank生成前应用：保留全通道z-score，只在`day_mask>0.5 && cosSOZ>cos(80°)`之外将输出COT置零，并给bank写入S/R/归一化/预测/几何/推理代码哈希。
- 修正已同步到服务器；不会改变正在训练的SimVP或已完成的R权重。远端旧脚本已备份，修正版远端`run_stages.py` SHA-256=`f021212867646ed8af347f5255c721af9074a11836c39566b86a982e9a9f49d2`。只读审计收据在本地`receipts/attempt12/R_INFERENCE_NORMALIZATION_AUDIT_attempt12.json`，变更记录在`receipts/attempt12/R_COT_INFERENCE_FIX_attempt12.json`。
- 下一步：按PBS≤3继续完成full和五折SimVP及预测bank；再按新方案先评估逐时效预测COT相对历史持续性（CPP仅作检索参考、验证集诊断），并完成三站有无预测COT的配对GHI头。当前未提交GHI头任务。

## 最新推进（2026-10-21，attempt12数据审计通过并启动正式重训）

- 最新队列推进：`215310.tc6000` 的R全量及5折训练已完成，控制器已登记 `R_READY_attempt12.json`，并提交SimVP fold_0作业 **215312.tc6000**（当前node22运行）。SimVP full `215311.tc6000` 当前第0轮约183/282 batch。实时未结束作业为既有`215308`、本任务`215311`与`215312`，正好3个；PBS `215310`为C态已完成。配对GHI头要等SimVP full/OOF模型完成。

- 最新实况：R的full及fold_0–fold_4六个模型已全部完成验证选模，`R_READY_attempt12.json` 为 `CORRECTED_SOLAR_S_SIX_R_COMPLETE`，test读取为0。SimVP full正式训练在node22 GPU4推进中；最近日志为第0轮139/282 batch，训练日志持续增长、GPU4约73.3 GiB且GPU利用率高。控制器在等待登记R阶段完成后启动fold_0。
- 实时未结束队列为既有 `215308.tc6000` 和本任务 `215311.tc6000` 两个；`215309` profile与`215310` R都已完成态。队列尚有名额，控制器会按依赖继续提交，且始终遵守三任务上限。

- 实际训练已进入迭代：R全train、fold_0、fold_1、fold_2均已完成验证选模（best epoch分别21、17、16、20），R PBS `215310.tc6000` 正继续训练剩余OOF折；SimVP正式full任务 `215311.tc6000` 已开始完整训练，GPU4当前显存约73 GiB/80 GiB。单卡profile测得full batch16，fold batch12；profile是工程实测，不作为精度结果。
- 本次实时队列仍为3个未结束：外部既有 `215308.tc6000`、本任务R `215310.tc6000` 和SimVP full `215311.tc6000`。账号并发到上限；控制器等待R/S依赖和空位后才会继续逐项提交OOF SimVP/GHI配对头。

- **修正后正式重训已开始。** profile `215309.tc6000` 已完成：A800-SXM4-80GB 单卡真实批次前反向和checkpoint重载均PASS；batch16峰值分配68.48 GiB、一步8.40 s且重载差0，完整S选择batch16、五个OOF折选择batch12。基于profile结果控制器开始顺序提交训练。
- `215310.tc6000`（node22 GPU3）正在训练修正后的R模型；全train R运行已生成并标记验证完成，现进入OOF折R训练。`215311.tc6000`（node22 GPU4）已启动修正后的SimVP full训练，日志和模型参数文件已创建。两项均使用A800，test/buffer保持关闭。GHI有/无COT配对头尚未启动，需等五折/全量S与R资产完成并生成验证预测后再进入。
- 实时全账户队列：`215308.tc6000`（既有direct_COT_HV_in）、`215310.tc6000`（本任务R）和`215311.tc6000`（本任务S full）三个未结束作业，恰好达到上限；profile `215309` 已为C态，不计入未结束。控制器在每个单独qsub前后均重新检查队列，不会超额提交。

- **attempt12最终数据审计已通过**：`DATA_AUDIT_PASS_attempt12.json` 标记 `PASS_PREPARED_ATTEMPT12_TRAIN_VAL_KEYS_LABELS_AGRI_GEOMETRY`。覆盖S 10,473帧（train 9,560 / val 913）、4,086 train与407 val序列；R 3,479行（train 3,178 / val 301）。日期分离通过；R/S重叠物理重建最大差0、S几何编码最大差`3.5763e-7`、COT目标最大差0；六套R统计只由各自train子集拟合，归一化回算误差均小于`1.34e-5`。test/buffer读取和GHI标签成员打开均为0。审计收据已下载至本地`receipts/attempt12/`。
- 审计通过后控制器已自动提交单卡profiling **215309.tc6000**，在node22/gpu2运行，实际分配 **NVIDIA A800-SXM4-80GB GPU 3**；用户指出归属学院服务器的GPU 0未被本任务使用。profile报告尚未生成，下一步由控制器据此顺序启动修正后的S/R训练及三站配对GHI头。
- 队列状态更正：`qstat -u slfu` 列出三条记录，但`215307.tc6000`已为完成态C，不计未结束；当前未结束任务实际为`215308.tc6000`（既有direct_COT_HV_in）和`215309.tc6000`（本任务profile），共2个。控制器也将C态过滤后再按≤3门槛提交。

- **阶段推进：attempt12已生成完整修正数据包。** S全train与五折共6套train-only归一化已完成；R全train与五折6套pack均已生成，完整包写出 `CORRECTED_SOLAR_TRAIN_VAL_DATA_READY`，覆盖S 10,473帧（4,086 train/407 val序列）和R 3,479行（3,178 train/301 val）。R/S重叠的物理几何重建最大差为0；全图辅助通道回算最大差`3.58e-7`。这些是构建产物自报值，仍等独立审计复核后才放行训练。
- 当前 PBS `215307.tc6000` 在node22运行，`audit_prepared_attempt12.py` 正以约86% CPU做独立审计；控制器 `140574` 仍处于 `WAITING_DATA`。实时账户队列只有 `215307`（本任务CPU审计）和 `215308`（既有direct_COT_HV_in）两个未结束作业。模型重训尚未启动；审计PASS后按计划自动进入profiling及训练阶段。

- 继续实时核验 attempt12：全train与五折的6套 SimVP train-only 归一化均已生成，R/SimVP时间索引文件也已开始落盘；构建进程在node22持续运行（约62% CPU），当前主要剩余步骤是完成R包和最终独立数据审计。PBS `215307.tc6000` 仍为R态，`QUEUE_STATE_attempt12.json` 仍是 `WAITING_DATA`，因此没有训练作业或新预测。
- 实时 `qstat -u slfu` 当前有两个未结束作业：`215307.tc6000`（本任务CPU数据准备，node22，8 CPU）和 `215308.tc6000`（已有 `direct_COT_HV_in`）。已结束的 `215306.tc6000` 不计入未结束数。没有提交额外作业；控制器需等待数据与审计门禁通过后才启动后续阶段。
- 目前可确认训练/验证太阳几何公式复核、精确UTC数据恢复、560条R独有源审计、缓存SHA与几何语义核验以及6套归一化均已完成；R pack构建、最终标签/样本键/AGRI/几何独立审计仍待完成。test/buffer读取仍为0。

- attempt10 PBS `215302.tc6000` 已完成10,473帧S场重建和六套train-only归一化，但R/S断言比较了R的16×16几何裁块与S的256×256全图sidecar，形状不同所以 `np.array_equal` 恒为False。这是代码尺寸比较错误，不是太阳公式或时间映射问题。全图和相同裁块的本地独立公式复算在5个时刻最大差为0；attempt10逐帧全场写入检查也已通过。
- 只读诊断 PBS `215304.tc6000` 的临时脚本也先按同样误解比较全图与裁块，因shape无法广播而退出；没有改写任何数据。诊断帮助确认了错误边界。
- 已在attempt12将断言修为取S全图sidecar的站点区域 `[103:119,118:134]` 后比较；同时仅在校验attempt10的10,473个缓存文件SHA及全图 `cosSOZ/cosRAA/day_mask` 与原始几何通道一致后复用缓存。node22实时检查表明attempt10缓存的10,473个data和aux文件都仍存在，避免重新生成大缓存。attempt12的19个Python源文件node22编译通过，PBS语法通过；源码tar SHA=`4eb2772724e7841218f8fabb205db200948eb073ab7ba6752a1a2f767308a824`，PBS SHA=`526b80064c7716294646cfdb75c92e5707a4bf21cb928c4d9eb980cc80094092`。
- 数据准备 PBS **215307.tc6000** 已在node22开始运行（8 CPU、0 GPU）。提交前slfu未结束队列为空，提交后仅该项在运行；控制器 PID `140574` 为 `WAITING_DATA`。当前等待修正后的R包生成及独立key/标签/AGRI/几何审计，门禁通过后自动启动训练。
- 当前仍没有模型重训或新GHI成绩。NOAA train/val几何公式核验、255帧恢复、560条R缺失源核验已通过；最终数据包独立审计待attempt12。test/buffer仍关闭。

本节取代attempt10“运行中”的旧状态；attempt10和attempt11只作为失败过程留档。

## 最新推进（2026-10-21，attempt9暴露旧缓存几何不一致；attempt10强制重建运行中）

- attempt9 数据准备 PBS `215301.tc6000` 在生成 S 六套归一化后，R/S重叠时刻几何一致性断言失败；没有生成DATA_READY或独立审计PASS，也没有启动模型训练。失败位置是重用的 attempt4 S缓存辅助几何与按修正公式重算的R裁块不一致。缓存SHA通过只能证明文件未损坏，不能证明缓存语义符合修正后的几何合同。
- 已建立隔离的 `code_pipeline_attempt10`：禁用attempt4缓存复用，强制重建10,473帧S数据与几何sidecar；重新拟合全train和五折S train-only归一化；在写入时增加全图SOZ、SOA−SAA、day-mask与sidecar一致性断言；独立审计增加所有S帧 sidecar/raw几何编码核验。attempt9失败产物和日志保留。19个Python文件本地及node22 swc/Python 3.10编译通过，PBS语法通过。attempt10源码归档SHA=`01c073f2df2f4f63005a7507d0ba0349a330de8742889192feb2d054aa5e8414`；PBS SHA=`bdd656a3b93506eae3957e81ddc45f0c54a1608bb06796f98a5c64abc0805ae6`。
- attempt10数据准备 PBS **215302.tc6000** 正在 node22 运行（8 CPU、0 GPU，2026-10-21 03:32之后提交；qstat显示队列运行态）。提交前 `qstat -u slfu` 只有 `215298.tc6000`（1 GPU）；提交后未结束作业共2个，满足≤3限制。控制器 PID `106300` 状态 `WAITING_DATA`，仅当DATA_READY及独立审计通过后才按序启动profile和模型训练。
- attempt9控制器已收到STOP标记，避免旧attempt继续轮询；目前仍可能在其60秒轮询周期内退出。test/buffer仍未读取。**截至目前，训练集与验证集太阳几何的独立公式核验和来源恢复已通过，但最终重建数据包还未通过独立审计，湖北三站配对重训尚未开始。**

以下 attempt9 数据准备状态记录作为历史留档，由本节取代。

## 最新推进（2026-10-21，attempt8失败已定位；attempt9数据准备运行中）

- attempt8 数据准备 PBS `215300.tc6000` 已结束，但未生成 DATA_READY 或独立数据审计文件，也没有启动任何模型重训。失败点不是560条来源映射：该审计已通过；而是构建脚本误读不存在的 `RESTORATION_WITNESS_ARCHIVE_AUDIT_attempt8.json`。唯一已核验的有效恢复见证审计文件是 `RESTORATION_WITNESS_ARCHIVE_AUDIT_attempt7.json`（255/255 请求哈希匹配、test/buffer读取为0）。
- 已建立隔离的 `code_pipeline_attempt9`，修正构建端、来源审计及最终数据审计所引用的见证审计文件，同时把全部新输出改到 attempt9 路径；attempt8目录、日志和失败记录均保留。19个Python源文件本地编译通过，node22上使用swc Python 3.10重新编译通过，PBS `bash -n` 通过。attempt9源码归档 SHA-256=`7811b7b1f39c99f15b8ca40a1b519011d9fe01ba22fd3c3dfc5885763c5dfe8d`；PBS文件 SHA-256=`4d5c39154451951dae22523a14125ced6a07c3c6ddfac5ecd8549d17cbc453c6`。
- 数据准备 PBS **215301.tc6000** 正在 node22 运行（8 CPU、0 GPU，开始时间 2026-10-21 03:32:21 集群时间）；脚本和产物目录均为 attempt9。PBS 显示名称仍是复制脚本里的 `hb_solar_data8`，只是调度器标签遗留，运行的实际文件名为 `prepare_data_attempt9.pbs`。提交前实时队列只有 `215298.tc6000`（另一个1-GPU作业）；提交后共2个未结束任务。
- attempt9 控制器 PID `89651` 处于 `WAITING_DATA`，只会在 `DATA_READY` 和独立键/标签/AGRI/几何审计 PASS 后启动profile及S/R/GHI训练。最终修正数据包还没有通过审计，故截至本条没有任何模型重训启动。
- NOAA独立几何核验（62,802个train/val点）、255帧精确UTC恢复、以及560条R非S缓存来源核验继续有效；test/buffer载荷仍关闭。

以下attempt8“构建运行中”记录现已由本节取代；保留为历史过程。

## 最新推进（2026-10-21，attempt8 的 560 条来源审计通过，数据构建运行中）

- attempt7 PBS `215299.tc6000` 已结束于数据构建阶段，没生成 `DATA_READY`，也没有启动模型训练。它暴露出来源清单覆盖口径错误：旧代码把“未在旧阶段R全场目录中”误当成“未在S序列缓存中”。实际有560个R行键不在S序列缓存；其中358条在旧阶段全场目录，另202条才需用已恢复的FD-201数据。数据源本身均可找到，失败是清单与路径绑定不完整。
- attempt8 新增源审计后，560/560个非S缓存键已核验：train 530、val 30；358个旧阶段全场源、202个恢复源；BJT→UTC键、split、20×256×256形状、有限值及SHA全部通过。报告为 `R_NON_S_CACHE_SOURCE_AUDIT_attempt8.json`，SHA-256=`94fb644a2938706ad615c35c069c5d327fe5d9f32d4dfc29f8f8d5df06984486`，test/buffer读取均为0。
- 数据准备 PBS `215300.tc6000` 在 node22 运行（8 CPU、0 GPU），已通过来源审计，当前执行修正S/R数据包构建；提交时账户仅有 `215298.tc6000` 一个未结束GPU作业。本轮队列控制器 PID `71226` 只会在 `DATA_READY` 和独立包审计通过后进入 profile/S/R/GHI训练阶段。
- attempt8 的19个Python文件哈希、远端编译、PBS语法及三个输入归档哈希均通过。源码tar SHA=`0b3ca8bc1d42e19dccef72cf7e2be515755ef5fc4ff28017c199d052904b2347`；PBS SHA=`81ad15335c3fabea0559962fc76e72468ff469d0de565a1a6985b448620f8571`。临时准备解包放在node22 `/dev/shm`，避免根盘空间不足。
- NOAA太阳几何公式对照和255帧精确UTC反射率恢复均已通过，但最终包独立审计仍在进行。当前仍没有模型重训作业或新GHI成绩；test/buffer继续关闭。


## 最新推进（2026-10-21，attempt7 数据准备已提交；几何及恢复见证核验通过）

- NOAA 几何公式与 62,802 个 train/val 点的独立对照已通过；恢复的 255 帧白天可见光数据及 202 个仅供 train/val 的 R 缺失源均完成哈希核对。R 输入覆盖 3,479 条，其中 3,277 条可与既有 S 缓存逐键对照；差异超过 1e-4 的 10 条均集中在已记录的 2026-01-01 可见光源差异，其他差异仅为浮点反归一化量级。
- attempt5 因 tc6000 登录节点没有 FD-201 的规范 `/home/Data_Pool_3/...` 挂载而失败；attempt6 修正数据源路径后，又因准备归档遗漏 `RESTORATION_REQUEST_WITNESS.json` 在起始处失败。两个失败目录和日志均保留，均未启动模型重训。
- 已核验远端 `RESTORATION_WITNESS_ARCHIVE_AUDIT_attempt7.json`：恢复见证状态 PASS、255/255 请求与恢复报告哈希相符、test/buffer 读取为0。归档 SHA-256=`59fd880be1b5f1e56a2964f0733ad068115ebda19977ef1686e82f60a6043f8f`；独立见证审计本地 SHA-256=`60a8d896b28bf346cdb00e62f335de3c277d3bef584bbfdf4efddce1f0bba10b`。
- 新建隔离的 `code_pipeline_attempt7` 和 `prepare_data_attempt7.pbs`，在 attempt6 准备包解压后额外解开已核验的见证归档；build 和独立 audit 都要求恢复报告、见证文件和 SHA 一致。源码包 SHA-256=`3d26269da353e6aff1a3f516f755e8801f10d2943bf75e21e049ddd79dbb8362`，PBS 脚本 SHA-256=`ad93be03dc4c701caf9413dd3bb2e91ac0f0629765cbe2e03a58b168c049d7cf`。18 个 Python 文件远端逐项哈希通过，PBS `bash -n` 与 Python 3.10 编译通过。
- 数据准备 PBS `215299.tc6000` 已实时提交并核验在 node22 运行（8 CPU、0 GPU）；提交前账户仅有 `215298.tc6000` 一个未结束 GPU 任务，提交后共2个。队列控制器 PID `49605` 等待 `DATA_READY` 和独立审计，不会跳过门禁。node22 `/tmp` 根盘余量约14 GB、`/dev/shm` 余量约164 GB；本轮大缓存仍写 `/dev/shm`，准备归档约4.4 GB。
- 当前模型重训仍未开始。数据准备成功后，独立审计必须再次核验键、标签、AGRI源、太阳几何、日期隔离与 test/buffer=0；只有 PASS 后控制器才按已批准合同依次 profile、重训修正后的 S/R，再做湖北三站有无 COT 配对 GHI 头。test载荷保持关闭。

## 最新推进（2026-10-21，attempt4键顺序断言修复；attempt5已提交）

- 实时复核 PBS `215292.tc6000`：`exit_status=1`，准备过程已完成 10,473 帧缓存和六套 SimVP 归一化，但在 R/CPP 时间键断言处停止；没有生成 `DATA_READY`，attempt4 没有启动任何模型重训。
- 定位到精确原因：R 行表和 CPP 侧车各有 3,479 个唯一 BJT 时刻，键集合完全一致（R-only=0、CPP-only=0），只是排列顺序不同；第一处顺序差在索引 792（R=`202508250615`，CPP=`202508220615`）。此前代码错误要求数组行顺序一致，虽然已有按键 LUT。
- 新建隔离的 `code_pipeline_attempt5` / `prepared_attempt5`。将构建端和独立审计改为校验唯一时间键集合一致、按 `time_bjt` 查表对齐，并让 SimVP indexes 依据 UTC/BJT manifest 映射物理帧路径；独立审计另行校验所有 10,473 组修正帧/几何文件 SHA-256。attempt4 的代码、缓存清单和失败日志保持原样。
- attempt5 仅在 node22 上 SHA-256 复核 attempt4 的完整 `/dev/shm` 缓存后才会复用；若缓存已不存在，则写到新的 attempt5 `/dev/shm` 目录。数据源、split、标签、模型和训练合同未改变。
- 首次重提因 PBS 文件是 Windows CRLF 被 qsub 拒绝，未产生作业；转为 Unix LF 后 `bash -n` 和远端 Python compileall 通过，远端源码包哈希与本地一致。随后实时查询账户全队列并成功提交数据准备 **215293.tc6000**。
- 当前实况：`215293.tc6000` 在 node22 运行，8 CPU / 0 GPU；`215290.tc6000` 为另一个已有训练任务、占 1 GPU。队列共 2 个未结束作业。attempt5 控制器 PID `117453` 处于 `WAITING_DATA`，仅当 `DATA_READY` 与独立 audit PASS 都存在时提交后续 profile/训练。当前没有模型重训完成或 COT 收益新结论；test/buffer仍关闭。
- attempt5 源码包 SHA-256=`8f815c7d87bcea8bcfe5991f408cf1d71d5168181b20e6bee9ffc1a59be39843`（注：这是本次最终归档哈希；以 `CODE_SHA256_attempt5.json` 为准）。PBS 脚本 SHA-256=`136eaac0df1dff22e9e07e179480da850a9d0a928973e8cf2bc3f11f29e53740`（以清单为准）。

## 最新推进（2026-10-21，attempt3空间不足失败；attempt4数据准备运行）

- **已确认：旧 attempt2 的数据准备 PBS 215289.tc6000 以 exit_status=1 结束，未生成 `prepared_attempt2/DATA_READY.json`，所以没有启动模型训练。** 失败点是 R/S AGRI 输入对齐断言，不能将该失败写成几何审计失败。
- 太阳几何已完成 train/val 全量 NOAA 独立核验；反射率从精确 UTC HDF 恢复已完成。两项只说明输入修复所需的上游审计通过，不等于数据包或模型训练通过。
- 初始源差异抽查修正：4条抽查中3条 R NPZ AGRI patch 与 canonical full-scene 裁块完全一致；1条（BJT 2026-01-01 08:00）C01–C06 不一致，C09–C15 一致。故目前只能说发现行级可见光源差异，**不能概括为 R 与 S 全面使用不同处理链**。
- 全量只读源审计已完成：3,479/3,479 个 canonical full-scene、`data16` 和旧 NPZ 输入文件均存在；`data16` 裁块与同 UTC 全图裁块 3,479/3,479 完全一致。旧 R NPZ 的 `patch` 有 10/3,479 行与 canonical 裁块不一致，10 行均为 2026-01-01 08:00–17:00 BJT 的连续训练时次；差异仅在 C01–C06，C09–C15 全部逐像元一致，最大绝对差 0.0944223。报告和逐行CSV在 `diagnostics_attempt2/R_SOURCE_AUDIT_attempt2.json`、`R_SOURCE_AUDIT_per_row_attempt2.csv`；审计只读 train/val 的 AGRI `patch`，未打开 NPZ 标签成员，test/buffer读取均为0。
- 为避免来源混杂，新的 `code_pipeline_attempt3/build_corrected_data.py` 已改为：重叠时次直接取已修复 S 全图缓存的站点裁块；R 独有时次从同一 canonical full-scene AGRI 根目录按正确空间索引裁块；对所有 R 行重算同一 NOAA 太阳几何并使用精确 HDF 恢复的新昼间可见光。旧 R NPZ 的标签/CPP mask 保持原映射，且 R train-only 归一化仍按各折单独拟合。该数据构建需等全量源审计确认所有 canonical 文件存在后执行。
- 新 attempt3 预处理代码、独立审计和隔离的 S/R/GHI 执行目录已完成静态检查并同步至服务器；远端源码哈希与本地清单一致，压缩包 SHA-256=`c2067ee3342980dae43b30f0603ba08a3c4b3f3609e8934c6ccd19fef4a81d17`。修复构建会统一从 canonical full-scene 产生所有R输入（与SimVP重叠时直接取已校正S缓存），并保留旧NPZ来源路径用于逐行差异记录；独立审计通过前控制器不会提交模型任务。
- attempt3 数据准备 PBS **215291.tc6000** 已以 `exit_status=1` 结束。唯一错误为 `OSError: [Errno 28] No space left on device`：它把约18 GB缓存写入node22已满的本地根盘 `/dev/nvme0n1p4`；失败时处理到1,792/10,473帧。该日志和失败缓存均保留，部分缓存移至node22 `/dev/shm/hubei_solar_corrected_20261021_attempt3_failed`，未用来冒充完整数据。
- 修复仅改变临时缓存位置和尝试编号：新代码把大缓存写入node22 `/dev/shm`（提交时可用224 GB），不改模型、标签、日期划分或数据来源。attempt4源码包本地/远端SHA-256一致=`f1c9f51ffaccaf05fa33d65aea9eedf3036ff326e372385fd4cd0e08f1df57a3`，远端bash语法和Python编译通过；队列控制器改为先完成单卡S性能测量再提交R任务。
- **attempt4 数据准备 PBS 215292.tc6000** 已实时确认在 node22 运行（8 CPU、0 GPU）；最新日志进度为 **8,448/10,473 帧**，新缓存已写入 `/dev/shm`（约49 GB），根盘空间错误没有复现，尚无 `DATA_READY.json`。训练/验证最终数据包的逐项几何审计仍待构建完成后执行，所以模型重训尚未开始。
- 太阳几何公式及独立 NOAA 参考核验已经通过（train/val共62,802个点，天顶角最大差`1.75e-13°`、方位角最大圆周差`2.72e-10°`）；但修复后实际数据包仍须在准备完成后由独立审计脚本复核，不能把公式核验等同于最终数据审计通过。
- 实时账户队列当前有两项未结束作业：`215290.tc6000`（其他 `direct_cot_P_U`）和本任务CPU数据准备 `215292.tc6000`。attempt4自动控制器进程PID 84262正在运行，状态为 `WAITING_DATA`；只有数据包和独立几何/键/标签审计都通过后才会自动启动训练。test/buffer载荷读取仍为0。

以下内容中 `attempt2` 的“正在等待DATA_READY/即将启动训练”表述已被本段取代，作为失败过程留档。

## 历史推进（2026-10-21，数据准备已排队）

- 认证前 SSH 短断恢复后，attempt3 续跑器完成官方 NOAA 参考对照（见上方 JSON），并在 FD-201 CPU 节点恢复原始白天反射率。恢复包核查结果：255 个唯一 UTC 帧、1,344,223 像元，255/255 均有 C13 原始源一致性证据，最大绝对差 0；报告标记 `test_payloads_read=0`、`buffer_payloads_read=0`。压缩包已传回 node22 并解压到独立 `restored_attempt2`，没有覆盖旧缓存。
- 修复后 train/val 数据构建 PBS **215289.tc6000** 已提交，脚本 `prepare_data_attempt2.pbs`，8 CPU、4小时、无 GPU；队列提交后回执：215284 `R`（其他 `direct_cot_P_U`，1 GPU），215289 `Q`（`hb_solar_data`）。PBS 215288 审计已完成。提交前执行器实时查了全账户队列，确认未结束作业少于3个；本任务没有终止任何作业。
- 当前本地自动续跑状态 `DATA_PREPARATION_SUBMITTED`。它正等待 `prepared_attempt2/DATA_READY.json`；只有数据包的manifest、配对键/标签/AGRI/几何审计通过后，才启动S/R/GHI配对重训控制器。此刻修复后的模型训练尚未开始。
## 最新进展（2026-10-21，本轮实况）

- 全量几何审计 PBS `215288.tc6000` 已完成，exit_status=0（node22 CPU-only）。覆盖 4,086 个 train 序列、407 个 val 序列，共 10,473 张唯一 AGRI 帧（train 9,560、val 913）和 3,479 条 R 样本（train 3,178、val 301）；train/val 日期键隔离通过，test/buffer 载荷读取均为 0。
- 审计报告列出需要从精确 UTC 原始 HDF 恢复 255 帧、1,344,223 个白天像元；检测到 28 帧旧 aux 几何不一致，因此旧数据不能直接用于修正后训练。
- 独立 NOAA 官方 JavaScript 几何参考核验已通过：train/val 共 62,802 个点，天顶角最大绝对差 `1.75e-13°`，方位角最大圆周差 `2.72e-10°`，大气折射关闭后比较几何角。报告为 `NOAA_ALL_TRAIN_VAL_REFERENCE_AUDIT.json`。
- 当前执行器 attempt3 已进入 `ALL_TRAIN_VAL_OFFICIAL_NOAA_REFERENCE_PASS`，会继续传输恢复请求至 FD-201、恢复白天反射率、重建 train/val 数据并按队列规则启动后续重训。最近 SSH KEX 前断连，故恢复、DATA_READY 与模型重训尚未完成；不要把公式/几何通过误报成重训已启动。
- 已有恢复需求归档与审计产物先保存在服务器原实验目录；下一阶段成功后再核验 DATA_READY、manifest/key/labels/AGRI/geometry 与 GPU 训练队列。PBS 仍最多3个未结束作业；215284 不是本任务，不得触碰。
# 湖北太阳几何修复与配对重训状态

## 当前状态（2026-10-21）

- NOAA/Meeus 太阳几何核心及预处理接入已完成；官方 JavaScript 交叉核验 864 个合成公式案例，最大角度差低于 1e-12°。本地回归脚本再次通过（864 cases）。这只证明公式实现，不等于实际训练/验证数据已审计。
- 全量实际数据审计 job **215288.tc6000**（node22，4 CPU、无 GPU）最后一次可连接时为运行态。需产出全量 geometry report 并通过独立 NOAA 对照后，才可继续恢复反射率和构建训练数据。
- PBS 215287 是部署路径错误导致的快速失败；215286 是 R pack 路径拼错并已在 10240/10473 帧处失败；215285 暴露旧 aux 与 raw 几何临近日落不一致。失败日志和证据均保留，未覆盖旧数据/权重。attempt2 本地执行器随后只在查询报告文件时遭遇 SSH key-exchange 断连而退出，未做远端写入、未 qsub。attempt3 已接续等待同一个 215288，不重复提交审计。
- `prepare_data_attempt2.pbs` 已修正为 `restored_attempt2`，日志改为 `prepare_data_attempt2.log`；已同步到服务器，远端/本地 SHA-256 均为 `185f9f908c1dbb9a03275f4ebf5db104834ebea98c7d711259ac656bed27314a`。
- SSH 连接存在间歇性 KEX 前断连；本地执行器 attempt3 已启动（tool session 94875），状态为 `RESUMING_WAIT_FOR_EXISTING_AUDIT_NO_RESUBMIT`；它只对明确发生在SSH认证前的KEX断连重试，避免命令执行结果未知时自动重放。审计成功后继续独立 NOAA 复算、恢复和数据准备。执行器只在审计和恢复依赖满足后推进，qsub 前后核对 slfu 全队列，绝不盲目重复提交。
- 最后一次成功读取的全队列快照：215284（其他 direct COT 作业）与 215288（本审计）运行中；215287 已完成。215288日志之后进到6400/10473帧（监控快照）。FD-201 CPU恢复节点只读探测通过，swc Python存在，根卷剩余211 GB。此快照在 SSH 恢复前会过期，任何新 qsub 仍必须实时查询。

## 尚未完成

1. 实际 train/val 全帧及 R 样本太阳几何核验、时间/日期隔离报告、官方 NOAA 全实际点复算。
2. FD-201 CPU 恢复旧掩膜误置零的白天反射率，并由修正版数据准备脚本生成修复后的 train/val。
3. 重建/重训受影响 S 与 R 模型，再重跑湖北三站有无预测 COT 的相同样本配对 GHI 实验。

**截至当前：正式全量审计未完成；数据恢复未完成；任何修正后的 GPU 训练和配对 GHI 重训均未开始，暂无本轮成绩。test/buffer payload reads = 0。**



## 最新推进（2026-10-21 05:02 BJT；修正OOF日期折分来源路径）

- OOF审计脚本曾引用不存在的 `prepared_attempt12/metadata/train_date_folds.csv`。已根据服务器真实目录修正为 `preparation_attempt1/metadata/train_date_folds.csv`，并同步 `audit_oof_bank_lineage.py` 到 zjnu-hpc。文件SHA-256本地/远端一致：`8ac2cf8b3930ab9ed29ea7292cf592def96980aa80662f8e6980f5e39f304d09`。
- 权威日期折分文件已读出并核对：250个日期记录，5折各50天。修正后本地编译通过；远端使用训练环境 `/public/home/slfu/miniconda3/envs/swc/bin/python` 编译通过，4项远端合成测试全部通过。此前用于核查的系统Python 2.7不适用，但没有PBS任务因此失败，也没有提交新任务。
- 实时队列仍有3个未结束任务：`215308.tc6000`（既有任务）、`215311.tc6000`（node22/GPU4，SimVP full，第3轮验证loss 0.2777633）及`215312.tc6000`（node22/GPU3，fold_0，第3轮验证loss 0.3109249）。S预测bank尚未产出，故实数据COT评估、OOF行级审计和GHI配对头均尚未开始；账户满额时不追加提交，控制器继续等空位。
- 本轮路径修复见 `receipts/attempt12/OOF_DATE_FOLD_PATH_FIX_attempt12.json`。test与buffer载荷读取仍为0。

## 最新实况（2026-10-21 04:48集群时钟；GPU训练与续跑控制器正常）

- SimVP full `215311.tc6000` 最新完成第3轮，验证loss `0.2777633`；fold_0 `215312.tc6000` 最新完成第4轮，验证loss `0.2888934`。两项仍在node22分别使用GPU4/GPU3。三次GPU采样显示两张卡SM利用率在动态变化（GPU4为98/90/17%，GPU3为7/5/79%），显存约73/55 GiB；不是作业未启动或GPU空闲的状态。
- PBS账户全队列仍有3项未结束（另有既有任务`215308.tc6000`），未提交新作业。远端控制器PID `140574`仍存活；源码会在每个阶段提交前重新查询全队列，满3项时等待，并按profile→R→full及五折S→COT评估/OOF审计→GHI顺序自动推进。
- 六个关键控制/评估脚本本地与远端SHA一致；预测bank暂未生成，因此预测COT指标、OOF来源核对和GHI有/无COT配对训练都尚未进入。实时收据：`receipts/attempt12/LIVE_TRAINING_GPU_CONTROLLER_20261021T0448CST.json`。

## 训练推进（2026-10-21 04:51集群时钟）

- full模型推进至第4轮，验证loss `0.2504645`；fold_0仍在第4轮（验证loss `0.2888934`，该轮于04:47完成，任务仍为R）。
- 队列仍为3项，未提交额外作业；`S_BANK_READY`仍未出现。依赖控制器保持运行，等当前任务释放资源后按队列推进其余S折。

## 训练推进（2026-10-21 04:54集群时钟）

- fold_0完成第5轮，验证loss `0.2640957`；full仍在第4轮，最新验证loss `0.2504645`，作业保持R。PBS队列仍有3项未结束，控制器存活；暂无S预测bank，后续COT/OOF/GHI阶段未启动。

## GHI配对选模流程审计（2026-10-21 04:57集群时钟）

- 已对将来运行的有/无预测COT GHI头做源码级审计：FA与FAC都为24,489参数；FA的COT输入固定为零，FAC接收train拟合标准化后的预测COT。每个站点和seed运行时断言两臂初始权重哈希相同，并复用同一组训练样本顺序、batch、优化器、学习率计划、损失与更新数；每个站点分别依据完整验证集RMSE选择best checkpoint。
- 独立评分器逐项检查站点/init/target/lead键顺序、标签与晴空量一致、FP64 RMSE/MAE/bias、checkpoint重载一致性、实际更新数及等站点等seed汇总，并计算目标日期/连续两日配对区间。两个评分脚本本地与远端SHA均一致；证据见`receipts/attempt12/GHI_PAIR_SELECTION_AUDIT_attempt12.json`。
- 这是待运行实验的公平性/选模合同核查，不是收益结果；GHI训练尚未开始，仍等待校正SimVP full和五折OOF预测bank及其COT/OOF审计。

## 训练推进（2026-10-21 04:59集群时钟）

- full模型完成第5轮，验证loss `0.2451660`；fold_0仍为第5轮，验证loss `0.2640957`。两项PBS作业仍为R，队列共3项未结束；控制器PID `140574`存活，S预测bank尚未生成。

## 训练推进（2026-10-21 05:00集群时钟）

- fold_0推进到第6轮，验证loss `0.2556693`；full推进到第5轮，验证loss `0.2451660`。两项任务仍在运行，未发现失败标记；S预测bank尚未完成，后续预测COT与GHI配对仍待上游模型结束。

## 训练推进（2026-10-21 05:00集群时钟；新折已接续）

- 既有作业`215308`已由PBS标记完成，控制器据此提交了S的fold_1：`215313.tc6000`于05:00:38启动，PBS分配node22，作业启动记录显示计算节点`gpu2`/局部CUDA设备1。账户未结束作业维持3项：full `215311`、fold_0 `215312`、fold_1 `215313`。
- 当时full最新完成第5轮（验证loss `0.2451660`），fold_0第6轮（`0.2556693`）；fold_1正初始化训练。S预测bank尚无完成标记，forecast-COT和GHI配对阶段尚未运行。收据见`receipts/attempt12/LIVE_TRAINING_20261021T0500CST.json`；test/buffer读取均为0。


## 实时训练快照（2026-10-21 05:06 CST）

- NOAA train/val 几何复算与 `DATA_AUDIT_PASS_attempt12.json` 仍为 PASS；恢复数据和最终修正包未发现新的对齐/来源问题。test/buffer 保持关闭。
- SimVP full (`215311.tc6000`) 正在 epoch 7（约 13%），最近完成 epoch 6 的验证 loss 为 `0.2669241`；当前最佳仍为 epoch 5 的 `0.2451660`。
- SimVP fold_0 (`215312.tc6000`) 正在 epoch 8（约 4%），最近完成 epoch 7 的验证 loss 为 `0.2666648`；当前最佳仍为 epoch 6 的 `0.2556693`。
- SimVP fold_1 (`215313.tc6000`) 正在 epoch 0（约 73%），尚无完成轮次指标。三项均在 node22 使用单 GPU；账户未结束队列为这三项，已达 PBS 上限。控制器 PID `140574` 存活，按队列名额自动推进。
- 暂无 `S_BANK_READY`、预测 COT 评估或 GHI 配对训练完成标记；配对训练仍等待 SimVP full 与五折 OOF 预测链。日志未见 Traceback、OOM 或作业崩溃；fold_1 有 `find_unused_parameters=True` 的非致命 DDP 性能提示。
- 轮次验证损失有波动，选模继续使用各 run 的最佳完整验证 checkpoint；此快照不是最终模型成绩，也不代表预报 COT/GHI 已有提升。


## 下游时效映射核验（2026-10-21 05:09 CST）

- 对 GHI 配对阶段将复用的 `hubei_forecast_correctutc_20261020/time_repair_v1/runtime_attempt3/pack/rows.csv` 做了只读行键审计：168,790 行（153,515 train、15,275 val），`target_time_bjt - init_time_bjt == lead_minutes` 的不匹配数为 0；lead 集合恰为 +15…+240 min 的 16 个时效。
- 因此配对组包代码的 `lead_minutes // 15 - 1` 能正确映射未来帧索引 0…15，不存在旧版 legacy 0…225 坐标造成的一帧偏移。pack manifest 标记 test/buffer payload reads 为 0；此次只读取行键/元数据，没有打开标签载荷。
- 05:09 队列仍为 `215311` full、`215312` fold_0、`215313` fold_1 三项运行；控制器存活，尚无 SimVP bank / forecast COT / GHI 配对完成标记。


## Forecast-COT共同支持回归修复（2026-10-21 05:13 CST）

- 下游评估器静态审查发现：`valid[ri, 0]` 的视图被逐方法有限值检查原地改写。多个初始化共享同一目标时刻时，前一个预测的 NaN 会错误削减后续预测的 CPP 共同有效支持。
- 已改为每个目标行复制有效掩膜，并加入重叠起报时刻 + 单像元 NaN 的回归用例。旧实现把该用例的支持从预期 512 像元降成 511；修复后恢复为512。
- 本地 Windows Python 和服务器 `swc` Python 环境均运行 `test_forecast_cot_diagnostics.py`：4/4通过。修正评估器 SHA-256：`9ef61f4d6c7ffff3c479c1598ba3fe18613d9299d1457dacc7dcc10f7a779325`；测试 SHA-256：`b8ccd00e2712e70a68abceb4e977d1bdf5f1b0864c0b0bcf331cad1c75243cf6`；本地与远端文件哈希一致。
- 修正版已同步到 attempt12 服务器代码目录。同步时 SimVP full/fold_0/fold_1 仍在运行，GHI/COT评估阶段尚未启动；故不会覆盖或重跑已有结果，后续评估将直接使用修正版。此次不提交新 PBS 任务，test/buffer载荷读取均为0。

## 实时训练快照（2026-10-21 05:15 CST）

- 实时读取 `DATA_AUDIT_PASS_attempt12.json`，状态仍为 `PASS_PREPARED_ATTEMPT12_TRAIN_VAL_KEYS_LABELS_AGRI_GEOMETRY`；S训练/验证帧10,473/913，R训练/验证行3,178/301，6套R角色均只用train拟合归一化。太阳几何：R复算最大绝对差0，S复算最大绝对差3.58e-7，S/R重叠最大差0，目标时刻映射差0；test与buffer载荷读取均为0。
- `qstat -u slfu` 实时仍有3项运行：full `215311.tc6000`（node22，局部GPU4）、fold_0 `215312.tc6000`（node22，局部GPU3）、fold_1 `215313.tc6000`（node22，局部GPU1），均为单GPU。训练日志最新完整轮次分别为7/8/1；full当前最佳验证loss 0.2411956（epoch 7），fold_0当前最佳0.2556693（epoch 6），fold_1刚完成epoch 1（0.5966635）。验证loss仅用于训练选模，尚不是COT/GHI结论。
- 队列控制器PID `140574`仍存活；其依赖顺序为full及fold_0…fold_4完成并生成预测bank后，执行修正版forecast-COT共同支持评价、OOF来源审计和配对GHI阶段。当前COT/GHI结果标记尚未生成；等待空出的PBS名额，不超3项提交。
- 共同支持回归修复及本快照对应源码/收据已由提交 `e80ef76` 推送到 `origin/main`；修复不改模型、训练数据或训练合同。

## 实时运行核验（2026-10-21 05:18 CST）

- 再次读取当前PBS队列和控制器：`215311.tc6000`、`215312.tc6000`、`215313.tc6000` 均仍为R，控制器PID `140574`仍存活；没有重复提交或超出3项上限。
- GPU采样确认本实验的node22局部GPU4/3/1分别对应full/fold_0/fold_1；瞬时SM利用率分别为93%/24%/34%，显存约73.3/55.2/55.2 GiB。GPU3在前一次瞬时采样显示0%，复查时已恢复计算，未见卡死证据。
- fold_0 新完成第9轮，验证loss `0.2526335`，较其此前最佳0.2556693更优；full仍最新第7轮、最佳0.2411956；fold_1仍最新第1轮。近期日志没有Traceback/OOM。暂时仍没有SimVP预测bank或forecast-COT/GHI结果标记。
- 额外对远端 attempt12 整个代码目录运行 `swc/bin/python -m compileall -q`，结果 `COMPILEALL_PASS`；仅生成字节码缓存，无数据载荷读取或PBS提交。

## 实时训练推进（2026-10-21 05:20 CST）

- 队列复核仍为三项运行（`215311` full、`215312` fold_0、`215313` fold_1），控制器PID `140574`存活；剩余fold和下游任务尚未排入。未见训练异常或OOM。
- full新完成第8轮，验证loss `0.2359206`，优于第7轮的当前旧最佳0.2411956；fold_0仍最新第9轮、最佳0.2526335；fold_1新完成第2轮，验证loss `0.3582111`。这些仅为SimVP验证选模信号，不是COT/GHI增益证据。
- 当前没有S预测bank、forecast-COT评价或配对GHI结果标记；test/buffer仍关闭。

## GHI配对初始化复核及训练推进（2026-10-21 05:26 CST）

- 在服务器 `swc` Python环境用CPU和合成输入复核attempt12当前FA/FAC初始化代码：两臂初始state SHA相同（`e225d257...f2de70`），初始输出最大差0；FAC在初始化时对两组不同合成COT输入的输出差为0。说明预测COT臂从无COT共同预测起点开始，只有训练学习后才可能产生COT增量。收据 `receipts/attempt12/GHI_PAIR_INITIALIZATION_WITNESS_attempt12.json`；该见证不读取标签、不训练、不代表任何GHI成绩。
- 实时PBS仍为3项R（full `215311`、fold_0 `215312`、fold_1 `215313`），controller `140574`存活。full最新第8轮（验证loss `0.2359206`）；fold_0到第10轮（`0.2494573`）；fold_1到第3轮（`0.3121215`）。暂无训练错误、OOM、S预测bank或forecast-COT/GHI结果。
- 服务器端公平初始化见证所用 `train_paired_forecast.py` SHA-256：`3b2319b5c72f3abb00f003f8c228d3cdd6bef1ab39fb72724ae7e8fd7bec53ab`。本轮没有PBS提交；test/buffer继续关闭。

## 实时训练复核（2026-10-21 05:28 CST）

- 全量SimVP完成第9轮，验证loss回升至 `0.2471407`（第8轮最佳 `0.2359206`），训练loss仍降至 `0.1527656`；目前只有一次未改善，未达到patience5停止条件，继续按完整验证集保存/选择best checkpoint。
- fold_0最新第10轮验证loss `0.2494573`；fold_1最新第3轮 `0.3121215`。PBS `215311/215312/215313`仍为R，controller `140574`存活；错误检索未发现Traceback、OOM或CUDA error。
- 尚未生成任一S预测bank、forecast-COT评估或GHI配对结果标记，test/buffer继续关闭。

## 实时训练复核（2026-10-21 05:31 CST）

- PBS全队列仍有3项运行，controller `140574`仍存活，completed/failed阶段状态无变化；未产生预测bank或下游评估标记。
- full最新第9轮验证loss `0.2471407`，当前best仍为第8轮 `0.2359206`；fold_0最新第11轮验证loss `0.2555291`，当前best为第10轮 `0.2494573`；fold_1新完成第4轮，验证loss `0.2856804`，较第3轮改善。训练日志错误扫描无Traceback/OOM/CUDA错误。
- 这只是SimVP验证选模过程；几何修正是否改善预测COT，须等冻结R的forecast-COT验证评估，不以训练或AGRI验证loss替代。

## 实时训练推进（2026-10-21 05:36 CST）

- PBS队列仍有3项运行：full `215311.tc6000`、fold_0 `215312.tc6000`、fold_1 `215313.tc6000`；控制器PID `140574`存活，attempt12状态为`RUNNING_AUTHORIZED_STAGES`，`failed={}`。
- 最新轮次：full第10轮（训练loss `0.1404086`、验证loss `0.2368077`）；fold_0第12轮（`0.1270051`、`0.2569005`）；fold_1第5轮（`0.2747162`、`0.2795315`）。fold_1训练日志继续更新；node22瞬时GPU采样显示GPU1为100%利用率、55,213 MiB显存，故上一快照中未更新的epoch日志不是卡死证据。
- 本次日志扫描未发现Traceback、OOM、CUDA error或RuntimeError。`S_BANK_READY`、forecast-COT评估及GHI配对完成标记均尚未生成；当前仍处于SimVP训练阶段，不能据此声称预报COT或GHI已提升。
- 详细实时查询结果保存在`receipts/attempt12/LIVE_TRAINING_20261021T0536CST.json`。队列已满，控制器等待空位后继续提交后续fold；test/buffer继续关闭。

## 实时训练推进（2026-10-21 05:41 CST）

- 实时PBS仍是full `215311.tc6000`、fold_0 `215312.tc6000`、fold_1 `215313.tc6000`三项运行，控制器继续存活，attempt12无失败阶段。
- full推进至第11轮（训练loss `0.1325294`、验证loss `0.2402363`）；fold_0推进至第13轮（`0.1226863`、`0.2608019`）。两者验证loss高于各自当前最佳轮次，仍按完整验证集best-checkpoint与既定patience流程训练。
- fold_1最新仍为第5轮（训练loss `0.2747162`、验证loss `0.2795315`）。05:40:28对node22采样时，其GPU1利用率100%、显存55,213 MiB；full/fold_0对应训练进程CPU占用分别112%/116%，表明任务仍在计算而非退出。此次未发现Traceback/OOM/CUDA error/RuntimeError。
- COT预测评估与GHI配对阶段仍未开始，完整预测bank标记不存在。证据见`receipts/attempt12/LIVE_TRAINING_20261021T0541CST.json`；test/buffer关闭。

## 实时训练推进（2026-10-21 05:49 CST）

- full `215311.tc6000`到第12轮（训练loss `0.1270183`、验证loss `0.2380386`）；fold_0 `215312.tc6000`到第14轮（`0.1189363`、`0.2586390`）；fold_1 `215313.tc6000`到第7轮（`0.2143708`、`0.2590017`）。fold_1刷新当前最佳验证loss；full/ fold_0仍以此前最佳checkpoint为准。
- `qstat -u slfu`仍为以上3项运行；控制器PID `140574`存活，`failed={}`。full和fold_0各有4轮未刷新最佳验证loss；训练按现有patience继续，并未手动终止或改合同。
- 所有任务日志均未发现Traceback/OOM/CUDA error/RuntimeError。S预测bank、forecast-COT指标与GHI配对完成标记仍不存在；当前还没有预报COT精度或GHI收益结论。实时证据见`receipts/attempt12/LIVE_TRAINING_20261021T0549CST.json`。

## 实时训练推进（2026-10-21 05:54 CST）

- full `215311.tc6000`最新第12轮（训练loss `0.1270183`、验证loss `0.2380386`）；fold_0 `215312.tc6000`最新第15轮（`0.1149914`、`0.2589198`）；fold_1 `215313.tc6000`最新第8轮（`0.1881666`、`0.2721530`）。PBS实时队列仍是这三项运行，控制器`140574`存活，阶段状态无失败。
- fold_0第15轮完成后仍处于PBS运行状态，暂未出现早停/完成标记；不依据patience猜测终止或重提。fold_1日志仍正常按约6分钟一轮更新。日志扫描无Traceback、OOM、CUDA error或RuntimeError。
- 预测bank及COT/GHI结果标记仍缺；下游尚未开始。实时查询收据为`receipts/attempt12/LIVE_TRAINING_20261021T0554CST.json`，test/buffer继续关闭。

## 训练到验证预测bank阶段（2026-10-21 05:57 CST）

- 检查PBS日志后确认此前full/fold_0的训练进程已正常完成早停，不是挂起：full在第13轮触发patience5，best验证loss为`0.2359206`（epoch 8）；fold_0在第15轮触发patience5，best验证loss为`0.2494573`（epoch 10）。日志均明确写出`TEST_CLOSED: skipped`。
- 两个PBS任务仍为R，因为已切换到验证预测bank推理：full完整验证407个窗口，fold_0为OOF验证823个窗口。两目录已有`pred_agri_patch.npy`、`true_agri_patch.npy`、`future_geometry_patch.npy`，但只有部分输出文件且无bank-ready标记，仍需等待写入/校验完成。
- fold_1保持训练中，第9轮进度约80%（309 steps中的248 steps）；其job log实时更新。PBS队列仍是full `215311`、fold_0 `215312`、fold_1 `215313`三项，控制器PID `140574`存活、失败为空。
- 当前没有完整`S_BANK_READY`或forecast-COT结果，COT与GHI阶段尚未开始；不以部分bank推理或AGRI验证loss替代COT指标。详细证据见`receipts/attempt12/LIVE_TRAINING_20261021T0557CST.json`。

## 验证预测bank与时间键独立核验（2026-10-21 06:00 CST）

- full与fold_0的预测bank均已通过阶段成功标记并完成只读数组检查：full 407个验证窗口、fold_0 OOF 823个窗口，AGRI数组形状分别为`[407/823,16,13,16,16]`，未来太阳几何为`[407/823,16,3,16,16]`；预测、真值和几何数组所有元素均为有限值。
- 独立比较bank `keys.csv`与原始val index：full 407/407、fold_0 823/823，样本序号、split和24帧源路径逐行完全一致。逐窗口把24个源文件名时刻按UTC+8转换后，历史末帧等于key初始化时刻、16个后续帧逐个等于key目标时刻，lead正好为+15…+240分钟；两组总错位数0。
- 两个bank报告均为train-only归一化、原始反归一化SimVP输出（无裁剪/插值），并记录test/buffer载荷读取为0。队列中full与fold_0已完成；fold_1、fold_2、fold_3运行中，共3个未结束任务，控制器继续工作。
- 这是预测源与时标的质量核验，不是COT精度结果。仍等待fold_1…fold_4 OOF bank齐备后统一运行forecast-COT和GHI配对评估。机器复算结果见`receipts/attempt12/FORECAST_BANK_AUDIT_20261021T0600CST.json`。

## 2026-10-21 06:12 CST 实时训练快照

- attempt12 的太阳几何/时标审计已通过后，重训任务仍在 `zjnu-hpc` 的 node22 正常运行；PBS `215313.tc6000`（fold_1）、`215314.tc6000`（fold_2）、`215315.tc6000`（fold_3）各用1张卡，账户未结束作业为3/3，控制器PID `140574` 状态 `RUNNING_AUTHORIZED_STAGES`，`failed={}`。
- fold_1第11轮进度283/309步（第10轮验证损失0.2682438）；fold_2第1轮206/304步；fold_3第2轮32/308步（第1轮验证损失0.5543598）。node22进程与GPU占用查询证明训练仍在计算；未发现Traceback、OOM、CUDA error或RuntimeError。
- full和fold_0的预测bank已就绪并通过06:00逐样本键及+15…+240分钟时间核验；其余fold bank尚未完成。预测COT评价和有/无COT GHI配对头尚未开始，因此目前没有新的COT精度或GHI收益结论。test和buffer读取计数均为0。
- 实时证据：`receipts/attempt12/LIVE_TRAINING_20261021T0612CST.json`。

## 2026-10-21 06:28 CST：fold_1 验证预测bank通过独立核验

- fold_1 作业 `215313.tc6000` 已完成，797个OOF验证窗口的 `S_BANK_READY` 和 `BANK_REPORT` 齐备。新增只读审计逐项核对源index与bank键、UTC文件名+8小时BJT映射、+15…+240分钟时效，以及AGRI预测/真值和未来几何数组的形状、有限值及哈希，均通过；test/buffer读取计数为0。
- 当前已有 `full`（407窗）、`fold_0`（823窗）和 `fold_1`（797窗）三个完整bank。PBS控制器已在fold_1释放资源后提交fold_4 `215316.tc6000`；此时fold_2 `215314`、fold_3 `215315`、fold_4 `215316`共3个任务运行，未超过并发上限。fold_2/3第3轮验证损失分别为0.2764325/0.2906781；fold_4开始训练。日志扫描没有Traceback、OOM、CUDA error或RuntimeError。
- COT精度评估和配对GHI头仍待fold_2…fold_4预测bank齐备后启动，当前尚无预报COT或GHI提升结论。审计代码及机器输出见`audit_forecast_banks_attempt12.py`和`receipts/attempt12/FORECAST_BANK_AUDIT_20261021T0625CST.json`；实时队列快照见`receipts/attempt12/LIVE_TRAINING_20261021T0628CST.json`。

## 实时重训推进（2026-10-21 06:42 CST）

- 实时PBS队列为 fold_2 `215314.tc6000`、fold_3 `215315.tc6000`、fold_4 `215316.tc6000`，均在 `zjnu-hpc` node22 各用1张卡；账户未结束任务为3/3。控制器PID `140574` 仍存活，状态 `RUNNING_AUTHORIZED_STAGES`，已完成 `profile/R/full/fold_0/fold_1`，失败列表为空，test载荷读取计数为0。
- 最近完成验证轮次：fold_2 epoch 4，train/val loss `0.2776097/0.2604916`；fold_3 epoch 5，`0.2534178/0.2673567`；fold_4 epoch 1，`0.8353159/0.5377611`。三者仍在继续训练，以上仅是训练选模损失，不是COT或GHI成绩。
- 已就绪预测bank仍为full、fold_0、fold_1；fold_2…fold_4尚未产出完整bank，因此forecast-COT评价与有/无预测COT的GHI配对训练均未开始。本轮日志未检出Traceback、OOM、CUDA error或RuntimeError。
- 太阳几何、train/val数据恢复与时间键审计依赖已通过；本轮不改训练合同、不重提任务，控制器将在bank齐备后按既定顺序自动进入COT与GHI阶段。实时收据：`receipts/attempt12/LIVE_TRAINING_20261021T0642CST.json`。

## 训练推进（2026-10-21 06:52 CST）

- 06:51:49 实时核验：PBS `215314/215315/215316.tc6000`（fold_2/3/4）仍在 node22 各用1张卡运行；队列为3/3。控制器PID `140574`存活，completed 为 `profile/R/full/fold_0/fold_1`，failed 为空，test载荷读取为0。
- 最近完整验证轮次：fold_2 epoch 6，train/val loss `0.2166484/0.2615145`；fold_3 epoch 7，`0.1899431/0.2680219`；fold_4 epoch 2，`0.5774720/0.3480843`。采样时三者分别继续到第7轮49/304步、第8轮28/308步和第3轮286/308步。验证损失仅用于checkpoint选择。
- ready预测bank仍为 full、fold_0、fold_1；预测COT评价与配对GHI训练尚未启动。日志错误扫描未发现Traceback、OOM、CUDA error或RuntimeError。收据：`receipts/attempt12/LIVE_TRAINING_20261021T0652CST.json`。

## 实时训练推进（2026-10-21 06:58 CST）

- 06:58:13 PBS复核：`215314/215315/215316.tc6000`（fold_2/3/4）仍在 node22 各用1张卡，队列为3/3；控制器PID `140574`存活，完成 `profile/R/full/fold_0/fold_1`，失败为空，test载荷读取为0。
- 最新完整验证轮次：fold_2 epoch 7，train/val loss `0.1868047/0.2495359`；fold_3 epoch 8，`0.1694441/0.2677006`；fold_4 epoch 3，`0.3610181/0.2700573`。采样时分别推进到第8轮0/304步、第9轮26/308步、第4轮278/308步。这些是AGRI模型选模损失，不是预报COT指标。
- ready预测bank仍为full、fold_0、fold_1；COT评价与GHI配对仍未开始。未发现Traceback、OOM、CUDA error或RuntimeError。详见`receipts/attempt12/LIVE_TRAINING_20261021T0658CST.json`。

## 2026-10-21 07:23 CST：提前验证 forecast-COT 并修正下游接口

- full-train SimVP验证bank（407窗）和corrected-R full checkpoint已就绪，因此在不占训练GPU的单线程CPU上，隔离生成FP32 forecast-COT预览bank并运行正式forecast-COT评价器。评价器通过：112个lead/scope/station指标行、1,164个精确配对的`(init,target,lead)`键；CPP common-valid support下全时效共297,604个有效16×16像元记录。test/buffer读取均为0。
- 预览全时效像元加权COT RMSE：SimVP预测AGRI→R为10.564，历史COT持平为9.229，离线真实未来AGRI→R为8.684。三站站点像元forecast/persistence RMSE分别为牛首12.118/9.512、王寨11.903/9.547、米庄10.659/9.090。该预览指向当前预测链仍较差，不能写成COT或GHI增益；CPP只是检索参考，且正式GPU FP16评估尚待完整bank齐备后复算。明细和解释见`receipts/attempt12/FORECAST_COT_PREVIEW_CPU_attempt12.md`及同名JSON/CSV。
- 预览暴露了正式下游调用的目录接口错误：评价器需要`R_runs_attempt12/full/{val_keys.csv,...}`，但原调用传入`R_runs_attempt12/full/val`，曾查找不存在的`full/val/val_keys.csv`。已把`run_stages.py`的`--r-validation`改为R运行根目录；原版已留存，修正版已同步服务器。两端`py_compile`通过；按修正参数重跑评价器通过（112行、1,164配对键），因此不会等到GHI阶段再失败。修复收据：`receipts/attempt12/RUN_STAGES_PATH_FIX_attempt12.json`；源文件原/新SHA-256分别为`f34c8762…e49402f6`和`23e08552…95a2146d`。
- 07:22实时PBS仍为fold_2/3/4三项运行（`215314/215315/215316.tc6000`，node22各1张卡）；controller `140574`存活、failed为空。最新验证轮次为fold_2 epoch10、fold_3 epoch11、fold_4 epoch7；三个预测bank尚未ready。未发现Traceback/OOM/CUDA error。正式GPU FP16 COT评价与配对GHI训练仍待这些OOF bank齐备后自动推进。

## 实时训练状态（2026-10-21 07:26 CST）

- 训练/验证几何审计通过：10,473个AGRI样本帧（train 9,560、val 913）与3,479条R样本（train 3,178、val 301）逐键核对；R几何最大差0，S几何最大差3.58e-7，目标时间差0，train/val无交叉。独立NOAA公式复核覆盖62,802个时空点（train 57,334、val 5,468），天顶角最大误差1.75e-13°、方位角最大环形误差2.72e-10°。
- corrected-R、full SimVP及fold_0/1阶段已完成；fold_2/3/4在node22各用1张GPU继续训练（PBS 215314/215315/215316.tc6000），控制器PID 140574仍运行，当前3/3并发，无失败；最近完整epoch分别为10/12/8。训练loss仅用于AGRI checkpoint选模。
- 正式forecast-COT最终评价和三站配对GHI头尚未启动；需等fold_2/3/4预测bank完成后按控制器顺序继续。test/buffer读取仍为0。实时收据：receipts/attempt12/LIVE_TRAINING_20261021T0726CST.json。

## 实时训练状态（2026-10-21 07:35 CST）

- PBS 215314/215315/215316.tc6000（fold_2/3/4）均在node22运行，各用1张卡；控制器PID 140574存活，3/3并发，failed为空，test读取0。最近完整epoch为fold_2=12、fold_3=13、fold_4=9；尚无fold_2…4预测bank，因此正式COT评价和配对GHI训练仍未开始。

## 2026-10-21 实时重训推进（服务器 07:45 CST）

- 训练集/验证集数据恢复和太阳几何审计已通过：S 10,473帧（train 9,560、val 913），R 3,479行（train 3,178、val 301）；R几何逐行差0，S最大差3.58e-7，目标时刻差0、train/val不交叉。独立NOAA公式复核62,802个时空点通过（天顶角最大误差1.75e-13°，方位角环形误差2.72e-10°）。
- 已经按修正后的数据和几何开始重训，不在等待：corrected R、全量SimVP及fold_0/1完成；PBS控制器PID 140574仍运行，attempt12状态`RUNNING_AUTHORIZED_STAGES`，失败列表为空，test载荷读取0。
- 当前PBS占用3/3：215314（fold_2，node22，1 GPU）训练已在第13轮按patience正常停止，正在完成OOF预测bank；215315（fold_3，node22，1 GPU）最新完成第15轮；215316（fold_4，node22，1 GPU）最新完成第10轮。现有完整S bank为full 407窗、fold_0 823窗、fold_1 797窗；fold_2/3/4的bank报告尚未齐备。
- 正式forecast-COT评价及FA/FAC配对GHI头尚未开始。控制器将等三个OOF bank生成后先做COT验证评分和来源审计，再自动进入三站配对GHI训练；无需再次确认，也不会另行重复提交。
- 本次实时收据：`receipts/attempt12/LIVE_TRAINING_20261021T0745CST.json`。此前CPU预览只用于提前诊断，不替代正式GPU/OOF结果；目前不能宣称修正后COT或GHI已有提升。

## 2026-10-21 实时推进（服务器 07:47 CST）

- 215314（fold_2）已完成第13轮训练并进入850个OOF验证窗的预测推理；`pred_agri_patch.npy`、`true_agri_patch.npy`、`future_geometry_patch.npy` 已出现，但 `BANK_REPORT.json` 与 `S_BANK_READY` 尚未生成。215315（fold_3）最新完成第15轮，215316（fold_4）最新完成第11轮。三项仍在node22运行，队列3/3；控制器PID 140574存活，状态`RUNNING_AUTHORIZED_STAGES`、failed为空。
- 对CPU FP32预览CSV按时效复核：forecast COT RMSE点估计在16个时效中有15个高于历史持平；日期块区间明确高于0的时效为+15、+30、+45、+75和+240 min。多数15分钟时效仅覆盖4个目标日期；+60/+120/+180/+240覆盖14–16日。因此这是定位问题的先导诊断，正式GPU/OOF结果仍待bank齐备，不能当作最终结论。
- 修正预览报告中的单位笔误：COT RMSE及其差值为无量纲，不是W m⁻²；数值未改。更新文件`receipts/attempt12/FORECAST_COT_PREVIEW_CPU_attempt12.md`。
- 本次状态收据：`receipts/attempt12/LIVE_TRAINING_20261021T0747CST.json`。

## 2026-10-21 07:52 CST 实时推进与 fold_2 bank 审计

- 修正后的 train/val 太阳几何与数据键审计已通过；attempt12 重训继续自动推进，无需重复提交。PBS 控制器 PID `140574` 存活，状态 `RUNNING_AUTHORIZED_STAGES`，失败列表为空，test payload 读取计数为 0。
- fold_2 `215314.tc6000` 已以 `exit_status=0` 完成，850 个 OOF 验证窗口的预测 bank 已生成。独立审计覆盖 full、fold_0、fold_1、fold_2 四个 bank，结果 `PASS_PARTIAL_CORRECTED_SOLAR_BANKS`：源索引与 bank 键逐行相符、UTC→BJT与 +15…+240 min 键一致，预测/真值/几何数组形状正确且有限；test/buffer读取均为 0。机器审计见 `receipts/attempt12/FORECAST_BANK_AUDIT_PARTIAL_20261021T0750CST.json`。
- 当前 fold_3 `215315.tc6000` 在 node22 单卡运行至 epoch 16（train/val loss `0.1123414/0.2782141`）；fold_4 `215316.tc6000` 在 node22 单卡运行至 epoch 12（`0.1233377/0.2478260`）。PBS保留表中另显示fold_2为C，实际未结束任务为2项，低于3项上限。当前日志无训练异常。
- 还需 fold_3、fold_4 bank完成后才运行完整forecast-COT评价；后续配对GHI训练仍未启动，控制器将按依赖自动提交。训练轮次损失不代表COT/GHI精度收益。
- 实时收据：`receipts/attempt12/LIVE_TRAINING_20261021T0752CST.json`。

## 2026-10-21 08:01 CST：太阳几何复核通过，修正后训练继续

- 训练集和验证集太阳几何核对已完成：NOAA 官方几何角公式复核覆盖 train 57,334 点、validation 5,468 点；天顶角最大差 `1.75e-13°`，方位角最大环形差 `2.72e-10°`。修正数据包审计覆盖 S 10,473 帧、R 3,479 行，键/标签/AGRI/几何一致且 train/val 不交叉；test/buffer 读取为0。
- 修正后的 SimVP 重训确已启动并在推进。PBS `215315.tc6000`（fold_3，node22，1 GPU）已正常完成，写出 809 个验证窗口的预测 bank；`215316.tc6000`（fold_4，node22，1 GPU）仍运行，最新第13轮 train/validation loss 为 `0.1169960/0.2502225`。当前实际未结束作业 1 个，低于3个上限；控制器 PID `140574` 存活，`failed={}`。
- 尚未完成的部分：fold_4 预测 bank尚未就绪，因此正式 forecast-COT 评价及三站有/无预测COT配对 GHI 头训练还没有开始。控制器将继续按依赖推进；当前日志未发现训练错误。训练损失不是 COT/GHI 精度结论。
- 实时收据：`receipts/attempt12/LIVE_TRAINING_20261021T0801CST.json`。

## 2026-10-21：从已有预览定位 forecast-COT 的薄弱时效

- 对现有 validation CPU FP32 预览按1小时分组，比较 SimVP→R 与历史COT持平的同键、同CPP参考支持误差。预测链的 lead-wise ΔRMSE（forecast−persistence）在全图为1–4小时 `+1.830/+0.568/+0.679/+3.073`；站点像元均值为 `+2.665/+1.264/+1.682/+4.114`。正值表示forecast更差，COT RMSE无量纲。
- 16个时效中全图有15个误差更高；站点像元和3×3邻域各有47/48个站点-时效比较更差。4小时段退化最大。这把下一步关注点收敛到COT外推链在站点邻域和长时效上的误差，但该汇总只是full-train模型的CPU预览，不能取代修正几何下的OOF正式评估，也不能单凭此判定是云团位移、增厚/消散还是R反演误差。
- 复核报告：`receipts/attempt12/FORECAST_COT_PREVIEW_HORIZON_DIAGNOSIS_attempt12.md`。正式评价继续等待fold_4 bank；不在OOF结果出炉前另起重复训练。
