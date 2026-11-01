# 2026-10-17 补充进度：PBS fold 输入准备

## 2026-10-17 03:32 BJT：折2进入Epoch 2；PBS环境依赖盘点

- FD-201服务器UTC 19:31:28：折2训练进程/三rank仍存活，已完成Epoch 1验证，日志记录`val_loss=0.55632`（best），随后进入Epoch 2；当时约25/282 batch。相较Epoch 0的`0.68810`验证损失有下降，但这是训练期SimVP云场指标，不能作为COT/GHI结果或测试结论。
- 折1全797条OOF预测仍运行于物理GPU3；三个memmap文件时间戳持续更新，日志仍标`train=0,val=797,test=0`，但尚无`keys.csv`、`BANK_REPORT.json`或完成/审计回执，不能消费该bank。未读取数组内容。
- 为评估利用学院PBS并行资源的可行性，检查`zjnu-hpc`的`swc`环境（Python 3.10）：当前仅确认torch和numpy可导入，`fvcore/iopath/yacs/pytorch_lightning/timm/cv2/omegaconf`均未安装；FD-201的完整swc环境及湖北工程目录在PBS登录节点不可见。PBS队列`slfu`仍有`214221.tc6000`一个Q态任务。故本轮未把未满足环境条件的折训练提交到PBS，也未改共享环境。
- 后续优先让FD-201上的OOF推理及折2自然完成；同时把PBS环境隔离修复作为并行准备项，先在用户私有路径建立可复现依赖bundle并通过导入/真实batch烟测，再考虑提交调度任务。任何继续并行只使用确认为本项目已分配的GPU，不占用其他账户进程。

## 2026-10-17 03:28 BJT：折1 OOF推理与折2训练继续推进

- FD-201 Luna只读快照（服务器UTC 19:27:52）：折1 OOF bank 主进程及两个DataLoader worker仍存活，使用物理GPU3；三个预分配memmap文件的mtime更新至UTC 19:27:48，但`keys.csv`、`BANK_REPORT.json`、exit code和审计回执仍未生成。当前日志确认索引为`train=0, val=797, test=0`；此处`val`是折外训练日期索引中的分区名，不是固定validation集合。未读取预测数组内容。
- 折2训练仍在FD-201物理GPU0–2运行，Epoch 1约185/282 batch，三rank存活，采样利用率约96/99/98%；Epoch 0指标为`val_loss=0.6881033`、`val_mae=0.6226599`，Epoch 1验证未结束。以上仅是AGRI场预测模型训练过程，不是COT/GHI技能结果。
- 本轮只追加状态记录；未登录未分配节点、未停止/重启现有进程、未读取test/buffer载荷，也未保存或复述用户提供的认证口令。下一步待OOF推理完整生成键表与报告后运行本地bank审计；在所有折外预测和FA/FAC前置门禁完成前不训练GHI头。

## 2026-10-17 03:20 BJT：fold1 OOF推理启动，fold2继续训练

- 复核 FD-201：fold2 的 SimVP S-fold 训练仍在物理 GPU0–2上运行；启动时通过fold-specific索引、train-only norm和阶段数据SHA回执门禁。最近日志为Epoch 0约78%（219/282 batch），三rank均存活，未见错误/OOM；该状态是训练过程，不是预测技能结论。
- fold1 的407条固定validation checkpoint复载审计此前已通过；本轮改用折外索引 `simvp_oof_index.csv`（797条，全部标记为该折索引的val集合、属于冻结训练日期），对应best checkpoint SHA-256=`e95a54f44c8d6e5b14935270f616e7a66c9d00921ef2c2ae4d9d2ebeac53871a`、norm SHA=`84c215afec383b4b85326c9283d64c02ae56e707129064df1dc2e80f8f74ac61`、index SHA=`b83386e2811991b4919d63f5f221f346d3acc82e1671da08eed91e14bf3d8f4a`。不把这797条和固定验证集混为一谈。
- fold1 OOF推理脚本先在物理GPU3完成2样本烟测：输出张量维度、数据重基址与train-only统计读取成功，报告中`test_payloads_read=0`；烟测全场normalized MSE/MAE仅作程序运行见证，不作为折外技能结论。随后已在FD-201 screen `hubei_s_oof_fold1_bank_20261017` 启动全部797条折外预测，输出目标为`/home/Data_Pool/zjnu/hubei_cot_independent_20261016/oof_training/S_oof_fold1_oof_bank_20261017_fd201_gpu3`，物理GPU3目前已分配给推理进程；尚未完成/尚无bank回执。
- 03:25 BJT只读刷新：fold1推理主进程及两个dataloader worker存活，显存约4.6 GiB；主进程累计读取约22.1 GB数据字节，输出预测张量文件已创建且仍在写入，尚无`keys.csv`或`BANK_REPORT.json`。fold2已进入Epoch 1、约108/282 batch，三rank存活且日志持续更新。FD-107可用密钥查询到六卡均已有显存进程（即使两卡瞬时util为0也不视为空卡）；FD-172当前密钥认证未通过。本轮未使用额外口令发起登录，未在未经调度的其他节点启动进程。
- 新增bank完整性审计脚本`tooling_20261017/audit_hubei_simvp_oof_bank_20261017.py`并已本地语法检查通过；它只核验折外索引、输出键、shape/有限值和`test_payloads_read=0`，不读取GHI标签。待完整bank生成后在FD-201运行并保存审计JSON。

### 03:27 BJT只读快照

- fold2已完成Epoch 0完整validation：`val_loss_epoch=0.6881033`、`val_mae=0.6226599`；当前Epoch 1约122/282 batch，torchrun与三rank仍存活，无错误/OOM。仅为场模型阶段指标，不作COT/GHI技能结论。
- fold1完整OOF bank仍在GPU3推理中，三输出memmap文件持续更新，进程约6分钟；无`keys.csv`、`BANK_REPORT.json`或退出码，故不能宣称完成。
- FD-107实时卡检查仍见六卡均有显存上下文，GPU4/5即使瞬时util为0仍视为已被占用；FD-172仅密钥认证未通过。没有在其他节点启动计算。
- 该OOF索引覆盖test/buffer数值载荷为0。推理沿用fold1冻结checkpoint、norm和16通道SimVP合同；没有改正在训练的fold2、没有使用其他用户GPU，也未读取/写入共享凭据。
- 下一步：等待GPU3的797条预测结束后跑bank键唯一性、shape、finite和test/buffer读取审计；fold2继续自然训练。完成fold1 bank与fold2训练/固定validation复载后，再按冻结分折合同排后续OOF折；不启动GHI头，直到全折OOF bank汇总和FA/FAC输入门禁通过。

# 2026-10-17 PBS主机UTC 18:37：train/val完整性验收仍在读取

- PBS `tc6000`现场快照：验证器PID 71073存活，累计运行16:51；`/proc/71073/io`的`rchar=59,320,179,054`，相较UTC 18:32:38约41.2 GB继续增长。尚无`stage_verification_receipt.json`，所以fold2–4的训练启动门禁仍未通过。
- `qstat -u slfu`当前仅`214221.tc6000`处于Q状态，另无未终态任务；本轮未提交PBS任务。train/val验证未通过之前不在任何节点绕开输入门禁启动后续折。
- 用户提供的非FD-248节点认证信息未写入文件；仅将已授权的只读核验用于服务器访问，不在工程记录中保存凭据。
- 下一步：等待当前单一验证器完成；一旦生成回执，独立核验23,468个train/val文件的大小与SHA、输入清单和状态，再运行fold输入准备器。之后仍按冻结合同提交后续训练并在每次提交前后查全账户PBS队列。

# 2026-10-17 train/val门禁通过；fold2已在FD-201进入启动阶段

- PBS端验收器完成：`stage_verification_receipt.json`状态为`PBS_TRAINVAL_STAGE_SHA256_AND_INDEX_REBASE_PASSED_TEST_CLOSED`，逐文件核验23,468个train/validation payload、66,508,197,403 bytes；test sequences=0、test/buffer payloads staged=0。source manifest SHA-256=`73e0f91955a7e8c07d69a1e6d79c900275ff000f350a826fb2fa04e79e47571a`；重基址后的主索引SHA-256=`5e2d7c1f8b1605fa15365c8eb3319fa4124141c3dc1e1d20d38cd4527d6d4644`。回执SHA-256=`7c122891d4332582e5f13efb6442eab79f1fda231adfe051add033532d18fe1e`。
- Fold2–4的PBS-path输入准备器通过，receipt状态`PBS_FOLD_INPUTS_REBASED_TEST_CLOSED`，各折索引只含train/val。receipt SHA-256=`59e66c6dbc7eea790dde0e7ef2b55025c12338b9089488c24cc29c0e2d0ae05d`。为每折补齐的train-only norm均与FD-201源文件SHA一致；prepared fold2为3,643条train/val样本。
- PBS单折脚本首次提交为`214512.tc6000`，因PBS中的旧SimVP checkout缺少`tools/train.py`而在5.5秒以exit_status=1结束；未进入模型训练、未产出GPU训练结果。检查后发现PBS `swc`环境还缺`fvcore`，所以没有继续改动公共环境或盲目重提。为不污染旧checkout，已将FD-201与PBS不同的官方OpenSTL源文件复制到隔离bundle `control/simvp_code_fd201_20261017/`；83个Python文件的源/副本哈希逐项一致，manifest SHA-256=`bb1d01c29b96673e233d619b471a55c2ca8c680c046f239a44a1dba44448261c`。此bundle在PBS当前环境尚未通过导入烟测（`ModuleNotFoundError: fvcore`），未用于训练。
- 同一逐文件验收器在FD-201对本地`/tmp/zjnu_hubei_simvp_stage_20261016`独立完成校验：23,468个文件、66,508,197,403 bytes、manifest哈希与PBS一致；test/buffer payloads=0。镜像验收脚本保留了原工具的`PBS_...`状态字符串，但receipt中的实际`stage_root`明确为FD-201本地`/tmp`路径；该receipt仅证明此本地stage的完整性。
- 使用已通过fold0/1实测的FD-201运行环境，fold2已按冻结合同启动：screen `hubei_s_oof_fold2_20261017`，torchrun PID 1496083，三个rank启动后仍在NFS RPC等待（`rpc_wait_bit_killable`）；截至本快照尚未出现训练日志或GPU显存分配，因此标记为“启动中/存储等待”，不是已训练。物理GPU 0–2已分配给该run；不重启、不抢占，先查明共享存储等待并观察是否自然恢复。run manifest记录seed42、每rank batch4、world size3及配置/索引/norm/trainer哈希。
- 当前PBS队列复核：`214221.tc6000`为Q，`214511.tc6000`为C，`214512.tc6000`为C；未在修复环境前再提交。FD-107六张A6000当前均有其他作业占用。FD-201 GPU3保持空闲作为资源余量；FD-201 GPU4/5属于其他进程，不触碰。
- 下一步：只读诊断fold2的NFS等待，等rank进入训练并核对实际GPU绑定/日志；若PBS环境可在不改共享环境的情况下补齐依赖并通过GPU smoke，再将后续折分散到PBS，仍保持每折3-rank与总GPU不超过8。队列每次qsub前后重查，test/buffer继续关闭。

# 2026-10-17 PBS主机UTC 18:26：train/val逐文件SHA验收仍在运行

- 在PBS登录节点`tc6000`复核：验收器PID 71073仍运行，启动后约5分56秒；`/proc/71073/io`累计`rchar=11,977,791,086`，较18:24:46的6,866,636,654持续增长。全量stage SHA验收尚未产出`stage_verification_receipt.json`，也未产出`pbs_fold_input_receipt.json`；因此尚不能准备/启动fold2–4训练。
- 验收命令只校验冻结manifest列出的train/validation payload（23,468个文件），test/buffer载荷仍关闭；当前未重启或并行重复哈希该任务。
- PBS全账户队列：`214221.tc6000`仍为Q，`214510.tc6000`为C；非终态1项，未提交新作业。满足stage验收后，按GPU总数不超过8且PBS未结束任务不超过3的规则，先排fold2/3的6卡作业；fold4保持3卡合同，待有足够资源再提交，不拆改rank/world size。
- 下一步：继续观察同一验收进程；回执出现后核对状态、manifest及文件数/字节数，再运行fold输入准备器并校验新回执。训练提交前后重新查询全队列。

# 2026-10-17 PBS主机UTC 18:16:44：stage仅剩41个data文件

- PBS stage目录65,536,131,746 bytes，data有11,693/11,734个NPY；最后一条tar接收流仍存活。相较PBS自报UTC 18:14:57，增加约219 MB和42个文件。暂不以目录总字节数作为合格依据，因为它已接近/略高于预期payload总量，但计数尚未达到manifest的精确目标。
- slfu队列保持214221（Q）、214510（R），2个非终态。接收结束后将对manifest中23,468个train/val文件逐个校大小和SHA，只有验收成功后才生成receipt和启动fold作业。

# 2026-10-17 PBS主机UTC 18:12:56：仅最后一路data分片在传

- PBS stage目录为66,045,942,818 bytes，data下11,600/11,734个NPY；最后一路tar receiver（对应source `data-2.list`）和FD-201 sender仍存活，其余三路已退出。余134个data文件，接收仍有增长，等待自然完成，不重启/中断。
- PBS队列仍为214221（Q）、214510（R），2项非终态；stage及fold receipt仍不存在。收到所有文件并确认发送/接收进程退出后再跑完整大小与SHA验收、fold收据和后续训练提交。

# 2026-10-17 PBS主机UTC 18:11：分片接收在收尾

- PBS stage为65,678,298,274 bytes，data目录11,531/11,734个NPY；相对该主机UTC 18:10:06增加约387 MB和67个文件。tar接收进程从4路降至2路，另外2路已结束；stage receipt仍未生成，不能启动折训练。
- `qstat -u slfu`仍为214221（Q）、214510（R），2个非终态任务。剩余203个data文件到齐后，等待最后接收进程退出，再做全量哈希验收；不以单纯目录大小/文件计数代替校验。

# 2026-10-17 控制器UTC 18:28：train payload传输接近完成

- PBS主机自报UTC 18:10:06：stage目录65,291,061,538 bytes，`data/`下11,464/11,734个NPY；与该主机18:08:49快照相比增加约761 MB和149个data文件。四路tar receiver仍存活，剩余270个data文件；未见stage/fold receipt，仍不启动依赖该stage的作业。
- slfu全账户PBS队列仍为214221（Q）和214510（R），共2个未结束任务，保留1席。待接收进程自然退出且源目标文件全部到齐后，立刻跑冻结stage逐文件SHA/size验收和fold manifest验收，再提交下一折。

# 2026-10-17 控制器UTC 18:26：PBS传输仍在增长

- PBS主机自报UTC 18:08:11：train/val stage目录64,170,396,834 bytes，`data/`含11,244/11,734个NPY；同一快照四个tar receiver均存活。相较Luna于18:07:57观察到的11,217个data NPY，约14秒增加27个文件。全量stage校验receipt仍未生成，当前约缺2.34GB/490个data文件，不能在PBS启动依赖此stage的折训练。
- `qstat -u slfu`有2个非终态任务：214221（Q）和214510（R）；提交资源限额余1席。没有提交新任务。
- FD-201 fold1四分片validation checkpoint复载审计已于服务器自报UTC 18:24通过，详情见下方记录。当前继续等待PBS传输完成；结束后逐文件验收train/val payload、完成fold索引重基址和输入receipt，再按队列容量启动后续折。

# 2026-10-17 02:24 BJT：fold1复载验证审计通过

- FD-201 主机自报UTC 18:24，fold1 best checkpoint 复载后的4卡验证推理已结束，独立校验器通过：407/407个冻结验证键完整且唯一，四分片分别102、102、102、101条；checkpoint、train-only norm、冻结index及train/val stage manifest哈希与预期一致。
- 审计状态为`HUBEI_SIMVP_FOLD1_CHECKPOINT_RELOAD_VALIDATION_AUDIT_PASSED_TEST_CLOSED`；test/buffer payload读取均为0。全256×256场、13个AGRI通道、16个lead上的冻结归一化空间指标为MSE 0.26769、MAE 0.31982（共5,548,015,616元素）。这只验证S模型全场验证复载与样本覆盖，不是站点GHI精度或论文最终结果。
- 审计报告：`/home/Data_Pool/zjnu/hubei_cot_independent_20261016/oof_training/S_oof_fold1_val_reload_audit_fd201_20261017/FOLD1_VALIDATION_RELOAD_AUDIT.json`。数据来源stage manifest SHA为`73e0f91955a7e8c07d69a1e6d79c900275ff000f350a826fb2fa04e79e47571a`，折1 best checkpoint SHA为`e95a54f44c8d6e5b14935270f616e7a66c9d00921ef2c2ae4d9d2ebeac53871a`。
- 首次自动审计调用因shell将多目录参数重复传递，审计器按单目录解析而拒绝；未改动或重跑推理。随后以正确分片目录参数独立重跑，407键审计通过。已修复wrapper、通过`bash -n`并同步FD-201；本地/服务器脚本SHA一致（`1a52c109fe461dc8f7f22ed30154913f96094e9c74be9d09d6ae8d6a71382e3b`）。
- 下一步：确认PBS train/val stage传输全量到齐并完成逐文件大小/SHA验收与fold输入receipt；在这些门禁通过后，按冻结计划用PBS启动fold2/3及fold4，不从本次验证审计结果推导GHI效果。全账户PBS队列提交前后都需核查，test/buffer仍关闭。

# 2026-10-17 01:48 BJT：fold1审计输入已安全暂存

- PBS主机自报01:48:04 BJT：全场train/val stage为52,909,685,184 bytes，`data/`下9,096/11,734个NPY；4个接收tar仍存活。预期payload 66,508,197,403 bytes，尚缺约13.60 GB；`stage_verification_receipt.json`仍不存在，故不得在PBS使用该stage。
- 已将fold1的train/val索引、train-only norm、fold元数据报告、配置及早停best checkpoint从FD-201暂存至PBS stage的`control/fold_1/`。五项文件均已完整传输；关键哈希与FD-201源记录一致：index `5d436d6ebc4c3a6d622d2288afb84e1770f8977451d80c3f7fc5e1f2922a5d17`、norm `84c215afec383b4b85326c9283d64c02ae56e707129064df1dc2e80f8f74ac61`、config `04c69ff8d802c97db896d13ed912c38788eceb3235dbb155f1c8ba6f66e2ba9d`、best checkpoint `e95a54f44c8d6e5b14935270f616e7a66c9d00921ef2c2ae4d9d2ebeac53871a`。完整文件清单也含在目标目录，传输过程未读test/buffer载荷。
- PBS全账户队列仍只有`214221.tc6000`（Q）一个未结束作业。没有提交作业；fold1 checkpoint仍须在stage全量SHA验收后通过PBS复载验证，才算审计完成。

# 2026-10-16 17:43–18:01 UTC（PBS / FD-201各自时钟）：stage传输仍在推进，fold1已早停待复载

- PBS主机自报UTC 17:43:37（BJT 01:43:37）：train/val stage为50,324,697,477 bytes；`data/`下8,630/11,734个NPY。四个接收`tar`进程仍存活；与UTC 17:43:05快照相比，stage增加约308 MB、58个数据文件。预期payload 66,508,197,403 bytes、23,468文件，尚未到齐；stage校验与fold输入receipt仍不存在，不能启动依赖该stage的训练或推理。
- `qstat -u slfu`仍只有214221.tc6000（Q）一个未结束作业；未提交新作业。
- Luna对FD-201的现场检查（服务器自报UTC 18:01:55）：S OOF fold1在Epoch 13、global step 3850后因连续5轮验证无改善早停，best验证loss 0.264；进程已退出，best/last checkpoint已写入。未发现fold1的全量冻结validation checkpoint-reload审计产物。该checkpoint目前只证明训练早停完成，不是经过复载审计的OOF bank，也不是GHI技能结论。
- 用户提供了非FD-248节点通用密码；仅作为后续必要的SSH认证选项，不记录凭据。本轮未用其尝试无关登录，也未在任何服务器直接启动GPU进程。
- 下一步：等待/核实传输完成；全量验收通过后先完成S fold1冻结validation复载审计的PBS作业安排，再按已批准fold2/3与fold4合同提交作业。每次qsub前后实时复核slfu全队列，继续保持test/buffer payload关闭。

# 2026-10-17 01:54 BJT：确认stage传输仍在前进

- PBS主机UTC 17:36:45：stage目录46,368,199,173 bytes，`data/`下7,876/11,734个`.npy`；相较PBS主机UTC 17:34:44增加1,132,950,400 bytes和216个data文件。4个receiver仍存活，传输未完成。
- `stage_verification_receipt.json`与`control/pbs_fold_input_receipt.json`均不存在；全量输入大小/SHA校验及索引重基址门禁尚未通过。PBS队列仍仅`214221.tc6000`（Q）。因此不提交折训练。
- 继续等传输结束后立即执行全量验收与fold输入收据生成；fold1仍按固定三rank合同自然运行，不改变卡数。

# 2026-10-17 01:53 BJT：stage继续增长

- 本机UTC 17:53:22；PBS主机时钟显示UTC 17:34:44。PBS stage为45,235,248,773 bytes，`data/`下7,660/11,734个`.npy`；相较PBS主机UTC 17:32:48快照，约116秒增加1.10 GB和209个data文件。4个tar receiver仍在运行，源清单仍目标66,508,197,403 bytes；未齐，不验收、不训练。
- PBS全账户队列保持`214221.tc6000`（Q）一项未结束任务；当前剩余名额不改变数据门禁。
- Luna只读FD-201快照UTC 17:52:10：fold1 Epoch 13、185/287 batch；GPU1–3各约19 GB，无checkpoint更新或完成标记，日志未见错误。GPU0空闲但不改变冻结的每折3-rank合同。
- 下一步继续传输；全量结束后运行大小与SHA校验、fold索引重基址及输入收据，再逐次提交后续PBS折并逐次复查队列。

# 2026-10-17 01:52 BJT：继续传输与资源快照

- PBS UTC 17:32:48：`hubei_independent_trainval_stage_20261017` 当前目录 44,138,365,445 bytes，`data/` 下 7,451/11,734 个 `.npy`；4 个 tar 接收进程仍存活。目标 payload 为 66,508,197,403 bytes，未全量到齐、未完成逐文件大小/SHA 验收，禁止生成验收收据或启动依赖该stage的训练。
- slfu PBS 队列仍仅有 `214221.tc6000`（Q，未结束）；其余两席可用，但继续等stage验收后再按已审计顺序逐项提交。
- FD-201 UTC 17:52:10 的Luna只读快照：fold1为Epoch 13、185/287 batch，3个rank持续运行，GPU1–3各约19 GB；未见完成标记、checkpoint更新或日志报错。GPU0物理卡采样空闲，但后续S折仍须遵守已冻结的每折三rank训练合同，不擅自改变world size或抢占当前折资源。
- FD-107 UTC 17:51:38：6张A6000均有显存占用（约13.9–41.5 GB），无可确认空卡；未启动新进程。
- 本轮仅核查状态；未修改服务器作业、未启动/停止进程，密码未写入工程文件。下一步等待train/val stage传完，先跑全量验收器，再依次使用PBS空余名额启动fold2/3及fold4。

## 01:50 BJT 传输继续推进

- PBS UTC 17:30:24：stage目录42,766,870,149 bytes，data目录7,189/11,734个NPY；相对UTC 17:29:54增加约281 MB和54个NPY。4路receiver仍在工作，未达到全量验收条件。
- slfu队列仍仅有214221(Q)未结束；FD-201 Luna UTC 17:48:59快照显示fold1已进入Epoch 13、73/287 batch，日志于17:48:56更新，三个rank仍在GPU1–3（约19 GB/卡）。未见完成/退出标记或错误；这是AGRI场预报S训练，不代表GHI技能。
- FD-248仅尝试已有SSH公钥认证（未尝试猜测/复用密码），返回`Permission denied (publickey,password)`；当前没有可用凭据，未核查该节点。FD-107、FD-172、FD-201及学院PBS登录均已完成授权范围内的只读资源检查。

## 01:30 BJT stage持续增长

- PBS UTC 17:29:54：stage目录42,486,178,821 bytes，data目录7,135/11,734个NPY；相比UTC 17:27:57增加约1.10 GB和209个data文件，接收仍在推进。全量payload预期66,508,197,403 bytes、23,468个文件；不足前禁止生成验收收据/开训。
- slfu队列仍为214221(Q)一个非终态PBS作业；fold2/3与fold4的两个预备名额仍可用，但此时不提交未满足stage-receipt门禁的作业。

## 01:48 BJT 传输/训练复查

- PBS UTC 17:27:57：stage目录41,382,214,405 bytes、18,685个文件，data已有6,926/11,734个NPY；四路接收tar仍存活。相对UTC 17:24:22增加约2.07 GB、397个data文件，传输仍推进，但全量SHA验收尚不能开始/不能宣布完成。
- slfu qstat当前只剩214221(Q)一个未结束作业；214507为完成态。项目最多3个非终态PBS作业，因此数据验收完成后可在限制内提交fold2/3和fold4两个并行作业。
- FD-201 Luna只读快照UTC 17:46:10：fold1处于Epoch 12、270/287，日志持续更新，torchrun及3个rank存活，未见异常或完成标记；GPU1–3各约18,998 MiB。训练未中断。

## 01:45 BJT 提交前静态修正

- 复核fold2/3 PBS启动门禁时发现：控制输入准备器默认一次生成fold2、3、4三折收据，而fold2/3脚本原先要求收据折集合必须“恰为{2,3}”，会与合法三折收据冲突并导致作业启动后退出。已修正为检查fold2和fold3均包含在已验收收据中，fold4仍由独立三卡脚本校验。
- 已通过fold2/3及单折PBS脚本`bash -n`，并通过stage verifier、fold-input preparer的Python语法编译检查；当前尚未提交任何作业，仍等待完整数据stage及其SHA验收收据。

## 01:44 BJT 实时刷新

- PBS（UTC 17:24:22）`hubei_independent_trainval_stage_20261017` 已增至39,308,534,533 bytes，data目录6,529/11,734个NPY；相较UTC 17:21:11增加约1.82 GB和348个文件。train/val源payload目标66,508,197,403 bytes，stage仍不完整，必须等传输结束后再做逐文件大小与SHA验收。
- slfu全账户PBS队列已从3个未结束任务降至1个：214221(Q)；214505和214507均已完成(C)。因此目前最多可再提交2个项目作业，但不在训练输入收据缺失时提前排空/占用GPU队列。
- PBS GPU资源只按调度器分配使用。现场qstat显示node21、node22、node23均有其他账户运行中的GPU任务，GPU队列另有大量等待任务；已准备的fold2/3六卡作业和fold4三卡作业仍是可审计的并行合同，待数据stage通过验收后串行提交并逐次复查全账户队列。
- FD-201（UTC 17:42:42）fold1训练主进程和rank仍存活，GPU1–3显存约19 GB/卡，无结束状态；较新的Luna快照（UTC 17:39:40）为Epoch 12、62/287 batch。FD-107（UTC 17:44:17）六卡分别有约13.9–41.5 GB计算显存占用；FD-172（UTC 17:40:43）五卡均在约79–100%利用率且有其他进程。没有从其他用户处抢卡或启动未获调度资源的进程。
- 下一步：保留fold1自然训练；监测正在增长的数据stage。传输完整后先完成stage和三折控制输入的校验、重基址与收据生成，再利用新释放的PBS名额并行启动fold2/3和fold4；启动后保留逐折训练、完整validation复载与OOF bank审计，不能将场模型loss当作GHI论文结果。

## 01:40 BJT 实时刷新

- FD-201（服务器UTC 17:40:09）fold1的torchrun及3个训练rank仍存活，使用GPU1–3；GPU2/3采样利用率100%，GPU1采样时为0%。Luna只读刷新（UTC 17:39:40）显示Epoch 12、62/287 batch，日志和进程正常，未见完成标记；这仍是AGRI外推场模型训练状态，不代表站点GHI预报技能。
- PBS `hubei_independent_trainval_stage_20261017`（UTC 17:21:11）增长至37,487,036,421 bytes，data目录有6,181/11,734个NPY，4个接收tar进程仍运行。相比此前UTC 17:20:51的37,287,520,261 bytes、6,144个NPY，20秒内增加约200 MB/37个文件；源目标66,508,197,403 payload bytes，仍未完整且未做全量逐文件SHA验收，不能启动后续折。
- PBS slfu全账户队列现场仍为214221(Q)、214505(R)、214507(H)，3个未结束作业，达到上限；没有提交新作业或更改队列。
- 使用用户提供的授权凭据只读登录FD-172成功（UTC 17:40:43）。5张RTX 3090均有其他账户的计算进程，采样GPU利用率约79%–100%；没有可安全分配给本项目的空卡，未启动任何进程。FD-107本次只读快照也显示6张卡均有显存进程。FD-201其他卡同样有进程；继续只使用fold1已占用的授权资源。
- 下一步：保持传输与fold1自然运行；stage完成后执行全量大小/SHA校验和fold输入复核。只有全量验收通过且PBS全账户未结束任务数降至2个或更少时，才按已审计脚本提交下一折作业，并在提交后复查队列。

## 01:37 BJT 实时刷新

- FD-201 fold1 Luna只读快照（UTC 17:36:51）：Epoch 11、262/287 batch；三个worker存活，无训练异常或OOM。仍以Epoch 8验证loss `0.27195`为当前best，尚无`training_complete.json`；best/last没有更新。GPU1–3运行fold1，GPU0/4/5另有显存进程。
- PBS目标快照（UTC 17:18:34）为36,004,231,685 bytes，data 5,899/11,734、aux 11,734/11,734；四路接收tar仍存活。相比17:16:58增加约0.93GB，传输在增长，仍未完成SHA验收。
- PBS全账户队列仍为214221(Q)、214505(R)、214507(H)，共3个未结束作业；未提交或改变作业。

## 01:35 BJT

- 发现PBS目标目录此前只有fold2–4配置及trainer，缺少三折各自的冻结索引、train-only归一化和折报告；因此尚无最终fold输入收据，PBS启动脚本会正确拒绝训练。
- 已从FD-201只读源将fold2、3、4各自的`simvp_trainval_index.csv`、`norm/simvp_train_only_norm.npz`、`report.json`复制到PBS stage `control/fold_N/`。9个文件源/目标SHA-256逐项相同；折报告仍为`METADATA_ONLY_FOLD_MANIFESTS_PASS_TEST_BUFFER_CLOSED`，并记录`test_buffer_payloads_read=0`。fold2–4配置源/目标SHA也一致。
- 该操作只复制train/validation的索引、归一化及元数据；没有读取测试值或payload，没有生成最终stage验证收据，也没有提交PBS作业。
- PBS目标UTC 17:16:58：目录35,071,727,749 bytes；data 5,722/11,734个NPY，aux 11,734/11,734个文件。四条data传输流继续增长，尚未完成全量逐文件SHA验收。
- PBS队列仍为214221(Q)、214505(R)、214507(H)，共3个未结束任务。跨FD-107/172/201的GPU快照均无可安全借用的空卡；保持使用调度器获批资源的规则。
- 下一步：保持数据流和fold1运行；数据到齐后先运行全量stage校验器，再重基址fold索引并生成最终收据；随后重新核查PBS任务数及GPU实际分配，才启动fold2–4训练。
