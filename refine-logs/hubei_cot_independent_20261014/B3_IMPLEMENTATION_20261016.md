# B3 预测场 S 实施合同（湖北）

日期：2026-10-16。适用范围：湖北三站独立实验的 B3 场外推阶段；不改变湖南实验、B1/B2 标签/指标或已冻结日期分配。本文只规定 S 的准备、训练与验证产物；test 和 buffer 有效载荷继续关闭。

## 数据与时间

- 严格使用冻结日期映射 `frozen_split_20261015_v1/day_assignment.csv`（SHA-256 `6ef8cbc7d2bb7d1b1d3defdfcba473cb171e06db0115c8206f7c289d8bad19c6`）。本合同只生成 train/val 读取索引，不触碰 test/buffer HDF/AGRI/COT/GHI 数值。
- 完整场为官方湖北 256×256、20通道、float32处理结果。已有全场帧22,193；从冻结 train/val 的原始 HDF 补齐缺失2,512帧（train 2,322、val 190）。补帧只能追加至独立目录，不改写官方原始数据。使用经审计的湖北裁剪、标定LUT、通道映射、太阳角与夜间掩膜。
- 训练输入/目标采用既有16通道合同：13个AGRI波段（源通道0–5、8–14）+ `cosSOZ`、`cosRAA`、`day_mask`；输出13个AGRI波段。固定8帧历史输入→16帧未来，采样间隔15分钟。初始化时刻为最后历史帧；目标为 `init+15,…,init+240 min`。窗口须同一BJT日期且24帧中心日间标志均为1。
- 不复用旧湖北S权重；从随机初始化训练。AGRI归一化仅由train日期的历史输入帧拟合；已知太阳几何归一化仅由train序列内全部时间帧拟合。采用固定排序后等距抽取最多2,048个唯一时次；记录抽样清单和统计量。validation只用于选模/描述，test/buffer不参与。

## 模型、训练和交付

- 主干沿用学院服务器原版OpenSTL `SimVP` + `gSTA` 实现，保持Hubei旧16通道模型结构：`hid_S=32, hid_T=256, N_T=8, N_S=4`，空间卷积核3；不加入新骨干、IFS、COT输入或额外损失。
- 优化器/训练按原配置：AdamW，`lr=1e-3`、`weight_decay=1e-4`、cosine schedule、5 epoch warmup、最多50 epoch；损失为归一化AGRI空间场MSE；以validation loss选best并保留last、每轮曲线、源代码/配置/数据清单哈希。
- 正式训练之前：真实train batch前向/反向、梯度有限、checkpoint保存重载一致、单卡吞吐/显存profile；单卡优先，确认稳定后再考虑其他卡。若任何一项失败，停止正式训练并保留诊断日志。
- 正式训练结束仅生成train/validation预测bank和S的validation诊断；不得运行OpenSTL默认test。后续R、站点GHI预报头训练与评价按主计划B3执行，且各自须另做来源与公平性审计。validation结果不能写作test性能。

## 当前状态

2026-10-16：June–July14 train/val范围内可取得的AGRI帧已补入独立包，不能表述为“逐时全齐”：原按96时次/日得到的44个差值，经原工程历史/实时源联合coverage audit核对后，28个对应UTC 17:00（BJT 01:00）的常规无产品时次，真正处于产品日历但在联合归档中缺失的是16个时次，明细见`AGRI_GAP_RECONCILIATION_20261016.md`。这些缺口没有插值或用近似时次替代。旧B3序列索引存在跨BJT日、跨split/缓冲日期窗口，旧`retry4`及其预测bank降级为探索性历史，不能作为正式结果或进入后续GHI预报头。严格元数据过滤后重新stage 4,086 train / 407 validation序列（每个窗口24帧均属同一BJT日、同一split且中心日间标志全为1），未stage test/buffer有效载荷；train-only归一化和单卡真实batch/profile均通过。四rank NCCL见证通过后，新正式run `b3_simvp_seed42_20261016_strict24_same_day_v3` 正在训练。运行状态、日志、失败的早期启动诊断及资源快照见`PROGRESS_20261016.md`。归一化训练损失/MAE不是物理尺度预报技能结果。

### 严格同日索引与run启动修订（2026-10-16）

- 权威严格索引：`/home/Data_Pool/zjnu/hubei_cot_independent_20261016/manifests/strict24frame_same_day_20261016_v2/report.json`；train 4,086、val 407。SHA-256：索引`4eddd1649e90cd1a76c48264e972561338953d791cca021b3eeadc1101d22c62`；本地stage索引`e91693d421029d664cf32dc833c4bba854a03260c8b3f3c7b13e3d4dd956c763`；stage报告`e76c689fa566d6bf7fb53b567d8dd6028779268b2885638cdd729760718f2832`。stage数据59,329,519,402 bytes；test/buffer载荷stage数为0。
- 新归一化文件`/home/Data_Pool/zjnu/hubei_cot_independent_20261016/norm/simvp_train_only_norm_strict24_same_day_20261016.npz`，SHA-256=`d958148e43d240903080e4dd7a0afe95818883e25d6933e72cd54b66dac4e993`。
- 启动wrapper本地/远端SHA一致为`df014418c00299893d538a7eef8bbf2dfd59e3dc7a46ab6c4858c3885fc81804`；冻结配置SHA=`81b7795840d6a7ff8a245e45ebc66b4a3d387ba3c2fd4dcf7a61c91fe4289931`。旧run `strict24_same_day_v1` 因未初始化distributed process group、`v2`因Lightning默认日志目录对该账号不可写，均在首个训练batch前退出；二者无训练checkpoint，原日志保留作启动诊断。
- 当前run从随机初始化，使用seed42、每rank batch4、4张RTX 4090（物理GPU1–4），50 epoch上限、5 epoch early stop；screen `hubei_s_strict24_v3`。GPU0和5有其他用户进程，不使用。启动log `/home/Data_Pool/zjnu/hubei_cot_independent_20261016/logs/b3_simvp_seed42_20261016_strict24_same_day_v3.log`。已通过2/2 sanity batches；进入Epoch 0后开始训练。此为工程训练状态，不是技能结果。

## 2026-10-16 profile 与缓存记录

- 全场补帧清单SHA-256=`ee334f5cd2590757d6642722001b00defceed2768651ab3dd6fb96bba50e7de7`；时序索引SHA-256=`0ef4b6bca2540e7ced9ba788d9f2686a05ce6179db01fee92622fe49b635b875`；train-only归一化SHA-256=`fcc4b04abb5b6ff5bd5f13bc15ee147b98a0ee29844eaa9025b66b90c2891d8e`。索引为5,153 train、509 val、test=0。
- RTX 4090单卡真实train batch profile通过：batch 1/2/4的峰值显存分别约4.61/9.21/18.38GB；batch4单步完整8→16前反向约3.10s，loss/梯度有限，保存重载预测最大绝对差0。profile只是一批工程验证，不是技能结果。batch4留下约6GB显存余量，选作每卡batch，之后扩至4张当前空闲卡。
- 为减少NFS等待，已将索引实际引用的11,734个全场和11,734个几何文件（合计66.5GB）复制到FD-201本机`/tmp/zjnu_hubei_simvp_stage_20261016/`；本地stage index SHA-256=`5ccdc3d99d0e6c3838701952dc3c407943f10efd178aa42b407ec297f8215a43`。缓存可重建、非持久；正式checkpoint/log写回`/home/Data_Pool/zjnu/hubei_cot_independent_20261016/`。
- 正式S训练只读冻结的train/val索引；test/buffer载荷始终不读。训练运行状态另记于`PROGRESS_20261016.md`。

## 训练后预测bank的四卡并行准备

- `scripts/infer_hubei_simvp_trainval_patches_20261016.py`现支持`--shard-index/--shard-count`，按每个split的canonical索引行号取模切为互斥分片；每个worker写入独立输出目录，并将全量split行号写入`keys.csv`，便于准确合并。保留test/buffer行拒绝、train-only归一化及checkpoint SHA记录。
- `scripts/merge_hubei_simvp_trainval_patch_shards_20261016.py`合并前核验所有shard ID齐全、索引/checkpoint/normalization哈希一致、train/val键与原冻结索引逐行相同、无重复/漏行、输出数组形状和值有限。合成CPU覆盖测试在Windows与FD-201均通过；远端推理脚本`--help`通过。
- 已部署的远端实现SHA-256：推理器当前为`a6164ff93d3c77124e62e4169dd0c16e9d766fdcd73b402e9f20a4b0925de5c5`（2026-10-16 06:14 BJT在FD-201与本地复核一致）；合并器`e6504f3c6a0fd163c31619f87a152c810cceffada272b3bbbb8990f828e8d56d`，合成测试器`9f8910d926ec266d8f14c5cb6e2ba31ec52bbd94b230fa694797d5efcd8f6598`。此前记录的推理器哈希`a573...`已被当前现场文件替代，不作为运行依据。
- 这只是推理基础设施准备，尚未生成正式bank；当前S训练期间不抢占其GPU。训练结束并核对best权重及early-stop状态后，用4个独立进程分别设置`CUDA_VISIBLE_DEVICES=1/2/3/4`、进程内`--device cuda:0`及`--shard-index=0..3 --shard-count=4`，每个写入唯一输出目录，之后运行merge QA。GPU0/5若仍由其他账号占用则不触碰。

## 2026-10-16 strict S→R预报pack完成与OOF门禁

- strict-stage best S bank、forecast-COT bank已分别通过4,086 train / 407 validation完整唯一键覆盖，持久产物位于`/home/Data_Pool/zjnu/hubei_cot_independent_20261016/forecast_bank/strictstage_best_v3_20261016/`和`forecast_cot_bank/strictstage_best_v3_20261016/`。COT由预测AGRI经冻结R生成，不是未来CPP观测；COT推理明确`test_payloads_read=0`、`ghi_labels_read=false`。
- 联合15分钟站点标签后生成forecast GHI heads train/validation pack，路径为`/home/Data_Pool/zjnu/hubei_cot_independent_20261016/forecast_pack/strictstage_best_v3_20261016/`；全体FA/FAC共同样本键共70,171（train 63,448、validation 6,723），lead为+15至+240 min，test/buffer值读取为0。pack只是特征/标签对齐与数据完整性产物，尚未训练GHI forecast head。
- train/validation输入域差异由`audit_hubei_forecast_bank_train_val_drift_20261016.py`进行描述性审计：以train-only AGRI通道标准差归一化后，S全场normalized RMSE train=0.32098、validation=0.40308；forecast COT mean train=7.22、validation=10.40。train输入是S和R已见日期的in-sample输出，而validation来自未见日期。该差异可能混合真实日期/天气难度变化和模型对训练日期的拟合，不能单凭此报告归因；但它已足以阻止将当前train pack直接用于确认性FA/FAC训练与公平validation比较。
- 下一阶段先生成预先固定的BJT日期组OOF S→R train输入：每个held-out train日期折的S和R训练、归一化统计拟合、checkpoint选择均不得接触该折日期；完整24帧仍限制同日/同split。每折validation模型选择固定使用原冻结validation日期。合并OOF输出后与full-train S→R validation bank核验键、lead、站点窗口、尺度/非有限值及漂移，再决定是否启动FA/FAC。资源未通过时停在OOF合同/实现准备，不用in-sample pack替代，也不读test/buffer。

## 2026-10-16 strict S 训练终态与复评状态

- FD-201日志`/home/Data_Pool/zjnu/hubei_cot_independent_20261016/logs/b3_simvp_seed42_20261016_strict24_same_day_v3.log`末尾显示Epoch15完成后因连续5轮未改善触发早停，`global_step=4096`，并明确输出`TEST_CLOSED: skipped by frozen Hubei train/validation contract`。当前进程和screen均不存在；不得将此前Epoch14运行快照继续称为当前状态。
- 已稳定核验checkpoint：`best.ckpt`与`best-epoch=10-val_loss=0.244.ckpt`同SHA `022795dc7b17f5b7ca545c630c1c666993233bfa215db3cce32efab6e7be2e86`；`last.ckpt` SHA `f0f40de83297500562fb1e40da421c9fd9a91eb8e32c7c1476c8488cddf68c33`。两者需要在统一完整validation上复评分；训练rank局部validation loss不能替代。
- strict index SHA仍为`4eddd1649e90cd1a76c48264e972561338953d791cca021b3eeadc1101d22c62`（train 4,086 / val 407，test/buffer为0），train-only norm SHA=`d958148e43d240903080e4dd7a0afe95818883e25d6933e72cd54b66dac4e993`。本轮真实checkpoint烟测在冷启动导入共享NFS上的Python包时处于`rpc_wait_bit_killable`，输出目录尚未创建；已结束该烟测，不代表模型前向失败。完整bank和验证评分尚未产生，之后应先解决运行环境/共享盘导入阻塞，再按空闲GPU重试；不访问test/buffer。

## 2026-10-16 validation复评分修复记录

- best/last第一次全量val复评的推理与数组、`keys.csv`写出均已完成，但脚本在生成`BANK_REPORT.json`时错误地访问未加载的train dataset，触发`KeyError: 'train'`。两目录未生成报告，故该次产物不作为有效评分；保留为诊断材料，不覆盖。
- 已修复`infer_hubei_simvp_trainval_patches_20261016.py`：`sequences_by_split`只汇总本次实际加载的split。新脚本本地与FD-201 SHA-256均为`82342dce378e6fe6ad6938e66aa6cb95a3272742c9a1cf168c2026a8d1cf5898`，本地语法编译检查通过。
- 修复后的best checkpoint复评分已在`/tmp/zjnu_hubei_rescore_best_v3_20261016/`完成并通过QA：407条val key唯一，pred/true形状`(407,16,13,16,16)`、geometry形状`(407,16,3,16,16)`，均float32且finite；test/buffer读取标记为0。全场（256×256、13通道、16 lead）train-normalized MSE=0.2346095940、MAE=0.2969769224。报告SHA=`e9102c42090f461668a70dd544fdc8690c252ab0a6c1b9d245453ef95365a36a`；checkpoint/norm/index SHA与上列冻结合同一致。该值是checkpoint复评用的验证损失，不是站点GHI技能。
- last checkpoint复评分已完成并通过同一QA：407条val key唯一；数组形状、float32和finite检查均通过；test/buffer=0。全场归一化MSE=0.2387419795、MAE=0.2906471719。报告SHA=`ed0e161a68cfad9e5a9a13f5db6bab1ff10f448e8e8a17873d300e656e3d132b`。依据预先规定的validation MSE选模，best（MSE 0.2346095940）优于last（0.2387419795）；该选模只适用于S验证损失，不代表站点GHI技能。
- 为后续大规模bank生成增加并部署可选`--stage-root`缓存映射：canonical index/key字段保持不变，只映射dataset内部AGRI/aux读取路径到FD-201本机stage。首轮smoke发现冻结index有一部分路径已直接指向本机stage，其余路径需从manifest映射；已扩展为两种形式均支持。当前本地与FD-201版本SHA=`a87f652c43198ec90ae006acbb924b6e7e56578a20b0d168cc2fdddcb3a14d71`，两条validation小样本一致性烟测正在重试（PID 1003953，GPU1，输出`/tmp/zjnu_hubei_stage_smoke_best_v2_20261016/`）。通过后再生成正式train/val forecast bank；test/buffer有效载荷仍未读取。

## 2026-10-16 五折 R OOF 训练准备与首折启动

- 五折OOF源索引与日期分配已通过metadata审计：250个固定train日期，每折50天；S严格同日序列支持4,086 train/407 val，折内held-out训练日期不进入对应S训练或归一化。CPP监督只覆盖155个train日期和16个固定val日期；不同折有效监督数见各`r_pack/PACK_QA.json`，不把无效CPP样本当晴空。
- 五折R train/val包均在`/home/Data_Pool/zjnu/hubei_cot_independent_20261016/manifests/oof_fold_manifests_20261016_v5/folds/fold_N/r_pack/`通过哈希、形状、有限值和mask审计。每折pack只提供对应held-out折以外的train监督及固定validation；无CPP监督的行保留在AGRI/COT推理映射中但从R loss/选模排除。test/buffer payload读取为0。
- 使用批准的COT模型实现（SHA=`8af2d0e101b492bf2096223fe788b7f3a8a9b5ece2544db45903b95397620e05`）进行fold0单卡GPU运行见证，状态`HUBEI_R_RUNTIME_WITNESS_PASSED_NO_FORMAL_TRAINING`；前反向、梯度、save/reload与FP64复算通过。RTX 4090上profile batch128约5,677 examples/s、每epoch至少17 updates；batch192不满足至少12 updates/epoch，故首折采用128。
- 2026-10-16 20:33 BJT启动R fold0正式训练：FD-201 GPU1，seed42，batch128，最多50 epoch/patience8；tmux `hubei_R_oof_fold0_20261016`，输出`/home/Data_Pool/zjnu/hubei_cot_independent_20261016/oof_training/R_fold0_seed42_20261016/`。启动后初查进程存活，run metadata已写入；必须等固定样本过拟合gate和epoch日志确认后才能宣称训练主体已运行。该模型用于held-out train dates的OOF COT生成，不是GHI预报技能。FA/FAC训练未开始，test/buffer仍关闭。
- batch=8 stage smoke v3已完成：8条validation样本键唯一，pred/true/future-geometry均为有限值，test/buffer读取标记为0；stage manifest SHA与冻结stage核验值一致，val映射引用19,536。与原路径best-v3全量validation输出的前8条逐元素比较，pred、truth和geometry最大/平均绝对差均为0。smoke报告SHA=`d267aa4c8ec7466b1a4e0ce2f3356f35c914bffe568de92af6c57f4a8cb43ebc`；pred SHA=`cf822ebf880f35cd6d57fc5774542908937cd03cef639e2e895af0212ce69af6`，truth SHA=`b1e38953c36cc9eb5b5e6f63503a50d588087dc13f3a2d311d45d7de2fc0425a`，geometry SHA=`881ecc20d58ff9a07765c8e16e3e70724e017d76fd370434cb3afbf456a79188`。这一门禁只验证缓存路径等价性，不是模型技能结果。
- 单卡吞吐profile已完成：前64 canonical train+64 val样本，batch 8/workers 4约4分钟；batch 16/20未见提速，batch20峰值显存约20.7/24.6 GiB，因此正式bank采用batch8以保留显存余量；batch8/workers8也未见稳定提速。三卡profile分别在GPU1–3运行互斥分片，每片64 train+64 val；三片行号并集精确覆盖0–191且无重叠，test/buffer=0，数组有限。约4分钟完成384样本，aggregate吞吐约为单卡profile的2.5–3倍。profile仅支持资源决策，不是模型结果。
- 正式forecast bank已在FD-201 GPU1/2/3启动，分别为`shard-index=0/1/2, shard-count=3`，进程内设备`cuda:0`、batch8、workers4，输出位于`/tmp/zjnu_hubei_forecast_bank_best_epoch10_strict24_v3_s0_20261016/`至`s2`。启动快照PID为1185133、1186008、1186931；预计覆盖冻结索引全部4,086 train + 407 val行，test/buffer载荷关闭。须待三片报告、键和数组QA后再合并并接续forecast-COT/R与GHI头。

## OOF S fold 训练启动记录（2026-10-16）

- 当前启动脚本`solar-cot-ifs/scripts/run_hubei_simvp_oof_fold_20261016.sh`在物理GPU选择与进程内CUDA设备编号间显式映射：选择物理GPU1,2,3时，`CUDA_VISIBLE_DEVICES`使训练器使用本地设备0,1,2。脚本本地与FD-201临时部署副本均通过`bash -n`，SHA-256=`bdfacab702b58b497203f42ec5bcb697d8aca18bc03164f646954c3c70038c86`。
- attempt2因传入旧物理编号1,2,3而在Lightning设备初始化时报错，未进入sanity/epoch，作为诊断失败保留，不视为训练。
- attempt3已按fold0日期专属索引、fold0 train-only norm及独立配置从随机初始化启动，输出位于`/home/Data_Pool/zjnu/hubei_cot_independent_20261016/oof_training/S_fold0_seed42_20261016_attempt3/`；物理GPU1–3对应训练器设备0–2，3 ranks、每rank batch4。2026-10-16 21:23:40 BJT，固定validation sanity 2/2完成并进入Epoch 0（284 batches）；尚无完整epoch、best checkpoint或held-out OOF预测，不形成模型技能结论。共享存储在数据读取阶段出现RPC等待，需观察实际进度。
- fold0完成并通过checkpoint、val选择与数据隔离审计后，才能同配置启动fold1–4；所有五折held-out预测键完整且无泄漏后再合并OOF S bank并接入相应OOF R输出。test/buffer有效载荷继续关闭。
- 2026-10-16 21:28 BJT状态复核：日志约Epoch 0 119/284（42%），rank PID与GPU UUID核对确认只占物理GPU1–3，每卡约18,998 MiB；GPU0原有PID 234965和GPU4/5他人任务未触碰。当前每步约2.05秒，尚无首个完整epoch/validation选择结果，故训练完成时间与模型质量暂不能判断。
