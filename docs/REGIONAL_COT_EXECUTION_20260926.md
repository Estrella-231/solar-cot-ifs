# 区域 COT 推进记录：2026-09-26

当前方案：`refine-logs/cot_dynamics_20260925/REGIONAL_STATION_COT_PLAN.md`。最新状态（2026-09-27）：区域CPP/AGRI配对、两版区域R探索训练及同像元诊断已完成；区域GHI三组pilot已提交PBS210854，当前排队，未读取test。新阶段合同及结果边界见 `docs/REGIONAL_COT_GHI_NEXT_20260927.md`。下文按日期保留历史快照。

## 实时核验与恢复

- FD-107 可达。原 PID 1561132 已退出；原日志最后报 `NetCDF: HDF error`。索引停在39/73分片，不能视为缓存完成。
- manifest SHA `9b6f5ff452a377a202db3d7dc305cebd74230cb084b6accf21df745b4b47dcf8`、湖南 grid SHA `54a38fcd217a74e0def533217580b48132ef25fee40ed9ed51063e4efdd42c62` 与原构建身份一致。
- 新低优先级 CPU 恢复进程 PID 2621810 使用原 builder；外层脚本最多重试5次，仅捕获 HDF 读取错误，其余错误立即失败。保留原已完成分片及哈希，未改标签或跳过失败帧。
- 本轮最新日志已成功写入编号40的分片，累计41/73分片、10,496帧，说明恢复已越过旧停止位置；剩余构建与验收尚未完成。该数字是本轮快照，不能当作未来状态。
- 日志：`/home/Data_Pool/zjnu/solar_cot_regional_20260925/cpp_bank_v1/resume_audit_20260926.log`。完成标记为原 `index.json` 的 `COMPLETE_EXPLORATORY_CPP_BANK`；验收标记另为 `acceptance_20260926.json` 的 `PASS_EXPLORATORY_CPP_BANK_INTEGRITY`。

验收会逐分片检查SHA、数组形状、有效数值范围、train/val唯一时刻、无交集及缺失名单闭合，并汇总月度CPP分布。它不证明日照/AGRI配对或历史CPP独立物理QA已通过。生产版本不与V2自动合并。

## 配对缓存准备与验证

新增 `scripts/pack_regional_agri_cpp_bank_20260926.py`，已同步训练服务器并通过服务器 Python 语法检查。它需要已完成的CPP验收标记，才允许构建配对缓存。

每个时刻读取原13 AGRI与3太阳几何，监督掩膜取CPP有效、13通道有效、day_mask及SOZ<80°的交集。无有效监督的帧留存登记，从R训练缓存排除；不改变正式GHI样本。图像、COT和掩膜按分片存成可映射NPY，训练时可复制到分配节点本地盘，减少共享盘重复读取。归一化只拟合train有效日间监督像元，val/test不参与。

已运行明确标注为合成数据的合同检查：train和val分别给出不同输入值，验证归一化只使用train；无效通道和监督掩膜一致；分片哈希及断点重入通过。这是代码检查，不能作为真实区域数据验收结果。真实全缓存配对尚未运行。

## 下一门槛

1. 等待恢复进程产出完整缓存和全部分片验收；出现不可恢复错误则记录具体源文件，不能用缺省值补标签。
2. 复制验收后的CPP缓存到训练服务器；运行真实配对缓存，检查月度/时刻/站点日照与云量覆盖、排除帧原因及train-only归一化。
3. 固定R_region初版合同后做新的单卡profile，逐档增加batch至吞吐饱和且保留显存余量。旧72帧batch16结果不当作完整训练最佳batch。
4. 提交前实时核验账户全队列，最多3个未结束PBS任务。本轮查询已有210628、210631两个运行作业；未提交或终止任何PBS任务。

区域R稳定后再推进静止/搬运/SimVP→R三基线与匹配GHI诊断；E启动条件已改为R质量可接受、数据合同稳定、基线失败模式明确，不要求搬运先有提升。工程与来源限制留在本记录，正文没有新增性能结论。

## 用户要求继续推进后的实时记录

FD-107 的恢复进程2621810已经结束；73个CPP分片、18,670帧完成。`acceptance_20260926.json` 为 `PASS_EXPLORATORY_CPP_BANK_INTEGRITY`，73个分片全部验收；train存在12,319/39,589帧（31.117%），val存在6,351/6,351帧。验收JSON已保存本地 `audits/cot_dynamics_20260924/regional_cpp_bank_acceptance_20260926.json`。原始产品缺失与物理QA缺口仍保留。

约3GB缓存正在通过本机中转复制到zjnu-hpc；两服务器直接SSH连接超时，没有据此更改网络或认证配置。目标目录为 `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/regional_station_cot_20260925/cpp_bank_v1`。

训练服务器接续CPU进程 PID43118 运行 `wait_pack_regional_bank_20260926.py`，目前已核对27/73目标分片SHA，状态为 `VERIFYING_TRANSFER`。这是实时快照。全部目标分片验收后，它才调用真实AGRI配对脚本，输出 `agri_cpp_pair_bank_v1`；不根据文件数量或源端完成标记提前放行。等待传输超过4小时会失败并保留状态。它不提交GPU训练。

接续状态：实验目录 `transfer_pair_status_20260926.json`；日志：`transfer_pair_20260926.log`。配对完成标记为输出缓存index中的 `COMPLETE_EXPLORATORY_REAL_AGRI_CPP_CACHE`，并生成BJT月度 `distribution_audit.json`。数据脚本已增加BJT split日期核验；源统计中的val/202506的12帧是UTC月份分组，固定manifest首个val帧为2025-06-30 21:00 UTC，即2025-07-01 05:00 BJT，不是训练期帧误入验证。

下一次检查应先核验传输和配对进程状态，再读取真实配对分布和train-only归一化；这些尚未完成时不提交区域R单卡profile。

## 2026-09-26 18:24 进度快照

- 全量真实AGRI–CPP配对缓存已生成，索引登记73个分片、9,781个可用于R的帧；`test_used` 字段为false。配对缓存不是训练结果，CPP来源的独立物理QA仍未闭环。
- PBS队列实时核验：账户当前2个未结束任务（210628、210709），未超过3个任务上限。profile作业 `210709.tc6000` 在node23运行，已运行约1分钟；`210628.tc6000` 为既有运行任务。
- `cotreg_fullprof` 已通过配对缓存完整性前置检查并进入节点本地NVMe复制阶段（缓存约42GB）。最新日志尚未出现profile逐batch结果，故不能说GPU测速或正式训练已经完成/启动。作业只做资源profile，不训练模型、不写checkpoint。
- profile脚本复制后会校验各分片SHA，再用真实train缓存按递增batch测端到端吞吐和显存，并保留显存余量。下一步依据实际数据选R_region训练batch；提交训练前再次查询全账户PBS队列。

## 2026-09-26 19:05 profile恢复记录

第一次全缓存profile在本地NVMe复制完成、进入DataLoader取样后失败，未测出batch吞吐或显存档位。根因是配对缓存的`records.json`包含被排除时刻，但NPY数组只压缩保存可用帧；profile误将记录在JSON中的位置当作压缩数组行号，触发`index 233 out of bounds for axis 0 with size 128`。这不是CPP/AGRI标签或缓存完整性失败。

已修正profile数据集索引，改用配对构建器写入的`array_row`，本地和训练环境Python 3.10语法检查通过，远端代码已核验包含该修复。重跑作业`210732.tc6000`已在node23进入R态；账户当前未结束任务为210628和210732，共2个。新作业还在启动/缓存复制阶段，需等待逐batch profile日志后才能确定训练batch。仍未启动R训练，未读取test。

## 2026-09-26 19:15 GPU占用快照

用户询问卡数与占用。PBS 210735明确分配`exec_gpus=node23-gpu/0`，即1张NVIDIA A100-SXM4-40GB；不是多卡训练。实时nvidia-smi快照中分配卡0为0/40536 MiB、0% GPU利用率，作业仍在本地缓存准备阶段，尚未进入训练profile kernel。node23卡1、卡2当时分别约100%、45%利用率且有显存占用，为节点其他作业，不归属于本任务。profile前两次中断：第一次数据行索引错误，第二次成功修复索引但在清理DataLoader时重复删除导致`UnboundLocalError`；第三次已加入安全清理并重跑，需等batch结果确认峰值显存/吞吐。

## 2026-09-26 19:20 完成全缓存单卡资源profile

第三次profile作业`210735.tc6000`已完成，分配node23 GPU0，1×A100-SXM4-40GB；全量配对缓存通过73分片SHA校验，profile报告`COMPLETE_SINGLE_GPU_R_PROFILE`，train 6,293帧、val 3,488帧，未读取test。实测batch 2/4/8/16/32/64/128吞吐约48.3/297.7/317.2/217.8/344.4/327.0/326.0序列每秒，对应峰值分配显存约0.33/0.53/0.94/1.75/3.37/6.61/13.10 GiB。当前短profile规则选batch32（最高吞吐且显存<75%）；该结果是资源初选，不代表最终训练batch或精度最优。作业此前两次失败均已定位并修复，第三次成功。

用户要求后续若训练未开始优先node22 80GB卡。形式化R_region全量训练目前尚未提交；node22在线GPU型号为A800-SXM4-80GB，但PBS节点当前附注`gv maint`，虽然节点状态显示free，需提交训练前再次核实PBS能否正常分配。正式训练脚本/预算还需从探索性profile结果形成独立配置；不将40GB短profile结果写成模型训练已启动。

## 2026-09-26 19:44 区域 R 全缓存探索训练已启动

用户要求未开训时优先使用node22。尽管`pbsnodes`仍显示`note=gv maint`，PBS实际已成功将作业分配到node22 GPU1；`qstat -f`确认`exec_gpus=node22-gpu/1`，型号A800-SXM4-80GB。提交前队列只有一个未结束作业；当前210628与210746运行，共2个，低于3个上限。

补齐并同步了`train_regional_r_full_20260926.py`及PBS入口`run_regional_r_full_20260926_node22.pbs`。代码本地/远端Python 3.10语法检查、PBS Bash语法检查通过。训练只读取配对缓存中的train/val；检查73分片SHA、train-only norm和索引状态；没有读取test。探索性配置seed42、batch32、最多50 epoch、patience8、masked Huber(log1p COT)，独立输出到`r256_full_exploratory_seed42_node22`。CPP来源生产者绑定及独立物理QA仍未闭环，结果不能作为独立COT真值或GHI技能结论。

PBS作业`210746.tc6000`已进入训练循环并完成epoch0：train Huber 0.07421、val Huber 0.02994（log1p COT），峰值已分配显存3.36 GiB，epoch耗时约51.5秒。A800 GPU1实时快照约4.2 GiB显存/54%利用率。当前是首轮工程性探索训练，需观察后续epoch和验证指标，单epoch值不作结论。

## 2026-09-26 19:56 训练资源与进度快照

作业210746仍在node22 GPU1运行；日志已到epoch20，当前val Huber 0.01666且最佳epoch为20，仍在早停预算内。`torch.cuda.max_memory_allocated`稳定约3.36 GiB；nvidia-smi中进程约4,219 MiB（含CUDA上下文/缓存，约80GB的5.2%），即时GPU利用率8%。这是单次瞬时采样，不代表整轮平均利用率。整景输入仍是16×256×256，当前低占用来自R网络通道宽度base16和batch32较小；并非裁剪到16×16。已验证的A100 profile中batch128约13.10 GiB且吞吐未超过batch32，因此本轮不在训练中途改变batch；后续若要充分用A800，可在本轮结束后单独profile更大batch并保持更新步数/优化预算公平。

## 后续资源优化

当前R256 seed42训练不中途更改batch32，避免改变同一训练的优化步数与梯度尺度。完成后在A800上用真实缓存做短profile，覆盖batch32/64/128/256（内存余量允许再增档），按完整数据吞吐而非显存占用选择；同时记录GPU利用率和DataLoader等待。A100实测的batch32约344序列/s，高于batch64约327和batch128约326，因此在A800尚无测量前不假定更大batch会更快。若只是想占满80GB而吞吐没有提升，不扩大批量。

用户补充资源偏好：后续独立A800 profile将batch上探至512，优先尝试达到约60GB左右显存占用，同时保留至少25%余量；以实测吞吐和验证稳定性决定是否用于下一轮，不在当前epoch26训练中途切换batch。

## 2026-09-26 20:10 首次全缓存R训练完成；A800大batch profile已排上

作业210746的日志以`REGIONAL_R_COMPLETE`结束，status和validation_metrics均为`COMPLETE_EXPLORATORY_REGIONAL_R`。实际epoch 0–44，patience=8触发早停，best epoch36，运行约1,868秒。val masked Huber(log1p COT)=0.01551；区域有效像元COT MAE/RMSE/bias=1.854/3.725/+0.114；Sili 5×5 MAE/RMSE=1.562/3.158，64×64=1.714/3.324；Zhujia 5×5=1.023/2.488，64×64=1.280/2.977。未发生预测>100比例；验证预测与指标文件已写盘。CPP仍只是检索参考，来源绑定及独立物理QA未闭环，不能解读为独立物理真值或GHI预报增益；test未读取。

按用户“显存多占一点”的要求，新增profile batch 32/64/128/256/384/512。PBS `210765.tc6000` 已在node22 GPU1（A800 80GB）运行，输出隔离到`profile_a800_batch_sweep_20260926`；提交前队列无未结束任务，当前仅该profile运行。profile将复用node22本地42GB缓存并重新逐分片验SHA；结果以吞吐最高且峰值占用低于75%选batch，512若OOM或超过余量阈值则记录并回退，不把“占满”当作目标本身。

## 2026-09-26 20:33 A800大batch profile完成

作业210765已C/exit_status=0，node22 GPU1 A800 80GB。实际缓存profile batch32/64/128/256/384/512吞吐分别约274/247/362/278/293/334序列/s，峰值分配显存约3.37/6.61/13.10/26.06/39.03/52.00 GiB。故最高短测吞吐是batch128；batch512占用约65.6%显存，吞吐比batch128低约7.7%，仍比A100原测batch32高约3%。若下一次R训练优先响应用户“多用显存”的偏好，batch512是可运行候选并保留约34%余量，但更新步数/epoch大幅减少，须用独立配置和完整验证评估，不能把它当作纯硬件提速。4步短测结果只作资源筛选，建议下一次跑小规模完整epoch对照batch128与512，再定正式批量；不为显存占用本身重复当前已完成训练。

## 2026-09-26 20:45 batch512长预算探索训练启动

按用户要求用更大batch并补足更新次数，提交PBS `210768.tc6000` 至node22 GPU1。batch512、epoch上限680、patience122，按训练集6293帧每epoch约13步，预设总更新量约8,840步，与batch32完成的45×197=8,865步接近；这是独立实验配置，仍用seed42，最多12小时。job已进入训练循环，epoch1完成：train Huber 0.12518、val Huber 0.08230；峰值PyTorch分配51.88 GiB，nvidia-smi约55,095 MiB且GPU利用率100%。初期loss不能与此前batch32模型直接比较，须看验证收敛趋势和最终空间误差。test未读取。qstat当前本账户2个运行作业（210768及另一现有作业），未超过3个上限。

## 2026-09-26 23:04 batch512训练进度快照

PBS210768仍为R态，node22 GPU1；已到epoch195/680，当前best epoch182、best val Huber约0.01538，当前epoch195 val约0.01562，连续未刷新13/122 patience。已运行约2小时19分，按当前约42秒/epoch估算，全预算约8小时上下；若验证指标继续改善，早停会重置。显存维持约55,095 MiB/81,920 MiB。10秒`nvidia-smi dmon`抽样SM利用率在0与97–100%间交替，属于当前batch计算与数据准备/验证阶段的间歇占用；单点0%不能代表整段空闲。账户当前2个运行作业，未超过3个任务上限。

## 2026-09-27 00:14 batch512训练进度

PBS210768仍在node22 GPU1运行，已完成epoch296/680；best epoch更新至291，best val Huber(log1p COT)=0.014619，当前epoch296为0.015456、stale=5/122。累计运行约3小时32分，资源余量约8小时28分。峰值分配显存仍51.88 GiB，nvidia-smi约55,095 MiB。10秒SM采样表现为多次83–100%计算与若干0%间隔，说明该批量仍存在批次间数据/同步等待，但作业持续推进且无OOM。账户两个GPU作业运行中，均在资源上限内。

## 2026-09-27 05:10 batch512区域R探索训练完成

实时核对远端状态文件 `status.json` 与 `validation_metrics.json`：PBS `210768.tc6000` 已从队列消失，远端训练状态为 `COMPLETE_EXPLORATORY_REGIONAL_R`；状态给出的训练耗时约8小时25分，最终日志到epoch679/680。最佳epoch599，best validation masked Huber(log1p COT)=0.01378192。保留的best.pt时间戳04:04，验证预测及指标于05:08写完。全区域验证像元212,757,698，COT MAE=1.6495、RMSE=3.2463、bias=-0.0321；站点5×5邻域Sili MAE/RMSE=1.3290/2.5186，Zhujia=0.9699/2.3704。预测COT>100比例为0。未使用test。状态明确CPP生产来源和独立物理QA未解决，因此此结果仅为探索性CPP参考拟合，不构成独立反演精度或GHI技能结论。输出含best.pt、metrics.jsonl、validation_metrics.json、validation_predictions.npz；下一步应做同split基线/旧R公平对照及云类/厚云条件分解，并审计极厚云预测压缩，测试集保持关闭。
