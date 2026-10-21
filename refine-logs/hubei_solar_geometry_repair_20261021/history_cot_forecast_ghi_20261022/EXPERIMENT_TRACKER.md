# 湖北历史 COT→GHI 预报配对实验跟踪

## 状态

| 阶段 | 状态 | 证据 |
|---|---|---|
| 冻结对照合同 | `PLAN_READY` | `EXPERIMENT_PLAN.md` |
| train/val 历史时间键、split、lead 审计 | `PASS_METADATA` | `history_coverage_metadata.json`；168,790行、4,458个init、6,271个历史帧时刻，无缺帧键/跨split/重复键，BJT-UTC映射全通过 |
| node22 暂存文件完整性 | `PASS_NODE22` | PBS `215503.tc6000` Exit_status=0；71,328个历史 data/aux 路径存在 |
| 冻结R真实输入冒烟 | `PASS_R_TRAIN_VAL_SMOKE` | train/val各一帧，输入/输出形状正确，有限非负COT，checkpoint/norm/code哈希已记录 |
| 历史COT bank | `PASS_BANK_AUDIT` | `215506.tc6000` exit_status=0；train/val源链接与哈希通过，匹配批形状R回放差值为0 |
| GHI小头profile / 4-GPU配对训练 | `PASS_PBS_215507` | node22 4张A800；1/2/4 worker profile与9组配对训练全部完成，Exit_status=0 |
| 保存预测FP64评分 | `PASS_SCORE_AND_INDEPENDENT_RECALC` | 15,275个验证行；18份预测键/顺序/标签/晴空因子/哈希通过；独立复算与主评分完全一致；test/buffer=0 |

## 实时资源

- 2026-10-22 00:10 CST，`qstat -u slfu` 提交前为0项，提交后为1项（`215503.tc6000`）。
- PBS分配 `node22/0-3`，GPU `node22-gpu/0`，实测 `NVIDIA A800-SXM4-80GB`；该短审计申请1 GPU，不占用其他用户任务。
- 当前任务用途仅为路径 stat 和两帧冻结R smoke；没有启动GHI训练，没有读取test/buffer，没有改动其他PBS任务。

## 2026-10-22 01:08 CST：四卡配对训练与独立验证评分完成

- PBS `215507.tc6000` 在浙师大学院服务器 node22 正常退出，`exit_status=0`，实际执行时长 00:13:47，分配 A800 GPU 0/3/4/5。训练及profile均未读取test/buffer。
- 1/2/4 worker profile 总吞吐分别为 651,737 / 704,951 / 938,328 examples/s；按“距实测峰值不超过5%时取最多worker”规则选4 worker。4 worker总吞吐比2 worker高约33.1%，比1 worker高约44.0%。四卡用于并行独立配对任务，不改变单模型batch=1536、30轮、行顺序、初始化或更新合同。
- 9组站点×种子配对（牛首、王寨、米庄×42/43/44）均完成，两臂各30轮。主评分按保存的验证预测 FP64 汇总：等站点、等种子 RMSE 由100.918降至99.202 W/m²，差值−1.715 W/m²，点估计降低1.70%；MAE 68.414→67.000 W/m²，bias +2.729→+3.305 W/m²。
- 按22个BJT目标日期做5,000次配对bootstrap，RMSE差值95%区间[−3.888,+0.310] W/m²、相对改善区间[−0.315%,+4.086%]，均跨零。九个站点×种子点差及三站平均、四个小时段的点差方向均有利，但没有分组区间；当前只报告探索性验证点估计，不声称稳定收益。
- 本实验加入的是过去8帧真实AGRI经冻结R反演的**历史COT**，不是未来预测COT；不能用1.70%替代attempt12的预测COT结果（95.703→95.609 W/m²、区间跨零）。R输出也不是独立观测。
- 新增 `audit_saved_history_cot_predictions.py` 对18份保存预测独立复算；15,275验证行、9组共同键顺序、标签和晴空因子、与主评分预测哈希全部通过，数值与主评分完全一致，test/buffer读取均0。证据与图表见 `RESULTS_ATTEMPT2_20261022.md`、`results_attempt2/`。
- 因日期块区间跨零，不在同一验证合同上继续加种子或搜索；更多seed不会增加独立目标日期。下一步若要正式确认，先冻结未用于本轮选模的新验证日期合同，再判断是否值得开新一轮配对训练。当前PBS作业已完成，后续长任务仍按实时profile多卡并行，单模型batch/更新合同保持不变。

## 2026-10-22 00:44 CST：四卡训练包代码验收

- 已修复两处会影响配对解释/评分的实现细节：历史COT标准化只用train历史帧中的有效日间像元拟合，无效COT在标准化后设为中性零；独立评分器按所有预测行直接汇总BJT日期键，避免日期列表索引越界。
- bank构建器现检查四卡分布式实际写入总帧数必须等于需求帧数；不足即失败，不生成完成收据。
- 本地全部Python脚本 `py_compile` 通过；`zjnu-hpc` 的 `swc` Python环境编译通过；四卡PBS脚本 `bash -n` 通过；评分日期聚类/有放回日期抽样维度微测通过。登录端无PyTorch，故GPU模型前向测试以node22 R smoke及PBS运行期检查为准。
- 计划提交1个PBS任务：node22申请4张A800、单任务最长6h；先四卡生成历史train/val bank，后单卡复核，再运行1/2/4 worker吞吐profile，按实测选择最高且距峰值≤5%的并行度，最后训练9组站点×种子配对（每组两臂顺序训练）。单模型batch=1536、30 epochs、相同行序/初始化/更新数保持不变。
- qsub前实时 `qstat -u slfu`（2026-10-22 00:44 CST）：无未结束任务；PBS `215504.tc6000` 已提交并在运行，提交后全账户1项。`qstat -f` 当前确认node22、请求`gpus=4`、已分配`node22-gpu/0,3,4,5`、6小时上限。计划、代码、审计均只读train/val，test/buffer仍关闭。
- 作业进入用户脚本后，PBS状态仍为`R`；初始 `nvidia-smi` 识别4张 `NVIDIA A800-SXM4-80GB`，当时GPU利用率为0%（处于启动/输入检查阶段），不能据此估算最终加速。配套15分钟heartbeat已更新为低频监控本实验，状态不变不重复通知。
- node22二次核验已看到torch distributed的4个rank进程，并在分配的GPU 0/3/4/5各占约1.0–1.16 GiB；该瞬时利用率采样仍为0%，表明仍在初始化/读帧阶段，尚未取得首个`BANK_PROGRESS`，不宣称已有吞吐结果。账户队列仍只有本任务1项。

## 2026-10-22：attempt1 bank生成成功，审计因FP16批形状差异停止

- PBS `215504.tc6000` 已终止完成态，`exit_status=1`；它已成功生成 `history_bank_attempt1` 的完整6,271帧train/val bank，四个rank总写入数与预期一致，bank report/数组哈希完整，test/buffer读取均为0。首轮独立R回放比较单帧推理和bank的64帧批推理，发现最大绝对差 `2.7007e-4 COT`，所以审计严格失败，GHI训练没有启动。
- 初步定位是验证程序没有与bank构建器使用相同的cuDNN确定性设置和batch形状；还未将这解释当作已证实。修订审计将对照同一rank的64帧输入批次并固定 `cudnn.deterministic=True`、`benchmark=False`，以保存的bank和原始train/val源独立重放。原bank、attempt1失败收据和日志均保留，不覆盖。
- 下一步先完成PBS `215506.tc6000` 的单GPU短审计（只读attempt1 bank及train/val源）；通过后再提交四卡长任务，仅运行1/2/4 worker profile、9组配对GHI训练和FP64验证评分。账户列表会显示已完成任务C状态；未结束任务计数必须读取状态列而非Session列。当前活动任务仅`215506`。历史COT是本实验诊断输入，不等同于预测未来COT的完整链路结论。

## 2026-10-22 00:38 CST：attempt2批形状复核已提交

- PBS `215506.tc6000` 正在node22单卡GPU0运行（1卡、30分钟上限），只复核已生成bank和对应train/val源，不重建bank、不训练GHI。提交前账户只有已完成态`215504/C`，未结束任务数为0；提交后未结束任务数为1。
- qstat列表中Session ID在第5列、State在靠近末尾的列；先前脚本误用第5列计数，把`C`作业计为活动。之后统一按`$(NF-1)`状态列计数，不影响当时的提交数量上限。
- `215506.tc6000`现已以`exit_status=0`完成。独立复核检查了168,790行和1,350,320条历史连接，6,271帧键/数组哈希全通过；train/val两组rank-0 64帧R输入和COT重放的最大绝对差均为0，test/buffer读取均为0。银行源问题排除，现可进入GHI训练阶段。

## 2026-10-22 00:39 CST：四卡profile与GHI配对训练已启动

- PBS `215507.tc6000` 已运行，qstat确认node22申请 `gpus=4`，实际分配GPU 0/3/4/5，6小时上限；提交前未结束任务0个，提交后1个。四卡用于并行吞吐profile及独立站点/seed配对任务，不改变单模型batch=1536、30轮和更新合同。
- 运行顺序为1/2/4 worker profile→按“离峰值吞吐不超过5%内取最多worker”选择并行度→9组FA_HIST/FAC_HIST配对训练→保存预测和FP64日期块验证评分。此处训练比较历史COT输入信号，不能冒充未来预测COT链路已验证增益。

## 冻结实验合同

- 两臂同用8帧历史AGRI、同一固定SimVP预测AGRI和太阳几何、相同标签行/掩膜、相同训练样本与更新数。
- 仅实验臂收到由这8帧真实历史AGRI经同一冻结修正R得到的历史COT；控制臂保留相同COT支路及参数，但其标准化COT输入固定为零。
- R不更新。COT只由过去输入帧生成；无未来COT/AGRI/CPP输入。
- 仅train/val；归一化仅由train拟合。站点牛首、王寨、米庄，种子42/43/44；每个配对相同初始化与batch顺序。
- 指标由保存预测重算；任何验证结果不称独立测试结论。

## 2026-10-22 00:13 CST：历史帧节点审计完成

- PBS `215503.tc6000` 已完成，`exit_status=0`，分配节点node22、A800 GPU0；任务只做train/val路径stat和冻结R推理smoke。
- `history_coverage_node22.json`：GHI行168,790、初始化4,458、历史帧键6,271；缺帧键0、跨split0、重复stage/样本键0、错误时间映射0；71,328个data/aux路径全部存在。test/buffer读数均0。
- `history_r_smoke.json`：train及val各抽一帧，R输入均[16,16,16]、输出[1,1,16,16]，输出有限非负；使用修正R checkpoint SHA `d8f25a...d6b5d0b`和train-only norm SHA `4db7bb...d3394446`。test/buffer读数均0。
- 下一步：在同一node22通过4卡并行构建历史AGRI/几何/COT帧库，实测1/2/4卡独立worker吞吐，再运行同数据同更新合同的FA-history vs FAC-history配对训练。


## 心跳只读核验 2026-10-21T17:02:36Z

- Luna实时读取完整slfu队列：215507=R，单个未结束PBS，分配GPU 0/3/4/5；没有提交、取消或重启任务。1/2 worker profile已分别完成，汇总吞吐651,737/704,951 examples/s；4 worker profile仍运行，尚无完整选择收据、9组训练完成或评分结果。
- node22只读活性检查：orchestrator PID15799，4个profile worker PID21007–21010均在，按节点进程elapsed刚启动约59秒；日志0字节尚不足判死。分配四卡当时均2MiB/0%，不把其他两卡的活动算到本任务。
- 工具观察UTC与节点报时约相差18分钟：观察17:02:36Z时node22报00:44:20+0800。判定本次worker存活时采用同节点elapsed，不用跨主机墙钟差假定已卡住十几分钟；不调整数据文件时间或训练输入。
- 持续按既定作业推进，无可复核新GHI结果或实质故障。本轮不重复提交，不改attempt1/2、模型或测试隔离规则。


## 心跳完成复核 2026-10-21T17:17:14Z

- Luna现场完整qstat未列未结束任务；215507-f已Unknown Job Id，但attempt2_complete、9/9配对orchestrator及FP64评分产物共同确认完成。未提交或重启作业。
- 另用verify_saved_history_predictions_20261022.py独立复核18份逐样本预测、36个best/last实际SHA、初始化/行序/更新配对，以及保存标签/晴空量逐键等于原GHI_pack_attempt12数组的实际float32输入；源数组SHA与pack report相符。原15,275共同val键、22个目标日期、5000次频数加权日期bootstrap均重现，数值差<1e-10。
- 复核结果保存在receipts_heartbeat_20261022T0117/independent_saved_prediction_check.json，附加报告入口沿用RESULTS_ATTEMPT2_20261022.md。完整预测/权重/源标签副本本地保留并已忽略Git，不把原始标签CSV或权重推入报告归档。
- 总体历史COT点改善1.6997%，95%日期差值区间[-3.88775,+0.30982]仍跨零；此处历史COT输入不等于未来预测COT。稳定收益和整体目标仍未证实，test/buffer保持关闭。当前有限作业监控已完成用途，结束该阶段自动轮询，不重复attempt1/2。
