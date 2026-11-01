# 湖北未来 COT 单因素辅助监督重训（attempt13）

## 目的

attempt12 已修复并核验太阳几何、恢复昼间可见光源并重训完整 SimVP→R 链；三站预测 GHI 的平均降幅只有 0.098%，日期块区间跨零。与此同时，SimVP→R 的验证 COT RMSE 在 +1、+2、+3、+4 h 均高于历史持续性。下一步直接检验一个可解释的训练改动：SimVP 是否因为只优化全域 AGRI 像元 MSE 而损失了对 R-COT 有用的局地云厚度信息。

本任务是验证筛选，不预设有收益，不替代 attempt12，也不改论文主表。CPP 是卫星检索参考，不是独立 COT 真值；全部结果仍为反复使用过的验证集诊断，不是独立测试。

## 历史同类实验与本轮定位

该训练思想已有湖南先导，不作为本轮的方法新颖性主张。`refine-logs/cot_aware_simvp_20261017/EXPERIMENT_PLAN.md` 和 `EXPERIMENT_TRACKER.md` 记录了两站湖南同预算配对训练：原 SimVP、等预算 AGRI-MSE 微调、AGRI+COT 混合辅助，以及后续 CPP-only 辅助。共同验证支持上的站点 3×3 COT RMSE 分别为 10.664161、10.750112、10.740627、10.743569；辅助损失未通过总体 COT RMSE/近零虚警放行门，CPP-only 的厚云 CSI 虽有小幅改善，近零虚警却显著增加。该结果是湖南旧月度划分下的验证证据，不是湖北数据结果。

因此 attempt13 的科学定位是**湖北三站独立区域复核**：使用湖北自己的 train/validation、修正太阳几何和对应冻结 R；不复用湖南权重、结果或数据。当前仓库未检到此前完成的湖北同类配对训练。若本轮改善，只能说明该受控设置在湖北验证集上的区域复现，不能把辅助损失说成首次提出的方法；若未改善，应与湖南负结果合并解释为跨区未复现。历史来源：`refine-logs/cot_aware_simvp_20261017/EXPERIMENT_TRACKER.md`（04:24、04:30、06:10 CST 记录）及 `EXPERIMENT_PLAN.md`。

## 单因素配对实验

| 臂 | 训练损失 | 作用 |
|---|---|---|
| MSE-control | 原 attempt12 的全域有效 AGRI MSE | 与候选保持新代码、初始化、训练顺序、模型选择和执行环境一致的重训基线 |
| COT-aux | 同一 AGRI MSE + λ × 冻结 R(COT(预测 AGRI)) 与目标时刻 CPP COT 的掩膜 log1p-MSE | 唯一变化为增加 COT 辅助项，检验是否保留更多能被固定 R 读取的云厚度信息 |

- 使用 attempt12 已审计通过的数据划分、恢复后的可见光输入、NOAA 几何、SimVP gSTA、8→16帧、16输入通道（13 AGRI + cosSOZ + cosRAA + day_mask）、13个 AGRI 输出通道。训练与验证均只用对应集合；test/buffer 继续关闭。
- 冻结 attempt12 修正后的 full R 权重与其 train-only 均值/标准差；`eval()`、参数 `requires_grad=False`。COT 梯度只穿过 R 回到 SimVP 预测 AGRI，不能更新 R。
- 每个目标时效的候选路径：SimVP 预测 AGRI 反标准化到物理通道值 → 截取与 R 相同的 16×16站点区域 → 拼接已知目标时刻太阳几何 → 按 R pack 的全部16通道 train-only 统计标准化（day_mask 不得改回0/1）→ 冻结 R 前向。只在目标时刻 CPP 有效像元上与对应 CPP COT 比较；不使用未来观测 AGRI 作为输入或COT标签，不用 R 输出自蒸馏目标，避免把 teacher 自身拟合误当作参考技能。
- 保留原 AGRI 监督的全域有效像元支持与实现。COT 损失使用有效掩膜上的 `log1p(COT)` MSE；先用 `where` 消除无效 NaN 再平方，全无效批次不产生梯度更新。按 lead 等权，保证长lead不因样本数或像元数额外加权。
- λ 仅用 train 日期固定按月抽取的32个窗口，在 SimVP 输出处计算 AGRI 和 COT 损失梯度范数比，预注册为 `0.1 × median(||grad L_AGRI|| / ||grad L_COT||)`，保存逐窗口结果后冻结；不查看验证结果来调 λ。
- 两臂均重新随机初始化，seed=42；构造并保存相同初始 SimVP 参数 SHA、训练样本顺序、batch 顺序、数据增强/DropPath随机流、优化器和学习率计划。固定 attempt12 合同的 AdamW、lr=1e-3、weight decay=1e-4、cosine、5 epoch warmup、50 epoch、batch16（如联合单卡profile不通过，则两臂共同降到同一个 profile 通过的 batch）。为满足“实际更新数完全一致”，两臂都完整训练同样的50轮，不做逐臂早停；每轮仍保存同一验证指标下的最佳checkpoint。训练/验证batch、学习率计划和更新总数完全匹配。
- 两臂统一按 CPP-common support 的验证集逐lead物理 COT RMSE选择 best checkpoint：先将冻结R的log1p输出变换回物理COT并限制在[0,100]，再以每个lead的CPP有效像元RMSE作等lead平均；两臂使用相同键、掩膜和指标。站点像元、3×3邻域、逐lead及厚度分层只在checkpoint锁定后报告，避免候选单独获得模型选择优势。

## 评估和放行门槛

1. 主终点：三站等权、逐lead等权的16×16物理COT RMSE，候选相对同次 MSE-control 的日期块配对差及95%区间；同时相对 attempt12 原模型和历史COT持续性报告。所有方法在精确相同 CPP-common 键/像元上比较。
2. 局地终点：站点像元和固定3×3 COT RMSE、MAE、bias；+15…+240 min逐lead曲线；CPP COT<1、1–10、10–30、≥30分层，预先固定COT≥30为厚云，报告厚云漏报/CSI与近零虚警。
3. AGRI护栏：整个验证场、13通道及逐lead原物理单位 RMSE。COT-aux 相对 matched MSE 的季节/整体等权 AGRI RMSE 若恶化超过1%，或厚云 CSI下降超过0.02，或近零虚警明显恶化，则失败。
4. COT候选只有同时优于matched MSE和历史持续性、日期块95%区间完全有利、湖北三站均不退化、AGRI/厚云/近零护栏全部通过，才放行生成五折OOF候选AGRI→R-COT银行及同键GHI头重训。否则保留负结果，停止这条COT辅助损失候选，不在验证集上调λ或继续扫权重。
5. 若放行，后续使用五折日期隔离的 SimVP/R 预测 train bank、full-train validation bank，按同样GHI头合同配对有无COT；PBS每次提交前实时查询完整 `slfu` 队列，最多3个未结束任务。不得把未来真实AGRI/CPP/R作为推理输入。

## 实施顺序与自动修复

1. 只读重验 attempt12 `DATA_READY`、独立数据审计、NOAA几何收据、full R checkpoint/归一化/CPP逐键映射和 attempt12 full S init/归一化哈希；检查train/val键唯一、目标16帧完整、日期隔离及 test/buffer 计数为0。
2. 构建独立 train/val CPP-COT target sidecar，按 `target_time_bjt` 精确连接每帧，另存键表、支持掩膜、数组SHA和支持覆盖；行顺序不作隐式假设。
3. 远端 `swc` 环境编译、NaN mask/坐标/通道标准化/冻结R梯度/CPP标签映射合成见证；真实batch检查R权重无梯度、SimVP梯度有限非零、预测与val target键一一对应。
4. 进入一次A800单卡真实profile，按稳定显存和吞吐确定两臂共同batch；完整profile通过后才执行成对重训。发现异常时保留日志和目录，修复最小单一原因，用新attempt/新输出路径重跑，不覆盖attempt12或旧失败证据。
5. 训练、checkpoint reload、预测银行、FP64独立复算和汇总都完成后更新tracker。任何PBS qsub之前和之后均需实时 `qstat -u slfu`；空队列快照过期后不得用于新提交。

## 当前状态

- attempt12 的恢复/几何/训练验证收据已通过并完整重训；不是本 attempt13 的训练结果。
- 本 attempt13 已获用户“恢复和核对通过后直接重训、错误直接修复”的明确授权，不再请求重复确认。
- attempt13 代码和配对运行脚本已在本地准备；登录节点看不到 `/dev/shm` 缓存属于节点本地路径，必须在PBS分配的node22上检查。若attempt12缓存不可见，则在attempt13独立路径按持久化来源恢复，不覆盖attempt12。提交前会再实时核验完整队列。
- 发现并修正选模指标实现：原验证累计误差在log1p空间，却标记为物理COT RMSE；当前改为expm1反变换后的物理COT RMSE，并按预注册等lead方式选模。辅助训练项仍是log1p MSE。
- CPP 目标的单位已在 attempt13 sidecar 中显式恢复：R pack 的 `target.npy` 是物理 COT 除以100，构建目标时乘回100，并检查源值位于 `[0,1]`；模型辅助损失再对物理目标计算 `log1p`。选模及保存字段均标为物理 COT RMSE，避免把无量纲 `log1p` 指标误报成物理量。
- 单步显存 profile 已按服务器 OpenSTL `SimVP.forward` 的真实递推结构实现：两次8帧预测，中间拼接未来8帧太阳几何，最后输出16帧；对中间及最终张量形状都有断言。profile 不再用单次前向替代真实训练路径。
- test/buffer payloads: closed; reads = 0.

### 2026-10-21 13:16 CST：attempt13 retry4 只读进度快照

- 13:16:55 CST 实时 `qstat -u slfu`：未结束作业2/3。PBS `215356.tc6000` 状态 `R`，node22，申请 `ppn=8:gpus=1`；PBS `215359.tc6000` 状态 `H`，依赖 `afterok:215356.tc6000`，用于后续验证评分。
- 证据目录：`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021/hubei_cot_auxiliary_retrain_attempt13_retry4/`。MSE-control 当前 epoch 3，241/282 batch（约85%）。epoch 2 完整验证后物理 `val_cot_rmse=15.24987`，刷新当前 best；该值是训练中途验证选模指标，不是最终测试或最终技能结论。
- COT-aux 臂尚未开始；当前 `runs/` 仅有 `mse_control`。`215359` 仍处于依赖挂起，未开始评分。未发现失败或完成标记；best/last checkpoint 为训练过程产物。
- 此次只读检查没有读取 test/buffer payload，也未提交或取消作业。登录节点无 `nvidia-smi`，故本次只据 PBS 资源字段确认分配 node22 的单张 GPU，未独立复核 GPU 型号/利用率。

## 2026-10-21 14:00 CST：独立代码审计与当前状态

- 独立审查结论为 WARN、没有阻断训练的核心逻辑错误。retry4 启动路径真实存在且绑定正确；目标时刻COT映射、冻结R梯度路径、物理单位COT选模和共同 CPP 像元支持均通过源码核对。详细问题及结论见 `../ATTEMPT13_CODE_AUDIT_20261021.md`。
- 审计发现，当前排队评分任务不含计划中的全域AGRI护栏和attempt12完整对照；两臂预测COT bank/几何/checkpoint哈希也未在评分器里做完整闭环验证。因此现有COT评分只能作本轮COT诊断，不能单独触发完整下游放行。训练完成后须用保存的模型与验证数据补做AGRI评分和来源哈希核验，再按三站门槛判断。
- 配对合同虽保持seed、初始参数哈希和更新次数一致，但没有保存逐batch/RNG顺序摘要，且`deterministic=False`；最终描述限于“同初始化、同配置、等更新数的配对比较”，不声称逐步随机轨迹完全相同。
- 14:00 CST 实时 `qstat -u slfu` 为2/3：`215356.tc6000` 为R，node22:ppn8:gpus1；`215359.tc6000` 为H，依赖`afterok:215356`。远端日志的MSE-control在epoch 9约208/282合并训练/验证batch；epoch 8的物理验证COT RMSE=11.78668为中途best，非最终或配对结果。COT-aux尚未启动，test/buffer未读取。本轮未改PBS作业。


## 2026-10-21 12:28+08:00 CST：历史核查与 attempt13 retry3 状态

- 用户询问的“AGRI 像元 MSE + 冻结 R-COT 辅助损失”此前已在湖南实际训练过，不是首次探索。历史记录 `refine-logs/cot_aware_simvp_20261017/EXPERIMENT_TRACKER.md` 的 04:24、04:30、05:59、06:10 CST 给出同一 3,140 窗口 / 46 CPP 日期公共验证支持：原 SimVP、等预算 AGRI-MSE 微调、混合标签 COT 辅助、CPP-only COT 辅助的站点 3×3 COT RMSE 依次为 10.664161、10.750112、10.740627、10.743569。辅助候选没有改善总体 COT RMSE；CPP-only 相对 MSE 差 −0.006542、日期块95%区间 [−0.030278,+0.012902] 跨零，厚云 CSI 有小幅改善，但近零虚警率显著增加 +0.001726 [ +0.000819,+0.002897 ]，预设总体门槛未通过。该湖南验证结果是负向/未证实结果，不是湖北结果或独立测试。
- 历史核查结论：当前 attempt13 是修正太阳几何、湖北自身 R/数据及湖北三站上的区域复核；检索未发现此前已完成的湖北同类配对训练。不把辅助损失称作新方法；若湖北也不改善，按跨区域未复现如实报告。
- 215337 因 hostname 字面断言失败；215340 因缺 `cv2` 退出；215352 的兼容 shim 没放在 Python 实际启动路径，导入预检退出。这三次均未进入训练，保留原日志和失败收据，test/buffer读取均为0。私有 `python_vendor` 已补齐依赖并将 timm API shim 放到其启动路径；conda 激活后的完全同序导入 smoke 现为 PASS。共享 `swc` 环境未修改。
- PBS `215355.tc6000` 于提交前实时核对全账户队列为0/3，提交后1/3；当前 qstat 状态 `R`，请求 node22:ppn8:gpus1、36小时。实际启动日志确认 node22-gpu/1，`nvidia-smi` 为 NVIDIA A800-SXM4-80GB（编号1，80GiB）。数据和 CPP 预检PASS：4,086 train、407 validation、19,481个物理COT目标与R pack精确核对，test/buffer读取0。
- 联合单步 profile 通过：batch16、峰值69.291GiB、19.30秒；SimVP梯度有限非零，冻结R参数无梯度。`PAIR_STARTED_attempt13.json` 写明两臂各50轮、每臂12,800更新、seed42同初始化。当前只看到 `START_ARM mse_control`，MSE臂训练已启动；COT-aux臂尚未开始，暂时无checkpoint和验证结果。现场收据 `receipts/LIVE_215355.json`，提交收据 `receipts/SUBMITTED_215355.json`。

## 2026-10-21 12:27 CST：历史试验核查与 attempt13 retry4 更正

历史核查报告：`../hubei_cot_auxiliary_retrain_attempt13_retry4/HISTORY_AUDIT_PHYSICS_CONSTRAINT_AND_PERSISTENCE_20261021.md`。仓库已直接比较未来 SimVP→R COT 与历史持平（湖北 attempt12，1/2/3/4 h 均值RMSE：10.255/9.787/10.194/12.063 对 8.946/9.238/9.617/9.086），并做过湖南时间变化、空间梯度、COT辅助监督；这些代理监督未改善湖南总体COT RMSE。严格守恒/输运/辐射闭合等物理定律约束尚未做。

更正执行记录：PBS `215355.tc6000` 于首轮进度被提前取消；`282` 是 Lightning 单个epoch训练与验证合并显示（256 train + 26 val），不是 optimizer updates，因此不构成合同超步数。中止属于执行者误判，不能登记为模型/训练失败；原日志及目录保留。已在新目录 `hubei_cot_auxiliary_retrain_attempt13_retry4/` 用同一训练源码、同一输入/R/CPP和50轮成对合同重启；PBS `215356.tc6000` 提交前后队列为0/1，node22单张A800，当前R。此时只完成启动阶段，无checkpoint或模型结论；test/buffer读取0。

## 2026-10-21 13:13 CST：配对训练进展与后续验证评分已排队

- 实时 `qstat -u slfu` 显示账户未结束任务 2/3：`215356.tc6000` 在 node22 GPU1 运行；`215359.tc6000` 为本轮验证推理/评分任务，状态 H，依赖 `afterok:215356.tc6000`。未终止或改动其他任务。
- `215356` 当前仍在 `mse_control` 臂，epoch 2 验证已完成，日志显示物理 `val_cot_rmse≈15.20`；epoch 3 约 34%，`cot_aux` 臂尚未开始。该值仅是单臂中途验证读数，不是成对结果或最终 COT 技能结论。
- 为避免训练结束后再等待准备，已新增 `run_attempt13_retry4_infer_score.py` 和 `attempt13_infer_score_retry4.pbs`，本地 Python 编译、远端 Python 编译及远端 `bash -n` 均通过。`215359` 将在配对训练成功后，对两臂各自锁定的最佳 SimVP checkpoint 仅推理 407 个 validation 窗口，再用同一冻结 R 生成物理 COT，并在完全共同 CPP 支持上进行 FP64 逐lead/站点/厚度评分，和历史持平及离线真实未来图参照对照。CPP 仍为检索参考；不读取 test/buffer。
- 推理任务请求 node22 单卡；执行时会检查配对完成收据、数据缓存和验证行键，缺任何必需项即失败退出，不覆盖已有结果。当前主要训练仍运行，尚无 attempt13 最终评分。

- 2026-10-21 13:15 CST（实时复核）：`qstat -u slfu` 为2/3未结束任务，`215356.tc6000`仍在node22 GPU1运行，依赖评分`215359.tc6000`保持H。训练日志采样显示MSE-control epoch 3约65%；COT-aux臂未开始，当前仍无成对比较结果。该采样只用于进度记录，不作模型结论；test/buffer继续关闭。
- 2026-10-21 13:16:55 CST（Luna实时复核）：账户2/3未结束任务；`215356.tc6000`在node22运行，`215359.tc6000`仍因afterok依赖处于H。
MSE-control已到epoch 3的241/282合并训练/验证batch，epoch 2的当前best物理验证COT RMSE为15.24987；COT-aux尚未开始。无配对分数，无失败/完成标记；test/buffer未读取。
- 2026-10-21 13:18:45 CST（实时复核）：`qstat -u slfu`显示2/3未结束任务：`215356.tc6000`运行中，`215359.tc6000`处于H等待依赖。MSE-control已完成epoch 3并进入epoch 4（约15%）；epoch 3验证物理COT RMSE=14.00365，更新为当前单臂best并保存checkpoint。COT-aux仍未开始，因此该值不是配对结果；test/buffer未读取。
