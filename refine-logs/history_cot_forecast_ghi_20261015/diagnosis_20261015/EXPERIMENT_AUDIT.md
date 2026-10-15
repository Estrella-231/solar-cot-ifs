# 独立实验完整性审计报告

**日期**：2026-10-15  
**审计器**：GPT-5.6-Sol ultra，新鲜同家族代理，只读  
**项目**：history COT forecast GHI attempt2  
**总体判定**：WARN  
**integrity_status**：warn  
**review_independence**：same-family  
**acceptance_status**：provisional

本审计只读取本地计划、代码、九个完成记录与九份逐样本预测，并只读检查远端九个 `curve.json`。未训练、未提交作业、未修改实验代码或结果、未打开 test 载荷。

没有发现伪造真值、预测生成真值、自归一化评分、幻影结果、test 载荷使用或跨臂样本不一致。保存的九组预测能够独立复现全部已报告 RMSE 与置信区间。

主要问题位于训练设计和主张边界：训练 batch 大于全部训练集，1200 次更新相当于约 2432.7 个有效 epoch；首次验证已经在约 202.7 个 epoch 后。COT 臂还拥有 6912 个 A 臂实际失活的输入权重，且五个 COT 运行在第一个验证点 step100 即达到已观测最优。因而现有结果能证明“该训练协议下保存的验证候选 RMSE 退化”，不足以证明历史/外推 COT 本身无效或必然导致 GHI 退化。

## A. Ground-truth provenance

**严重度：LOW｜状态：PASS**

- GHI 来源是湖南四里、竺家站权威工作簿中的真实辐照度记录。工作簿解析读取 `结果集` sheet、`DATA_DATE` 和 288 个 `VAL0005…VAL2400` 五分钟区间结束字段；四里映射到 `四里光伏.xlsx`。证据：`audits/agri_spatial_payload_20260908_v1_sources/build_hunan_station16_samples.py:49-68,126-168`。
- M0 封存链逐文件校验 SHA，逐条检查 `target-init=lead`、标签时刻为 `target-10/-5/0 min`，并验证三条五分钟 GHI 的算术均值等于 15 分钟标签。证据：`scripts/prepare_ghi_base_cohort.py:30-38,54-80`。
- `kt` 的分母是 pvlib/Ineichen 计算的三个五分钟中点晴空 GHI 均值；`kt=observed_GHI/clear_sky_GHI`，不裁剪。证据：`scripts/add_interval_kt.py:31-63`。
- 标签经 `scripts/join_forecast_labels.py:38-55` 排除 test 行并绑定 `(station,init,target)`，再由 `scripts/pack_head_training.py:150-184` 检查 canonical lead、时区、三条区间结束记录和 `GHI=clear×kt`。
- `audits/head_pack_trainval_20260908.json` 的 `state=COMPLETE`、`source_labels_sha256=01587f...`、`arrays.ghi.sha256=b819c9...`；其文件 SHA-256 为 `3de1c9...`，与 `results_attempt2/source_manifests_audit.json:source_hashes.base_pack_audit_sha256` 一致。
- 训练代码只把 `x/cot/support/lead` 送入头部；`ghi` 只进入损失和评价，`clear` 只作为可提前计算的目标时刻晴空乘数。证据：`refine-logs/history_cot_forecast_ghi_20261015/train_attempt2_as_run.py:79-91,156-164`。
- 未来 COT 是模型派生输入：历史 AGRI→冻结 R，或历史 AGRI→冻结 SimVP→冻结 R/输运/冻结 hybrid；它不是 GHI 真值或 CPP 真值。证据：`scripts/build_sili_cot_dynamics_bank_20260927.py:241-258`、`scripts/train_sili_history_repeat_control_20260928.py:46-61`。

**LOW｜WARN**：工作簿的 W/m² 单位在源 schema 中写为根据 `IRRADIANCE` 字段和值域推断，而非工作簿显式单位元数据，见 `audits/agri_spatial_payload_20260908_v1_sources/build_hunan_station16_samples.py:153-167`。这不构成预测生成真值，但应保留来源限定。

## B. Score normalization

**严重度：LOW｜状态：PASS**

- AGRI 是 SimVP 训练标准化空间中的预测值；供 R 使用前会反标准化为物理 AGRI，再按 R 的训练归一化处理。证据：`scripts/build_regional_cot_ghi_pilot_20260927.py:222-239`。
- COT 为 `log1p(COT)`；transport 先 `expm1` 输运，再转回 `log1p`。证据：`scripts/train_sili_transport_thickness_pilot_20260927.py:25-38`。
- transport/hybrid 分别只用训练行及有效 support 拟合均值、标准差：`2.130404/1.380155` 和 `2.128328/1.410245`。证据：`train_attempt2_as_run.py:147-158`；JSON：`results_attempt2/source_audit.json:cot_norm_train_only`。
- 模型输出经 Softplus 得到非负 kt，再乘区间晴空 GHI；训练损失是原始物理 GHI 误差除以固定常数 1000 后的 MSE。证据：`train_attempt2_as_run.py:31-41,161-164`。
- RMSE、MAE、bias 全部在 W/m² 原始尺度上计算，分母是样本数。`reduction_percent` 的分母是 A 的 RMSE，不是模型自身最大值、均值或输出方差。证据：`train_attempt2_as_run.py:94-96,211-228`。
- 真值未裁剪；预测只受 Softplus 的非负约束。九组预测均有限且非负。验证真值范围 `0–1223 W/m²`。
- 没有 CPP mask 用于删除 GHI 评价行；transport 不可达处用共同 support 通道和零值编码。证据：`train_attempt2_as_run.py:35-41,73-91`。

**MEDIUM｜WARN**：保存的验证 `clear` 最小仅 `2.605×10⁻⁴ W/m²`，真实 kt 最大约 `20.995`，而九组预测 kt 最大约 `1.02–1.44`。这没有污染原始 GHI 指标，但说明低太阳高度的极端 kt 目标很难由当前头拟合。

## C. Result existence and consistency

**严重度：LOW｜状态：PASS**

九份 `predictions.npz` 均存在且已逐一以 `allow_pickle=False` 加载。每份准确包含：

`pred_ghi, true_ghi, row_ids, lead_minutes, target_day, init_time, target_time`

共同核验结果：

- 每份均为 5855 行；七个数组长度一致。
- 九份的 `row_ids/true_ghi/lead_minutes/target_day/init_time/target_time` 按位完全相同。
- `row_ids` 有 5855 个唯一值，范围 `538053–637885`。
- `true_ghi` 全有限，SHA-256 为 `436d1b...`。
- 16 个 canonical lead 为 `15…240 min`，各 lead 行数依次为 `358,357,356,364,365,364,367,368,373,370,377,376,374,369,363,354`。
- `target_time-init_time == lead_minutes` 全部成立。文件中 `init_time` 保存为 BJT `+08:00`，`target_time` 保存为 UTC `+00:00`；时区感知解析后相等，但混用表示方式容易被后续无时区代码误读。
- `paired_pack.npz:val_rows/val_ghi/(val_lead+1)*15/val_dates/val_init/val_target` 与九份预测对应数组按位一致。
- 九组 `pred_ghi` 哈希各不相同，不是复制结果。

逐运行独立重算与 `complete.json`、`summary.json:runs` 完全一致：

| 运行 | best step | RMSE | MAE | bias |
|---|---:|---:|---:|---:|
| A42 | 200 | 156.901 | 111.716 | -38.570 |
| T42 | 100 | 161.484 | 112.093 | -38.988 |
| H42 | 100 | 161.934 | 112.336 | -38.260 |
| A43 | 200 | 154.886 | 109.558 | -34.393 |
| T43 | 200 | 162.113 | 111.135 | -35.392 |
| H43 | 100 | 162.857 | 114.193 | -43.395 |
| A44 | 300 | 160.123 | 113.598 | -38.833 |
| T44 | 100 | 161.472 | 111.419 | -36.503 |
| H44 | 100 | 161.159 | 111.788 | -40.022 |

- 三 seed RMSE 均值 A/T/H=`157.303235/161.689429/161.983286`。
- transport−A=`+4.386194 W/m²`，95% CI `[0.080460,9.476395]`。
- hybrid−A=`+4.680051 W/m²`，95% CI `[0.715139,9.450159]`。
- `results_attempt2/independent_score.json:max_difference=1.1369×10⁻¹³`。
- `results_attempt2/lead_summary.csv` 的 34 行与 `independent_score.json:results` 数值差为零。
- `RESULTS.md:11-21` 的全部总体、+1/+2/+3/+4h 数字、百分比和 CI 均匹配实际文件。
- `summary.json:test_used=false`、`source_audit.json:test_payload_read=false`、`independent_score.json:test_payload_read=false`；训练入口不接受 test split。

**MEDIUM｜WARN：指标叙述不完整。**

`RESULTS.md:3` 的“两种 COT 臂均退化”只有对 RMSE 成立。三 seed 平均 MAE 为：

- A `111.6238`
- transport `111.5489`，比 A 小 `0.0749 W/m²`
- hybrid `112.7723`

transport 的 MAE 是极小、未做 CI 且种子不稳定的改善，不能称有效增益，但正文应写成“两种 COT 臂总体 RMSE 均退化”。计划要求报告 RMSE/MAE/bias 与 16 lead，见 `EXPERIMENT_PLAN.md:10`；当前每 seed 总体 MAE/bias 仅在 `complete.json` 中，16-lead MAE/bias 未保存。

**LOW｜WARN**：九份预测未保存 `support` mask；只能通过共同 `row_ids` 和 `paired_pack.npz:val_support` 重建。预测文件也未保存自身 SHA 清单，完整性依赖当前文件间一致性，而非不可变输出 manifest。

## D. Dead code and execution path

**严重度：MEDIUM｜状态：WARN**

实际 PBS 入口见 `run_attempt2.pbs:23-31`；它编译并执行训练脚本，未调用 `preflight.py` 或 `independent_score.py`。两者是作业前后独立步骤。

训练主路径全部实际可达：

- `prepare()`：`train_attempt2_as_run.py:45-91`
- 真实梯度/开关见证：`train_attempt2_as_run.py:172-177`
- profile：`train_attempt2_as_run.py:179-189`
- 九臂训练与验证选模：`train_attempt2_as_run.py:191-210`
- 汇总/bootstrap：`train_attempt2_as_run.py:211-230`

**MEDIUM｜WARN：as-run 来源校验分支实际为空。**

`train_attempt2_as_run.py:120-123` 查找的是历史别名：

- `simvp_checkpoint_sha256`
- `r_checkpoint_sha256`

三个 bank manifest 的实际字段是：

- `backbone_checkpoint_sha256`
- `regional_R_checkpoint_sha256`
- `manifest_sha256`
- `r_normalization_sha256`
- `base_pack_audit_sha256`

因此两次 `if key in meta and key in m` 均为假，训练过程中没有真正执行跨 bank 来源比较。结果文档在 `RESULTS.md:37` 已如实披露。后置 `audit_source_manifests.py:7-14` 使用正确字段复核通过，`source_manifests_audit.json:state=PASS_S_R_SPLIT_NORMALIZATION_LABEL_SOURCE_MATCH`，故当前产物来源一致，但这是事后修复，不应把 as-run 分支称为已通过。

**LOW｜WARN**：`preflight.json` 与 results 内副本哈希一致，但 launcher 没有调用 preflight，也没有单独执行收据。训练代码自身重新完成了主要行/时间/来源检查，因此不是结果失效点。

`independent_score.py:12-35` 确实加载全部九份预测、比较共同键并复算 RMSE/bootstrap；本次审计又独立复现了其结果。

## E. Scope and claim ceiling

**严重度：HIGH｜状态：WARN**

实际范围：

- 区域：湖南。
- 站点：仅 Sili/四里，中心 `(163,148)`。
- 空间：原生 64×64 station crop 的中心 `31:34,31:34`，即 3×3。
- 旧 train：332 个 sequence、4041 行、262 个 BJT target 日期，`2024-04-02–2025-06-30`。
- 旧 validation：449 个 sequence、5855 行、60 个 BJT target 日期，`2025-07-01–2025-09-10`。
- 配置：A、transport、hybrid。
- 种子：42/43/44，共九次训练。
- 每臂名义 1200 optimizer updates；保存候选均在 step100–300。
- 同一旧 validation 同时用于 checkpoint 选择、报告点估计和日期 bootstrap。
- 不是新月度独立训练、不是两站结果、不是 test、不是部署评价。

`summary.json:scope`、`RESULTS.md:25-30,40-43` 和 `EXPERIMENT_TRACKER.md:18-19` 基本准确地限制了主张。允许的最高表述是：

> 在四里站旧划分的探索性验证上，使用当前 3×3 头、当前 COT 表示和每100步选模时，transport/hybrid 的三种子平均 RMSE 分别比零 COT 对照高 2.79% 和 2.98%。

不能据此写成 COT 普遍无效、湖南全年两站结论、新月度独立结果、test 结果或业务收益结论。

## F. Evaluation type

**严重度：INFO｜状态：PASS**

`evaluation_type: real_gt`

评价目标是工作簿中的真实站点 GHI。COT 是模型派生预测特征，CPP 仅用于上游 R/hybrid 的训练或诊断；本次 GHI 得分不是 `synthetic_proxy`、`self_supervised_proxy`、`simulation_only` 或 `human_eval`。

## 专项因果、对齐、训练审计

### 1. `seq → row → station` 对齐

**严重度：LOW｜状态：PASS**

- 上游 bank 用 split-local sequence index 保存 `sequence_indices.npy` 和对应 `head_row_ids.npy`：`scripts/build_sili_cot_dynamics_bank_20260927.py:171-191`。
- attempt2 通过 sequence 值建立显式 lookup，不按行位置连接：`train_attempt2_as_run.py:47-53,75-89`。
- 每条 row 又回查 `rows.csv` 的 `cache_index/station/split`：`train_attempt2_as_run.py:130-145`。
- 5855 个保存 row id 全唯一，且与 `paired_pack.npz:val_rows` 一致。

没有发现 sequence/row 位置拼接错误。

### 2. 站点和 3×3 crop

**严重度：LOW｜状态：PASS**

- 上游站点顺序为 `[(163,148),(73,150)]`：`scripts/build_sili_cot_dynamics_bank_20260927.py:21`。
- 64×64 crop 是 `[row-32:row+32,col-32:col+32]`：`scripts/build_sili_cot_dynamics_bank_20260927.py:167-168`。
- attempt2 始终取 station index0，并取 `31:34`：`train_attempt2_as_run.py:62-71`。
- 所以中心为局部索引 `[32,32]`，3×3 无 off-by-one。

### 3. 时间轴、legacy/canonical lead

**严重度：LOW｜状态：PASS**

- Causal dataset 强制24帧、15分钟连续、BJT/UTC一致，8历史+16未来几何：`scripts/build_geometry24_forecast_cot_20260924.py:107-130`。
- SimVP 第0个输出对应未来第1帧，即 canonical +15 min；bank 明示 `lead_minutes=15…240`：`scripts/build_sili_cot_dynamics_bank_20260927.py:297-305`。
- attempt2 的 pack lead 是0…15，保存时转为 `(lead+1)*15`：`train_attempt2_as_run.py:85-91,204`。
- `target-init=(lead+1)*15` 和标签时刻 `target-10/-5/0` 被逐行断言：`train_attempt2_as_run.py:136-146`。
- 所有保存行的时差已再次独立验证。

没有发现 legacy lead0 被误当同刻，也没有 +15/+240 丢失。

### 4. COT 时间轴与 causal prefix

**严重度：LOW｜状态：PASS**

- 历史为8帧，future transport/hybrid为16帧，拼成24通道：`train_attempt2_as_run.py:56-74,88`。
- `warp_batch` 明确生成 lead1…16：`scripts/train_sili_transport_thickness_pilot_20260927.py:25-38`。
- 对目标 lead `l=0…15`，prefix 条件为通道索引 `<=8+l`：八个历史通道及未来 +15 到当前 target，目标之后通道全部置零。证据：`train_attempt2_as_run.py:35-40`。
- 梯度见证断言 post-target 通道梯度为零：`train_attempt2_as_run.py:172-177`。

没有发现使用目标之后预测 COT 的实现错误。

### 5. 未来观测泄漏

**严重度：LOW｜状态：PASS**

- Causal input builder只读取八帧历史 AGRI；未来只读取可提前计算的 nominal geometry：`scripts/build_geometry24_forecast_cot_20260924.py:42-49,119-130`。
- 上游 bank 目录确实还含 oracle/CPP 数组，但 attempt2 只打开 `cot_local_history/cot_local_forecast/motion_local32/image_local`，没有打开 `cot_local_oracle/cot_local_cpp`。证据：`train_attempt2_as_run.py:54-71`。
- hybrid 的 CPP 是上游监督目标，不是本次推理输入；冻结 hybrid 实际输入仍是历史 COT、预测 COT、输运、lead和历史运动：`scripts/train_sili_history_repeat_control_20260928.py:46-61`。
- test split 在 CausalInputs 中被显式拒绝：`scripts/build_geometry24_forecast_cot_20260924.py:47-49`。

### 6. support mask

**严重度：LOW｜状态：PASS**

- 八个历史通道 support 全真；16个 transport 通道使用无 wrap 的 in-domain mask。证据：`train_attempt2_as_run.py:65-75`。
- 相同 mask 通道传给 A、transport、hybrid，unsupported 值显式置零：`train_attempt2_as_run.py:35-41,89`。
- `paired_pack.npz` 中 train/val support 分别为 `[4041,24,3,3]`、`[5855,24,3,3]`。
- validation future support 平均有效率约 `95.94%`；849行至少有一个未来无效像元。所有方法仍保留相同 GHI 行。

**MEDIUM｜WARN**：A 并非严格的“仅 forecast AGRI+geometry”。它仍收到24个由历史 C13 运动和 transport 边界生成的 support 通道。证据：`train_attempt2_as_run.py:39-41,160`。因此应称为“零 COT 值、共享运动 support 的 A 对照”。

### 7. 随机初始化、行顺序和冻结梯度

**严重度：LOW｜状态：PASS**

- 每 seed 在每 arm 前重新设置 Python/NumPy/Torch/CUDA 随机种子：`train_attempt2_as_run.py:23-25,191-195`。
- 同 seed 三臂 `initial_sha256` 完全相同。
- 每 seed 的训练 stream 在 arm 循环外生成，三臂使用相同行序：`train_attempt2_as_run.py:191-199`。
- 上游 hybrid 使用 `.eval().requires_grad_(False)` 且 prepare 在 `@torch.no_grad()` 下运行：`train_attempt2_as_run.py:44-45,124-129`。
- optimizer 只接收新 `Head` 参数：`train_attempt2_as_run.py:194-200`。
- A 对 COT 数值独立、AC prefix 内有输入敏感度的运行时断言通过：`train_attempt2_as_run.py:172-177`。

### 8. batch composition 和样本重复

**严重度：HIGH｜状态：WARN**

训练集仅4041行，却选择 batch8192：

- profile 通过 `arange(batch)%len(train)` 制造重复样本，并用包含重复项的 `5×batch/time` 作为吞吐：`train_attempt2_as_run.py:179-188`。
- 正式 stream 将完整随机排列反复拼接后切成8192行 batch：`train_attempt2_as_run.py:191-199`。
- 每个 optimizer step 包含全部4041个唯一行，每行出现至少2次，部分出现3次，少数 step 中出现4次。
- 每 seed 总输入次数为 `9,830,400`，每行总出现2432或2433次。
- 1200 updates 等价约 `2432.67` 个全数据 epoch。
- 每100步验证一次，相邻验证点间约 `202.72` 个全数据 epoch。

这使“batch8192吞吐最优”主要反映重复样本吞吐，并非更多唯一数据吞吐，是本实验最明显的训练协议缺陷。

### 9. 名义参数量与有效容量公平性

**严重度：HIGH｜状态：WARN**

三臂总参数都为51073，形式上相同；但 A 的24个 COT 通道永久为零：

- 首层为 `Conv2d(64,32,3)`：`train_attempt2_as_run.py:28-33`。
- A 的24个 COT 输入通道对应 `24×32×3×3=6912` 个首层权重永远不影响输出、梯度恒零。
- A 的有效参数上限约44161；AC 两臂可使用全部51073。
- 6912 个差异约占总参数13.53%。

因此“相同架构/总参数量”成立，“有效容量严格匹配”不成立。该差异与 COT 两臂更强过拟合一致，但尚不能证明它是负 RMSE 的唯一原因。

此外，causal prefix 使后期 future-COT 通道只有少量独立训练行：

`4041,3790,3539,3287,3031,2773,2521,2267,2017,1768,1517,1263,1011,757,505,255`

最后 +240 通道的288个首层权重只在255个唯一训练行中激活，构成长时效过拟合风险。

### 10. 训练曲线和 checkpoint 判断

**严重度：HIGH｜状态：WARN**

已直接读取远端九个 `curve.json`。每个均有12条记录、step=`100…1200`、`n=5855`。

- A best step：`200/200/300`；last RMSE=`178.78/177.14/179.50`。
- transport best step：`100/200/100`；last RMSE=`223.15/213.05/203.78`。
- hybrid best step：`100/100/100`；last RMSE=`223.28/237.23/256.82`。
- 九臂当前 batch loss 总体下降，而验证 RMSE 在100–300步后大幅上升，构成明确过拟合。
- 保存 checkpoint 的选择逻辑是每100步取最低验证 RMSE，并在训练后重载逐元素核对预测：`train_attempt2_as_run.py:200-210`。
- 已观测网格上不存在遗漏的更晚好 checkpoint。

**HIGH｜WARN**：五个 COT 运行在第一个验证点 step100 即最优，代码没有 step0 或 step1–99 的验证记录。首次验证已约等于203个 epoch，因此不能排除更早、更好的 COT checkpoint。当前只证明“100步验证网格上的最佳候选仍弱于 A”。

### 11. validation 重复使用

**严重度：HIGH｜状态：WARN**

- 九个 GHI head 均在同一5855行 validation 上选 checkpoint并报告最终指标：`train_attempt2_as_run.py:200-210`。
- CI 也在同一 validation 的60个 target BJT 日期上 bootstrap：`train_attempt2_as_run.py:211-228`。
- hybrid update96 又是先在同一旧 validation bank上按 CPP patch thick RMSE 选择：`scripts/train_sili_history_repeat_control_20260928.py:167-212`。
- `RESULTS.md:42` 已承认 validation选模和旧 split 限制。

置信区间应视为 post-selection 描述性区间，不能作为独立确认性统计证据。

### 12. split过滤可复核性

**严重度：MEDIUM｜状态：WARN**

代码正确检查从 `init−105 min` 到 `init+240 min` 的24帧日期均属于新 monthly metadata 的 train/val 集合：`train_attempt2_as_run.py:104-117`。

以本地权威 `experiments/monthly_block_20261002/manifests_v1/day_assignment.csv` 对 `paired_pack.npz` 重算：

- 332个旧train sequence：0个触及test/buffer；
- 449个旧val sequence：0个触及test/buffer；
- 旧train中304个落新train日、28个落新val日；
- 旧val中427个落新train日、22个落新val日。

这证实它保留旧角色而不是新月度训练。但实际服务器输入 `day_assignment.csv` 没有随结果复制，也没有保存 SHA，因此只能确认与本地权威表一致时通过，无法证明服务器文件的逐字节身份。

### 13. 标签权重

**严重度：MEDIUM｜状态：WARN**

代码加载并检查 `weight>0`，见 `train_attempt2_as_run.py:79-82`，但更新损失没有使用 weight，见 `train_attempt2_as_run.py:161-164`。本次单站16 lead 的训练行数很接近（每 lead 249–258），影响预计较小，但 source pack 的 inverse-frequency 权重实际未生效。

### 14. 上游物理来源限制

**严重度：HIGH｜状态：WARN**

`results_attempt2/source_audit.json:cot_bank_manifest.r_reference_QA` 明确为：

`historical CPP producer/physical QA unresolved; exploratory only`

因此可以把 COT 当作冻结模型派生的预测特征，不能把 transport/hybrid 的结果解释为独立真实 COT 物理结论。

## 已确认、排除和未决事项

### 已确认的训练机制

- 当前保存的 COT 两臂总体 RMSE 确实高于 A。
- 九臂均明显过拟合，COT 臂更早且更严重。
- batch8192 导致每步重复完整训练集，1200步约2432.7个 epoch。
- A 与 AC 名义参数相同，但 A 有6912个永久失活的 COT 输入权重。
- 长 lead 的新增 COT 通道只有很少唯一训练行。
- 同一 validation 被用于选模、报告和 CI。
- transport 的平均 MAE有极小改善，故“均退化”必须限定为 RMSE。

### 已排除

- 模型输出生成 GHI 真值。
- 观测 GHI 进入模型输入。
- 模型自身统计量归一化评分。
- 九臂使用不同 row/GT/lead/date 支持。
- canonical lead 的 +15/off-by-one 错位。
- 3×3 crop 中心 off-by-one。
- post-target COT 通道泄漏。
- test、未来真实 AGRI、CPP 或 oracle 数组进入 attempt2 GHI 头。
- 同 seed 三臂随机初始化或训练 stream 不同。
- S/R/hybrid 在 GHI 训练中被反向传播。
- 100步网格后存在更好 checkpoint。
- 已知四里2024-07-01清晨固定值段污染本次 validation；本次验证日期为2025-07至09。选中的2024-07-01训练 sequence 位于下午，也不与已知清晨固定段重合。

### 未解决风险

- step1–99 是否存在更优 COT checkpoint。
- 有效容量差异是否是负 RMSE 的主因。
- COT 外推误差、过境空间错位、表示冗余、融合结构和优化设置各自的贡献。
- 上游 regional R 的 CPP producer/物理 QA。
- 服务器实际 day-assignment 文件的哈希身份。
- 上游 bank payload 的逐值时间/空间对齐；当前仅由代码、manifest/hash和下游键一致性支持。
- remote `best.pt` 文件哈希与 `complete.json:checkpoint_sha256` 未在本次权限范围内独立重算。
- 单站、旧 split、60个重叠窗口日期对新月度两站泛化能力。
- 保存预测缺少 support mask、clear、pred_kt、sequence id 和输出哈希清单。

## 建议的只读下一步

1. 从现有 `paired_pack.npz` 与九份预测重算每 lead 的 MAE/bias及日期块区间，补齐计划要求，不训练。
2. 按 target BJT 日期、lead、真实 GHI、clear、support缺失率、transport/hybrid COT强度和变化率分解配对误差。
3. 对高退化日期绘制四里 3×3 COT/GHI 轨迹、站点穿越时空剖面和预测 AGRI 云场，检查云过境错位。
4. 只读核对服务器 `day_assignment.csv` SHA、九个 `best.pt` SHA 和三个 bank 的关键数组 header/抽样 sequence-lead 值。
5. 搜索是否留有 step1–99 的状态快照或临时 checkpoint；若没有，现有产物无法只读判断早期最优。
6. 后续如重新实验，应将 batch 限制在唯一训练行数以内、在首100步密集验证，并加入真正的有效容量对照；这属于新实验，不是本审计结论。

review_independence: same-family  
acceptance_status: provisional

---

## 第二轮只读审计附录：早期重放、并集选点、案例对齐与诊断图

### 总体判定

**总体 severity：HIGH｜status：WARN｜integrity status：provisional**

新增产物改变了第一轮结论的解释边界，但不改变总体 WARN、旧划分四里站 validation-only 和 same-family/provisional 判据：

- 原检查点下，transport/hybrid 的平均 RMSE 确实比 A 高 `4.3862/4.6801 W/m²`。
- 对三臂统一补充早期候选后，transport/hybrid 数值上分别比 A 低 `1.7192/0.9658 W/m²`，但两个日期块95%区间均跨零。
- 因此，“COT 臂 RMSE 更差”只能描述原来的每100步选模结果，不能再解释为 COT 特征本身有害。
- 新结果仍是在同一 validation 上事后扩展候选、选模、估计效应和区间；它不是独立增益确认、test 结果、新月度结果或两站结果。

### 1. step100 重放复现

**severity：LOW｜status：PASS，附 provenance WARN**

执行路径：

- `diagnosis_20261015/replay_early_training.py:12-17` 直接加载原 `paired_pack.npz` 和 train-only COT normalization。
- `replay_early_training.py:24` 对所有九组统一使用候选步 `{1,2,5,10,20,30,40,50,75,100}`。
- `replay_early_training.py:25-30` 按 seed 重建与原实验相同的8192行训练 stream，重置模型初始化，并断言 `initial_sha256` 等于原运行。
- `replay_early_training.py:31-39` 复用原1200步 cosine schedule 的前100步，并在每个候选点保存预测。
- `replay_pbs.log:13-22` 记录九组 `step100_difference 0.0` 和 `EARLY_REPLAY_COMPLETE`。
- `early_replay_attempt1/summary.json:state=COMPLETE_EXACT_PREFIX_REPLAY`、`new_test_read=false`。

逐组比较 replay 与原 `results_attempt2/<arm>_seed<seed>/curve.json` 的 step100：

| 运行 | step100 RMSE | ΔRMSE | ΔMAE | Δbias | n |
|---|---:|---:|---:|---:|---:|
| A42 | 159.978852814510 | 0 | 0 | 0 | 5855 |
| transport42 | 161.483765442626 | 0 | 0 | 0 | 5855 |
| hybrid42 | 161.933566579898 | 0 | 0 | 0 | 5855 |
| A43 | 163.350661176654 | 0 | 0 | 0 | 5855 |
| transport43 | 168.418679821588 | 0 | 0 | 0 | 5855 |
| hybrid43 | 162.857224468577 | 0 | 0 | 0 | 5855 |
| A44 | 160.645327979787 | 0 | 0 | 0 | 5855 |
| transport44 | 161.471774558585 | 0 | 0 | 0 | 5855 |
| hybrid44 | 161.159066725927 | 0 | 0 | 0 | 5855 |

全部90个 `pred_step*.npz` 均只有 `pred_ghi,true_ghi,row_ids` 三个数组；每份均为5855行，row和GT与原 validation 逐元素相同。所有预测独立重算的 RMSE/MAE/bias 与对应 `curve.json` 差为零。

数组级证据边界：

- 原最佳步恰为100的五组——transport42、hybrid42、hybrid43、transport44、hybrid44——其 replay step100 `pred_ghi` 与原保存预测逐元素完全相同。
- A42、A43、A44、transport43 的原 step100 预测没有保存，只能证明 curve 指标完全相同，不能证明预测逐元素相同。
- `replay_early_training.py:40` 实际只断言 RMSE 差 `<0.001`，并未要求零差或比较 MAE/bias/数组。因此 `summary.json:state` 中的 “EXACT” 应理解为当前观测到的指标一致，不能泛化为九组完整状态、权重和逐样本预测均已证明相同。
- replay 的 `train_batch_rmse` 与由原保存 `loss` 反算的值最大差约 `9.22×10⁻⁶ W/m²`，来自 float32 loss 序列化后再开方，不构成训练漂移。

**severity：MEDIUM｜status：WARN：执行 provenance 不完整。**

`replay_early_training.py:9` 从可变的服务器脚本导入 `Head/seed/metrics`，没有保存或校验导入脚本哈希。当前本地脚本与 `train_attempt2_as_run.py` 的这三个 AST 定义一致，step100 行为也完全复现，但这仍不是远端源码身份收据。

此外：

- `replay_run.pbs:21` 应生成 `replay_allocation.csv`，本地诊断包未包含该文件。
- 现有 `allocation.csv:1-2` 是 `run.pbs:21` 所对应的案例提取任务分配记录，不是 replay 分配记录。
- `replay_pbs.log` 有完成标志，但未附独立 exit receipt。
- `early_selection_diagnostic.json` 和 `early_selected_predictions.npz` 没有保存 producer 脚本或文件哈希清单。

这些问题不推翻数值一致性，但限制执行链的不可变复核。

### 2. 并集 checkpoint 评价公平性

**severity：LOW｜status：PASS，附 post-hoc HIGH/WARN**

每个 arm/seed 的唯一候选集合都相同：

```text
1, 2, 5, 10, 20, 30, 40, 50, 75,
100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200
```

即每组21个候选。step100 在 replay/original 中重复，但三臂均重复一次，不产生额外机会。

公平性核验：

- **初始化**：`replay_early_training.py:28-29` 对每臂重置相同 seed，且初始哈希与原运行相符。
- **训练 stream**：`replay_early_training.py:25-27` 在 arm 循环外按 seed 构造，三臂使用相同行序。
- **模型、优化器、LR和更新数**：`replay_early_training.py:28-36` 相同。
- **验证支持**：所有90份早期预测使用完全相同的5855个 `row_ids` 和 `true_ghi`。
- **选模指标**：逐 seed、逐 arm 最小 validation RMSE。
- **候选机会**：三臂各21个，没有额外候选。
- **tie rule**：没有发生相同 RMSE 的并列，未保存 tie rule 不影响当前选择。
- **保存预测**：`early_selected_predictions.npz:A/transport/hybrid` 的九行预测均与其声明来源逐元素相同；`row_ids,true_ghi,lead_minutes,target_day` 与原 validation 相同。

独立枚举21个候选得到：

| arm | seed42 | seed43 | seed44 |
|---|---:|---:|---:|
| A | step200 | step200 | step75 |
| transport | step40 | step20 | step40 |
| hybrid | step40 | step50 | step30 |

与 `early_selection_diagnostic.json:selection` 完全一致。所有六个 COT 运行的最佳点都早于100步；A44 也在75步优于原 step300。

**MEDIUM｜WARN**：`early_selected_predictions.npz` 没有显式 `seed` 数组、`init_time`、`target_time` 或输出哈希；当前可由逐元素匹配确定其第0/1/2行对应 seed42/43/44，但文件本身不自描述。

**MEDIUM｜WARN**：未找到生成 `early_selection_diagnostic.json` 和 `early_selected_predictions.npz` 的保存脚本。`replay_early_training.py:39-44` 只写早期预测、曲线和 replay summary；`plot_diagnosis.py:16,75-79` 只读取并集产物。因此当前可以证明其数值正确，但不能完整复现其生成命令。

### 3. 并集选点的独立重算结果

**severity：LOW｜status：PASS for arithmetic**

逐 seed 指标：

| arm/seed | step | RMSE | MAE | bias |
|---|---:|---:|---:|---:|
| A42 | 200 | 156.900961 | 111.715816 | -38.569545 |
| A43 | 200 | 154.886074 | 109.557932 | -34.393404 |
| A44 | 75 | 159.486408 | 112.215909 | -36.189917 |
| transport42 | 40 | 154.540596 | 109.007526 | -18.238213 |
| transport43 | 20 | 155.952954 | 114.888668 | -29.110213 |
| transport44 | 40 | 155.622342 | 109.847139 | -20.873942 |
| hybrid42 | 40 | 155.494826 | 111.420393 | -25.952120 |
| hybrid43 | 50 | 155.426008 | 110.443670 | -23.579902 |
| hybrid44 | 30 | 157.455243 | 115.366278 | -29.652084 |

三 seed 平均：

| arm | RMSE | MAE | bias |
|---|---:|---:|---:|
| A | 157.091148 | 111.163219 | -36.384289 |
| transport | 155.371964 | 111.247777 | -22.740789 |
| hybrid | 156.125359 | 112.410114 | -26.394702 |

按 `independent_score.py:16-32` 的相同60个 target-date、2000次配对日期块 bootstrap 独立重算，差值定义为 `COT−A`：

| 对比 | RMSE差及95% CI | MAE差及95% CI | bias差及95% CI |
|---|---|---|---|
| transport−A | `-1.719184 [-5.559575, 2.197678]` | `+0.084558 [-2.317437, 2.302517]` | `+13.643499 [7.712494, 20.083478]` |
| hybrid−A | `-0.965789 [-4.561323, 2.532427]` | `+1.246895 [-1.217554, 3.512871]` | `+9.989586 [3.762542, 16.697742]` |

RMSE点估计和区间与 `early_selection_diagnostic.json:results.transport/hybrid` 完全一致。对应 RMSE 降幅为：

- transport：`1.094386%`
- hybrid：`0.614795%`

MAE和bias区间是本次审计从数组补算的诊断值，未存入原 JSON。正 bias 差表示 COT 臂的负偏差绝对值变小，不能直接等同于确认性性能增益。

### 4. 对第一轮解释的修订

**severity：HIGH｜status：WARN**

原检查点结果仍是事实：

```text
A          157.303235
transport  161.689429  （+4.386194）
hybrid     161.983286  （+4.680051）
```

并集选点变为：

```text
A          157.091148
transport  155.371964  （-1.719184）
hybrid     156.125359  （-0.965789）
```

transport/hybrid 相对 A 的符号分别移动约 `-6.1054/-5.6458 W/m²`。这是在数据、模型和三臂候选机会保持一致时，仅改变早期验证密度产生的结果。

据此可确认：

- 原每100步验证一次的协议错过了六个 COT 运行在20–50步的更低 RMSE。
- 4041行训练、batch8192 使每步约相当于 `2.027` 遍训练数据；20–50步已约等于 `40.5–101.4` 个 epoch。
- 第一轮“首次验证已太晚”和快速过拟合判断得到加强。
- 原“COT 臂总体 RMSE 更差”是粗选模网格下的 checkpoint 结果，不能作为 COT 表示本身退化的因果证据。

仍不能确认：

- COT 带来真实泛化增益；
- transport 或 hybrid 在独立数据上优于 A；
- 负结果完全由验证频率造成；
- COT质量、有效容量、融合结构和优化速度各自的贡献。

日期块区间没有计入“先在同一 validation 上搜索21个候选”的选择不确定性，因而仍是 post-selection 描述性区间。

推荐修订表述：

> 在四里站旧划分的同一5855行验证集上，将三种方法统一扩展到相同的21个 checkpoint 候选后，transport 和 hybrid 相对零 COT 对照的三种子平均 RMSE 点估计分别低 `1.72` 和 `0.97 W/m²`，但日期块95%区间均跨零。该事后诊断表明原先“COT 臂 RMSE 更差”的结论对每100步一次的粗选模网格敏感；当前证据既不能确认 COT 增益，也不能支持 COT 普遍有害。

必须同时保留：

> 这是同一 validation 上的事后 checkpoint 诊断，不是独立增益确认、test 结果、新月度结果或两站结果。

### 5. `alignment_audit.json` 和四个案例的证据边界

#### 5.1 时间、row、lead 与 pack 对齐

**severity：LOW｜status：PASS for four sampled cases**

`extract_real_cases.py:35-44,57-61` 对四个案例完成：

- 24个 `relpaths` 为8历史+16未来，15分钟连续；
- UTC文件名加8小时后，index7等于初始化时刻；
- index8–23等于 `init+15…240 min`；
- 每案例16个 `row_ids` 与原预测和 `paired_pack.npz:val_rows` 逐位相同；
- pack lead 为0–15，对应保存 lead 15–240分钟。

具体时间范围：

| sequence | init BJT | 24帧 BJT 范围 |
|---:|---|---|
| 2312 | 2025-08-05 09:30 | 07:45–13:30 |
| 762 | 2025-07-12 15:15 | 13:30–19:15 |
| 530 | 2025-07-09 11:15 | 09:30–15:15 |
| 3959 | 2025-09-03 11:00 | 09:15–15:00 |

没有发现 UTC/BJT、legacy/canonical lead 或 row串行错误。

#### 5.2 空间和输入对齐

| JSON key | 实际建立的证据 | 证据边界 | 判定 |
|---|---|---|---|
| `station_global_rc=[163,148]`, `crop64_top_left=[131,116]`, `station_local_rc=[32,32]`, `crop3_slice=31:34` | 内部裁剪算术和3×3中心一致 | 这些数值由 `extract_real_cases.py:50,60-65` 硬编码；没有重新读取 grid_static、站点经纬度或原HDF row/col | MEDIUM/WARN |
| `target_geometry_max_abs_diff=0` | 16个未来时效完整64×64 geometry 与 forecast image bank 一致，见 `extract_real_cases.py:53-54` | 不证明13个 forecast AGRI通道物理正确，也不证明 forecast 等于未来观测 | LOW/PASS |
| `history_motion_max_abs_diff=0` | 四案例用8历史帧重算 motion 与 bank 一致，见 `extract_real_cases.py:55-56` | 不证明 motion 是真实云团运动 | LOW/PASS |
| `transport_pack_max_log_diff=0`, `hybrid_pack_max_log_diff=0` | 重放64×64输出的当前 lead 中心3×3与 head pack 一致，见 `extract_real_cases.py:51-52,57-60` | 只覆盖四案例，不证明预测 COT 的物理准确性 | LOW/PASS |
| `image_pack_max_abs_diff=0` | image bank 当前 lead 的16输入通道中心3×3与 pack 一致，见 `extract_real_cases.py:61` | 案例 NPZ 只保留三个 RGB AGRI通道和 geometry，不能单靠案例文件重核其余10个 AGRI通道 | MEDIUM/WARN |
| `history_R_replay_max_log_diff=0` | 历史 AGRI 经当前冻结 R 的输出与 history bank 一致，见 `extract_real_cases.py:46-50,65` | 不证明 R 输出是真实 COT | LOW/PASS |
| `input_future_observation=false` | transport/hybrid 数据流只使用 history/forecast bank 和 motion，见 `extract_real_cases.py:51-52` | JSON布尔值本身是硬编码；案例文件实际包含未来观测用于诊断 | MEDIUM/WARN |
| `actual_future_used_for_diagnostic_only=true` | 未来观测只用于 RGB、same-time R代理和 geometry核对，见 `extract_real_cases.py:40-50,53-64` | 只证明当前所读 producer 路径 | LOW/PASS |
| `test_payload_read=false` | producer仅选择 manifest `val`、val bank 和 pack的 `val_*`，见 `extract_real_cases.py:20,24-29,35-36` | 不是系统级文件访问日志 | LOW/PASS |

第一轮“3×3中心无 off-by-one”和“案例路径无未来观测输入”判定保持 PASS；“全 bank payload 已逐值对齐”仍未解决，只从未核验提升为四个有意选择案例的抽样证据。

#### 5.3 NPZ数组含义

- `rgb,rgb_valid`：16帧未来观测 AGRI 的 C03/C02/C01及有效掩膜，仅供事后诊断。
- `forecast_rgb`：SimVP预测 AGRI 的同三通道反标准化值。
- `oracle_cot`：冻结 R 对未来观测 AGRI 的 same-time输出；它不是实测 COT，数组名 `oracle` 不应被解释为物理真值。
- `oracle_history`：冻结 R 对8历史帧 AGRI的输出；四案例与 `history` 逐元素相同。
- `transport`：历史末帧 COT按 `motion` 输运所得16帧。
- `hybrid`：冻结 history-control 模型输出。
- `transport_support`：输运反向采样仍在64×64域内的 mask；它不是 AGRI/CPP有效性或COT质量 mask。
- `cpp,cpp_valid`：CPP检索参考及其有效掩膜，不是独立真值。
- `geometry`：未来16时效确定性太阳几何。
- `relpaths`：24帧UTC相对路径，文件本身没有显式时区字段。
- `motion`：`[dy,dx]`，像元/15分钟。

案例文件未保存 `init_time,lead_minutes,true_ghi,pred_ghi`；必须结合 `selected_cases.json`、原预测和 `paired_pack.npz` 才能复算指标。

### 6. 案例数值与案例选择标签

#### 6.1 数值一致性

**severity：LOW｜status：PASS**

四个案例均满足：

- row、lead、support和geometry一致；
- transport/hybrid中心3×3与 pack 的 log-COT最大差分别不超过 `5.96×10⁻⁸/2.98×10⁻⁸`；
- `selected_cases.json:metrics` 的隐含定义为先逐 seed 算16时效指标，再对三个 seed 的指标取平均；
- 所有案例 RMSE/MAE/bias 重算最大差低于 `6×10⁻¹⁴`；
- `delta_rmse` 精确等于 `(RMSE_transport+RMSE_hybrid)/2−RMSE_A`；
- `ghi_range` 精确一致；
- `cpp_peak` 是中心像元16时效有效CPP峰值，保存物理数组重算差不超过 `1.53×10⁻⁵`。

#### 6.2 “最大改善/最大退化”标签错误或缺少隐藏筛选条件

**severity：HIGH｜status：WARN**

按 `selected_cases.json` 自身使用的 `delta_rmse` 定义，对全部449个 validation sequence 重算：

- 真正最大退化是 sequence `660`，init `2025-07-11T07:45+08:00`，`delta_rmse=+179.7172 W/m²`。
- 真正最大改善是 sequence `2037`，init `2025-08-01T12:45+08:00`，`delta_rmse=-70.5000 W/m²`。
- sequence2312 的 `+70.6658` 是全体和260个完整16-lead sequence中的第5大退化。
- sequence762 的 `-51.9410` 是全体第4大、完整16-lead sequence第3大改善。
- sequence530 可称高CPP参考案例，但没有证据证明是最厚云案例。
- sequence3959 的中心CPP峰值 `0.03372` 支持 near-zero reference，但不能证明它是随机或代表性 control。

仓库内没有生成 `selected_cases.json` 的 producer或候选表：

- `extract_real_cases.py:35` 只读取该 JSON。
- `plot_diagnosis.py:14` 也只读取它。

若存在额外质量、可视性或时段筛选，必须保存筛选规则和候选全集。否则应将标签改为“高退化案例”“强改善案例”，不得称 largest。

### 7. `global_decomposition.json`

**数值 severity：LOW｜status：PASS；解释 severity：MEDIUM｜status：WARN**

所有 subgroup 的 n 和三臂 RMSE/MAE/bias均可重算，最大差约 `2.2×10⁻¹³`。

根据数组精确反推，其未写明的定义为：

- `support_full/zero/partial`：当前目标 lead 的 `val_support[row,8+lead]` 3×3分别全真、全假或部分有效，n=`5593/220/42`。
- `kt_ge_09/kt_lt_04`：`val_ghi/val_clear >=0.9` 或 `<0.4`，n=`3034/785`。
- `cot_ge30/cot_lt01`：transport臂当前 target lead 的中心像元 `expm1(val_cot_transport[row,8+lead,1,1]) >=30` 或 `<0.1`，n=`211/1962`。

限制：

- 文件没有 definitions、单位、seed聚合规则或 producer。
- `cot_*` 使用 transport 自身输入分组，不是 CPP或未来R独立参考。
- `cot_lt01` 混入所有220个 `support_zero` 行。
- `kt_*` 使用目标期观测 GHI，是事后分层，不能用于业务条件选择。
- `support_partial` 只有42行；所有 subgroup 均无独立区间。
- 该文件使用原检查点预测，不是早期并集选点预测。

因此只能用于描述性诊断，不能证明 COT 在某种天气下存在因果收益。

### 8. 两张诊断图的只读图审

#### 8.1 `training_and_early_checkpoint_diagnosis.png`

**severity：MEDIUM｜status：WARN**

图像尺寸 `2440×771`，三面板总体可读。以下内容准确：

- 左图区分验证实线和当前训练批次虚线。
- 中图时效为更新步数，三臂使用相同早期网格。
- 右图原检查点 `+4.39/+4.68` 和补早期 `-1.72/-0.97` 与数组一致。
- 右图早期两个95%区间都跨零。
- 底部说明明确指出同一 validation、post-hoc、不能作为独立增益确认；该说明准确。
- `plot_diagnosis.py:75-82` 使用的点估计和区间来源正确。

必须修复：

1. 底部“第100步逐组精确复现”表述过强。应改成“九组 step100 的 RMSE/MAE/bias 一致；其中五组有原保存数组并逐元素一致，另外四组无原 step100 数组可核”。
2. 混合臂补早期的 `-0.97` 标签负号与零线/误差线重叠，成图近似显示为 `0.97`。负号决定方向，必须移动文字、加白底或改用 Unicode `−`。
3. 中图阴影在 `plot_diagnosis.py:71-72` 是三 seed 最小–最大范围，不是 CI；图注必须明确写出。
4. 左图虚线是当步8192条重复批次的 RMSE，不是全训练集指标；建议在图注直接说明，避免把它当作完整 train curve。

可选修复：

- 在右图标签中直接附 `95% CI跨0`，减少读者只看点估计造成误读。
- 给三个面板添加 `(a)(b)(c)`。
- 将原 transport CI下限 `0.080` 标在图注中；当前分辨率下视觉上近似接触零线。

#### 8.2 `case_2312_real_cloud_cot_ghi.png`

**severity：HIGH｜status：WARN before external use**

图像尺寸 `2025×2124`。以下内容通过：

- 初始化 `2025-08-05 09:30 BJT`；
- 顶部 `+60/10:30、+120/11:30、+180/12:30、+240/13:30` 全部正确；
- `plot_diagnosis.py:37` 的目标时间换算正确；
- 观测 AGRI与预测 AGRI分行显示，并使用相同通道、相同拉伸；
- GHI横轴为未来时效，黑线为以目标时刻结束的15分钟实测均值；
- 底部GHI使用原检查点三 seed 均值，阴影为 seed范围，图注已说明；
- `rgb_stretch.json:low/high/gamma/shared_across_all_cases` 与 `plot_diagnosis.py:17-22` 一致，四案例共用同一拉伸。

必须修复：

1. `plot_diagnosis.py:38` 色条写“物理 COT”，证据不支持。R和CPP的物理QA尚未解决，应改成“R/模型 COT尺度（无量纲）”或“检索代理 COT”。
2. 行标签“真实 AGRI→R”容易被读成真实 COT。建议改为“观测 AGRI→冻结R（事后代理）”。
3. 图注明确增加：“未来观测 AGRI仅用于事后诊断，不是模型输入；R与CPP均非独立 COT真值。”
4. 若正文或图注称 sequence2312 为“最大退化”，必须改为“高退化案例”或补充真实筛选规则。
5. 与早期选点诊断并列展示时，应在面板内更醒目地写明“GHI曲线使用原检查点”，避免读者误认为它使用并集选点预测。

可选修复：

- 写明四列只是16个时效中的4个展示时效。
- 增加灰色区域表示 `transport_support=false` 的图例。
- `rgb_stretch.json` 可补充 low/high 来自四个所选观测案例的2%/98%分位数及裁剪比例。
- 将“伪彩色”进一步注明为 C03/C02/C01 reflectance proxy。

### 9. 对第一轮 PASS/WARN 的影响

- **A Ground truth provenance：不变，PASS。** 新产物没有改变 GHI真实观测来源；案例中的 `oracle_cot` 仍是模型代理。
- **B Score normalization：不变，PASS。**
- **C Result consistency：原结果不变；新增 replay/union 数值 PASS，但 provenance WARN。**
- **D Execution path：新增 replay 路径可追到代码和日志；union selection及两个诊断 JSON producer缺失，维持 WARN。**
- **E Scope/claim ceiling：维持 HIGH/WARN，且必须把“COT退化”缩窄到原检查点协议。**
- **F Evaluation type：仍为 `real_gt`，但只是旧划分四里站 validation。**
- **时间/lead/3×3对齐：四案例进一步支持 PASS。**
- **全 bank逐值对齐：仍未全局证明。**
- **训练协议：从“疑似漏掉早期最优”升级为“确认六个 COT运行的更优点位于20–50步”。**
- **科学增益：仍未确认。**

### 10. 已确认、已排除、未解决

#### 已确认

- 九组 step100 validation RMSE/MAE/bias 与原 curve 完全一致。
- 五组有原 step100预测可逐元素确认；四组只可确认指标。
- 三臂并集候选网格、初始化、stream、validation row/GT和选择指标公平一致。
- 任一 arm 都没有额外候选机会。
- 六个 COT运行的并集最优点均在20–50步。
- 并集选点下 COT 两臂的 RMSE点估计转为轻微改善，但区间跨零。
- 四案例的时间、row、lead、中心crop、support、geometry及 pack handoff一致。
- `selected_cases.json` 的两个 largest 标签不是全 validation 极值。
- `global_decomposition.json` 数值可复现，但定义和 producer未保存。
- 诊断图对时效、观测/预测 AGRI和 post-hoc CI跨零的主体说明基本正确。

#### 已排除

- 并集结果由 COT 臂获得更多候选造成。
- 并集结果由不同 validation rows或GT造成。
- step100 指标差异由随机重放漂移造成。
- 四案例中存在15分钟时轴、UTC/BJT或中心3×3 off-by-one。
- 当前所读案例 producer把未来观测 AGRI送入 transport/hybrid输入。
- 右图“补早期后两个区间跨零”的说明错误。

#### 未解决

- 在独立 selection/test 数据上，COT是否真正改善 GHI。
- 选点不确定性计入后区间是否仍跨零或进一步变宽。
- 有效容量、过拟合、COT质量、融合结构和优化速度的各自贡献。
- 其余445个 validation sequence 及训练 bank 的逐值空间/时间对齐。
- 四里全局像元位置相对 grid_static、站点经纬度和原HDF索引的再次独立核验。
- R/CPP的物理 COT质量。
- replay远端导入脚本身份、完整退出收据和文件哈希清单。
- `selected_cases.json` 是否存在未保存的额外筛选条件。
- 新月度、竺家站、test和部署泛化。

review_independence: same-family  
acceptance_status: provisional
