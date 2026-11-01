## 2026-10-30 执行简化：不默认追加内部隔离与审查层

下方为旧候选的评价与调度历史，不再作为所有新实验通用准入门。采用[简化方式](../../docs/execution_simplification_20261030/CURRENT_POLICY.md)：可直接评价全train级联/冻结头的因果GHI预报；不以缺少OOF、CPP指标未过门、某分站CI跨零或无新日期为通用停止理由。旧AC−AH多重判据回答的是特定历史/未来COT分解，不能覆盖其他有效问题，也不把旧未通过记录改成通过。

---

# 从预测 COT 到未来 GHI 的验证门

更新：2026-10-19。此文件只规定后续工程实验，不改论文正文，也不打开 test。

## 要回答的问题

在真实部署式输入下，加入由预测 AGRI 反演得到的预测 COT，能否改善两站 +15 至 +240 min 的 15 min GHI 预报？COT 指标变好本身不算回答了这个问题。

## 预测链与配对对照

- 所有可部署臂均从初始化时刻可用的 8 帧历史 AGRI 产生 16 帧未来 AGRI；未来输入不得使用真实目标 AGRI 或 CPP。
- 逐像元冻结反演器只处理预测 AGRI，得到对应时效的预测 COT。历史 COT 可由历史 AGRI 经同一冻结反演器计算。
- 站点 GHI 头使用相同的 3×3 站点邻域、相同的 AGRI/太阳几何、相同 15 min 目标标签、站点、初始化/目标键和白天掩膜。每个目标时效的三帧窗口以该目标帧结束；早期窗口可包含初始化前的历史帧。
- 固定 SimVP 预测 AGRI 后使用三个同容量配对臂：A 不输入 COT；AH 仅输入初始化前已观测 AGRI 经冻结 R 得到的历史 COT，并将预测未来 COT置零；AC 输入同一批历史 COT及`R(predicted AGRI)`得到的未来预测 COT。三组使用相同样本、初始化、日期顺序、优化器、轮数、标签、归一化和配对种子，分别独立拟合头部。
- `AC−AH` 是未来预测 COT 的主对比，隔离“未来 COT”相对历史反演 COT的增量；`AH−A`量化历史COT作用；`AC−A`保留与既有“无COT vs +预测COT”结果一致的总输入对照。只看AC−A无法分辨增益来自历史还是未来COT，因此不单独据此宣称未来COT有效。
- 只有当前 COT 辅助 SimVP 通过 COT/AGRI/厚云护栏后，才追加一个独立的上游方法对照；报告中分开“加 COT 输入”与“改变 AGRI 预测训练目标”的效应，不能混为单一 COT 增益。

## 防止训练输入泄漏

GHI 头训练样本的预测 AGRI/COT 必须由未见过该初始化日期的上游模型产生。优先使用按日期分组的两折交叉拟合：每折上游模型仅在另一折训练日期上拟合，再为留出折生成 8→16 预测；验证集只由最终 train-only 上游 checkpoint 生成。若算力暂时只允许同一 SimVP 对训练样本做 in-sample 推理，该结果只能标为筛选 pilot，不能作为论文主结果；之后必须补交叉拟合或独立训练期证据。

## 评价范围和停止规则

- 使用固定月度合同的 train/validation 日期和统一标签；不读取 test/buffer 载荷。validation 日期分布在 26 个连续月份、范围为 2024-04 至 2026-06，按月留出而非连续一年；应明确写“跨 26 个月的分散验证日”，不得称作连续一年验证。
- 主指标为两站等权、共同样本上的 GHI RMSE；同时报告站点、+15 至 +240 min 分时效、MAE、偏差和目标日期块配对 bootstrap 区间。所有指标由保存的逐行预测和标签独立重算。
- 真实未来 AGRI→R 只可作为单列 oracle 信息上限诊断，不能进入可部署模型或代替 forecast-COT 结果。
- 实际可用的未来GHI收益要求 AC−A 在配对验证中稳定为负、目标日期块区间不跨0、两站没有不可接受退化，且早期时效不被远期平均掩盖。要把收益归因于“未来预测COT本身”，还要求 AC−AH 同样显示稳定增益；若只有AC−A改善而AC−AH不改善，只能说明整体COT侧车有效，不能归因于未来COT。筛选pilot通过后，仍需用上游日期留出/交叉拟合预测训练输入和独立冻结评价确认；否则保留为探索结果。
- 为定位可能的COT噪声，另按目标帧站点中心的**预测**COT分层为 `<1、1–10、10–30、≥30`，并按 `+15–60、+75–120、+135–240 min` 汇总A/AH/AC误差。该分层只作验证集描述诊断，不用于按某一层挑模型，也不将预测COT区间称为CPP真值云类；重叠预报实例不作独立显著性样本。

## 当前门状态

截至 2026-10-19 03:20 CST，M1 SimVP+COT 辅助训练仍处于 PBS R 状态，step-2000 checkpoint 与正式验证 JSON 已落盘；候选 `val_equal_season_rmse=0.2856219593`，同合同 MSE 对照 step-2000 为 `0.2856157556`，候选暂高约 0.0022%（AGRI 护栏仍在 +1% 内）。这只是 AGRI 中期检查，不是 COT 独立审计或 GHI 收益。截至 03:40 CST，M1训练记录到 step 2800/4000；step-2000验证已完成，但 step-4000同掩膜 COT/厚云护栏尚未完成。截至 03:39 CST，bank 已按恢复边界 6,976/33,253 重启，PBS 214883 已在 node22 GPU0 运行，拟固定 batch=64；test_payload_opened=false、future_observed_agri_read=false。吞吐profile：batch=64单批64序列耗时24.70秒、2.59序列/秒、峰值CUDA分配28.94 GiB；batch=96单批96序列耗时48.40秒、1.98序列/秒、峰值43.38 GiB；batch=128瞬时显存70.2/81.9 GiB，低于15%余量门槛。按实测速度选择batch64，保留显存和速度余量，不再继续调批次。账户 PBS 3/3（214221 Q、214871 R、214883 R），不得提交新作业。本文档是准备好的下一阶段合同，不代表 GHI 训练已启动。旧日期合同的 `forecast_ghi_window_20261004_v1` 以及使用真实 AGRI 历史帧的 `r_geometry_downstream_ghi_20261016` 均不能代替本未来预测对照。

2026-10-19 补充：已按固定月度序列清单与 `ghi_head_pack_new_r_v1/rows.csv` 建立 metadata-only 精确配对，产物为 `results/future_ghi_crosswalk.json` 和 `results/future_ghi_crosswalk_keys.npz`。候选 1,230,720 个 station×init×lead 键中，781,504 个 train/val forecast-instance 键找到同 split 的 GHI 标签（train 714,914；val 66,590），split mismatch 为 0；逐条独立重建 init/target UTC、站点、lead 和原始 GHI 行号的复核通过，见 `results/future_ghi_crosswalk_independent_audit.json`。源帧/标签每个目标可因不同初始化时刻重复出现，这是不同预报实例；独立不确定性仍按目标日期块处理。此处只确认键与合同，不是模型成绩，也没有读取任何 AGRI/CPP/GHI 数组、test 或 buffer 载荷。下一步是待当前 M1门和算力允许后，为 train/val 同一批初始化生成 frozen-SimVP 的 8→16 预测输入、由预测 AGRI 经冻结R生成逐时效 COT，再按这些键做 A/AC GHI pilot；若训练预测仍为 in-sample，该pilot只作筛选，不能作为论文主结果。


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

04:02 CST状态：PBS 214221=Q、214871=R、214883=R（3/3）。M1 step 3700/4000，最终验证与COT审计尚未生成；预测AGRI→冻结R bank为9792/33253，仍在I/O阶段且无完成manifest。test/buffer/未来真实AGRI载荷仍关闭。等待当前任务完成后先做独立COT同掩膜审计与bank完整性检查，再决定CPP-only消融和GHI A/AH/AC pilot；当前没有新的GHI收益结论。

04:03 CST复核：M1已推进到step 3800/4000；bank的durable进度仍9792/33253，但数据数组于04:03更新、index于04:02更新，处于下一512条进度flush前的写入阶段，不能把未刷新的progress当作停滞。PBS全队列仍3/3，test/buffer保持关闭。等M1最终评估审计与bank完整manifest；门禁前不启动后续训练。

04:04 CST复核：bank progress已前进到10304/33253（329,728个forecast instances，BUILDING；test_payload_opened=false、future_observed_agri_read=false）。M1最近durable快照step 3800/4000（04:02），PBS仍R，等待百步保存点及最终同掩膜审计。全账户PBS 3/3，无新增作业或重启。

04:07 CST状态：M1 durable step 3900/4000，GPU仍活跃；最终验证/审计未生成。bank durable 10304/33253，约2分钟未刷新，但该bank按512序列flush，且PID仍存活、逻辑读取增长；共享存储等待明显，尚未满足重启条件。全PBS 3/3；test/buffer/未来真实AGRI仍关闭。

04:09 CST更新：M1已到4000/4000更新，但step4000验证、终态标记和同掩膜COT审计还未落盘；训练不能算完。bank已到10816/33253（+512），排除真实未来AGRI/test的标记为false。队列仍满3/3，保持两项运行任务，后续门禁仍是M1审计、bank完整性，再进入未来GHI A/AH/AC对照。

04:10 CST核查：M1在step4000后仍处于验证/后处理中（GPU与worker进程活动），但没有step4000验证或终态审计文件。bank仍10816/33253，当前尚未到下一持久化flush，作业存活。全队列仍满3/3；测试与buffer排除继续生效。

04:11 CST更新：M1验证后处理持续活动，日志无错误匹配，但终态文件未生成。bank虽仍10816/33253 durable进度，预测数组有更新且GPU/读取活动恢复，处于批次处理中。PBS 3/3；test/buffer/未来真实AGRI仍未读取。等待实际审计收据和bank manifest后再推进下一训练。

04:13 CST M1 step-4000 AGRI验证：候选季节等权RMSE 0.2856077859，MSE对照0.2856004358，同3140验证窗；候选仅高0.00257%，因此当前AGRI护栏通过但两者实际持平。COT共同支持审计和GHI收益均待评，训练不得据此判正向。forecast-COT bank 11328/33253、3.35序列/秒，继续构建；PBS 3/3。

04:17 CST live checkpoint: PBS full-account queue is 214221=Q, 214871=R, 214883=R (3/3). M1 has saved `step_4000.pt` and `validation_step_4000.json` (3,140 validation windows; equal-season AGRI RMSE 0.2856077859 versus matched MSE 0.2856004358, +0.00257%). The candidate COT scoring directory `cot_score_aux_4000/` exists but contains no score files yet; original/MSE COT scores, independent audit, and `training_complete.json` are absent, so M1 remains in postprocessing and is not complete. Node22 shows PID19003 on GPU1 using 34.5 GiB; GPU1 sampled 6% utilization at 04:16, consistent with scoring/postprocessing rather than a failed job. Forecast-COT bank is durable at 11840/33253 (378,880 forecast instances), BUILDING; progress last flushed 04:14 and job PID107603 remains alive on GPU0 using 37.3 GiB. Its test/future-observed-AGRI exclusion flags remain false. Queue is full; do not submit or restart. Next gates remain final same-mask COT scores plus independent audit, and complete bank manifest/integrity before building packs and submitting the paired A/AH/AC GHI pilot.

04:19 CST new artifact: M1 produced `cot_score_aux_4000/cot_aux_validation_score.json`, status `PASS_CPP_COMMON_SUPPORT_FP64`, on the monthly-block validation split (3,140 sequences; 46 CPP-supported dates; zero test reads). Candidate center-3×3 station/lead-macro COT RMSE is 10.74063 and full-patch RMSE is 11.02875; pooled thick-cloud CSI is 0.42517. These are candidate-only figures: original-SimVP and MSE-step-4000 scores on the exact same support plus the independent audit are still pending, so no COT gain/loss conclusion is made. The forecast-COT bank advanced from 11,840 to 12,352/33,253 (395,264 instances; 2.686 sequences/s latest batch), remains BUILDING, and both exclusion flags are false. PBS remains full (214221=Q, 214871=R, 214883=R); do not submit/restart. Continue to the same-mask comparisons and audit, then complete bank validation before the matched-input GHI pilot.

04:20 CST Luna cross-check: bank progressed 11,840→12,352 during 04:16:47–04:18:03; by 04:19:33 durable progress was still 12,352, but forecast arrays were modified at 04:19 and PID107603 logical reads increased ~9.4 GB (last 8 s +41 MB), with shared-filesystem lock wait and no physical read increment. Treat as active batch before the next 512-sequence flush, not a confirmed stall. M1 candidate score/forecast arrays are present; `cot_score_original_simvp/` exists but is empty, MSE score directory is absent, and independent audit/`training_complete.json` are still absent. PID19003 plus four workers remain active (workers ~79–83% CPU; GPU1 sampled ~8%). Queue at 04:20:01 remains 214221=Q, 214871=R, 214883=R (3/3); all exclusion flags remain closed. Keep current jobs running; next work is still same-mask original/MSE comparison and audit, then bank completion/integrity and paired GHI screening when a PBS slot frees.

04:23 CST downstream-alignment check: reran `test_future_ghi_context.py` locally; `FUTURE_GHI_CAUSAL_PACK_WITNESS_PASS`. The synthetic witness verifies the three-frame T−30/T−15/T context, observed-history versus forecast source indices at +15/+30/+240 min, and the COT history-only mask boundary; it reads no project payload and does not replace the future pack's key-level audit. This removes a code-level time-context ambiguity before interpreting the planned A/AH/AC comparison. Remote 04:22:59 check still shows queue 3/3; only the candidate COT score is present, bank durable progress is 12,864/33,253, and the forecast arrays' mtime is 04:22, so the bank process remains actively writing ahead of its next durable flush. Do not restart either live job.

04:24 CST exact same-mask preliminary comparison: `cot_score_original_simvp` is now present and its `validation_dataset_indices.npy` is byte-identical to the COT-aux candidate indices. On the common 3,140-sequence / 46-CPP-date support, original SimVP center-3×3 station-lead macro COT RMSE is 10.6641613 versus 10.7406269 for COT auxiliary (candidate is +0.0764656 worse); full-patch RMSE is 10.9526937 versus 11.0287499 (+0.0760561 worse). Pooled thick-cloud CSI is 0.4139968 versus 0.4251665 (+0.0111697 better). These are point comparisons, not yet the paired date-block interval; MSE-step-4000 score and independent FP64 audit are still missing. The mixed-label COT auxiliary candidate therefore does not yet improve overall COT RMSE, despite a small CSI increase. Keep the planned CPP-only single-factor noise ablation as the next diagnostic after M1 audit; keep the independent original-SimVP forecast bank running because it is required for the direct downstream A/AH/AC GHI test.

04:26 CST follow-up: original-SimVP COT score is present and byte-identical validation indices to the aux candidate were confirmed; center-3×3 RMSE 10.6641613 (original) vs 10.7406269 (aux), pooled thick CSI 0.4139968 vs 0.4251665. MSE-step-4000 score remains in the same live M1 postprocessing sequence; no independent audit or completion marker yet. Forecast bank advanced another 512 to 13,376/33,253 (428,032 forecast instances; latest batch 2.62 sequences/s), still BUILDING with test/future-observed-AGRI flags false. PBS remains 3/3. Once MSE score lands, run the prewritten FP64 date-block audit; retain the direct forecast-COT bank for the later A/AH/AC GHI comparison.

04:28 CST verified progress: forecast-COT bank durable count advanced 13,376→13,888/33,253 (444,416 forecast instances; latest batch 27.85 s, 2.30 sequences/s), so the earlier ~2-minute durable gap was the normal 512-sequence flush cycle. M1 original-SimVP score remains complete; MSE-step-4000 score is still pending while job 214871 remains R and GPU1/process are present. Bank job 214883 remains R; test/future-observed-AGRI exclusions remain false. Queue remains 3/3 (214221 Q, 214871 R, 214883 R). Keep both jobs running; execute the paired date-block audit immediately after MSE scoring and `training_complete.json` appear.

04:30 CST three-arm score completion: MSE-step-4000 `cot_aux_validation_score.json` and `training_complete.json` are present; the training PBS script has launched `audit_cot_aux_validation.py` in the same allocation (live PID29195 at 04:29:27; independent audit JSON not yet written). Same-support center-3×3 RMSE: original 10.6641613, equal-budget MSE 10.7501119, mixed-label COT auxiliary 10.7406269; full-patch: 10.9526937, 11.0389034, 11.0287499. Pooled thick CSI: 0.4139968, 0.4238940, 0.4251665. Thus both fine-tunes are worse than original on point RMSE; COT auxiliary has a small thick-CSI gain, whose date-block interval and low-COT/AGRI guards remain pending. Do not select or launch CPP-only from validation point estimates alone; use the completed independent audit to decide. Forecast bank remains at 13,888/33,253, process alive, with exclusions false; PBS remains 3/3.

04:35 CST CPP-only noise ablation submitted as PBS `214884.tc6000` after M1's independent audit completed and the full account queue showed two unfinished jobs (214221=Q, 214883=R); post-submit queue is 3/3 including 214884=R. The candidate changes only the COT target source to CPP-only on common valid support; it holds seed 42, 4,000 updates, batch 8, AGRI objective, frozen R and the calibrated mixed-M1 lambda fixed. Server checksum verification and `CPP_ONLY_TARGET_WITNESS_PASS` completed in the job prologue; unique output path was clean. No test/buffer or future observed AGRI input is used. M1 gate outcome: overall fail; candidate COT-RMSE was +0.07647 versus original in center 3×3 (paired 95% CI [-0.09532,+0.24643]) and near-zero false alarm increased versus MSE (paired 95% CI [+0.00054,+0.00195]); thick CSI improved versus MSE but not enough to pass all gates. The separate original-SimVP forecast-COT bank remains at 13,888/33,253 and is required for the direct GHI A/AH/AC pilot.

## 2026-10-19 06:10 CST 续跑状态

CPP-only COT 辅助 SimVP 已完成4,000步及同掩膜三模型评分。独立审计 `seed42_cot_aux_cpp_only_v1/independent_cot_audit.json` 状态为 `PASS_INDEPENDENT_FP64_PAIRED_COT_AUDIT`：3,140个验证窗口、46个目标日期块、5,224,460个共同有效CPP像元，test载荷读取数为0。候选未通过预设筛选门：中心3×3 COT RMSE相对原SimVP增加0.0794（95%日期块区间[-0.1059,+0.2606]），相对等预算MSE仅减少0.0065且区间[-0.0303,+0.0129]跨零；厚云CSI相对MSE增加0.00307，但近零COT误报率增加0.00173且区间[+0.00082,+0.00290]不跨零。AGRI季节等权RMSE相对MSE变化+0.00095%，在1%护栏内。CPP是反演参考产品，不是独立物理真值。故该CPP-only候选仅留作诊断，不替换SimVP，也不追加该上游方法臂到GHI筛选。

当前只推进“原SimVP未来AGRI→冻结R→未来预测COT是否能改善GHI”的独立输入源筛选。forecast bank于06:16已到30,784/33,253，状态BUILDING；test_payload_opened=false、future_observed_agri_read=false。账户未结束PBS任务为214221=Q、214883=R（2/3），尚无bank_manifest，故当前等待剩余bank并在完整性门通过后再提交 `run_future_ghi_a_ac.pbs`。该筛选会明确标作pilot（训练期上游预测为in-sample），不能作为论文主结果；正式结论仍需日期交叉拟合或独立训练期上游输入。

06:30 CST续跑：bank_manifest状态`COMPLETE_FORECAST_AGRI_R_TRAIN_VAL_BANK`，rows=33,253、forecast_instances=1,064,096；索引独立检查得到train 30,436、val 2,817、seq_id重复数0，六个npy数组大小与声明shape相符。所有test/buffer/future-observed-AGRI/CPP载荷标记均false。按全账户PBS实时队列（214221=Q，214883=C）及脚本校验、唯一输出目录检查后，已提交GHI A/AH/AC筛选任务214888；提交后为214221=Q、214888=R。该运行是训练集上游预测in-sample的pilot；只有未来COT相对AH的AC−AH与总对照AC−A均通过日期块配对门，才进入下一阶段的交叉拟合/独立训练期方案，pilot本身不作为论文主结果。

## 2026-10-19 06:57 CST：当前GHI pilot与噪声收缩候选

修正标签pack路径后的PBS 214889运行正常，当前在CPU构建完整GHI因果输入包；train keys 90,073/714,914（12.6%），无完整manifest，因此尚无GHI头部结果。该pilot仍是in-sample上游预测筛选。若完整未来COT相对历史COT未能通过既定GHI门，下一单因素候选为 refine-logs/cot_aware_simvp_20261017/FUTURE_COT_ATTENUATION_PLAN.md中的AC50：只将未来标准化COT异常乘0.5，保留历史COT、AGRI、几何、头、样本和预算不变。其独立代码已准备；当前结果前不提交。若完整AC已通过，则先推进交叉拟合/独立训练期上游输入确认，不以在样本pilot代替。


07:00 CST：未来COT收缩AC50的独立代码、合成审计与提交脚本已备妥并同步服务器；所有7项SHA256及bash语法检查通过。主pilot 214889仍在打包，93,525/714,914 train键，尚无结果或manifest。只有当前pilot的AC−AH未通过未来COT GHI增量门时才提交AC50；若通过则转日期交叉拟合确认。
