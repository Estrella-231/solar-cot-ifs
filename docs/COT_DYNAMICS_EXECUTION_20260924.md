# COT dynamics 执行交接（2026-09-24）

本页只记录工程状态；科学问题、实验组、停止条件见 [`refine-logs/cot_dynamics_20260924/EXPERIMENT_PLAN.md`](../refine-logs/cot_dynamics_20260924/EXPERIMENT_PLAN.md)。状态可能随 PBS 和服务器产物变化，重启前必须实时复查，不能把本页的作业号当完成证明。

2026-09-24 完成快照：修复版 SimVP `210506.tc6000`、预测 bank `210593.tc6000`、配对 GHI 作业 `210598.tc6000` 均已完成并验收；后两者 PBS 退出码0。oracle train/val bank 已完整且不可部署。B1结果与日期级配对复核见 [`results/cot_dynamics_20260924/ghi_b1/README.md`](../results/cot_dynamics_20260924/ghi_b1/README.md)。E 未训练；大视场反演与源随动外推准确性仍是门禁，服务器当前状态须实时复查。

## 服务器与输入

- SSH alias：`zjnu-hpc`；用户 `slfu`；Hunan AGRI/aux 根目录：`/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed`。
- 第二篇工程：`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907`；本次隔离结果：其下 `experiments/cot_dynamics_20260924/`。
- 修复版 backbone 工程：`/public/home/slfu/ttzhou/swc/irradiance_forecast/HunanBackboneTriad_20260921`；**模型源码必须使用**其 `geometry24_v2/triad_models.py`，与 checkpoint 的 `model_code_sha256` 一致。其根目录旧 `triad_models.py` 与 geometry24 权重不匹配。
- 主 SimVP：`geometry24_v2/runs/simvp_geometry24_formal_e12_v1/best.pt`；同目录 `training_complete.json` 已记录12轮、41,520步，best权重SHA-256为 `32e24c620e3ad2939f8a4dea50599c4f657d8072193462707e5e7ca0feaf28fb`。AFNO 只作次要稳健性对照：`geometry24_v2/runs/afno_transformer_geometry24_formal_e12_v1/best.pt`。
- 冻结 R、历史 COT：第二篇工程的 `configs/r_frozen_repaired_seed42.json` 与 `data/history_cot_trajectory_real_agri_rpair_20260923/original_R/`。R SHA-256 为 `0cbdb932f49945259d5f1704316edaf6481be7cfdb54692236f829988925799c`，缓存已标 FP32、train/val、test=false。
- GHI 标签/行合同：`data/head_pack_trainval_20260908_v1/`。本轮必须重建图像输入 sidecar，不得沿用其中旧 SimVP 预测图的 `x.npy`，也不得沿用旧 `solarresnet_v5_input_trainval_20260918_v1`。

## 已完成与实测门禁

- `audits/b0_contract_v2.json`：train/val 27,675/4,258 个序列、538,053/99,849 个 GHI 行，全量起报/目标时刻、16 lead、3 条5分钟标签、站点中心 `[8,8]` PASS；按06:00、12:00、18:00 BJT固定时刻见证几何，test未读。
- `p16_cpu_train_pilot_64/` 与 `p16_cpu_pilot_16/`：历史 C13 相位相关平移的 P16 演示。+240分钟站点中心5×5有效来源覆盖分别50.4%（固定64条train）和51.3%（固定16条val），不能当完整4小时 COT 预报。
- `audits/p16_vs_p0_oracle_{train_fixed64,val_fixed16}.json`：在每个时效 **P16 有效来源像元完全相同的集合** 上，用冻结 R(真实未来 AGRI) 作离线参考，比较历史末帧保持不动 P0 与历史 C13 相位相关平移 P16；脚本和本地报告见 `scripts/evaluate_p16_vs_p0_oracle_20260924.py`、`audits/cot_dynamics_20260924/`。val 固定16序列的站点中心5×5，+15分钟 MAE 为 P0 0.188、P16 0.156，+240分钟为0.782、0.791，单位均为 `log(1+COT)`；train 固定64序列的对应数值为0.211/0.207、0.860/0.866。val 在前150分钟多有优势，train 未稳定复现；长时效半数来源无效，不能用此小样本筛选 E 或宣称 GHI 增益。真实未来 AGRI 仅在离线参照侧，未送入 P0/P16。
- `audits/cot_dynamics_20260924/source_following_native_r_*.json`：额外固定样本pilot按历史 C13 速度去最后一张历史 AGRI 的上游位置裁16×16，再用冻结 R 原生输入反演，避免把 R 改在32/64输入上或跨窗口拼接。val/train上游patch覆盖100%/98.4%，但+240分钟中心5×5相对P0误差在val更差、train略好，尚不能放行E。CPU重复原位R与既有GPU历史bank最大差0.0032/0.0051 log(1+COT)，零偏移一致性以记录的容忍度而非位级相等验收。
- B1 完成结果：A0/AH/AI/O_diag两站等权RMSE分别147.639/148.852/148.773/130.109 W/m²；完整同键验证99,849行、test=false。O_diag−AI为−18.664 W/m²；72个起报日期配对bootstrap 95%区间[−23.062,−14.327]，AI−AH的区间[−2.301,+2.048]跨零。O_diag只表明本链路可利用更好的未来COT，不能部署，亦非物理COT真实性证明。逐样本预测与审核JSON在 `results/cot_dynamics_20260924/ghi_b1/`，bootstrap在 `audits/cot_dynamics_20260924/ghi_b1_day_bootstrap_v1.json`。
- `audits/r_halo_train_cpu_witness.json` 与 `audits/r_tiling_train_cpu_witness.json`：原生16×16 R改为直接32/64，中心16平均差0.614/0.613 log1p(COT)；九块16重叠拼接的未锚定中心差0.325，强锚定后接缝跳变0.428，而原生内部相邻梯度0.154。val固定案例同方向。**E 的大视场 COT 输入门禁失败；不得直接把 R 改在大patch上推理或拼接后宣称同一反演合同。**
- B0 小样本 GPU：`210576` 因误用旧模型定义严格载权失败；修复哈希绑定后 `210577` 完成当前 best SimVP 2条、AFNO 4条 val 的24几何→16预测图→R全链路。`210578` 因 PBS 变量错误在模型前失败；`210579` 修复后在一张 A100 40GB 上完成 batch 8/16/24/32 的64序列 profile，峰值4.95/9.85/14.76/19.67 GiB、全流程吞吐3.45/3.67/3.34/3.15序列/秒；batch16暂优。短样本吞吐含启动/缓存，正式运行记录仍需核对。
- 真实未来 AGRI→R 仅用于 oracle 诊断：CPU 两序列小样本通过；PBS `210580` 已提交，先跑固定256序列GPU profile，再顺序生成 train/val 全量缓存。任何 `real_future_cot_log1p.npy` 禁止进入部署组推理。
- [真实验证 AGRI 的视场诊断图](../figures/cot_dynamics_20260924/r_halo_actual_val_sili_v2/r_halo_actual_val_sili.png) 固定取四个验证序列的历史末帧；三个白天四里案例中，原生16×16 R 与同权重32×32 R 的共同中心16差值 MAE 为0.213/0.147/0.425 `log(1+COT)`。第三行案例是夜间，图内标明。图旁 `source_arrays.npz` 和 `provenance.json` 保存原值及展示色标；这证明视场不一致，不能作为 GHI 预报提升证据。

## 接续顺序

1. `qstat -u slfu` 计入全部未结束状态（R/Q/H等），未结束任务数达到3不得 `qsub`；同时核对当前GPU总数不超过8。禁止停掉其他任务让路。
2. SimVP 完成标记、best权重哈希及 `210506.tc6000` 成功退出已核验；如 checkpoint 或源码后续改动，须重新核验。
3. `scripts/run_cot_dynamics_geometry24_bank_20260924.pbs` 已以 `210593.tc6000` 完成：一张卡、batch16，独立 train/val SimVP→R COT 与13+3通道预测图 bank，并在同一作业内按 GHI 行键打包修复版图像 sidecar。两份 `complete.json`、sidecar `audit.json`、哈希和序列索引均已验收。
4. `scripts/run_cot_dynamics_ghi_b1_20260924.pbs` 已以 `210598.tc6000` 完成：同头、同样本、同种子训练 A0/AH/AI；O_diag重用其批量、COT标准化及图像sidecar；逐行配对指标已从保存预测重算。`run_cot_dynamics_ghi_oracle_20260924.pbs` 仅作独立重跑备份。
5. 下一步先处理E的入口：原生16 R 的上游图裁剪虽解决固定样本来源覆盖，却未显示一致的长期 COT 精度。扩大E训练前需验证上游非站点patch分布、完整train/val的有效性与比P0/AP更好的反演参考误差；不要用越界零填充伪装晴空。新E若要宣称GHI增益，仍必须从同一E初始化做物理锚定和GHI指导的配对训练、各自重训H并与A0/AH/AI比较。

所有新建脚本原件位于本仓库 `scripts/`，已用的服务器副本位于 `experiments/cot_dynamics_20260924/scripts/`。旧AFNO-COT bank 采用 `geos[:16]`，把历史几何错配给未来图；旧配对GHI差值不得当新结果。论文中不要写服务器路径、作业号或修复过程。
