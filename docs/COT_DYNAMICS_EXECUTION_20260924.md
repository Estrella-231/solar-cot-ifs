# COT dynamics 执行交接（2026-09-24）

本页只记录工程状态；科学问题、实验组、停止条件见 [`refine-logs/cot_dynamics_20260924/EXPERIMENT_PLAN.md`](../refine-logs/cot_dynamics_20260924/EXPERIMENT_PLAN.md)。状态可能随 PBS 和服务器产物变化，重启前必须实时复查，不能把本页的作业号当完成证明。

## 服务器与输入

- SSH alias：`zjnu-hpc`；用户 `slfu`；Hunan AGRI/aux 根目录：`/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed`。
- 第二篇工程：`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907`；本次隔离结果：其下 `experiments/cot_dynamics_20260924/`。
- 修复版 backbone 工程：`/public/home/slfu/ttzhou/swc/irradiance_forecast/HunanBackboneTriad_20260921`；**模型源码必须使用**其 `geometry24_v2/triad_models.py`，与 checkpoint 的 `model_code_sha256` 一致。其根目录旧 `triad_models.py` 与 geometry24 权重不匹配。
- 主 SimVP：`geometry24_v2/runs/simvp_geometry24_formal_e12_v1/best.pt`，须等待同目录 `training_complete.json`。AFNO 只作次要稳健性对照：`geometry24_v2/runs/afno_transformer_geometry24_formal_e12_v1/best.pt`。
- 冻结 R、历史 COT：第二篇工程的 `configs/r_frozen_repaired_seed42.json` 与 `data/history_cot_trajectory_real_agri_rpair_20260923/original_R/`。R SHA-256 为 `0cbdb932f49945259d5f1704316edaf6481be7cfdb54692236f829988925799c`，缓存已标 FP32、train/val、test=false。
- GHI 标签/行合同：`data/head_pack_trainval_20260908_v1/`。本轮必须重建图像输入 sidecar，不得沿用其中旧 SimVP 预测图的 `x.npy`，也不得沿用旧 `solarresnet_v5_input_trainval_20260918_v1`。

## 已完成与实测门禁

- `audits/b0_contract_v2.json`：train/val 27,675/4,258 个序列、538,053/99,849 个 GHI 行，全量起报/目标时刻、16 lead、3 条5分钟标签、站点中心 `[8,8]` PASS；按06:00、12:00、18:00 BJT固定时刻见证几何，test未读。
- `p16_cpu_train_pilot_64/` 与 `p16_cpu_pilot_16/`：历史 C13 相位相关平移的 P16 演示。+240分钟站点中心5×5有效来源覆盖分别50.4%（固定64条train）和51.3%（固定16条val），不能当完整4小时 COT 预报。
- `audits/r_halo_train_cpu_witness.json` 与 `audits/r_tiling_train_cpu_witness.json`：原生16×16 R改为直接32/64，中心16平均差0.614/0.613 log1p(COT)；九块16重叠拼接的未锚定中心差0.325，强锚定后接缝跳变0.428，而原生内部相邻梯度0.154。val固定案例同方向。**E 的大视场 COT 输入门禁失败；不得直接把 R 改在大patch上推理或拼接后宣称同一反演合同。**
- B0 小样本 GPU：`210576` 因误用旧模型定义严格载权失败；修复哈希绑定后 `210577` 完成当前 best SimVP 2条、AFNO 4条 val 的24几何→16预测图→R全链路。`210578` 因 PBS 变量错误在模型前失败；`210579` 修复后在一张 A100 40GB 上完成 batch 8/16/24/32 的64序列 profile，峰值4.95/9.85/14.76/19.67 GiB、全流程吞吐3.45/3.67/3.34/3.15序列/秒；batch16暂优。短样本吞吐含启动/缓存，正式运行记录仍需核对。
- 真实未来 AGRI→R 仅用于 oracle 诊断：CPU 两序列小样本通过；PBS `210580` 已提交，先跑固定256序列GPU profile，再顺序生成 train/val 全量缓存。任何 `real_future_cot_log1p.npy` 禁止进入部署组推理。
- [真实验证 AGRI 的视场诊断图](../figures/cot_dynamics_20260924/r_halo_actual_val_sili_v2/r_halo_actual_val_sili.png) 固定取四个验证序列的历史末帧；三个白天四里案例中，原生16×16 R 与同权重32×32 R 的共同中心16差值 MAE 为0.213/0.147/0.425 `log(1+COT)`。第三行案例是夜间，图内标明。图旁 `source_arrays.npz` 和 `provenance.json` 保存原值及展示色标；这证明视场不一致，不能作为 GHI 预报提升证据。

## 接续顺序

1. `qstat -u slfu` 计入全部未结束状态（R/Q/H等），未结束任务数达到3不得 `qsub`；同时核对当前GPU总数不超过8。禁止停掉其他任务让路。
2. 核验 SimVP `training_complete.json`、best 权重哈希及 job `210506.tc6000` 最终状态。当前 best 不能提前冻结为正式 S。
3. 仅在正式 SimVP 完成后，提交已通过 `bash -n`/`py_compile` 的 `scripts/run_cot_dynamics_geometry24_bank_20260924.pbs`，用一张卡、batch16 生成独立 train/val SimVP→R COT 与13+3通道预测图 bank，并在同一作业内按原 GHI 行键打包修复版图像 sidecar；以两份 `complete.json`、sidecar `audit.json`、哈希和序列索引为准。
4. 两份 SimVP bank、sidecar 和 oracle train/val bank 完成后，提交 `scripts/run_cot_dynamics_ghi_b1_20260924.pbs`；同一作业先以同头、同样本、同种子训练 A0/AH/AI，再强制重用 AI 的批量、COT 标准化及图像 sidecar 训练 O_diag，最后从保存预测重算配对指标。O_diag 只可单列诊断。`run_cot_dynamics_ghi_oracle_20260924.pbs` 仅作独立重跑备份，不与组合脚本同时提交。
5. `scripts/evaluate_geometry24_cot_ghi_pair_20260924.py` 必须从保存的预测和相同 `pack_row` 重算两站等权 GHI RMSE、MAE、bias、16 lead。若 O_diag 在匹配条件下无明确潜力，先查 R/H/标签对应，停止 E 扩展；若有，再解决上游 COT halo 门禁，不能用 P16 越界零填充伪装晴空。

所有新建脚本原件位于本仓库 `scripts/`，已用的服务器副本位于 `experiments/cot_dynamics_20260924/scripts/`。旧AFNO-COT bank 采用 `geos[:16]`，把历史几何错配给未来图；旧配对GHI差值不得当新结果。论文中不要写服务器路径、作业号或修复过程。
