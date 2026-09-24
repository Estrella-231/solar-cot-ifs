# COT 运动与演变实验跟踪表

日期：2026-09-24。B0 的数据合同、P16 边界支持度及 R 大视场一致性已实测；修复版 geometry24 单卡 pilot/profile 已完成，oracle bank 已提交，SimVP正式模型仍在训练，未训练 E/GHI 头、未打开 test。服务器验收见 `experiments/cot_dynamics_20260924/audits/`；本地实现见 `scripts/*20260924.py`。

| Run ID | 阶段 | 目的/系统 | Split | 主检查 | 优先级 | 状态 | 前置条件 |
|---|---|---|---|---|---|---|---|
| D000 | M0 | geometry24全链路、输入方向/时刻见证、来源锁定 | train/val | 24几何与16预测逐帧对应；无test | MUST | PARTIAL：逐行合同审计PASS、S/AFNO小样本bankPASS；S正式训练未完成 | 选定修复版S训练完成 |
| D001 | M0 | halo覆盖、原生patch与拼接R一致性、单卡profile | train/val pilot | 位移源覆盖/拼接误差/吞吐 | MUST | GATE FAIL：直接大裁剪与重叠16拼接均不稳定；单卡profile完成 | D000 |
| D010 | M1 | A0/AH/AI/O_diag匹配来源训练H | train/val | 同键GHI RMSE/MAE/bias，理想参考潜力 | MUST | PLANNED | D000/D001，新bank |
| D011 | M1 | P0持续与AP纯搬运 | train/val | COT空间/轨迹/越界，零训练参照 | MUST | PARTIAL：16个val/64个train序列 P16 支持度 pilot；非正式 AP | D000/D001 |
| D020 | M2 | E_phys与其H；AP的H | train/val | AD/AP相对AH/AI的GHI与COT误差 | MUST | CONDITIONAL | M1有信号、覆盖足够 |
| D030 | M3 | 同起点E_phys_cont/E_task，固定H反传 | train/val | 梯度见证、相同步数、COT漂移 | MUST | CONDITIONAL | D020通过 |
| D031 | M3 | 固定两种E、从相同初始化重训新H | train/val | AT相对AD/AH/AI的配对增益 | MUST | CONDITIONAL | D030 |
| D040 | M4 | 删除搬运与同容量自由场控制 | train/val | 机制与物理语义归因 | CONDITIONAL | NOT_STARTED | D031通过 |
| D041 | M4 | seed42/43/44配对复核 | train/val | 原始点估计、日期配对CI、seed差异 | CONDITIONAL | NOT_STARTED | 配置/超参数已冻结 |
| D042 | M4 | 图像底图AFNO复核/去边缘项/加ramp项 | train/val | 每次仅一因素，次要结果 | OPTIONAL | NOT_STARTED | 核心结论成立且有明确诊断需求 |
| D050 | final | 独立时间test与双语论文结果 | test | 预登记冻结协议 | CONDITIONAL | LOCKED | 来源/实现/多seed/审计要求满足 |

已核验证据（2026-09-24）：`b0_contract.json` 验收 train/val 27,675/4,258 个序列及 538,053/99,849 个 GHI 行，24 帧几何、BJT/UTC、目标标签三条5分钟记录、两站 `[8,8]` 均PASS，test未读。`p16_cpu_pilot_16/pilot.json` 在固定16条val序列、两站上用历史C13估计平移；中心5×5有效源覆盖由+15分钟99.4%降至+240分钟51.3%，全patch由89.7%降至51.6%；它只证明当前16×16历史场缺上游信息，不代表运动预报精度。`r_halo_cpu_witness.json` 的8个固定val历史站点案例中，原生16×16 R与直接32×32/64×64 R的共同16×16平均绝对差为0.332/0.318 log1p(COT)，故直接改大R输入不合格；需验证锚定原生区域的拼接策略。合成平移方向测试得到 `(dy,dx)=(+3,-4)`。

追加 `r_tiling_cpu_witness.json`：相同8个历史站点案例，用九个重叠16×16窗口拼接32×32，未锚定的中心16与原生R平均差0.165 log1p(COT)，重叠像元多次估计的平均标准差0.143；强行锚定原生中心后边界两侧平均跳变0.291，而原生内部相邻梯度均值0.085。约3.4倍的接缝使此拼接也不能直接放行。按方案先停止扩大 E；A0/AH/AI/O_diag的正确geometry24配对对照不依赖halo，仍可独立推进。

训练期固定样本复核：`r_halo_train_cpu_witness.json` 中直接32/64裁剪相对原生16的中心16平均差0.614/0.613；`r_tiling_train_cpu_witness.json` 中重叠拼接未锚定中心差0.325，锚定后接缝0.428、原生内部梯度0.154；两者同样未过门禁。`p16_cpu_train_pilot_64/pilot.json` 的64条固定train序列中心5×5来源覆盖由+15分钟98.3%降至+240分钟50.4%，全patch覆盖见原报告。这些是源区域/反演合同诊断，不能当COT预报技能或GHI提升。

GPU pilot第一次 `210576` 因误载旧 `triad_models.py` strict加载失败，无产物；修复为checkpoint哈希绑定的 `geometry24_v2/triad_models.py` 后，`210577` 完成2条SimVP和4条AFNO验证小样本的完整24帧几何、16预测图及冻结R前向。`210578` PBS脚本未定义变量、前向未开始；修复后 `210579` 在A100 40GB单卡完成64序列×batch8/16/24/32 profile，峰值4.95/9.85/14.76/19.67 GiB，全流程吞吐3.45/3.67/3.34/3.15序列/s（含启动及缓存，样本短）；暂选batch16，不因显存更高而选择更慢档。未来真实AGRI→R只作oracle的GPU job `210580` 已在qsub前全账户未结束数2时提交；完成以产物为准。修复版SimVP仍训练中，不能把当前best当最终冻结权重。未创建持续监控。工程门槛为提案中的预登记候选值，不是已观察到的结果。
