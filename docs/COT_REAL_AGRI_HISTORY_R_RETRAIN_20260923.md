# 新 R 历史 COT 侧车的 GHI 头配对重训

## 问题和实验边界

固定 GHI 头替换实验表明：更换历史 COT 后，下游预测只发生极小变化。为检查 GHI 头是否需要适配新的历史 COT 数值分布，本试验把原 R 与真实 AGRI 继续训练得到的新 R 放入相同的头部训练流程，两个 GHI 头均从随机初始化开始。

主比较为冻结 v5 SolarResNet 因果轨迹头的 seed42、10 epoch pilot。每个站点分别训练 `A_ST`、`AC_H`、`AC_F`、`AC_ST`：其中 `AC_H` 只使用历史 COT，`AC_ST` 使用历史 COT 加同一冻结 forecast-COT 侧车的因果前缀；`A_ST` 和 `AC_F` 是核验不依赖历史 R 的参照。所有组保持相同网络、样本行、初始化 seed、标签、mask、优化器、学习率、batch 和 epoch 上限。此实验是既有 B1 seed42 pilot 的新配对诊断，不冒充 50 epoch/多 seed 主结果。

两套历史 COT 均由相同湖南冻结 SimVP 历史 AGRI loader，从每个初始化时刻之前的 8 帧**真实 AGRI**重新经 R 反演，逐像元存成 FP32 `log1p(COT)`；重新生成 train 与 validation 两个 split，test 不读取。原 R 与新 R 使用相同的冻结 R 输入归一化和推理精度。候选新 R checkpoint 为 `18d285c2b1349ca9890c48109b3a149520638674bae69293ff9480938fcfe069`，起点为原冻结 R `0cbdb932f49945259d5f1704316edaf6481be7cfdb54692236f829988925799c`。

为隔离“历史 R 的替换”，两条路线共用现有冻结 future-COT 侧车（FP32、由冻结 SimVP 预测 AGRI 输入原 R 生成）。因此 `AC_ST` 中的历史和未来 COT 暂时分别来自新旧 R；该组只回答历史 COT 反演变更在固定未来分支下的影响，不能代表完整部署时新 R 同时作用于历史与预测 AGRI。`AC_H` 是更直接的历史 R 对照。

训练头会对各路线各自的历史/未来 COT 训练输入重新拟合一个 train-only 标量 mean/std；归一化只用于数值尺度，不是全图空间汇总特征。GHI 标签只作监督和验证。由于候选 R 在同一 2025-07 至 09 验证期依据 CPP COT 选模，本 pilot 的 GHI validation 是探索性诊断，不是独立时间泛化证据。CPP 生产者绑定仍未核实，沿用项目批准的探索性门禁；结果不能直接写成论文主张。

## 执行

侧车构建：`scripts/build_real_agri_history_cot_trainval_pair_20260923.py`。它在全新目录中生成原/新 R 的 train、val COT，并记录 checkpoint、脚本和数据哈希；若缺行、原新样本数不一致或输出目录已存在则停止。

重训：`scripts/run_real_agri_r_history_retrain_pair_20260923.pbs`。输出分别写入：

```text
data/history_cot_trajectory_real_agri_rpair_20260923/
runs/causal_cot_v5_real_agri_Rpair_seed42_20260923/original_R/
runs/causal_cot_v5_real_agri_Rpair_seed42_20260923/continued_R/
```

独立验证审计入口为 `scripts/audit_real_agri_history_r_head_retrain_pair_20260923.py`：它核对两边逐行 pack 键、真值、晴空值、站点和 lead，并从保存的 `pred_kt` 用 FP64 重算指标；对 `A_ST/AC_F` 做不依赖历史 R 的 placebo 核验；对两站等权 RMSE 报告初始化日成组 bootstrap 区间。

当前验证结果、样本/标签独立审计、初始化日 bootstrap 和模型权重哈希将在作业完成后补入本记录。测试集维持关闭；pilot 完成后先审计逐样本预测和同 cohort，再决定是否扩大 seeds 或训练轮数。
