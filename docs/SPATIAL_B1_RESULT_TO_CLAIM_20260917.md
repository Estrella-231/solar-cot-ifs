# 空间 COT B1 Result-to-Claim

判定：claim_supported=no；confidence=high仅针对“不放行主张”的判断。评审为GPT-5.6-Sol ultra、same-family/provisional；完整性审计未运行。确定性证据检查确认四个引用数值都存在，但不证明科学主张正确。

本次只支持描述性陈述：seed42、10epoch验证pilot中，AC_map的两站等权RMSE为148.6723 W/m²，是三组最低；相对A_sp只低0.2181 W/m²（0.146%），低于预设1%门槛且分站方向相反。因此不能写“空间COT提高预报精度”，也不应直接用seed43/44尝试挽救未过门槛的信号。

下一步优先级：

1. 对现有checkpoint做零额外训练的输入激活、归一化、对齐、首层权重和zero/permutation敏感性检查。
2. 输入确实被利用后，仅做seed42短预算的AC_shuffle重训和A_proxy有效容量对照。
3. 若AC_map与shuffle或A_proxy没有清楚差异，停止H14空间优越性路线并记录负结果。
4. 只有上述控制清楚有利时，才扩展seed43/44；暂不换更复杂骨干。若日后换头，使用A/AC × old/new head因子设计。

CPP历史生产链问题不妨碍上述探索性工程诊断，但在物理解释和最终论文主张前仍须闭环。
