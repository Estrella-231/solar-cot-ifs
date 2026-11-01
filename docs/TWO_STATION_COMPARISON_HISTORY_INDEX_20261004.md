# 四里与竺家历史对照索引

2026-10-04按本地报告、结果JSON、Git历史及聊天检索整理。正的差值表示加入COT后GHI RMSE变差；单位W/m²。每一行只作该实验内部比较，不作跨实验绝对排名。全部为旧合同验证诊断，不是新月度划分的独立测试成绩。

| 实验 | 四里差值 | 竺家差值 | 来源 |
|---|---:|---:|---|
| 09-08 原kt损失、四统计量original | +1.755 | +7.768 | COT_FAILURE_CAUSES_20260908.md |
| 09-17 完整16×16静态图AC_map−A_sp | +1.029 | −1.465 | SPATIAL_B1_PILOT_RESULTS_20260917.md |
| v5 因果轨迹，仅历史AC_H−A_ST | −2.257 | −5.100 | HISTORICAL_COT_GHI_SIGNAL_20260923.md |
| v5 因果轨迹，仅未来AC_F−A_ST | −1.393 | +6.909 | results/cot_real_agri_20260922/v5_fixed_head_complete_snapshot.json |
| v5 因果轨迹，历史+未来AC_ST−A_ST | −2.431 | +3.640 | 同上 |
| 09-27 区域R256 pilot forecast−zero | −7.087 | +5.946 | results/regional_station_cot_20260927/ghi_pilot/ |
| 09-27 区域R256 pilot oracle−zero | −12.093 | +3.326 | 同上 |

## 最关键的四组对照

| GHI头输入 | 四里RMSE | 竺家RMSE |
|---|---:|---:|
| A_ST 无COT | 167.795 | 133.782 |
| AC_H 仅历史COT | 165.537 | 128.683 |
| AC_F 仅未来预测COT | 166.401 | 140.691 |
| AC_ST 历史+未来预测COT | 165.364 | 137.423 |

以上为同一v5 seed42四组验证快照。相比AC_H，加入未来预测分支后的AC_ST，竺家RMSE增加8.740、四里减少0.174；不同组分别训练，差异不能解释为固定模型的纯输入因果效应。

## 另一个有价值的对照：真实目标AGRI空间范围

`experiments/station_window_selection_20261003_v1/summary.json`记录60组、两站、三seed的真实目标时刻AGRI辐照度反演诊断。不是未来云图外推预报。

| 窗口 | 四里AC−A | 竺家AC−A |
|---|---:|---:|
| 1×1 | −1.093 | −1.340 |
| 2×2 | −0.740 | −1.334 |
| 4×4 | +0.079 | −2.858 |
| 16×16 | −0.857 | −0.193 |
| 64×64 | −4.518 | +0.513 |

总体COT增量的日期配对区间均跨零。竺家在小窗口可改善、64窗口略退化；窗口改变了AGRI及COT支持，不能把尺寸效应全部归为COT。

## 仓库演进和聊天证据

- `596d07b`：COT诊断与服务器状态；包含早期loss与表示问题记录。
- `654d13e`：记录真实历史COT的GHI验证信号。
- `a45db46`：分站COT策略与区域pilot退化诊断。
- `8a21007`：竺家容量匹配残差审计。
- 聊天“新建 COT 云物理论文工程”：配对区域GHI pilot前两小时/四小时的差异，来源与上述09-27 pilot相同，不是新的独立实验。
- 聊天“第二篇论文方案把控 (2)”：真实目标AGRI窗口60组结果及后续独立预测链窗口实验。后者在读取的聊天记录中仅为已启动，不能混成已完成结果。

## 现有图

- `figures/cot_pipeline_20260918/fig_actual_static_cot_gain_diagnosis_20260918.png`：16×16静态空间图两站对照，历史四统计量只读保留。
- `results/two_station_cot_degradation_20261004/two_station_comparison.png`：09-27 R256 pilot两站总体/逐lead对照。
- `results/regional_station_cot_20260927/ghi_pilot_event_audit_20260928/matched_event_cot_ghi_trajectories.png`：两站选定事件COT/GHI轨迹，不能替代整体频率。

核心结论：没有证据支持“竺家加入任何COT都会退化”。v5历史COT和预测COT的分离对照最值得优先复核；预测质量、融合和训练差异仍未被单一实验完全拆开。
