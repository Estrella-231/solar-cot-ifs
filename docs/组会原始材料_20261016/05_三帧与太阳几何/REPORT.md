# 配对 R 太阳几何与下游 GHI 结果

## 问题

AGRI→COT 反演模型（R）中的太阳几何修正，是否会改变 COT 对站点 GHI 的下游价值？我们先在同一批真实 AGRI 验证场景上比较两个 R 模型，再比较三个其余条件配平的 GHI 头部。

## 两个 R 模型的真实云场比较

图中使用了 6 个确定性验证场景：每个站点各取最小近零 CPP 参考中心、最大厚云 CPP 参考中心，以及旧 R 与修正 R 差异最大的一例。每一列都使用同一张真实 AGRI C03/C02/C01 假彩色图、同一个 CPP 反演参考、同一个站点像元和共享的色标。CPP 是反演参考，不是独立的 COT 真值。

物理 COT 的分布确实发生了变化，但方向取决于场景。四里厚云参考案例中，中心估计由 57.67 变为 52.61，而参考值为 68.75。竺家厚云参考案例中，由 36.58 变为 42.52，而参考值为 69.75。在差异最大的例子中，修正后的估计在四里更接近参考（59.50→40.81，参考 41.79），但在竺家反而更偏离参考（57.44→40.81，参考 56.13）。因此这些例子显示的是真实的输入—响应变化，而不是一致性的精度提升。

在完整的配对 R 验证对比中，修正减旧的站点等权 RMSE 为 +0.086 个 COT 单位（95% 日期簇区间为 −0.282 到 +0.438）。该修正略微降低了汇总 MAE，同时提高了 RMSE。各站点的变化方向相反，因此 R 的验证结果没有确立可靠的总体增益。

## 三个配对的 GHI 分支

所有分支使用同样的 51,929 条训练/验证记录（训练 47,666、验证 4,263）、两个站点、15 分钟区间标签，以及 T−30/T−15/T 三个时刻的真实 AGRI。每个站点独立训练，使用同样的 3×3 局地头部、41,665 个参数、30 个 epoch、batch 1,792 和配对种子 42/43/44。R 模型为冻结的。三个分支为：

1. `A_history`：AGRI 与修正后的标称 UTC 太阳几何，标准化的 COT 槽位固定为零。
2. `AC_history_old_R`：同样的输入，加上刚重训的原始几何 R 给出的 COT。
3. `AC_history_corrected_R`：同样的输入，加上配对的修正几何 R 给出的 COT。

主估计量是站点 RMSE 的均值，在两个站点和三个种子上等权平均。置信区间使用 2,000 次按北京时间目标日期成簇的配对 bootstrap 重采样，条件是“按验证集选择 checkpoint”。

| 分支或配对对照 | 验证 RMSE (W/m²) | ΔRMSE (W/m²) | 相对变化 | 95% 日期簇区间 |
|---|---:|---:|---:|---:|
| 仅 AGRI | 108.236 | — | — | — |
| AGRI + 旧公式 R-COT | 107.818 | −0.418 相对仅 AGRI | −0.39% | [−2.236, +1.136] |
| AGRI + 修正几何 R-COT | 107.784 | −0.452 相对仅 AGRI | −0.42% | [−2.294, +1.101] |
| 修正几何 R-COT − 旧公式 R-COT | — | −0.034 | −0.03% | [−0.365, +0.244] |

两个 COT 分支的点 RMSE 都略低于仅 AGRI，但三个区间都包含零。修正 R 与旧 R 之间的差异尤其小且不稳定。在四里，修正 R 比旧 R 误差低 0.355 W/m²；在竺家，误差高 0.288 W/m²。三个配对种子平均下来，修正减旧的 RMSE 分别为 −0.054、+0.158、−0.204 W/m²。

两个 COT 分支的 MAE 都更低：站点/种子等权均值由仅 AGRI 的 66.43 W/m² 降到旧 R-COT 的 64.26 和修正 R-COT 的 63.83。但它们的平均偏差变得更负（分别为 −0.46、−2.50、−2.99 W/m²），而且本实验没有为 MAE 定义单独的不确定度区间。这是描述性的次要结果，不能作为总体稳健增益的证据。

剔除 50 条源端标记为长时间零值的记录后，敏感性评分给出修正 R-COT 减仅 AGRI 的 ΔRMSE 为 −0.855 W/m²，95% 日期簇区间为 [−2.653, +0.801]。其区间同样跨零；这个仅评分用的子集不改变训练或 checkpoint 选择。

## 解读

这个配平实验没有显示修正后的太阳几何能修复 COT 到 GHI 的下游映射。旧 R 与修正 R 的输出在真实云场上确有差异，但在两者之间切换，GHI RMSE 平均只变化 −0.034 W/m²，且不确定度跨越正负两种符号。加入任一种 R-COT 历史都能得到一个略有利（更低）的 RMSE 点估计，但配对日期不确定度太宽，无法把它与“无变化”区分开。MAE 与偏差的变化也表明，COT 改变了误差形态，而不是一致地把误差纠正掉。

这一结果只支持一个很窄的结论：在这套两站、仅验证的合同下，太阳几何修正没有产生可测量的下游 GHI 改善，这三帧观测 COT 的额外价值仍未解决。它既不能说明 COT 没有物理价值，也不能预测未来预报链路的表现。

GHI 输入包含标称时刻 T 的真实 AGRI 帧，而卫星实际扫描可能发生在 T 之后。这是对同一 15 分钟 GHI 区间的离线同步重建，既不是因果预报，也不是 SimVP→R 的部署链路。没有读取测试载荷，没有改动论文正文。

## 复现与证据

- 冻结的实验定义：[`PLAN.md`](PLAN.md)
- 实际执行记录：[`STATUS.md`](STATUS.md)
- 真实 AGRI/R 图与选中案例：[`visuals/paired_r_real_clouds.png`](visuals/paired_r_real_clouds.png)、[`visuals/paired_r_real_agri_cases.json`](visuals/paired_r_real_agri_cases.json)
- GHI 对比图：[`results/paired_three_arm_v1/analysis/three_arm_ghi_validation.png`](results/paired_three_arm_v1/analysis/three_arm_ghi_validation.png)
- 独立 FP64 评分与配对 bootstrap：[`results/paired_three_arm_v1/analysis/independent_score.json`](results/paired_three_arm_v1/analysis/independent_score.json)
- 第二次按行加权的 FP64 复算：[`results/paired_three_arm_v1/analysis/secondary_row_weighted_score.json`](results/paired_three_arm_v1/analysis/secondary_row_weighted_score.json)
- 行对齐的验证预测：[`results/paired_three_arm_v1/analysis/paired_validation_scoring.npz`](results/paired_three_arm_v1/analysis/paired_validation_scoring.npz)
- R bank 映射与验证预测一致性：[`audit/paired_r_cot_bank_manifest.json`](audit/paired_r_cot_bank_manifest.json)、[`audit/r_bank_vs_saved_predictions.json`](audit/r_bank_vs_saved_predictions.json)
- 作业输出位于 `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/r_geometry_downstream_ghi_20261016`；PBS 作业为 `214295.tc6000`（R-COT bank）和 `214305.tc6000`（三分支 GHI 运行）。

bank、checkpoint 和完整的逐次运行训练轨迹仍保留在服务器上。仓库中包含查阅该结果所需的源脚本、仅验证预测、精简评分、图、清单和日志。
