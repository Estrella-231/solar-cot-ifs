# 竺家残差头 GHI 目标函数对照（2026-09-28）

## 问题与目的

此前竺家 AGRI＋COT 残差组的训练目标在 `kt` 空间计算加权平方误差，但模型最后按 GHI RMSE 评价和选模。因为 `GHI = kt × GHI_clear`，`kt` 误差对应的 GHI 平方误差还包含 `GHI_clear²`。本实验只改训练目标，检验这项口径差异是否妨碍残差学习。

## 对照和固定项

- AGRI-only 残差与 AGRI＋COT 残差使用同一竺家冻结基线、同一残差结构、相同初始化、数据顺序、batch=512、seed=42、学习率 `3e-4`、600 次更新和 ±0.15 `kt` 修正界限；零初始化修正作为验证选模的 step-0 回退。
- AGRI-only 对照将 COT 数值置零，保留因果有效时间掩码。两臂的可训练参数量相同，COT编码器共享同一初始化。
- 唯一计划变量是损失：

  `w × [((pred_kt - target_kt) × GHI_clear / 1000)² + 0.05 × (delta_kt × GHI_clear / 1000)²]`

  其中第二项把修正惩罚也换算到同一归一化 GHI 尺度。验证仍使用未加权 GHI RMSE 选择 checkpoint，与此前容量对照口径一致。
- 四里模型及湖南 SimVP、COT 反演器均冻结；只使用现有 train/validation bank，test 保持关闭。

## 数据与来源

训练bank为 `regional_ghi_pilot_bank_20260927_v1`。竺家冻结基线及COT编码器来自 `zhujia_cot_residual_seed42_20260927_v2`；提交前已在服务器核对哈希，正式完成文件会记录实际权重哈希：

- `zero_best.pt`: `ff087540eea05335619d1386469b3f90da7bf9c795c9d95b9ba03e943ef87cc6`
- `fusion_best.pt`: `29a5390cd8fd615c420c9bc6f41c877b622dde3db4315d3eafb88326d09485`
- 固定四里 `forecast_best.pt`: `4d243d0c8c3be33cff83ad8307c6d2e1bcde59d8a9bae6cac17bfd8e87ea7d4c`

## 执行记录与结果

- PBS 210978 在载入模型前失败：初版脚本误将 regional pilot 目录作为 `source-run`，其中没有 `fusion_best.pt`。日志和失败目录保留。
- PBS 210979 也在训练脚本启动前退出：PBS 中手写的源 checkpoint 哈希门禁与实际哈希不符。已在服务器重新计算并保存真实哈希，失败日志保留。
- PBS 210980 于 2026-09-28 00:43 左右在 node22/GPU7 的 A800 80GB 上完成，退出码0，walltime 9分38秒。作业内先各跑3步 smoke，完成标记、基线回退及配对检查通过，再进入正式训练。
- 正式bank含485个竺家训练序列、127个验证序列。AGRI-only和AGRI＋COT两组各完成600次更新。验证两臂使用相同的1624个站点/时效行，覆盖70个初始化日；`test_used=false`。
- 两组最佳检查点都选中step 0，预测逐样本精确等于冻结基线：RMSE 127.2954、MAE 93.0143、bias −11.7035 W/m²。AGRI-only与COT两臂所有16个时效的RMSE也都与基线相同；5000次按日期配对bootstrap中三组RMSE差值区间均为 `[0, 0]`，因为最佳输出完全相同。
- 训练末100步平均损失为AGRI-only 0.01881、AGRI＋COT 0.01072，但验证最优始终停留在step 0。两臂在有限pilot上的后续训练都不能泛化，COT组后段验证误差明显升高。保存曲线、逐样本输出和完整配对审计见 `results/zhujia_ghi_objective_residual_20260928_v3/`。

## 结论与下一步

本次 GHI 尺度损失实验没有解除竺家退化：在当前小型pilot和残差结构下，GHI尺度训练目标仍选回基线。因此，先前发现的kt/GHI目标差异不足以解释现有失败；训练/验证失配是直接观测到的现象，但不能据此把唯一原因断定为样本量。当前竺家继续保留无COT基线，四里不变。

下一项值得做的是扩大竺家训练与验证的日期覆盖、减少只在少量相邻天气过程上重复取样，并保持独立验证日期；先确认无COT基线和AGRI容量对照也能稳定选模，再重新评估COT增量。当前结构不要继续加更新数或直接进入多种子/独立test。

## 结论门禁

完成后核对两组训练步数、相同的验证行/真值/时效、step-0精确回退、验证GHI指标、逐时效误差和按初始化日期配对bootstrap。该实验仅能判断当前竺家pilot配置下，改用GHI尺度训练目标是否改善COT残差候选；单种子验证结果不构成独立测试或业务部署证据。
