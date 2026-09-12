# 成对 A/AC validation pilot：全部结果

日期：2026-09-08。PBS 207616.tc6000 正常结束，exit_status=0，walltime=00:07:05，单张 A100。无测试集结果。

两套训练：original 保留538053条，QC1保留537605条；验证集均为相同99849条。QC1仅为疑似平直段质量敏感性，排除448条训练标签并重拟合head train-only norm和权重；其他大kt、损失、S/R与原始pack不变。两套×A/AC×两档LR全部执行，不按收益挑cohort。

A为预测AGRI加共同几何/站点/lead；AC额外加入冻结R的4维log1p(COT)统计量。每组在相同两档LR内按验证终点选模，下表所有数值均为GHI RMSE，单位W/m²。终点为两站各自全16 lead RMSE的均值。epoch采用0起点。

## 全部8个候选

|训练分支|组|LR|最佳epoch|两站等权RMSE|四里RMSE|竺家RMSE|
|---|---|---|---|---|---|---|
|qc1|A|0.0003|12|149.905825|169.689407|130.122243|
|qc1|A|0.001|3|151.683323|169.965863|133.400783|
|qc1|AC|0.0003|6|152.600103|167.483949|137.716257|
|qc1|AC|0.001|3|151.248653|169.297761|133.199545|
|original|A|0.0003|12|149.817451|167.568568|132.066335|
|original|A|0.001|8|151.739899|170.661787|132.818011|
|original|AC|0.0003|15|154.613512|174.886170|134.340855|
|original|AC|0.001|7|154.578853|169.323474|139.834232|

## 各分支选定LR后的配对结果

|训练分支|A|AC|AC−A|相对误差变化|四里AC−A|竺家AC−A|
|---|---|---|---|---|---|---|
|qc1|149.905825|151.248653|+1.342828|+0.8958%|-0.391646|+3.077303|
|original|149.817451|154.578853|+4.761402|+3.1781%|+1.754906|+7.767897|

正差值表示AC更差。两套结果均未支持本次配置下的COT增益。单种子、经val选择LR的探索结果不构成显著性检验，也不能证明COT普遍无效。QC1没有把方向变为有利；两cohort差异不能仅归因于删去的标签，因为norm/权重与训练排列也随之变化。

## 核验与后续

- 全部8组checkpoint及预测SHA通过；所有预测的验证pack_row、GHI真值、clear-sky、站点、lead与原pack逐值相同；非负有限kt通过；NumPy float64重算与训练报告差<1e-8。
- 独立CPU checkpoint前向见证已通过 `PASS_4_SELECTED_CPU_CHECKPOINT_WITNESSES`，另存 `audits/head_paired_reload_20260908.json`。4个选定模型各取固定2站×4 lead共8行，严格重载并应用对应QC变换；CPU FP32与保存GPU BF16的最大kt绝对差0.005589，均满足事先固定的0.02×max(1,abs(saved_kt))容差。这是抽样模型绑定，不能称全量CPU重放。
- 指标来源 `audits/head_paired_pilot_20260908.json`；存在性预检查 `audits/head_paired_evidence_precheck_20260908.json`（5/5 verified，仅证明值存在）。
- P202进入无收益诊断：先核对通道、归一化、COT饱和与forecast输入域。无明确实现错误不开展大型模型搜索；暂不放行P203归因扩展/P204正式多种子/P205 test。
- IFS release及完整step仍缺失，不能启动AN/ACN；本次不能检验COT/IFS互补。

## 服务器证据

工程根：`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907`。

原分支：`runs/head_A_AC_seed42_20260908_v1`；QC1：`runs/head_A_AC_qc1_seed42_20260908_v1`。每套含4个候选的best.pt、validation_predictions.npz、complete.json与全部epoch记录，完整路径和SHA见指标JSON。日志：`head_pair_pbs.log`、`head_pair_qc.log`、`head_pair_original.log`。
