# 历史COT外推 → GHI配对实验执行表

- 2026-10-15：用户明确授权以已有历史COT外推结果做有无COT预报配对，覆盖旧候选进入GHI的停止条件。本轮旧合同四里站探索性验证，不修改论文。
- 远端元数据预检PASS：按新日期表排除窗口24帧触及test/buffer者，保留旧train/val角色。train332/512序列、4041行；val449/512序列、5855行。图像bank有完整同sequence覆盖，head_row_ids→旧pack→CSV的序列映射一致；R/S冻结，hybrid固定update96。test载荷读取0。
- 预检初次发现CSV使用validation而数组使用val，训练前已统一显式映射，不改变split归属；修订预检通过。目标swc环境py_compile和PBS bash语法通过。
- 提交前slfu完整队列仅214108运行；提交214125.tc6000后214108 R、214125 Q，未结束2项。请求node22单A800/8CPU/4小时。未启动额外GPU任务。
- 运行目录：/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/history_cot_forecast_ghi_20261015_v1；日志pbs_attempt1.log；输出results_attempt1。
- 3×3相同GHI头，A/AC-T/AC-H×seed42/43/44，共9臂，每臂1200更新；先真实数据梯度见证与单卡profile，再训练。最终必须保存验证逐行预测并FP64重算总体和16lead RMSE/MAE/bias、目标日期块CI。
- 当前状态：已提交排队，实际启动/训练进度待Luna实时确认；监控已委派。

## attempt1修复与attempt2
214125在训练前退出1：NumPy isin传入set导致ids为空，而独立预检使用list得到正确样本。将运行代码同样显式转list，保留train_attempt1_as_run.py、pbs_attempt1.log与results_attempt1。修复不改样本、模型或预算。实时全账户仅214108未结束后提交214126.tc6000，提交后两项未结束（214108R/214126Q）。新日志pbs_attempt2.log、输出results_attempt2；Luna正在跟踪启动。

Luna实时确认214126已在node22 GPU3 A800 80GB运行，日志已达到PREPARE train 64/332；启动卡占用2MiB，数据准备阶段采样约3427MiB。尚未出现profile/TRAIN时不称训练已开始。

Luna确认214126已通过prepare、梯度见证、profile并进入实际训练；seed42三臂各1200更新均完成，当前seed43 A step100，completed_runs=3。实际GPU3 A800采样733MiB/43%；小模型按吞吐profile选batch，不以占满显存为唯一目标。等待三seed全量结果后独立复算，不发布中途最优数字。

## 完成
214126 exit_status=0，九臂训练与重载均完成。本地独立FP64复算PASS：总体A157.303、输运161.689、混合161.983 W/m²；两AC分别退化2.79%/2.98%，日期配对CI均为正。60个目标日期，5855行；不支持本pilot预报增益，封存负结果不自动加模型/调参。完整结果见RESULTS.md，所有16lead在results_attempt2/lead_summary.csv；test关闭。

## 输入与训练诊断更新
用户要求检查退化原因后，214153完成四例真实AGRI/R-COT回源（96帧，alignment审计各项最大差0）；214162完成原前100步重放，九组step100精确复现。发现原每100步验证相隔约203遍训练数据，5/6 COT模型早期较好点早于首次验证。补查候选点估计改善1.09%/0.61%但CI跨0，只作事后诊断。原负结果解释收窄为原检查点集合；云消散/增厚漏报与训练过拟合并存，不认定COT本身无用。图件与报告见diagnosis_20261015/DIAGNOSIS.md。

- 2026-10-15：经输入和训练诊断后，准备 attempt3。选模组/最终计分组按初始化日期哈希固定分离（四分之一/四分之三），每10更新检查一次，9臂相同合同训练1200更新；保留attempt2。脚本已由旧训练程序独立派生并通过本地/远端swc py_compile、远端bash -n；CPU日期切分预检PASS（12个选择日期/1383行，48个计分日期/4472行，16个lead两侧均有样本）。PBS 214190.tc6000已运行于node22 GPU0，提交前slfu队列为空，提交后仅该作业；训练输出路径与SHA见preflight_attempt3.json。初始日志完成train准备332/332，已开始val准备。新结果仍属旧split四里站内部验证pilot。


### attempt3完成：更密集选模 + 留出初始化日期计分

- PBS `214190.tc6000` 已完成，`exit_status=0`，运行于 `node22` 的 `GPU0`；总用时约339.1秒。A、历史COT输运、历史+混合修正三臂×3个seed共9次训练，每臂1200更新；没有新建test载荷读取。
- 选模与计分按初始化BJT日期确定性分组：12个日期/1383行仅用于选checkpoint，另48个日期/4472行仅用于最终计分。每10步验证；独立FP64脚本核对9组完全相同的逐行键、标签、时效及目标日，状态`PASS_INDEPENDENT_FP64_RECOMPUTE`。目标日期块配对bootstrap 10,000次。
- 总体（+15至+240 min合并）A RMSE 158.76 W/m²；历史输运COT 167.22，较A增加8.46 W/m²（RMSE增加5.33%，95%日期块CI [+2.55,+15.49]）；混合修正COT 166.79，增加8.04 W/m²（RMSE增加5.06%，CI [+2.35,+14.76]）。两CI均完全大于0；三seed各自的总体RMSE也均退化。+1h点估计略好但区间跨零，+2至+4h总体变差。
- 这是旧split、单站Sili验证pilot；它比attempt2更严格隔离了选模日期与计分日期，但不是新月度划分、不是独立test，也不能推翻同期COT反演收益或证明COT物理量无用。混合臂还使用冻结SimVP辅助，不能归因成纯COT头收益。
- 报告：`attempt3_results.md`；独立数字：`independent_score_attempt3.json`；图：`forecast_ghi_attempt3_dateheldout.png/.pdf`；日志和GPU分配：`pbs_attempt3.log`、`allocation_attempt3.csv`；所有模型输出打包：`attempt3_artifacts.tar.gz`。
- 结论：在修正检查点选择过疏并做日期隔离后，当前已有的COT外推输入仍未改善这条GHI预报链路；因此保留负结果，不据此扩seed或改论文结论。
