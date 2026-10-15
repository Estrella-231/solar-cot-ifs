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
