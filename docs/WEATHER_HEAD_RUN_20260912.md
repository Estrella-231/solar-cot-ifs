# 天气分类头执行记录

最新：v2训练与12候选分组重算已完成，日志WEATHER_HEAD_AND_COMPARISON_COMPLETE；complete.json、paired_cot_effect.json及逐行概率已归档并通过本地SHA/混淆矩阵/计数复算。准确率67.57%，macro-F1=0.6650，多数类基线53.39%。最终结果见WEATHER_HEAD_RESULTS_20260912.md；下文Q/R描述为启动时历史记录。

## I/O修正后的v2启动

208399未进入GPU训练：约8分钟日志停CONTRACT，GPU1仅3MiB且无本任务compute进程；Python83431的wchan为lock_blkgrp_sync，gdb栈为memmove→NumPy contig_to_contig/AssignArray/FromArray，定位在共享FS mmap数组的np.array复制。gdb已detach，随后仅qdel本任务，服务器确认C。没有epoch或训练checkpoint，不算模型失败或数据内容错误。

新代码仅将mmap读取与二次复制改为普通np.load顺序载入，并增加加载/profile进度。保留v1源码、日志、提交回执、数据合同。科学协议、标签、划分、随机种子、优化器、网络不变。

v2提交208402.tc6000成功，提交前无未结束任务（208399为C）；申请仍1A100。最新入口scripts/run_weather_head_20260912_v2.pbs、日志weather_head_20260912_v2_pbs.log、输出runs/weather_coverage_head_seed42_20260912_v2、提交回执audits/weather_head_submission_20260912_v2.json、代码清单audits/weather_head_launch_sha256_20260912_v2.json。完成状态以v2保存结果为准。

2026-09-12按用户更正，实际分类为晴天、多云、阴天，不训练云相分类。协议见WEATHER_HEAD_PROTOCOL_20260912.md。

固定43122个train/validation独立站点目标，全部完成源读取：18487 READ、24628 SOURCE_MISSING、7 READ_ERROR。有效像元至少231/256的云量标签为：

|原划分|晴天|多云|阴天|无标签|
|---|---:|---:|---:|---:|
|train|2385|1301|7783|24667|
|validation|1775|1397|3812|2|

缺测不当晴天，分类训练/准确率仅用有标签样本；预测天气分组保留全部99849验证预测行。标签是源CPP区域云量参考，不是地面全天气象真值。源覆盖限制保留。

PBS208399.tc6000提交成功，提交前账户队列为空，提交后仅本任务Q；申请1A100/8CPU/40min上限。提交成功不表示训练完成。CPU见证独立Luna执行退出0、WEATHER_CPU_WITNESS_PASS；额外云量阈值、无效覆盖、水冰互换不影响类别检查通过。GPU profile及实际训练以服务器日志为准。

服务器根：`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907`。

- 协议：`docs/WEATHER_HEAD_PROTOCOL_20260912.md`
- 源标签与映射：`data/weather_clp_trainval_20260912_v1/`
- 提交回执：`audits/weather_head_submission_20260912_v1.json`
- 代码SHA清单：`audits/weather_head_launch_sha256_20260912.json`
- 日志：`weather_head_20260912_v1_pbs.log`
- 模型、逐行分类预测、分类指标与COT分组结果：`runs/weather_coverage_head_seed42_20260912_v1/`

用户未要求持续监控，仅本轮启动/结果有界检查由Luna执行；不建立小时自动任务。test未打开。已知源码/标签门禁不放宽；本轮新增头是用户明确授权的独立诊断实验。
