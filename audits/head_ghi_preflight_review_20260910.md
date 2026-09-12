# 独立部署前复核

子任务ghi_protocol_check只读复核：新core与旧pilot相比仅目标函数、相应日志/门禁文字及移除旧main；Head、seed、epoch排列、AdamW、选模/早停一致。QC1过滤、x/g/c变换和组权重与旧QC脚本一致。

独立代理在zjnu-hpc按协议原命令执行CPU --cpu-check，退出0，PASS_OBJECTIVE_VALUE_GRADIENT_AND_ARCHITECTURE，loss=1.3651920557022095。没有提交PBS、没有修改文件。

主代理在提交前落实两项建议：补充baseline_profile_sha256/profile_reused_from_original明确沿用旧profile；新NumPy汇总补充旧kt候选同键重算、差中差与同LR AC−A。后者由主代理检查并通过语法编译，未冒充独立代理已执行数值验收；实际验收在PBS训练完成后运行。新CPU重载脚本使用原Head实现和事前固定8行/容差。

PBS208020运行初检通过，原始pack已加载为538053train/99849val。此记录不宣称8候选训练或最终重载已完成。没有恢复持续监控。
