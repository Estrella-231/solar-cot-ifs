# 复现与执行说明

主计划见EXPERIMENT_PLAN.md；状态见EXPERIMENT_TRACKER.md与PIPELINE_STATE.json。所有训练、审计与恢复输出另存新目录，旧结果保留。

- `scripts/solar_geometry_noaa.py`为项目新入口；本目录code中的同名文件为冻结实验副本。
- `noaa_main_reference.js`来自 https://gml.noaa.gov/grad/solcalc/main.js 。验证时显式关闭折射，仅比较几何天顶角及北起顺时针方位角；哈希见FORMULA_REFERENCE_UNIT_AUDIT.json。
- `code/audit_and_prepare_geometry_parallel.py`计算所有实际train/val网格；旧aux/raw不一致单独记录，全量新几何须通过官方独立计算，不能因为旧aux不一致而跳过样本。
- `code/augment_restoration_requests.py`从已有AGRI数组提取C13参照，`restore_daylit_reflectance.py`按真实HDF索引恢复被旧夜间掩膜置零、但新几何属于白天的反射率。C13不匹配的HDF版本拒绝采用；不使用合成云、插值或伪造云高。
- `code/build_corrected_data.py`检查24帧时间及fold隔离，在node22单份本地cache保存新输入；重新拟合各折训练统计和R标签有效掩膜。旧的GHI行/标签/晴空量保持一致。
- `code/advance_pipeline.py`是本地确定性依赖执行器，只启动本次授权的修复流程；任何未知提交结果或数据错误均停止并写PIPELINE_STATE.json，人工定位后另存attempt，不盲目重试。
- `code/queue_controller.py`只调度本次明确列出的profile、R、六份S与GHI，最多3个账户未结束PBS任务、8张GPU。`STOP_QUEUE`仅停止新提交，不取消已有或其他项目作业。作业/训练结果以实际完成文件和保存预测为准。
- `code/run_stages.py`保持原架构、全模型globalbatch16/折模型12、S50epoch/patience5、R50epoch/patience8、GHI30epoch预算。优先单A800，实际profile不能容纳原globalbatch时才分两卡。
- `score_predictions.py`依据保存的逐样本预测重算；模型选择使用完整验证均值。只用train/val，test和buffer不读取。

公式修正会影响旧模型训练输入，因此不能仅给旧checkpoint换新角度并把结果当作完整修复。是否提高COT/GHI必须等本轮配对结果。
