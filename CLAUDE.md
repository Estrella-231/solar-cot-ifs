# 2026-11-02 用户确定：真实图训练GHI头

当前主路线使用真实同期AGRI三帧训练的既有GHI头，冻结后接预测AGRI及预测COT进行预报。不再用预测图训练或微调GHI头；不得把这类变化隐含在“全train重训”中。此决定覆盖此前“两条训练路径均可默认推进”的表述。train/validation/test外层日期清单与其他数据规则不变。

Details: docs/GHI_HEAD_ROUTE_20261102.md (repository root).

---

# 第二篇工作入口

更新：2026-10-30。先读`AGENTS.md`、`IRRADIANCE_ERROR_PREVENTION.md`和`docs/execution_simplification_20261030/CURRENT_POLICY.md`，再读当前任务目录的方案和tracker。旧的“planning_only_no_training_launched”“先重新P200审计”状态已过时，原文保存在`docs/execution_simplification_20261030/runtime_fix_before/CLAUDE.md`。

## 当前执行默认值

- 普通级联使用全部train和已核实资产；U/H、U-only R、OOF及多层内部划分均非默认要求。外层train/validation/test清单不变。
- 有实际未来预测与有效GHI标签即可做配对评价；辅助COT指标不作普遍准入门槛。真实同期输入训练的冻结头和预测输入训练的头均可评价，按实际训练域报告。
- validation可用于开发和选模。分站、分时效及区间完整报告，不默认要求所有项都改善。不能把验证复用说成独立test，也不因复用而停止所有开发。
- 新接口只检查受影响的时间、空间、标签、单位、归一化和数值链路；未变的来源引用已有核验，不另等审批或审查收据。
- 长任务优先有效的多卡配置，保持全局batch、更新预算和学习率计划；全账户未结束PBS≤3。例行监控交Luna，不自动恢复旧作业或定时。

## 结果与执行入口

- 总表：`docs/experiment_results_20261030/SUMMARY.md`。
- 最新完整湖南验证：`refine-logs/hunan_dydiff_20261028/full_validation_20261029/RESULTS.md`，SimVP预测COT使GHI RMSE由127.421降至125.340 W/m²（1.633%）；竺家单站区间跨零，旧负结果同样保留。
- 交接：`docs/AI_EXECUTION_HANDOFF.md`。
- 修复落实：`docs/execution_simplification_20261030/IMPLEMENTATION.md`。

普通湖北预报头使用`scripts/train_hubei_ghi_heads.py --input-mode forecast`，默认接受全train预测数据包；只有明确研究OOF时才加`--require-oof`，复现五折时再加`--oof-fold-count 5`。具体数据、预算和输出目录由当前实验方案决定。

test/buffer数值载荷关闭；真实未来AGRI/CPP不得作为可部署推理输入。两省资产独立，时间、站点索引、15分钟右闭标签及共同评价样本规则保留。已完成的模型、分数、原判定和论文正文不随执行简化改写。本文不声明实时PBS状态，当前运行情况须现场查询。
