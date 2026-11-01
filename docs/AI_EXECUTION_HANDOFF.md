# 2026-11-02 用户确定：真实图训练GHI头

执行时同时遵守仓库AGENTS的“研究规则纠错机制”：每轮完整比较后复查关键前提，出现旧新结果矛盾时优先做低成本判别；不把当前训练域选择升格为永久科学禁令。旧限制审计见[REPORT](rule_correction_20261102/REPORT.md)。本次规则整理不启动训练或恢复定时。

当前主路线使用真实同期AGRI三帧训练的既有GHI头，冻结后接预测AGRI及预测COT进行预报。不再用预测图训练或微调GHI头；不得把这类变化隐含在“全train重训”中。此决定覆盖此前“两条训练路径均可默认推进”的表述。train/validation/test外层日期清单与其他数据规则不变。

Details: docs/GHI_HEAD_ROUTE_20261102.md (repository root).

---

# 第二篇执行交接

更新：2026-10-30。当前任务按仓库`AGENTS.md`和[简化执行方式](execution_simplification_20261030/CURRENT_POLICY.md)执行。原交接把所有任务退回P200、反复复核不变资产和单卡先行，现已替换；旧文件完整保存在`execution_simplification_20261030/runtime_fix_before/docs/AI_EXECUTION_HANDOFF.md`。

## 从哪里继续

1. 读取当前任务目录的方案、tracker和[结果总表](experiment_results_20261030/SUMMARY.md)，确认已经做过什么、当前问题及唯一新增因素。不要根据9月的“尚未训练”状态重建整条链。
2. 普通实验默认使用原外层train/validation/test划分和全部train，复用已核实的上游权重、归一化及保存输出。上游在train上的预测可用于下游训练，不默认追加U/H、U-only R、五折OOF或多层校准留出。明确研究cross-fitting时才单独使用OOF。
3. 对本次真正改变的接口做小样本检查：UTC/BJT、站点像元、初始化与目标时刻、15分钟标签、单位、train归一化、模型重载和有限值。未变的资产引用已有核验；有新差异再定位受影响环节。
4. 有有效未来预测与GHI标签时直接做同样本有无COT评价。CPP覆盖、COT RMSE/CSI和文献口径用于解释，不普遍阻断GHI。允许真实同期图像训练的冻结头接预测图，也允许预测域重训；注明训练域和帧数。
5. 按当前问题确定主要比较，保存逐样本预测并复算，报告两站/三站、各时效及不确定性。辅助指标不全部改善时解释差异，不把它们追加成继续开发的通行证。旧实验依原判据报告，不事后改判。
6. validation继续用于开发与选模，保留候选与负结果；不必每个想法都等新观测。最终独立确认与开发验证分开，不打开封存test/buffer载荷。

## 代码与运行

湖北普通预测数据包可由`scripts/build_hubei_forecast_ghi_heads_pack_20261016.py`在不指定`--oof-train-bank`时构建；`scripts/train_hubei_ghi_heads.py --input-mode forecast`默认接受该全train级联输入。仅复现OOF设计时指定`--require-oof`；需要原五折数时再指定`--oof-fold-count 5`。模型、损失、预算、归一化及评价行没有随该开关改变。

长训练、批量推理和评价优先使用有效的多卡配置，保持全局batch、更新数和学习率计划。每次提交前后查全账户队列，未结束PBS最多3个；核对调度分配与物理占用，不操作他人作业。保留已有断点；例行监控交Luna。本文不恢复已停任务或定时。

本地仓库为`F:/CODE_Classify/辐照度预报/solar-cot-ifs`；历史登记服务器工程根为`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907`。具体实验可能使用独立目录，以当前runner和tracker为准。历史登记不等于当前可达或已经同步，远端修改前核对相应文件，避免覆盖其他窗口改动。

## 结果和必须保留的边界

湖南完整SimVP冻结验证已有127.421→125.340 W/m²（1.633%）的总体改善，见[完整结果](../refine-logs/hunan_dydiff_20261028/full_validation_20261029/RESULTS.md)。它使用真实同期三帧训练后冻结的配对GHI头，输入实际预测图；属于开发验证，竺家单站区间跨零。不要因缺少OOF或未在预报域重训而丢弃该有效结果，也不要用它改写其他设置的负结果。

时间/空间/标签、方法间同样本与掩膜、train拟合归一化及未来推理因果性仍是必要检查。真实未来AGRI/CPP可作训练监督或单列离线参考，不能进入可部署推理；CPP是反演参考。IFS来源未明确时可继续COT实验，不虚构实时IFS收益。

本次交接修复不修改论文正文。以后修改正文时，英文LaTeX与中文Markdown须同步，编译并查看实际页面；工程路径和内部修复留在工程文档。
