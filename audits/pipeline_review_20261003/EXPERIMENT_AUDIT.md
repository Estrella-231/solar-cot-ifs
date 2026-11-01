# 全链路实验完整性复核

日期：2026-10-03。技能：experiment-audit。审查者：GPT-5.6-Sol ultra，fresh agent `/root/pipeline_integrity_review`。review_independence=same-family；acceptance_status=provisional。

## Overall verdict: WARN

未发现提供文件中的新SimVP完成数字自相矛盾或将新合同端到端结果伪称完成。审查为限定文件的静态代码与保存JSON审查，未读test、未提交作业，也未独立读取旧R服务器status。

| 检查 | 状态 | 独立审查结论及证据 |
|---|---|---|
| A 真值来源 | WARN | S是真实AGRI，H是区间站点观测；R是物理QA尚未闭环的CPP参考监督。`scripts/join_forecast_labels.py:40-53`，`scripts/train_regional_r_full_20260926.py:156-164`。 |
| B 分数归一化 | PASS | 未发现用模型自身输出统计归一化分数；S/R使用train-only统计，GHI的/1000为固定尺度。 |
| C 结果存在性 | WARN | 新S的completion、epoch10、run_metadata自洽；旧R服务器status未纳入本次独立复核。`audits/handoff_20261003/simvp_training_complete.json:2-6`。 |
| D 无调用指标代码 | PASS | 本次列出的R metric_triplet、H evaluate均被调用和保存；不代表全仓库每个历史函数均检查。`scripts/train_regional_r_full_20260926.py:204-223`，`scripts/train_regional_cot_ghi_pilot_20260927.py:204-248`。 |
| E 证据范围 | WARN | 新S单seed、12epoch、val-only；旧H为512/128序列单seedpilot，不能证明新合同端到端稳健收益。 |
| F 评价类型 | PASS | S/H是真实观测评价；R是CPP retrieval-reference proxy；未来真实AGRI→R是明确隔离的不可部署oracle诊断。 |

## 优先发现

1. 下游脚本硬绑定旧日期/旧样本数/旧manifest和权重，不能只换新S就称新合同全链。见 `pack_regional_agri_cpp_bank_20260926.py:140-145`、`train_regional_r_full_20260926.py:143-145`、`build_regional_cot_ghi_pilot_20260927.py:95-130`、`join_forecast_labels.py:20-21,55`。
2. 区域H继续使用kt-MSE（`train_regional_cot_ghi_pilot_20260927.py:200-205`）。历史95%～99%的低clear损失集中证据来自20260910旧统计头，不是区域头实测；需独立重算区域头，不能跨版本引用该比例。
3. R监督SOZ<80°、day_mask阈值85°、H白天标签SZA<90°的支持域不同。差异已确认，影响幅度待诊断，不能测试后删行。
4. 新S只保存聚合像元验证指标，尚未保存可重算的逐样本/逐lead/物理通道/站点指标。0.368815不支持GHI改善主张。
5. H预测文件缺少自包含init/target/clear，需要回连旧base pack；新合同正式输出应保存完整行键和标签口径。
6. 两段递归及pooled MSE可能产生远时效平滑/站点误差，是待验证的架构限制，不是已证明的单一失败原因。

详细设计判断与执行优先级见 [处理链复查](../../docs/PIPELINE_DESIGN_REVIEW_20261003.md)。其中扫描时刻、阴影投影、区间积分等为主执行者补充的待核验物理/工程分析，不冒充本独立代理已经完成的数据验证。
