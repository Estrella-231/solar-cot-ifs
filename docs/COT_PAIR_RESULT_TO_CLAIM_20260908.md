# 固定 COT 配对诊断：Result-to-Claim 复审（2026-09-08）

```text
claim_supported: no
confidence: high
review_independence: same-family
acceptance_status: provisional
integrity_status: unavailable
deterministic_saved_result_audit: PASS_INDEPENDENT_SAVED_COT_RESULTS_AUDIT
evidence_precheck: 5/5 verified (evidence existence only)
decision: CLOSE_THIS_CONFIGURATION_COT_POSITIVE_EXPANSION
test_status: unopened
```

`confidence: high` 只表示：对“新配对诊断没有推翻 P202 不放行结论，并且应按原止损规则关闭当前配置的 COT 正向扩展”有高信心。它不表示已证明 COT 普遍无效，也不表示已证明外推域检索退化是 A/AC 负收益的唯一原因。`integrity_status: unavailable` 表示仓库中没有正式 `EXPERIMENT_AUDIT.json`；专门的保存数组审计虽然通过，但只验证数值、键、参考和掩膜，不重跑模型，也不能把语义结论升级为 PASS。由于本次是同模型家族复审，结论保持 `provisional`。

## intended_claim

显式 COT 能提高固定湖南 SimVP 外推下的站点 GHI 预报精度，并能与满足因果发布时间约束的 IFS 形成互补。

## verdict

完成的 COT-reference 配对诊断**不改变**此前 P202 的 `claim_supported: no` 判定。它补齐了当时允许的最后一项有界输入域诊断，并在当前冻结的 S、R、CPP-reference 和样本合同下发现明确的描述性事实：使用 SimVP 外推 AGRI 时，冻结 R 的参考检索误差高于使用真实 AGRI 时的误差。这使“forecast-domain 检索退化可能限制显式 COT 分支”成为有直接证据的解释候选，但没有把该解释提升为 GHI 负收益的因果机制。

P202 的直接下游终点仍然是决定性证据：seed42、validation-only 的 8 次 `A/AC × 两档 LR × original/QC1` pilot 中，按预定规则选模后，AC 相对 A 的两站等权 GHI RMSE 在 original 中恶化 `+3.1781%`，在 QC1 中恶化 `+0.8958%`；没有达到至少 1% 改善及两站均无明显退化的放行门。新诊断不计算 GHI、不训练头，也不提供能逆转这一直接结果的证据。

IFS 部分仍无可评价证据。已审计的湖南站点缓存只有 step 1–4，`release_fields=[]`，状态为 `BLOCKED_NO_AUDITED_RELEASE_RECORDS`；后续有界文档复查也没有发现适用于当前湖南 train/validation 范围的新 release-aware 来源。因此 `AN/ACN` 和四格交互没有运行，COT 与因果 IFS 的互补性仍未测试。

按 [EXPERIMENT_PLAN](../refine-logs/EXPERIMENT_PLAN.md) 的 P202 止损规则，当前配置现在应关闭正向扩展：不启动 P203 wide/AL，不启动 P204 多种子，不打开 P205 test，也不转向网络、损失、样本或超参数搜索。没有发现足以授权一次修复复跑的具体实现错误。

## what_results_support

1. **固定 validation 样本中的 forecast-input 检索退化。** 协议在读取误差前固定每站 32 个等间隔排序 target，共 64 个；其中 58 个 target 可配出 872 个 target/lead 对，152 个缺失 lead 和 6 个完全无 forecast 的 target 原样保留，未按覆盖、天气、CPP 值或误差换样本。
2. **同配对比较方向清楚。** 872 对共 219,310 个有效像元贡献上，physical COT RMSE 为 `R(real AGRI) 4.293595531910919` 对 `R(forecast AGRI) 10.131401625214263`。四里为 `4.659453035215319 → 12.608411155302491`，竺家为 `3.890977783952497 → 6.780822399830006`。两站 × 16 lead 的 32 组中，forecast 端的 physical COT MAE/RMSE 与 log1p MAE/RMSE 均为 32/32 数值更高。
3. **主要混杂项在本诊断中受控。** 两支的 R 权重、train normalization、通道顺序、三几何、CPP reference 和原 mask 相同，仅 13 个 AGRI 输入不同；872 对几何逐值相同。两支均以 CPU float32、batch 32、eval/no_grad 重跑，没有使用旧 GPU COT cache。两支均没有非有限值填补，且没有 clipping。
4. **保存结果的数值链可复核。** 独立 NumPy float64 审计没有导入诊断 runner 或 Torch，重算 635 个浮点指标的最大绝对差为 0；872 个保存键与冻结 inventory 一致，重复 target 的 real 预测、reference 和 mask 逐值相同。5/5 数值存在性预检查通过，但只证明被引用值存在。
5. **已保存时间键未显示输入时刻错位。** 872 对均满足 `target_time - init_time = lead_minutes`，保存的 `(station, init, target)` 均唯一；当前证据没有发现可解释结果的 cohort 或输入时刻 bug。
6. **v1 失败不污染 v2 数值。** 诊断 v1 在读取 checkpoint metadata 后，因相对 `meta.pack` 路径未按工程根解析而在 COT 数值计算前停止。v2 只增加 `root / relative_path` 解析，保留原协议、selection、模型、batch 和评价口径，并写入新目录；原失败产物仍保留。
7. **旧 cache warning 仍是数值告警，不是已定位的实现错误。** 原 cache 重放最大 log1p 差 `0.001033306` 仍超过冻结的 `0.001` 门；同时改变 benchmark 和 cuDNN TF32 后的 GPU/CPU 差异不能单独归因于其中一个设置，也没有证据说明该差异造成 P202 负收益。本配对诊断两支使用相同 CPU 路径，因此没有把旧 cache 后端差异混入 real/forecast 对比。

单独报告的 58 个 unique-real target RMSE `4.2879244141718456` 使用 14,571 个有效像元、每个 target 只计一次。它与 872 对中跨 lead 重复 target 的 forecast 汇总权重不同，不能直接比较；本复审没有作这种比较。

## what_results_dont_support

- 不支持“当前四维显式 `log1p(COT)` 统计分支提高固定湖南 SimVP 外推下的站点 GHI 精度”。直接 A/AC pilot 的总体结果方向相反。
- 不支持 COT 与 IFS 的增量互补或超加性交互；因果 IFS 输入合同仍未通过，相关模型未运行。
- 不证明 forecast-domain 检索退化是 AC 负收益的唯一原因，也不证明检索 RMSE 与 GHI RMSE 之间存在因果中介关系。本诊断没有读取 GHI。
- 不证明误差随 lead 单调增长或形成可推广的 lead 曲线。同一 target 跨多个 lead 重复，而不同 lead 的可用 target 数也不同；872 对不是 872 个独立真值样本。
- 不支持正式显著性、独立时间 test 泛化或部署性能。R 曾在该 validation 上选模，抽样也是 validation 描述；test 保持未读。
- 不支持把 `R(real AGRI)` 当作可部署 oracle。真实未来 AGRI 只用于诊断参考，不能进入业务推理。
- 不支持把 CPP-reference 称为无误差物理真值，也不支持声称 teacher/source provenance 已闭合。source NC 生成批次与实际生成 checkpoint 的绑定仍未认证。
- 不支持“所有 COT 表示都无效”。结论只覆盖一个冻结 S、一个冻结 R、一个四维显式统计分支和当前小头合同。
- 不支持把旧 cache 的超阈值差异定性为 TF32 错误、实现错误或 P202 负收益原因；该告警仍应原样保留。

## missing_evidence

以下是原复合主张若要成立仍然缺少的证据；在 P202 已失败且未发现具体实现错误的情况下，它们是主张缺口，不是继续扩展当前配置的执行清单。

1. 多种子、日期块置信区间、冻结方案后的独立时间 test，以及同 CPP 监督隐特征与容量控制均未运行。它们原本只应在 P202 方向门通过后用于正式确认与归因。
2. 当前没有把逐 target 的检索误差变化与相同样本的 A/AC GHI 预测差异作预注册、独立的机制检验；即使补做相关分析，也只能提供关联诊断，不能自动证明因果。
3. CPP source NC、生成批次和 R teacher checkpoint 的完整 provenance binding 尚未闭合，因此只能称 CPP-reference-supervised COT 表示。
4. IFS 仍缺 `release_time <= init_time` 的可审计记录、覆盖所需时效的同 cycle 原生 step、变量单位与 SSRD 前一步差分证据；没有共同 cohort 的 A/AC/AN/ACN 四格预测，自然也没有互补性或交互置信区间。
5. 没有跨模型家族的语义审查；本次同家族结论只能作为 provisional gate。

## suggested_claim_revision

建议将当前工程结论固定为：

> 在预先固定的湖南 validation 目标、相同 CPP-reference 与原 mask 下，冻结 R 使用 SimVP 外推 AGRI 时的 COT 参考检索误差数值高于使用真实 AGRI 时；同配对池化 RMSE 为 10.131402 对 4.293596，两站全部 32 个 station/lead 组的四项误差方向一致。与此独立地，seed42 的 original/QC1 A/AC validation pilot 中加入预注册四维显式 COT 统计未降低两站等权 GHI RMSE。因此当前配置未通过 P202，配对诊断不改变无收益判定，只支持 forecast-domain 检索退化这一有界描述。结果不证明该退化是 GHI 负收益的唯一机制，不证明 COT 普遍无效；因果 IFS 互补性尚未测试。

在 teacher/source binding 完成前，使用“冻结 R 导出的、CPP-reference-supervised 显式 `log1p(COT)` 表示”，不要使用“独立认证的 COT 真值管线”。

## concrete_next_action

1. 将当前 S/R/四维 COT/小头合同标记为 **P202 closed, positive expansion stopped**，保留 original/QC1 的 GHI 负结果与本次配对诊断作为失败边界证据。
2. 不运行 P203 wide/AL、P204 多种子、P205 test，也不以本诊断为由尝试新架构、loss、样本筛选、容差或超参数。test 继续封闭。
3. IFS 保持独立阻塞；只有出现覆盖当前湖南历史范围的具体新 manifest、下载来源或 release-aware 记录时，才做一次限定文件与字段的只读核验。不要重复扫描现有 step1–4 cache，也不要用旧湖北 2026-06/07 合同替代。
4. 当前没有满足“明确、可复现实现错误”的重开条件。以后若出现新的确定性证据定位到通道、normalization、特征装配或模型实际消费错误，应另立名称并重新走冻结协议；现有后端 warning、v1 路径解析失败和本次域退化本身均不满足该条件。
5. 论文或计划中的正向复合主张应撤回到上述有界负结果与未决 IFS 状态；在没有新资产事件前，本配置无需继续计算。

## evidence

- [冻结配对协议](COT_PAIR_DIAGNOSTIC_PROTOCOL_20260908.md)
- `audits/cot_pair_diag_20260908_v2/report.json` 与同目录 `cot_pair_predictions_20260908_v1.npz`
- `audits/cot_pair_results_audit_20260908.json`
- [配对结果独立复核](COT_PAIR_RESULTS_REVIEW_20260908.md)
- [配对诊断结果汇总](COT_PAIR_DIAGNOSTIC_RESULTS_20260908.md)
- `audits/cot_pair_evidence_precheck_20260908.json`
- [此前 P202 Result-to-Claim](HEAD_PAIRED_RESULT_TO_CLAIM_20260908.md)
- [R cache 数值后端诊断](R_CACHE_NUMERIC_RESULTS_20260908.md)
- [实验计划与止损规则](../refine-logs/EXPERIMENT_PLAN.md)
- `audits/ifs_station_source_20260907.json` 与 [IFS 来源线索有界复查](IFS_SOURCE_FOLLOWUP_20260908.md)
