# 成对 A/AC validation pilot：Result-to-Claim 判定

> 后续证据已更新：207626 数值见证和固定目标 COT 配对诊断均已完成，不能再按本页旧 next_experiments_needed 重复启动。最新复审见 [COT_PAIR_RESULT_TO_CLAIM_20260908.md](COT_PAIR_RESULT_TO_CLAIM_20260908.md)，实际执行顺序见 [HOURLY_NEXT_ACTIONS_20260908.md](HOURLY_NEXT_ACTIONS_20260908.md)。本页原始 pilot 判定和当时缺口保留以便溯源。

日期：2026-09-08

```text
claim_supported: no
confidence: high
review_independence: same-family
acceptance_status: provisional
integrity_status: unavailable
diagnostic_status: COMPLETE_WITH_RELOAD_TOLERANCE_WARNING
```

`confidence: high` 仅表示对本次 **P202 不放行** 判定有高信心；不表示已经证明 COT 普遍无效。`integrity_status: unavailable` 表示仓库没有正式的 `EXPERIMENT_AUDIT.json`，但本次专门的逐预测重算、哈希和抽样 checkpoint 前向见证均已通过。由于语义审查来自同模型家族，结论按规则保持 `provisional`。

## intended_claim

显式 COT 能提高固定湖南 SimVP 外推下的站点 GHI 预报精度，并能与满足因果发布时间约束的 IFS 形成互补。

## verdict

本次结果不支持该复合主张。P202 直接检验的无 IFS 对比 `AC vs A` 在 original 和 QC1 两个预先保留的训练口径中均为负收益，因此主张的第一个必要部分没有通过探索性放行门。IFS 的 release time 和完整 step 合同仍未通过，`AN/ACN` 没有运行，互补性部分没有被检验。

这是一项 seed42、validation-only、经各组相同两档学习率选模的探索性 pilot。判定是“当前证据不能支持正向主张”，不是“已经证明任何 COT 方法都无效”。test 未读取，也不应因本结果提前打开。

## what_results_support

两套口径均完整运行 `A/AC × lr{3e-4,1e-3}`，固定 seed42、batch 2048 和相同训练预算。QC1 仅隔离 448 条 train 标签，并重新拟合该口径的 head train-only normalization 和 32 组训练权重；validation 未排除任何行。两套对比均使用同一 99,849 条 validation 行、相同键、真值和掩膜。

各组按预定的两站等权 validation GHI RMSE 选择自身最佳学习率后：

| train cohort | A LR | AC LR | A RMSE | AC RMSE | AC−A | AC 相对变化 |
|---|---:|---:|---:|---:|---:|---:|
| QC1 | 0.0003 | 0.001 | 149.905825 | 151.248653 | +1.342828 W/m² | +0.8958%（更差） |
| original | 0.0003 | 0.001 | 149.817451 | 154.578853 | +4.761402 W/m² | +3.1781%（更差） |

逐站主终点如下，正差值表示 AC 更差：

| train cohort | station | n | A RMSE | AC RMSE | AC−A | 相对变化 |
|---|---|---:|---:|---:|---:|---:|
| QC1 | 四里（station 0） | 49,806 | 169.689407 | 169.297761 | −0.391646 W/m² | −0.2308% |
| QC1 | 竺家（station 1） | 50,043 | 130.122243 | 133.199545 | +3.077303 W/m² | +2.3649% |
| original | 四里（station 0） | 49,806 | 167.568568 | 169.323474 | +1.754906 W/m² | +1.0473% |
| original | 竺家（station 1） | 50,043 | 132.066335 | 139.834232 | +7.767897 W/m² | +5.8818% |

original 中 AC 的 16 个逐 lead RMSE 全部高于 A；QC1 中 AC 在 12/16 个 lead 更差、4/16 个 lead略好。QC1 的四里微小改善不足 1%，同时竺家退化超过 1%，因此既没有达到预设的总体至少 1% RMSE 降低，也没有满足“两站均无明显退化”的方向门。

QC1 将 `AC−A` 的负向差距从 +4.761402 缩小到 +1.342828 W/m²，但没有把方向反转。由于两个口径分别重拟合 normalization、训练权重并重新训练，不能把差距缩小单独归因于 448 条标签，也不能据此确认源记录故障或解释 COT 机制。

证据完整性方面：5/5 数值存在性预检查通过；8 套保存预测的指标均由 NumPy float64 独立重算，checkpoint/prediction SHA、validation 键、真值和掩膜一致。4 个选定 checkpoint 的 CPU 前向见证也通过，但其范围是每模型固定 2 站 × 4 lead 的 8 行，不是全 validation CPU 重放。

负结果后的有界 COT 特征诊断已核对 637,902 行 `c` 特征及六个固定 cache 序列。它没有发现 AGRI/COT 通道顺序、S→物理量→R normalization、`log1p(COT)` 定义、head train-only normalization、QC1 仿射重拟合或 pack 特征映射的明确错误；四维特征均有限，未见全 patch 近常量或输出上限裁剪。该诊断同时保留一个未关闭的数值 warning：R 的 CPU FP32 重载相对既存 GPU cache 在六序列上的最大 `log1p(COT)` 差为 0.008137。本诊断先行使用的最大绝对差门 `0.001` 未通过；随后参考的 `0.01` 来自旧的 CPU FP32 对 GPU FP16 合同，而当前 forecast cache 的 R forward 位于 autocast 外并记录为 FP32，不能事后沿用不同数值合同把本次检查升级为 PASS。

## what_results_dont_support

- 不支持“当前四维显式 `log1p(COT)` 统计分支提高整体湖南 validation GHI 精度”。
- 不支持 COT 与 IFS 的增量互补或超加性交互；IFS 分支尚未具备因果运行合同。
- 不支持正式论文收益、显著性或独立时间 test 泛化；这里只有一个训练种子，pilot 也未运行日期块 bootstrap。
- 不支持“COT 普遍无效”或“COT 导致误差”的普遍结论；这里只覆盖一个冻结 S、一个冻结 R、一个显式四统计量表示和一个小头合同。
- 不支持把负收益归因于通道错误、归一化、检索饱和、forecast-domain shift、容量、优化或标签 QC；这些机制尚未被隔离。
- 当前有界诊断虽未发现明确的特征链实现错误，也不支持宣称 R 的 CPU/GPU FP32 数值路径已完全复现；0.001 门未过且旧 FP16 的 0.01 门不适用于当前合同。
- 不支持把当前 R 输出称为独立认证的物理真值 COT。CPP 只作检索监督参考，且 source NC generation batch 与实际训练 checkpoint 的上游绑定仍为 `UNVERIFIED`。

## missing_evidence

1. R 的后端数值差异尚未闭合。需要用六个已经固定的 cache 序列和完全相同输入/权重，在 GPU 上记录 cuDNN、TF32 及相关后端设置，分别与 CPU 和旧 cache 比较。不得改输入、权重或阈值，也不得把观察结果后再选择的容差当作预定放行门。
2. 当前特征分布诊断不能替代真实 AGRI 与 predicted AGRI 上、相同 target/CPP reference 的检索误差对照。若不新增该配对证据，forecast-domain 检索退化及其与 GHI 负收益的关系应保持未决。
3. COT teacher/source 的完整 provenance binding 未闭合。即使代码和本次 checkpoint 哈希一致，也不能把参考产品来源与该 checkpoint 的训练数据生成链描述成已完全核实。
4. IFS 的 `release_time <= init_time`、同 cycle 完整 step、变量单位和累计量差分合同未通过，因而 C2 没有可用证据。
5. 多种子、日期块置信区间、wide 容量控制和同 CPP 监督隐特征控制均未运行；它们是正式正向归因所需证据，但在 P202 无收益且没有具体实现错误时不应继续扩展。

## suggested_claim_revision

建议当前工程表述为：

> 在固定湖南 SimVP、固定检索器和预注册四维 `log1p(COT)` 统计分支下，单种子 validation pilot 中加入显式 COT 未降低两站等权 GHI RMSE；该结果在 original 与疑似平直段 QC1 两种训练口径下方向一致，QC1 仅缩小退化幅度而未恢复收益。该结果只否定本配置的 P202 放行，不证明 COT 普遍无效；因果 IFS 互补性尚未测试。

在 teacher/source binding 完成前，宜称“冻结 R 导出的显式 `log1p(COT)` 表示”或“CPP-reference-supervised COT 表示”，不宜写成已独立认证的 COT 真值管线。

## next_experiments_needed

1. 先做一次有界 GPU 推理见证：仅对六个已固定 cache 序列重放 R，记录 cuDNN/TF32 开关及其他相关后端设置，比较 CPU、GPU 和旧 cache；不训练、不覆盖 cache、不改变研究配置，也不追认新的事后容差。输出应保持 warning，直到数值合同与差异来源被明确解释。
2. 后端见证之后，再用冻结 R 在预先固定的 train/validation 同一 target、同一有效 CPP 像元上，配对比较 `R(real AGRI)` 与 `R(predicted AGRI)` 的检索误差。未来真实 AGRI/CPP 只作诊断参考，不进入推理输入、选模或 GHI cohort；同一 target 的多初始化必须显式处理。
3. 若上述检查仍没有具体实现错误：关闭本配置的 COT 正向扩展，保留负结果；不启动 P203 的 wide/AL 控制，不启动 P204 多种子正式扩展，不打开 P205 test，也不转向架构、损失、样本或超参数搜索。
4. 仅当诊断定位到可复现的具体实现错误，例如错误通道顺序、错误 normalization、COT 特征装配错误、AC 实际未正确消费 COT 或已证实的 R 后端配置错误时，才允许一次独立命名的修复。修复后冻结其他条件，重复相同 original/QC1、A/AC、两档 LR、seed42、batch 和预算的 P202 矩阵，并重新进行预测重算、checkpoint 见证和 result-to-claim；不得同时加入第二项方法改动。
5. IFS release/native-step/单位合同通过前，不运行 AN/ACN，不声称互补。即使 IFS 单独资产后来可用，当前 C2 仍需新的同 cohort 四格配对证据才能判断。

## evidence

- `audits/head_paired_pilot_20260908.json`：8 套候选、选模结果、逐站/逐 lead 指标和独立重算审计。
- `audits/head_paired_evidence_precheck_20260908.json`：5/5 数值存在性检查；只证明引用值存在。
- `audits/head_paired_reload_20260908.json`：4 个选定 checkpoint 的固定 8 行 CPU FP32 前向见证及全 8 套哈希/validation 身份复验。
- `audits/cot_feature_diagnostics_20260908.json` 与 `docs/COT_FEATURE_DIAGNOSTICS_20260908.md`：通道、normalization、特征分布与六序列 R 重载诊断；状态为 `COMPLETE_WITH_RELOAD_TOLERANCE_WARNING`，不是无条件 PASS。
- `docs/QC_PILOT_DECISION_20260908.md`：结果出现前冻结的 original/QC1 双分支合同。
- `refine-logs/EXPERIMENT_PLAN.md` 与 `refine-logs/FINAL_PROPOSAL.md`：主张、实际意义门、P202 止损和 IFS 因果合同。
