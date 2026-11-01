# 固定 GHI 头下的预测 COT 替换诊断（验证集）

## 问题与比较

固定 seed42 的 AC GHI 头，对每个验证预测实例保持**目标时刻真实 AGRI、太阳几何、站点、GHI 标签和模型权重不变**。只替换 COT 输入：

- 参照：同一目标时刻的真实 AGRI → 冻结 R 得到的同期 COT；
- 候选：SimVP 预测 AGRI → 同一冻结 R 得到的预测 COT。

这隔离的是“预测 COT 替换对固定 GHI 头的影响”。候选使用真实目标 AGRI，因此属于离线 oracle-AGRI 归因，**不是可部署的端到端 GHI 预报分数**；它没有衡量预测 AGRI 的误差，也没有训练任何新头。

## 样本与对齐

使用现有预测bank中的271个CPP支持验证窗口；GHI打分没有再套CPP像元掩膜或进行二次筛选，但窗口队列本身受CPP支持覆盖限制，因此不是全验证集自然分布。每个预测实例按 `(station, target_time_utc)` 连接冻结 GHI pack；重复目标时刻在不同初始化和 lead 下保留，因为其预测 COT 可能不同。共得到 7,005 个 forecast-instance 行（四里 3,314、竺家 3,691），涉及 529 个唯一站点—目标时刻键、271 个初始化窗口和 8 个 BJT 目标日期。窗口有重叠，不能把 7,005 行视为独立样本。

## 结果

| 站点 | 同期 R-COT RMSE | 预测 COT + 真实 AGRI RMSE | 差值 |
|---|---:|---:|---:|
| 四里 | 71.16 | 75.89 | +4.73 W m⁻² |
| 竺家 | 79.41 | 84.96 | +5.55 W m⁻² |
| 两站 RMSE 等权均值 | 75.29 | 80.43 | +5.14 W m⁻² |

独立 FP64 复算的两站等权差值为 **+5.14 W m⁻²**，按北京时间目标日期成块、配对抽样 5,000 次所得 95% 区间为 **[−3.98, 11.91] W m⁻²**，跨过 0。当前只有 8 个 BJT 日期，不能据此作显著性结论。

逐 lead 的误差差值在短时效较小且方向不一，随后扩大：四里 +240 min 为 +21.36 W m⁻²，竺家 +240 min 为 +12.08 W m⁻²。该模式与预测 COT 长时效退化相符，但受日期少、预测窗重叠和真实 AGRI oracle 条件限制，只作为错误传播线索。

## 可复核文件

- `paired_predictions.csv`：逐预测实例、两种 COT 输入下的固定头输出。
- `fp64_independent_audit.json`：独立 FP64 指标、5,000 次目标日期块配对 bootstrap。
- `lead_metrics_fp64.csv`：按站点和时效的 RMSE。
- `key_alignment_receipt.json` 与 `key_alignment.csv`：全 bank 连接覆盖和键映射。
- `audit_fp64.py`：独立重算脚本。
- PBS 作业 `213859.tc6000` 完成；远端推理来源和 checkpoint 哈希记录于原始 `summary.json`。

## 结论边界与下一步

这项归因说明：当 AGRI 和太阳几何已是真实目标时，把预测 COT送入现有 AC 头，整体表现方向上更差，尤其长 lead；目前日期块区间仍很宽。它不能回答“预测 AGRI + 预测 COT 的整体 GHI 头会怎样”。下一步是为相同 forecast 实例保存站点 3×3 预测 AGRI，再分别做固定头的 AGRI-only 替换和 AGRI+COT 联合替换；只有这些共同支持离线归因闭合后，才依据 T17 重新判断是否值得训练 forecast-GHI 头。新 test 未打开。


### Lead-wise figure

![Fixed-head lead-wise GHI RMSE difference](forecast_cot_fixed_head_lead_sensitivity.png)

Positive values mean replacing same-time R-COT with forecast COT worsens the fixed head's GHI RMSE. Points are lead-wise estimates only; no per-lead confidence intervals are shown. The overall paired 95% date-block interval is in `fp64_independent_audit.json`. Alt text: Two station curves start near zero at short leads and rise at longer leads; Sili reaches about +21 W m⁻² and Zhujia about +12 W m⁻² at +240 min. Actual target AGRI and solar geometry are held fixed, so the plot is an offline COT-only attribution, not an operational forecast score.

Figure design followed general scientific visualization guidance: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065, v2. https://doi.org/10.48550/arXiv.2609.00065.


