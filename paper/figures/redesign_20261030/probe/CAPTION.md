# Conditional information figure — caption and use

## English caption

**COT information retained or lost when observed imagery is replaced by forecasts.** (a) GHI error of the frozen forecast baseline (AGRI plus eight historical COT frames) and fixed linear residual probes. The context probe uses predicted AGRI, solar geometry, historical COT, forecast lead and the frozen baseline output; its future-COT slots are set to zero. The two augmented probes use either COT retrieved from predicted AGRI or COT retrieved from actual future AGRI. (b) Paired differences from the frozen baseline; negative values indicate lower error. (c) Cross-substitution of the COT source at fitting and inference, with the original fitted probes reused without refitting. Values are GHI RMSE, with differences from the frozen baseline in parentheses. The dashed right column requires future observations. (d) Increment relative to the context probe within the four lead intervals; the first interval contains +15, +30, +45 and +60 min. All panels use the same 32,838 station–initialization–target records spanning 50 target dates between 6 August 2025 and 12 July 2026 in Hubei (not continuous daily coverage). RMSE is calculated separately for each of three stations and each of three frozen-baseline model sources, then averaged with equal weights. Whiskers are paired 95% percentile intervals from 5,000 target-date cluster bootstrap resamples. The probes use three-fold date-block out-of-fold prediction within development dates, with an excluded day on either side of each held-out block. *Actual-image COT is a frozen retrieval from actual future AGRI, used only for offline diagnosis, not an independent COT truth or a deployable forecast input. Orange denotes predicted future COT and purple denotes actual-future COT used offline. Shared colors across figures denote source roles, not identical weights or samples. These are development diagnostics, not independent test results.

## 中文对照

**观测图像替换为预报图像时，COT中保留与损失的信息。** (a) 冻结预报基线（AGRI加8帧历史COT）与固定线性残差探针的GHI误差。上下文探针使用预测AGRI、太阳几何、历史COT、预报时效及冻结基线输出，将未来COT特征槽置零；两个增强探针分别加入预测AGRI反演的COT或实际未来AGRI反演的COT。(b) 相对冻结基线的配对差值，负值表示误差降低。(c) 复用原探针而不重新拟合，交叉替换拟合与推理时的COT来源。单元格为GHI RMSE，括号内为相对冻结基线的差值；虚线框右列需要未来观测。(d) 四个预报时效区间内相对上下文探针的增量；首区间包含+15、+30、+45和+60 min。各面板使用湖北同一组32,838条站点—初始化—目标记录，覆盖2025年8月6日至2026年7月12日之间的50个目标日期，并非连续逐日覆盖。先对三站与三份冻结基线模型来源分别计算RMSE，再等权平均。误差线为5,000次按目标日期成簇重采样得到的配对95%百分位区间。探针仅在开发日期内部进行三折日期块折外预测，拟合排除留出日期及其前后各一天。*实际图像COT来自冻结反演器处理实际未来AGRI，仅用于离线诊断，不是独立COT真值，也不是可部署的预报输入。橙色表示预测未来COT，紫色表示仅用于离线诊断的实际未来COT。同色表示来源角色，不代表跨图同权重或同样本。此图属于开发诊断，不是独立测试结果。

## Scientific interpretation

The actual-future reference can carry GHI-related information that is not retained by the predicted input. Both probes with causal inference inputs in panel (a) are worse than the frozen baseline. The small improvement of predicted COT over the weaker context probe cannot be presented as a forecast upgrade. Panel (c) additionally shows that simply learning a mapping on the actual reference does not preserve its performance when predicted COT is supplied. This supports a joint source-quality and mapping-adaptation problem; it does not identify a unique causal failure mechanism. The actual-image proxy includes finite nonnegative retrievals above the COT training-reference range of 100, as documented in the original common-support audit; those values are not independent confirmation of cloud optical thickness.

## Figure provenance and review

- All numeric values are loaded by `plot_conditional_information.py` from immutable saved score JSON files; plotted values are exported as `plotted_values.csv`.
- Point estimates and full-horizon confidence intervals are independently reproduced from the archived date-level squared-error sums/counts before plotting. The verification differences and source SHA-256 values are saved in `source_manifest.json`.
- The separate 12-date AGRI×COT coherence experiment was read for context but is **not** included in this figure, avoiding a mixed-cohort comparison.
- Outputs: 177 × 133 mm vector PDF, editable-text SVG, 300 dpi PNG, and 150 dpi print-size preview. No raw data, model, mask, threshold or training was changed.
- This figure is a candidate for the forecast-information diagnostic section or supplement. It is **not** a replacement for the main paired prospective GHI evaluation.
- Rendering review found and corrected overlap of the numeric annotations and upper confidence intervals in panel (b). All 95% intervals are visible; the zero reference is retained and source-specific offline status is stated in the caption.
- Skills used: `scientific-visualization` and `paper-figure`. The generic recommendation for bars/radar charts was not used because paired estimates and source substitution are the relevant scientific quantities.

## Alt text

Four-panel diagnostic figure. Absolute errors show that adding predicted COT to a context probe changes RMSE only slightly, and both causal probes remain worse than the frozen forecast baseline. Actual-future-image COT gives substantially lower error only in an offline probe. A two-by-two matrix shows that this lower error is lost when the same mapping receives predicted COT. Hourly intervals show that the gap between actual and predicted COT information widens at longer forecast leads.
