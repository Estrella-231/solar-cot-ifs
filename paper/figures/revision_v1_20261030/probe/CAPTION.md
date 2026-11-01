# Conditional COT information: captions and interpretation

## English

**COT information under forecast and observed-image inputs.** (a) GHI RMSE and paired differences from the frozen forecast baseline, which already uses historical COT. Context is a fixed linear residual probe using predicted AGRI, geometry, historical COT, lead and baseline output; the two augmented probes additionally receive future COT. (b) Reuse of the fitted probes under cross-substitution of the COT source. Entries are RMSE; parentheses give differences from the frozen baseline. The dashed inference column requires future observations. (c) Differences from the context probe by forecast horizon; 0–1 h includes +15 to +60 min. All panels use 32,838 common records from 50 development dates with date-block out-of-fold fitting. Scores average the station-specific RMSEs for three stations and three frozen-baseline sources equally. Whiskers are paired 95% percentile intervals from 5,000 target-date-cluster bootstrap resamples. Observed-image COT is a frozen retrieval from actual future AGRI, used offline; it is neither independent cloud truth nor a deployable input. Negative differences indicate lower error.

## 中文

**预测图与真实图输入下的COT信息。** (a) GHI RMSE及相对冻结预报基线的配对差值；该基线已使用历史COT。上下文组为固定线性残差探针，输入预测AGRI、几何、历史COT、时效和基线输出；另外两组加入未来COT。(b) 复用已拟合探针并交叉替换COT来源；单元格为RMSE，括号为相对冻结基线的差值，虚线框推理列需要未来观测。(c) 各预报时效区间相对上下文探针的差值，0–1 h包含+15至+60 min。所有面板使用50个开发日期的32,838条共同记录，按日期块做折外拟合。先对三站与三份冻结基线来源分别计算RMSE，再等权平均。误差线为5,000次目标日期簇bootstrap的配对95%百分位区间。真实图COT是冻结反演器处理实际未来AGRI得到的代理，只作离线诊断，既非独立云真值，也非可部署输入。负差值表示误差较低。

## Interpretation for the text

Predicted COT reduces RMSE relative to the context probe by only 0.542 W/m², with a paired interval spanning zero, and both causal-input probes perform worse than the existing frozen baseline. Actual-future-image COT permits a substantially lower diagnostic error, but transferring that fitted mapping to predicted COT loses this advantage. This is consistent with both source-information loss and limited mapping adaptation; it does not identify a unique cause or establish an independent forecast upgrade. The four-hour pattern must not be called a change in COT physical accuracy: its ordinate is GHI error difference.

The 50 target dates span 2025-08-06 through 2026-07-12, not continuous daily coverage. The source probes are deterministic ridge fits; the three source identities refer to existing GHI baseline trainings, not three new random probe seeds. Source substitutions were posthoc diagnostics using the already fitted mappings, not preregistered primary comparisons. The separate 12-date AGRI×COT source-coherence experiment is not mixed into this figure.

## Alt text

A combined score-and-interval panel shows higher error for context and predicted-COT probes than for the frozen baseline, and lower error only when actual future image-derived COT is used offline. A two-by-two matrix shows that a mapping fitted to observed-image COT loses its low error when supplied with predicted COT. A horizon panel shows much larger offline observed-COT increments than predicted-COT increments.

## Verification

The full-horizon and four hourly point estimates and intervals for both the primary probes and source substitutions were recomputed from saved daily squared-error sums and counts. Maximum point-estimate difference is zero; maximum interval difference is below 3×10⁻¹⁴ W/m². PNG and PDF-render layout checks cover axes, all intervals, the offline-column outline, annotations and legends. Exported numerical values are in `plotted_values.csv`; exact source hashes and verification errors are in `source_manifest.json`. No filtering, new fitting, inference, training or test-data access occurred.
