# Forecast cloud-regime and event figures

These figures use only the completed `refine-logs/forecast_thick_cloud_events_20261102` analysis. No rescoring, new inference, training, sample change, or reconstructed cloud texture is involved.

## Deliverables

- `forecast_regimes.pdf/png/svg`: 180 × 112 mm. Paired absolute GHI errors and their differences on common reference-covered forecast records. Both regions use the same axes. The four thickness bins and unique station–target/date counts remain visible; moderate-COT degradation in Hubei and thick-COT intervals crossing zero are retained.
- `forecast_events.pdf/png/svg`: 180 × 190 mm. The existing Mizhuang improvement and Sili degradation cases use the same three preselected target times. Observed AGRI and CPP retain 16 × 16 context; Hubei predicted COT has 16 × 16 native pixels, while Hunan displays the actually archived 3 × 3 crop. The latter is enlarged for visibility without interpolation or invented spatial detail. GHI y-scales differ (0–160 and 0–800 W m⁻²), explicitly noted in the figure. COT curves share 0–70; all COT images share 0–80, with an upper-extension marker. Native gaps remain gaps.

Use `C:/Python314/python.exe -X utf8 make_figures.py` to regenerate from the preserved local source files. PNGs are opaque RGB at 400 dpi; PDFs use embedded TrueType fonts and SVG text remains editable. These are general manuscript layouts, not a claim of compliance with an unspecified journal. Parent paper width is 177 mm, giving a 0.9833 scale factor.

## Suggested English captions

**Forecast regimes.** Forecast GHI errors conditional on quality-valid station-pixel CPP COT. Panels (a,c) compare models without COT and with predicted COT, using equal-station and equal-seed RMSE; (b,d) show their paired difference (with minus without COT), with 95% intervals from 5,000 target-date-block bootstrap draws. Negative differences favour COT. Counts denote distinct station–target times and target dates, not the number of overlapping forecasts. The shaded row denotes thick reference COT. CPP supplies an offline retrieval reference; the common forecast records and 15-minute GHI labels are unchanged. Both thick-cloud intervals include zero, while the moderate-COT Hubei subgroup deteriorates.

**Forecast events.** Actual cloud and irradiance evolution for (a) the improving Mizhuang reference segment on 26 September 2025 and (b) the degrading Sili segment on 15 June 2025. Left: observed AGRI composites, CPP reference COT and forecast-image-derived COT at three selected target times; right: station-pixel COT and 15-minute GHI trajectories from one initialization. Crosses locate the station pixel, and dashed boxes locate its 3 × 3 irradiance-model input. Hunan forecast COT shows only the archived 3 × 3 crop, enlarged without interpolation; other maps have 16 × 16 native pixels. RGB uses identical C03/C02/C01 square-root reflectance mapping, and COT maps share a 0–80 scale. Grey CPP fields and broken curves indicate missing data. Shading marks the selected consecutive thick-reference endpoints; it does not imply that the whole 15-minute interval was optically thick. GHI curves and displayed RMSE use seed 42, with different GHI y-scales across cases. The selected cases are conditional examples, not estimates of improvement frequency. Future observations are used only for diagnosis.

## 对应中文图注

**预报分层。** 按质量有效的站点像元 CPP 云光学厚度对预报 GHI 误差进行条件分层。(a,c)为不加 COT 与加入预测 COT 的 RMSE，先计算各站、各随机种子的分数再等权平均；(b,d)为两者的配对差值（加 COT 减无 COT），误差线为 5,000 次目标日期块重抽样的 95% 区间。负值表示加入 COT 后误差降低。计数为不同站点—目标时次及目标日期数，并非重叠起报条数。浅色行标记厚云参考分层。CPP 只作为离线检索参考，共同预报样本与 15 分钟 GHI 标签不变。两省厚云差值区间均包含零，湖北中等厚度组出现退化。

**预报个例。** (a)2025年9月26日米庄改善参考段与(b)2025年6月15日四里退化参考段中的真实云场及辐照度变化。左侧为三个选定目标时刻的真实 AGRI 合成图、CPP 参考 COT 和预测图反演 COT；右侧为同一起报的站点像元 COT 和 15 分钟 GHI 时序。十字标记站点像元，虚框标记其 3×3 辐照度模型输入。湖南预测 COT 仅显示实际保存的 3×3 裁块，放大展示但不插值；其他图保留 16×16 原生像元。RGB 统一采用 C03/C02/C01 反射率平方根映射，COT 图统一使用 0–80 色标。灰色 CPP 场和曲线断点表示缺测。底色标记所选连续厚云参考端点，不代表整个 15 分钟区间始终厚云。GHI 曲线和所示 RMSE 使用 seed 42，两例 GHI 纵轴尺度不同。个例按改善或退化条件选取，不用于估计改善频率。未来观测只作离线诊断。

## Visual and source review

- Both outputs were opened and inspected after export; first-draft label collisions were corrected.
- Palette is consistent with the existing paper figures: no COT `#2878A5`, forecast COT `#D87836`, observed-image R `#8064A2`, reference/observation dark.
- Method identity is redundant in shape and line style, with a labelled zero reference for paired effects.
- No cloud image was generated, smoothed, selectively stretched or upsampled to imply additional detail.
- Source paths and SHA-256 values are recorded in `FIGURE_MANIFEST.json`. Full case extracts remain in the existing analysis directory; Git figures need not carry those raw arrays.
- The workflow used the scientific-visualization skill. Its current reference was checked at https://arxiv.org/abs/2609.00065: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. https://doi.org/10.48550/arXiv.2609.00065. This is a software/workflow citation, not evidence for a COT mechanism.
