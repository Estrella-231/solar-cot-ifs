# Real cloud event: figure captions and use boundary

## Figure A — cloud-source mechanism (main-text candidate)

**English.** Cloud-source differences and the associated irradiance response for an error-blind, calendar-selected event in Hubei, initialized at 12:00 BJT on 6 August 2025. (a) Native 16 × 16 AGRI composites and COT retrieved from observed and SimVP-predicted imagery at the first, central and last forecast times. The composites assign C03/C02/C01 to red/green/blue, using a common 0–1 reflectance range and square-root display transformation. The cross marks the Niushou station pixel; the dashed box encloses its 3 × 3 GHI input. COT has a common linear scale and is dimensionless. (b) COT at the station pixel through all 16 forecast times. (c) Observed right-ending 15-min GHI, the existing AGRI-plus-historical-COT baseline, and the same reference-trained residual mapping evaluated with forecast versus observed-image COT. Predicted curves are means over three frozen baseline-source variants for display, not ensemble scores; missing scored GHI intervals remain gaps. Observed future imagery and its R-derived COT are offline diagnostic inputs, and R-COT is not independent cloud truth. This training-date out-of-fold case illustrates a source mismatch, not its frequency or an independent forecast gain. The reference-derived COT occasionally exceeds the retrieval supervision range of 100; these outputs are retained rather than clipped.

**中文。** 湖北一个按日历规则选取、未参考误差选例的云事件，初始化时刻为2025年8月6日北京时间12:00。(a) 第一个、中间和最后一个预报时刻的原生16 × 16 AGRI合成图，以及真实图和SimVP预测图分别经反演器得到的COT。合成图将C03/C02/C01分别赋给红/绿/蓝通道，所有图使用共同的0–1反射率范围与平方根显示变换。十字为牛首站点像元，虚线框为进入GHI模型的3 × 3邻域。COT为无量纲量，并共用线性色标。(b) 全部16个预报时刻的站点像元COT。(c) 以目标时刻为右端点的15分钟平均GHI观测、已有AGRI与历史COT基线，以及同一个由真实图COT训练的残差映射在预测COT和真实图COT输入下的输出。预测曲线为三份冻结基线来源的均值，仅用于展示，不作为集成模型评分；缺少评分标签的区间保留断线。真实未来图及其反演COT仅作离线诊断，R-COT并非独立云真值。该训练日期内的折外案例说明输入来源差异，不能估计其发生频率或证明独立预报收益。少量反演COT超过监督参考范围100，图中保留这些输出而不截断。

## Figure B — all three station responses (supplement candidate)

**English.** COT-source and GHI trajectories for Niushou, Wangzhai and Mizhuang during the same calendar-selected event initialized at 12:00 BJT on 6 August 2025. (a–c) Station-pixel COT retrieved from observed future AGRI (purple dashed, offline) and SimVP-predicted AGRI (orange solid). (d–f) Observed 15-min GHI, the AGRI-plus-historical-COT baseline and the two source evaluations of the same reference-trained residual mapping. The axes are shared within each row. Niushou, Wangzhai and Mizhuang contain 13, 14 and 11 scored GHI intervals, respectively; missing intervals remain gaps. All conventions and the offline, training-date out-of-fold interpretation are the same as Figure A.

**中文。** 同一日历规则案例中牛首、王寨、米庄三站的COT来源与GHI轨迹；初始化时刻为2025年8月6日北京时间12:00。(a–c) 真实未来AGRI反演COT（紫色虚线，仅离线）与SimVP预测AGRI反演COT（橙色实线）的站点像元值。(d–f) 15分钟平均GHI观测、AGRI与历史COT基线，以及同一真实图COT训练残差映射在两种COT来源下的输出。每排共享坐标范围。牛首、王寨、米庄分别有13、14、11个可评分GHI区间，缺项保留断线。其他显示约定及离线、训练日期折外诊断的解释边界与图A相同。

## Integration note

These figures answer a forecast-source question, so they belong with forecast limitations or mechanism discussion, not among same-time estimation gain results. They must not silently replace the old synchronous Sili event while keeping its old caption. The gray baseline already contains eight historical R-COT frames; it is **not** a no-COT control. Orange and purple are the same fitted reference-trained residual mapping under two future COT input sources; they are **not** independently trained with-COT/no-COT heads. Their favorable separation in this one case does not establish general superiority. The response also remains biased high with observed COT, so the plots do not identify SimVP as the sole error source.

## What was changed and verified

- Reorganized observed imagery, forecast imagery, and their separate COT outputs into paired columns; spatial and temporal explanations are now connected in one figure.
- Removed long internal titles and repeated method legends; used 8–10 pt text at 177 mm width, native pixels, a shared COT scale, and direct local-input boxes.
- The old 0–0.7 RGB display clipped 35.46% of observed visible-channel values versus 0.21% of predicted values. The shared 0–1 display clips **0%** in either source for this saved case, preserving visible thick-cloud texture without changing data.
- Retained all 16 times in curves, including missing-label gaps and isolated valid intervals; added markers so an isolated prediction is not accidentally hidden by a line-only style.
- Kept the full COT plotting range, including values above100; no smoothing, selective enhancement, imputation or new inference.
- Source arrays and nine saved cross-source prediction files are hashed in `PROVENANCE.json`; plotted values are exported in `plotted_timeseries.csv`.
- Both PNGs were inspected at intended print dimensions and the first draft's clipped colorbar label and upper COT limit were corrected.

Rebuild: run `plot_real_case.py` with NumPy and Matplotlib. PDF and SVG preserve vector text/curves; image panels retain their actual raster resolution. The figures are not bound to a specific journal template.
