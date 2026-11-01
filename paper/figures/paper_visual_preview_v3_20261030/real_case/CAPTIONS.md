# Figure captions and interpretation

## Main figure: `fig_cloud_source_mechanism`

**English.** Cloud imagery, retrieved optical thickness and irradiance response for Niushou on 6 August 2025, initialized at 12:00 BJT. (a) Columns show +15, +120 and +240 min. Rows show observed AGRI, observed-image R-COT, SimVP-predicted AGRI and forecast-image R-COT, respectively. Composites use C03/C02/C01 and the same square-root display mapping over reflectance 0–1. Native 16×16 pixels are retained; the cross and dashed box mark the station and its 3×3 GHI input. (b) Station-pixel COT. (c) Observed target-ending 15-min GHI, the AGRI-plus-historical-COT baseline and a fixed residual mapping supplied with either forecast-image or observed-image COT. Predicted curves average three frozen-baseline source variants for display; missing labels remain gaps. The event was selected by calendar eligibility without inspecting errors. Future observations are offline diagnostic inputs, and R-COT is a retrieval proxy; values above its training reference range of 100 are retained. This development-date out-of-fold event does not establish independent forecast benefit.

**中文。** 2025年8月6日牛首站云图、反演光学厚度与辐照度响应，初始化为北京时间12:00。(a) 三列依次为+15、+120和+240 min；四行依次为真实AGRI、真实图反演R-COT、SimVP预测AGRI及预测图反演R-COT。合成图使用C03/C02/C01，统一采用反射率0–1上的平方根显示变换。保留原生16×16像元；十字和虚线框分别标记站点及其3×3 GHI输入。(b) 站点像元COT。(c) 以目标时刻为右端点的15分钟平均GHI观测、AGRI加历史COT基线，以及同一固定残差映射分别输入预测图COT和真实图COT的输出。预测曲线为三份冻结基线来源的展示均值，缺失标签保留断线。事件按日历有效性选择，未参考误差。未来观测只作离线诊断，R-COT是反演代理；超过训练参考范围100的输出予以保留。该开发日期折外案例不构成独立预报收益证据。

## Supplement: `fig_three_station_source_response`

**English.** Station-pixel COT (a–c) and GHI (d–f) at Niushou, Wangzhai and Mizhuang in the same calendar-selected event. Purple dashed curves use observed future AGRI-derived COT offline; orange solid curves use SimVP-derived COT. The same reference-trained residual mapping is used for the two GHI source evaluations. Gray is the existing AGRI-plus-historical-COT baseline, not a no-COT control. The stations have 13, 14 and 11 scored GHI intervals, respectively; missing intervals remain gaps. Within-row scales are shared. All source, aggregation and interpretation conventions follow the main event figure.

**中文。** 同一日历规则事件中牛首、王寨、米庄的站点像元COT（a–c）及GHI（d–f）。紫色虚线使用实际未来AGRI反演的COT，仅作离线诊断；橙色实线使用SimVP图像反演COT。两种GHI来源评估使用同一真实图COT训练的残差映射。灰色为已有AGRI加历史COT基线，并非无COT对照。三站分别有13、14、11个可评分GHI区间，缺项保持断线；每排共享坐标范围。来源、汇总与解释约定同主事件图。

## Supported interpretation

The observed-image retrieval increases strongly while the forecast-image retrieval remains low. Under the fixed reference-trained mapping, the observed-image source produces a lower GHI trajectory, closer to this event's observed decrease, while a substantial high bias remains. This visual evidence is consistent with loss of irradiance-relevant information in the forecast source. It cannot determine whether the unique cause is displacement, thickness evolution, retrieval behavior, timing, or cloud shadows. It also does not establish the frequency of this event or the population benefit of the mapping.

## Alt text

Three time columns and four source rows compare native observed and forecast satellite composites and their COT retrievals. At the middle and final times, observed imagery yields a high-COT region near the marked Niushou station, while forecast imagery yields a much lower COT field. Two aligned time-series panels show increasing observed-image COT alongside falling observed GHI. Forecast-COT input misses most of that GHI decrease; observed-image COT input improves the response in this case but remains biased high.

## Verification

- The main PNG and a raster render of the exported PDF were visually inspected; source-group labels, station boxes, full colorbar, axes and legends remain visible.
- The three-station supplement PNG was inspected. Missing intervals are not connected and isolated valid points remain visible.
- RGB source values outside the shared 0–1 range: 0% for both sources. No spatial smoothing, image-specific enhancement, interpolation or new prediction was applied.
- `plotted_timeseries.csv` contains 48 station-time rows. `image_source_arrays.npz` is an exact copy of the saved native source arrays. Original prediction files and the source reader are hashed in `PROVENANCE.json`.
- Dimensions: main 177×175 mm; supplement 177×116 mm. These are provisional manuscript sizes, not a claim of compliance with an unselected journal.
