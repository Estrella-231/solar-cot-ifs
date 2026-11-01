# 主文真实数据图：2026-11-02专业排版

本目录为当前main.tex/main_zh.md使用的6张真实数据图。图2、图3的方法示意图继续使用相邻journal_style_20261102目录。本次仅调整视觉编码与版式，不改变实验、评价样本、数值、区间或结论。

| 主图 | 文件 | 表达方式 | 复现脚本 |
|---|---|---|---|
| 1 | study_regions_coverage | 真实省界定位、湖北站点放大、完整有效日期条 | build_coverage.py |
| 4 | cot_quality | 同一有效像元计数图、离散厚度类偏差点 | build_quality_forecast.py |
| 5 | ghi_gain | 逐站配对点、总体独立分区、配对差95%区间 | build_estimation.py |
| 6 | ghi_thickness | 离散厚度类配对点、降幅95%区间及样本量 | build_estimation.py |
| 7 | current_observed_events | 每行一个日期，左侧真实云图，右侧COT/GHI曲线 | build_events.py |
| 8 | regional_forecast_lead | 两省逐时效曲线、配对差及完整日期块区间 | build_quality_forecast.py |

所有主图导出PDF与高清PNG；除图1外同时保留SVG。字体为Times New Roman，数学字体为STIX；白底、细轴、淡灰网格，蓝色表示不加COT，橙色表示加COT，紫色表示COT或配对差。空间快照保留原生16×16像元与既定变换，不做锐化或生成式补绘。

图7另存共同GHI纵轴的敏感性展示current_observed_events_common_ghi_axis，主图沿用原两日期0–150和0–500 W/m²的尺度，双语图注明示。案例、缺测间断及CPP点均未重新选择。

## 数据来源

- 图1：data_coverage_20261102/usable_dates.csv、date_counts.csv及既有站点坐标。新增省界为[Natural Earth Admin-1](https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-admin-1-states-provinces/)，本地源、地址及哈希见coverage_checks.json。
- 图4–6：refine-logs/hubei_same_time_geometry_20261102中的INDEPENDENT_FP64_RESULTS.json、COT_QUALITY_AND_GHI_STRATA.json及图4对应已保存有效像元数组，具体路径见脚本。
- 图7：current_events_20261102/event_trajectories.csv、selected_snapshot_fields.npz及EVENT_SELECTION_AND_PROVENANCE.json。
- 图8：regional_forecast_20261102/lead_metrics.csv；完整复用32条区域—时效行和原有区间，不重新重采样。

数据核验分别记录在coverage_checks.json、estimation_checks.json、events_checks.json、quality_forecast_checks.json。论文整合、编译和逐页检查见../../revision_records/20261102_data_publication/README.md。
