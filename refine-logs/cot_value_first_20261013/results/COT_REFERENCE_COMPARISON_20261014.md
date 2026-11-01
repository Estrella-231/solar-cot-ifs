# COT预测评价与时空匹配文献对照

日期：2026-10-14（BJT）  
用途：工程评价记录，供后续方法章节取舍；不是论文结论，也不把跨论文误差作排名。

## 对照矩阵

| 研究 | 实际任务与参考 | 空间/时间配对 | 评价内容 | 可借鉴到本项目 | 不应照搬/不可直接比较 |
|---|---|---|---|---|---|
| Yi et al. (2025), GOES-16云光学厚度与地表太阳辐照度临近预报 | COT图像未来预测，并另行评价由COT得到的太阳辐照度；预测COT与对应时效卫星观测COT对比 | 按预测有效时刻逐lead对齐；论文按区域场和局地站点范围展示/评价 | COT用RMSE、MAE、MBE、SSIM、FS；同时单独报告SSI预报 | 采用逐lead、多尺度、与持续性/基线比较；COT质量和辐照度应用收益分开报告 | 不把GOES-16的参考产品、区域、3小时预报范围、训练/验证条件或分数移植到FY-4B湖南；其“观测COT”与本项目CPP参考不是同一个参考体系 |
| Sun et al. (2025), FY-4A/AGRI COT/CER检索及垂直结构影响 | 这是卫星COT/CER检索验证和敏感性研究，不是未来预报 | MODIS 1 km数据聚合至AGRI 4 km足迹；MODIS过境与最近AGRI全盘扫描配对，时间偏差≤±7.5 min | MODIS对照散点与分布；报告R²、MAE、RMSE并讨论空间尺度/部分像元填充 | 把网格/足迹和扫描时差写明；跨传感器比较先做空间支撑匹配，并显式报告时间偏差 | ±7.5 min是MODIS跨传感器容差，不能用来放宽本项目同源AGRI–CPP的目标扫描配对；它不提供COT forecast skill基准，也不能与本项目forecast RMSE排序 |
| 本项目：湖南冻结SimVP→新月度R | S→R未来COT与同目标时刻CPP参考比较；持续性R为业务基线；真实未来AGRI→R只作离线检索参考 | +15…+240 min逐lead；AGRI与CPP同名义扫描目标键，不跨扫描插值；站点中心、3×3和16×16分尺度；只在共同有效像元评分 | FP64 RMSE/MAE/bias/FS、固定COT≥30混淆、训练尺度log1p-COT SSIM、日期块描述区间 | 同时保留绝对误差、基线skill、厚云和结构指标；报告支持数与独立日期；把参考缺测保留为缺测 | CPP是同源卫星检索参考而非地面真值；当前仅8个独立验证目标日期，指标是有限验证诊断，不能宣称稳定/显著泛化或GHI预测收益 |

## 当前项目证据摘要

- 主3×3点估计：逐lead先按站点等权、再对16个lead等权，S→R RMSE=25.538，Persistence-R=28.252，差=-2.714 COT；8日日期块95%描述区间[-6.476, 0.493]包含0。
- 厚云COT≥30：S→R CSI=0.082，持续性=0.189；总RMSE较低不能替代厚云识别评价。
- log1p-COT SSIM：train CPP有效像元确定q99.9=74.0397，按11×11完整有效窗口评分，S→R=0.2356，持续性=0.2468；S→R未优于持续性。有效中心77,400个，占全部名义候选中心24.8%。
- 2025-08-21单例图由机械规则选出（最早具有两站、全部16 lead的3×3 CPP支持的BJT目标日，序列键再字典序打破平局）。单例仅用于查看曲线形态；CPP不是独立真值，未来AGRI→R不是可部署预测。
- 因只有8个独立目标日且空间/参考体系不等同于文献，当前不报告跨论文RMSE排序；后续若写论文，优先表述评价设计差异与对齐原则。

## 产物

- 单例曲线：`cot_forecast_case_20261014_v1/cot_forecast_case_2025-08-21.png`（另有PDF/SVG）及其CSV、选择收据。
- 汇总结果：`cot_forecast_eval_20261014_v3/COT_FORECAST_VALIDATION_RESULTS_20261014.md`。
- SSIM细节：`cot_forecast_eval_20261014_v5/cot_ssim_metrics.json`、`cot_ssim_by_window.csv`。
- 数据/聚合口径及跨方法配对：见上述主结果和 `FORECAST_COT_VAL_BANK_RESULTS_20261014.md`；后者是跨lead池化的补充统计，不替代逐lead等权主终点。

## 参考文献

1. Yi, Y. et al. (2025). *Nowcasting of Surface Solar Irradiance Based on Cloud Optical Thickness from GOES-16*. Remote Sensing, 17(16), 2861. https://doi.org/10.3390/rs17162861
2. Sun, J. et al. (2025). *Impact of cloud vertical structure perturbations on the retrieval of cloud optical thickness and effective radius from FY4A/AGRI*. Atmospheric Chemistry and Physics, 25, 16347–16361. https://doi.org/10.5194/acp-25-16347-2025

## 图表工作流说明

此工程诊断图的可视化流程参考了 Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, arXiv:2609.00065, https://doi.org/10.48550/arXiv.2609.00065。该条目仅说明图表设计/审阅工作流来源，不是本研究方法或科学证据。
