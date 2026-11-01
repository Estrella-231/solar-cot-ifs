# 预报厚云分析的论文整合建议（尚未写入正文）

## 放置位置与图件

在预报总体结果后加入“Cloud-regime dependence and continuous events / 云况依赖与连续过程”。主文放一张分层统计图及一张紧凑的改善/退化事件对照；完整四例、逐站逐时效、覆盖与事件清单放补充材料。不要与同期35.51%厚云结果混用。正式插入时再统一图号、首次引用位置及中英章节，并编译检查实际版面。

候选主图：`figures/forecast_cpp_strata.pdf`。候选主文案例：湖北米庄改善、湖南四里退化；`figures/four_real_cloud_cases.pdf`四页完整场图作补充。后续紧凑主图可在已有真实数据上排版，不调用生成模型填云图或缺失场。湖南预测面板必须标注实际保存的3×3裁块，不能与16×16真实背景隐含视为同一视场。

## 中文结果段落候选

在质量有效的站点像元CPP参考上，按目标时刻光学厚度进行离线分层。厚云定义沿用COT≥30，地面标签仍为目标时刻结束的15分钟平均GHI。湖北厚云支持内的RMSE由72.479降至61.281 W/m²，点降幅15.45%；湖南由92.134降至89.464 W/m²，点降幅2.90%。相应日期块差值95%区间为[−25.927,+1.089]和[−5.821,+0.690] W/m²，均包含零。湖北厚云覆盖139个不同站点—目标时次、6个日期，湖南覆盖165个、25个日期。湖北中等厚度组则由86.965升至94.089 W/m²。结果表明COT相关修正具有云况依赖，不能从总体改善推导所有厚云和其他云况均受益。

按连续15分钟参考时刻归并，得到湖北75段、湖南137段站点厚云参考连续段；其中分别仅18段、7段含至少两个连续参考时刻。单时刻段主要限制过程分析，不能视为大量独立云团。每省各选日历最早的合格改善与退化段，展示真实AGRI、CPP、真实图和预测图反演COT及同一起报的GHI曲线。湖北米庄案例中，加入COT减轻了部分高估；湖南四里案例则显示局地增厚与后期光学厚度下降未被充分再现，两种GHI预测均未跟上观测恢复。这些是条件选例的解释性证据，未确定唯一误差来源。

## Matching English draft

Forecast errors were stratified by target-time COT at station pixels with quality-valid CPP retrievals. Thick conditions retained the existing threshold of COT ≥ 30, while GHI labels remained target-ending 15-minute means. Within this support, RMSE decreased from 72.479 to 61.281 W m⁻² in Hubei (15.45%) and from 92.134 to 89.464 W m⁻² in Hunan (2.90%). The corresponding 95% target-date-block intervals for the paired differences were [−25.927, +1.089] and [−5.821, +0.690] W m⁻²; both included zero. Thick-cloud support comprised 139 distinct station–target observations on six dates in Hubei and 165 observations on 25 dates in Hunan. In contrast, Hubei's moderate-COT subgroup increased from 86.965 to 94.089 W m⁻². Thus, the COT-associated correction varied with the reference cloud regime, and the overall improvement did not establish uniform benefit in thick clouds or other regimes.

Joining consecutive 15-minute thick-reference times yielded 75 station reference segments in Hubei and 137 in Hunan, of which only 18 and seven, respectively, contained at least two consecutive reference times. The many single-time segments limit process-level interpretation and do not represent independent cloud objects. The earliest eligible improving and degrading segment in each region was used to illustrate observed AGRI, CPP, observed- and forecast-image R-COT, and GHI trajectories from one initialization. The Mizhuang case showed a reduction in part of the GHI overestimation, whereas the Sili case showed incomplete representation of renewed local thickening and subsequent optical thinning; both irradiance forecasts missed much of the observed recovery. These conditionally selected examples provide explanatory evidence without identifying a unique error source.

## 图注必须说明

- CPP是卫星检索参考；未来真实AGRI/R仅用于离线图示。
- 分层日期块区间、独特目标/日期与重复起报的区别。
- 案例按结果方向分组后日历最早选取，非随机样本，不能估计改善频率。
- 案例曲线固定seed42；主表是三seed、站点等权，图中个例分数不是总体分数。
- 原生像元、统一RGB变换、统一COT色标、缺参考灰色与缺测断线。
