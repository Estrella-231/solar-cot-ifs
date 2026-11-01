# 当前三帧同期GHI模型：真实云事件

## 建议正文（英文）

Figure~\ref{fig:events} illustrates contrasting responses to the COT representation at Niushou. On 26 September 2025, the mean of the three seed-specific RMSEs decreases from 27.8 to 19.3 W\,m$^{-2}$. The COT-assisted model raises several underestimated late-morning values and lowers overestimated early-afternoon values, rather than applying a uniform reduction to irradiance. On 27 September, RMSE increases from 92.7 to 100.9 W\,m$^{-2}$. Both models miss the rapid rise in observed irradiance near 14:00, while the COT-assisted estimate is lower. Retrieved COT also exceeds the CPP reference at most available times in this second case. These processes illustrate useful and unsuccessful corrections by the current model; they do not isolate cloud displacement, retrieval error, or temporal representativeness as a unique cause.

## 建议正文（中文）

图6展示牛首站引入COT后两种不同的响应。2025年9月26日，三个随机种子各自计算后平均的RMSE由27.8降至19.3 W/m²。加入COT后，模型上调上午部分偏低估计、下调午后部分偏高估计，并非始终降低辐照度。9月27日，RMSE则由92.7升至100.9 W/m²。两个模型均未捕捉14:00附近实测辐照度的快速上升，加COT模型的估计更低；该日大部分可用时刻的反演COT也高于CPP参考。两例说明当前模型既可能作出有益修正，也可能修正失败，但不能据此将误差唯一归因于云位置、反演偏差或时间代表性。

## 选例方法插入语（英文）

For the illustrative Niushou events, we use a fixed 10:30--14:30 BJT window and require at least 12 of its 17 target intervals, a retrieved station-pixel COT maximum of at least 30, and a COT range of at least 5. We then select the earliest eligible date in each of the improving and non-improving RMSE groups. This direction-stratified selection provides contrasting examples, not an estimate of the frequency of benefit; the adjacent dates are not independent weather-event replications. The retrieved-COT selection threshold is distinct from the CPP-based thickness strata.

## 选例方法插入语（中文）

牛首站示例采用固定北京时间10:30—14:30窗口，要求17个目标区间中至少12个有效，站点像元反演COT最大值不低于30且变化范围不小于5。随后分别从RMSE改善和未改善组中选取最早的合格日期。这种按误差变化方向分组的选例用于展示对比过程，不能估计收益出现频率；相邻两日也不视为独立天气事件的重复验证。这里使用反演COT筛选，区别于采用CPP参考划分的厚度子集。

## 图注（英文）

Observed cloud fields and paired same-time GHI estimates at Niushou on (a) 26 and (b) 27 September 2025. Each column contains AGRI composites and COT fields at 10:30, 12:30 and 14:30 BJT, followed by station-pixel COT and 15-minute mean GHI trajectories. Composites use C03/C02/C01 with the same square-root mapping of reflectance 0--1. Native $16\times16$ pixels are retained; the cross and dashed box mark the station and its $3\times3$ input. Each displayed image is the final frame of an observed three-frame model input. COT has a common scale; GHI axes span 0--150 and 0--500 W\,m$^{-2}$. Sparse CPP retrieval references are shown as unconnected crosses. Model curves average three seeds for display; annotated RMSEs average the seed-specific scores. Missing GHI intervals remain gaps. Cases are the earliest eligible dates from the improving and non-improving groups under the stated completeness and retrieved-COT criteria.

## 图注（中文）

牛首站2025年(a)9月26日和(b)9月27日的真实云场与有无COT的同期GHI估计。每组依次展示北京时间10:30、12:30和14:30的AGRI合成图与COT场，下方为站点像元COT及15分钟平均GHI轨迹。合成图使用C03/C02/C01，统一采用0—1反射率的平方根映射；保留原生16×16像元，十字和虚线框分别标记站点及其3×3输入范围。展示的每张图均为真实三帧模型输入的末帧。COT采用统一尺度，GHI纵轴分别为0—150和0—500 W/m²。稀疏CPP检索参考以不连线的叉号标记。模型曲线为三个种子的展示均值，标注RMSE为各种子单独计算后平均；缺失GHI区间保持断线。两例分别取既定完整性和反演COT条件下改善组与未改善组最早的合格日期。

## 补充观察（不必全部写进正文）

- 9月26日：15个共同GHI区间，12个CPP站点参考配对；当前R相对CPP的站点COT平均偏差为−14.617、RMSE为15.997。较厚过程仍可获得GHI改善，不要求COT先成为无偏真值。
- 9月27日：16个共同GHI区间，17个CPP站点参考配对；当前R的COT平均偏差为+6.850、RMSE为8.012。该事件COT RMSE比前一日低，并不意味着下游GHI收益更大；这是两个不同场景的例证，不能据此做一般因果断言。
- 14:00目标区间实测GHI为484.2 W/m²，无COT/有COT三种子展示均值为164.009/142.532 W/m²。两者均低估这一快速上升。AGRI为卫星采样，GHI为目标15分钟均值，不能直接把峰值差全部归因于同刻云厚度。
- 图中RGB均为实际观测合成，不是AI生成；COT为当前冻结反演器输出，不是CPP真值。
