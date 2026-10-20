# 湖北修正COT：真实云图、轨迹与冻结GHI头归因

范围：407个验证序列，15,275个站点—初始化—目标时刻记录，22个目标日期，三站三seed。实际BJT时间修复后的复用验证。真实未来输入替换仅为离线诊断，不是预报成绩，也不是独立test。

## 完整样本上的固定权重输入替换

| 输入/处理 | 等站等seed RMSE (W/m²) |
|---|---:|
| FA_pred | 102.606987 |
| FAC_pred | 104.294007 |
| FAC_observedCOT | 85.981141 |
| FAC_zeroCOTslot | 175.563688 |
| FAC_permutedCOT | 246.368540 |
| FA_observedAGRI | 79.746138 |
| FAC_observedAGRI_predCOT | 90.561372 |
| FAC_observedAGRI_observedCOT | 80.143786 |

逐站、逐seed原始表见runtime_attempt3/fixed_head_diagnosis/head_interventions.csv。差值区间见runtime_attempt3/fixed_head_diagnosis/summary.json；本地独立复算见runtime_attempt3/fixed_head_diagnosis/local_independent_analysis.json。

## 真实图和曲线

每站两例：最大COT误差云事件，以及另一日期的中位误差活跃云事件。选择只看COT轨迹，不按GHI增益筛选。案例曲线为三seed预测均值，仅作诊断，不替代逐seed等权主指标；无效GHI标签断线，CPP有效的离散时刻用菱形标记，夜间图像留灰。所有帧使用相同反射率拉伸；青框为3×3输入，白星为站点。COT采用log1p色距，地图中COT≥100饱和为顶色；曲线保留超过100的实际值。云图是目标时刻快照，GHI是该时刻结束的15分钟区间均值。

- Niushou 202509261000，largest_COT_error；站点COT RMSE=29.786，真实图R峰值=88.622，预测图R峰值=47.808。
- Niushou 202510051400，median_active_event；站点COT RMSE=6.008，真实图R峰值=27.293，预测图R峰值=19.620。
- Wangzhai 202509261000，largest_COT_error；站点COT RMSE=30.333，真实图R峰值=87.932，预测图R峰值=48.751。
- Wangzhai 202511061100，median_active_event；站点COT RMSE=6.973，真实图R峰值=55.242，预测图R峰值=49.422。
- Mizhuang 202509261000，largest_COT_error；站点COT RMSE=35.749，真实图R峰值=105.241，预测图R峰值=52.200。
- Mizhuang 202512021115，median_active_event；站点COT RMSE=7.042，真实图R峰值=18.576，预测图R峰值=8.084。

## 时间修复的剩余限制

本轮只修复8小时标签时钟绑定；冻结S/R太阳几何仍保留源均时差漏乘180/pi的公式误差，天顶角最大偏差3.236度，不能称几何全链已修好。详见SOLAR_FORMULA_AUDIT.json和FINDINGS.md。

## 解释边界

真实图反演COT与预测图反演COT来自同一个修正R，可测量外推导致的表示差异，但不是独立物理真值；有CPP有效值的子集另列比较，不填补参考缺口。替换真实COT与预测AGRI可能构成不一致的训练外组合，替换不改善本身不能证明头部无能力；替换改善只能说明该冻结头在这个诊断条件下有响应。训练均值归一化槽置零及跨日期同站同lead置换，测量的是同一个头的输入依赖，与两个独立训练头之差不同。峰值时间偏移只能称站点COT信号偏移，不能把局部峰值跳动当作已追踪同一云团。
