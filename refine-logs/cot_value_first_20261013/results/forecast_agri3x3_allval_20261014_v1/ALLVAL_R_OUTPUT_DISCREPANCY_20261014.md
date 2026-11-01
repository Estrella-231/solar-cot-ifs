# 全月度验证窗：SimVP→R 与同期未来 AGRI→R 输出差异

T36已覆盖全部3,140个新月度validation序列。以下由保存的物理COT输出作FP64汇总，按两站×16个lead等权平均；季节按目标时刻从UTC转换到BJT后确定。完整站点、季节、lead、1×1/3×3/16×16表见 `allval_R_output_discrepancy_by_bjt_season_fp64.csv`。

| BJT季节 | 3×3输出差RMSE | MAE | 平均偏差（forecast R−target R） |
|---|---:|---:|---:|
| DJF | 17.29 | 9.17 | −2.30 |
| MAM | 15.23 | 7.91 | −1.07 |
| JJA | 13.52 | 7.12 | −1.55 |
| SON | 19.86 | 7.66 | −3.62 |

3×3 RMSE按站点/季节（COT物理单位）：

| 站点 | DJF | MAM | JJA | SON |
|---|---:|---:|---:|---:|
| 四里 | 17.42 | 17.22 | 14.80 | 19.72 |
| 竺家 | 17.17 | 13.24 | 12.25 | 20.01 |

两站×四季等权的3×3输出差RMSE由+15 min的23.29降至+120 min的12.48，再升至+240 min的20.34。这个非单调曲线是两个输入场经同一R后的预测差异，不能解释为外推物理COT误差随时效先降后升；滑动验证窗口彼此重叠，季节各仅来自两个指定验证日期块。

**解释边界：** `target R`表示同一冻结反演器作用于真实未来AGRI的输出，不是独立COT真值。因此上表用于定位SimVP图像误差如何改变R输出，并显示这种响应随季节/lead变化；不能作为COT forecast accuracy。物理COT与CPP的比较只能在CPP原有稀疏有效掩膜上单独解释。稀疏支持上的厚云及到达诊断见原B3报告，样本仅8个BJT日期且重叠。

NPZ、本地profile与独立审计均绑定到同一bank SHA；test未读取。

### 时间分组复核

旧 `allval_R_output_discrepancy_by_season_lead_fp64.csv` 中的season实际由UTC目标月份计算，之前误标为BJT季节。独立NumPy复算后，384个站点×季节×lead×尺度组与旧CSV逐项一致（RMSE最大差6.4e−14），确认差异来自季节时区分组而非bank数值。上表及新CSV改按BJT目标月份重新分组；旧CSV保留作历史复算收据，不用于季节结论。全时段结果和CPP参考评分不受此季节标签修正影响。复算脚本与收据为 `recompute_r_output_discrepancy.py`、`independent_R_output_discrepancy_recompute.json`。
