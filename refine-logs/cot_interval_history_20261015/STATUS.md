# 执行状态：2026-10-15

- 已完成源白天连续零值标记与原工作簿日间1378个单元格回查。源表显式0、无设备状态；未删正式标签或训练行。
- 已定位太阳几何均时差缺少rad2deg，171个真实窗口复现并通过独立SPA核查。只用于本轮新GHI几何；冻结R/S未改变。
- 未训练的两帧准备保留为data_attempt1。正式三帧T−30/T−15/T为data_attempt2：train47666、val4263。验证实际扫描支持全部核验；18条训练记录缺实际扫描元数据，按名义时间使用并明示。
- PBS214194.tc6000：node22单A800，batch1792，两站×两组×三seed共12臂完成，390/420更新，各30轮；CPU重载及两种独立FP64计分/日期bootstrap通过。例行监控由Luna执行。
- 主估计为两站、三seed的RMSE差均值：AC−A=−0.258 W/m²（0.24%点估计改善），95%目标日期簇区间[−2.149,+1.571]。没有稳定增益，不自动扩展或调参。
- 评分排除可疑零值后点差−0.505 W/m²，区间仍跨零；原训练保持完整。四里34条可疑零值约占无COT验证平方误差52%，不能据此直接判故障。
- 固定头重复当前COT的历史扰动表明历史被使用；保留改善与退化各站各一例的真实三帧AGRI/R-COT及GHI图。选例不影响主评分。
- [完整报告](../../results/cot_interval_history_20261015/REPORT.md)。预测、来源/权重哈希、源核查及代码均保存；test未读，论文未改。

工程根：`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/cot_interval_history_20261015_v1/`。正式输出`results_attempt1`，诊断`diagnostics_attempt1`。原工作簿核查在FD-201；来源目录见报告。PBS最终历史已被清理，未将Unknown Job Id当作失败，也未声称读到最终exit_status。
