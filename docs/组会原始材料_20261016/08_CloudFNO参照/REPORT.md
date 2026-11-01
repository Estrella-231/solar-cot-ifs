# 历史 COT 外推与 Cloud-FNO：1–4小时描述性对照

本次只提取已保存的旧验证指标，没有训练、没有读取新 test 载荷。
Cloud-FNO 数字由用户提供论文第7页 Figure 2(s–v) 标注逐项核对；对应各小时，而非0到该时效累计平均，也非MODIS对照表。
本项目采用20260928 v5 full_history运行：512个验证窗口、72个BJT初始化日；原分析同时包含64×64逐像元和站点(32,32)评价，本图只展示站点像元。评价为物理COT误差，非log域误差，非先对图取空间平均；CPP有效与输运可达范围取交集。
历史COT输运：历史C13估运动，输运历史末帧R-COT；8帧混合修正还使用未来SimVP→R COT等输入，不能称纯历史COT输入模型。
冻结厚云准则最终选中step0输运。update96是按原记录保留的最佳学习候选（未选中）；四个时效统一用同一候选，不逐时效挑模型。

## 全云况物理 COT RMSE

|范围/方法|+1 h|+2 h|+3 h|+4 h|
|---|---:|---:|---:|---:|
|paper_image / Cloud-FNO|6.750|8.090|8.930|9.610|
|patch / Persistence|7.663|9.969|11.175|11.987|
|patch / Transport (selected)|7.419|9.799|11.179|11.988|
|patch / 8-frame hybrid (candidate)|8.428|9.869|10.969|11.019|
|station / Persistence|6.633|8.515|10.242|10.477|
|station / Transport (selected)|6.151|8.909|10.646|11.133|
|station / 8-frame hybrid (candidate)|7.155|9.037|9.622|9.497|

## 解读与边界

在本项目相同验证样本上，混合修正的1小时空间误差高于输运，4小时空间/站点误差低于输运。全天候平均误差改善不等同厚云改善，故不能忽略原选模结果。
Cloud-FNO为另一地区、另一产品与分辨率的2017年测试评价；本项目为四里旧划分验证。图分面且共享纵轴，仅展示数值量级及随时效变化，不能计算成相对Cloud-FNO的模型提升或将旧结果冒充新月度结果。
本项目CPP是检索参考，不是独立云物理真值；这也不是GHI精度。

## 复核与复现

逐时效数值从training_complete.json提取，与day_metrics.jsonl相应update的lead_metrics逐项一致；总时段RMSE另从日期级平方误差和/样本数用FP64复算通过。当前没有重新读取逐样本预测，不能把这一提取称为逐样本重新评估；逐时效CI未重算。
hourly_metrics.csv包含MAE、样本数、厚云RMSE及厚云数量。图为hourly_comparison.png / .pdf。
运行：python scripts/plot_history_cot_cloud_fno_20261015.py
论文 DOI：https://doi.org/10.1029/2025GL119553
