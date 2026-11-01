# 新月度 R 云识别与公开基准的口径对照

日期：2026-10-13（BJT）；仅验证集，未读取test。模型为月度新split seed42 R。参考 CLP 从对应 CPP NetCDF 按冻结站点网格裁块；1,398个验证样本、356,169个共同有效像元全部获得 CLP（覆盖率100%），网格坐标最大误差0°。

## 云/晴空识别

以预测物理COT≥0.1判有云，以同像元 CPP CLP∈{1,2}判有云、CLP=0判晴空。对有效掩膜的完全相同像元计算 POD、FAR、HR和KSS。

| 来源/样本 | 云POD | 晴空POD | 云FAR | HR | KSS |
|---|---:|---:|---:|---:|---:|
| 本项目：R的COT阈值 vs CPP CLP，验证站点像元，N=356,169 | 0.925 | 0.740 | 0.123 | 0.863 | 0.665 |
| FY-4B业务CLM质检：FY-4B CLM vs MYD35，全圆盘2024-12，N=92,528,101 | 0.922 | 0.816 | 0.045 | 0.901 | 0.737 |

两个云POD接近，但这是不同参考和样本条件下的背景对照，不是头对头排名。我们的COT阈值有较高云检出率，但晴空POD较低、云FAR较高；这与模型COT偏向把一部分参考晴空像元预测成弱云的情况一致。站点分项：四里云POD/晴空POD/FAR/HR/KSS=`0.979/0.685/0.103/0.902/0.664`；竺家=`0.857/0.777/0.149/0.825/0.634`。

参考资料的FY-4B CLM质检采用2024年12月全圆盘数据，与1 km MYD35做时间±5分钟、最近邻空间匹配；报告总体云POD 0.922、晴空POD 0.816、云FAR 0.045、HR 0.901、KSS 0.737。本项目目标则是由COT连续回归器阈值化得到的云判别，参考为同一CPP检索流程输出的CLP，空间限于两站、CPP有效且日间的16×16像元。参考和覆盖差异意味着不能从相近的云POD推导模型达到业务CLM水平，也不能把FAR差异完全归因于模型。

## COT误差及厚云

本项目同一验证集上COT RMSE/MAE为9.074/4.646（CPP参考）；厚云参考COT≥30的RMSE为14.129。FY-4A/AGRI一项全时段检索研究以官方MODIS云产品为基准，报告COT RMSE 12.79；FY-4B夜间COT方法以CALIPSO为参考，在不同太阳天顶角样本报告RMSE 1.26和1.99。两者的卫星/算法、参考、昼夜条件、像元匹配和COT样本分布均不同，只能列作量级背景，不能比较优劣。公开数值也没有给出与本项目一致的COT≥30 POD/FAR/CSI，因此厚云识别CSI=0.504目前是项目内结果，暂无同阈值公开数字可直接对照。

## 结论与后续可比性

当前已完成的是“连续COT预测阈值化结果 vs CPP CLP”的严格同样本验证，以及与公开FY-4B CLM全盘质检数字的口径说明。CPP CLP和COT来自共同CPP产品链，不能视作独立真值；本项目的对照也不是使用FY-4B官方CLM产品。若需要真正同条件比较业务产品，应再获取对应目标时刻的FY-4B CLM文件，并按相同空间点、太阳日间掩膜和时空匹配规则计算同一组指标。

参考：

1. 国家卫星气象中心，FY-4B AGRI L2云检测产品（CLM）2024年12月质检报告：[PDF](https://img.nsmc.org.cn/QAS/FY4B/CLM/FY4B_AGRIX_CLMQ_L2_20241231_MONH_REPOT_V0.pdf)。
2. Zhao et al., *Image-based retrieval of all-day cloud physical parameters for FY4A/AGRI and its application over the Tibetan Plateau*：[arXiv](https://arxiv.org/abs/2405.19336)。
3. *Retrieval of Cloud Optical Thickness During Nighttime from FY-4B AGRI Using a Convolutional Neural Network*：[Remote Sensing](https://www.mdpi.com/2072-4292/17/5/737)。

机器可读结果：`cot_cloud_vs_cpp_clp_monthly_R_seed42.json`；逐像元固定COT阈值结果：`cot_cloud_detection_monthly_R_seed42_validation.json`；CLP提取覆盖/网格审计：`clp_reference_monthly_R_seed42.audit.json`。两步的复算脚本分别为 `extract_validation_clp.py`、`score_cot_cloud_vs_cpp_clp.py`。
