# COT外推到达行为：完整共同支持验证窗口

bank窗口数 271；同时覆盖两站全部16 lead且每站每lead至少有1个CPP共同有效3×3像元的窗口数 8；目标日期 20250821。

到达定义为每个窗口/站点中，固定3×3共同有效像元的物理COT均值首次达到COT≥30的离散lead。未插值；未达到记为240分钟内miss。仅在参考与方法都达到阈值时计算到达误差，miss另列。

| 站点 | 方法 | 参考厚云窗口 | 方法命中 | 240 min miss | 命中样本到达误差均值/中位数(min) | 峰值幅度误差均值(COT) |
|---|---|---:|---:|---:|---:|---:|
| Sili | S→R forecast | 8 | 1 | 7 | -105.0 / -105.0 | -30.35 |
| Sili | Persistence-R | 8 | 1 | 7 | -105.0 / -105.0 | -30.49 |
| Sili | Future observed AGRI→R (offline) | 8 | 1 | 7 | -120.0 / -120.0 | -25.08 |
| Zhujia | S→R forecast | 8 | 0 | 8 | — | -43.89 |
| Zhujia | Persistence-R | 8 | 0 | 8 | — | -43.39 |
| Zhujia | Future observed AGRI→R (offline) | 8 | 0 | 8 | — | -34.67 |

## 限制与解释

这些是CPP有支持的重叠窗口数，不是独立云事件数；不计算显著性区间。检测数只针对参考厚云窗口。方法未越过阈值的情况按miss记录，不能只在成功命中子集报告平均到达误差。离线真实未来AGRI→R用于辨别反演链和外推链差异，不能作为业务预报成绩。CPP不是独立真值。

本诊断把单例曲线扩展到完整共同支持窗口，但时间/空间支持仍窄。结合主指标（SimVP→R整体RMSE点估计略好于持续性、区间跨0；厚云CSI与SSIM较差），证据仍不支持外推头稳定有skill；不据此启动GHI forecast head训练。机器结果与逐窗明细分别见`arrival_cohort_diagnostic.json`及`arrival_cohort_by_window.csv`。test载荷读取为0。
