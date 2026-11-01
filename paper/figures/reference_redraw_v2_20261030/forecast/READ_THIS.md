# 先看原图，再重排：预报结果图 V2

直接参照：Papachristopoulou et al., *Effects of clouds and aerosols on downwelling surface solar irradiance nowcasting and short-term forecasting*, AMT 17, 1851–1877 (2024), **Fig.11，PDF第17页／印刷1867页**。[原文](https://amt.copernicus.org/articles/17/1851/2024/)。截图沿用原作者CC BY 4.0图像，只做PDF裁切与对照页显示缩放，未重绘其数据。

先看 `reference_old_new_comparison.png`：上为原论文截图，中为旧版，下为新绘图。新图为 `forecast_ghi_v2.png/pdf/svg`。

## 具体照什么改

1. **原图2×2紧凑排版**：从177×132mm变为177×112mm，减少高图例、大标题与外部站名重复所占空间。面板边界、宽度与行间距统一。
2. **原图细边框与浅网格**：统一四边细框，绝对误差和逐时效差值使用同一15–240分钟横轴网格。方法线贯穿使用同色和不同点型。
3. **原图轴名短、解释留图注**：删去轴名中重复的“from last observed frame”，改为Forecast horizon；长比较含义由面板名、线例、图注分工。
4. **原图共享底部图例**：四种输入共用一个底部图例，避免顶端挤占绘图区。历史/未来两种作用仍分别展示，右下共用左下站点顺序。
5. **保留本研究需要的科学信息**：作者四格为RMSE/rRMSE/R/FS；本研究不为相似外观临时计算冗余指标。保留绝对GHI误差、未来COT配对差以及两类逐站增量，完整95%区间不裁切，三seed点也保留。

## 双语图注

**English.** Paired GHI forecast evaluation on 15,275 common station–initialization–target records from 22 validation target dates. (a) Four model variants over +15–240 min from the last observed frame. (b) Forecast-COT correction minus the matched correction control, with pointwise 95% paired target-date bootstrap intervals. (c,d) Historical-COT-minus-no-COT and forecast-COT-minus-control differences for Niushou, Wangzhai, Mizhuang and their equal-station mean; both panels share this station order. Diamonds average three seed-specific scores; open circles show individual seeds; whiskers are 95% intervals from 5,000 paired date-block draws. Negative differences improve RMSE. Panel scales differ. The correction models share the frozen historical-COT base. Future observed AGRI/CPP is not an inference input. This is development validation, not an independent test.

**中文。** 22个验证目标日期、15,275条共同站点—初始化—目标时刻记录上的GHI预报评价。(a) 四种模型在相对历史末帧+15～240分钟的RMSE。(b) 未来COT修正减同容量控制修正，阴影为逐时效目标日期块配对95%区间。(c,d) 历史COT减无COT、未来COT修正减控制修正；两面板行序相同，分别为牛首、王寨、米庄和三站等权平均。菱形为三个种子分别评分后的均值，空心圆为单种子，横线为5,000次配对日期块bootstrap的95%区间；差值负向改善。面板坐标范围不同。后两模型共享冻结历史COT基底，推理不含未来真实AGRI或CPP。本图为开发验证，不是独立测试。

## 不变项与检查

直接读取V1已经复核的CSV，无训练、无重新选样、无重评分、无改变CI。复制的三份CSV与V1哈希相同。所有16个时效保留；177mm输出宽度下最小字体8pt。PNG、PDF实际渲染和原图—旧图—新图对照均已查看；检查过底部标签重叠并修复。
