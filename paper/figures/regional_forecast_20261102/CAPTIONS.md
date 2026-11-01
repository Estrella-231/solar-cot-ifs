# 可插入主稿的双语图注

## English caption

Lead-dependent GHI forecast errors with independently trained regional DyDiff predictors and observed-image-trained irradiance models. Panels (a,b) show RMSE without COT and with predicted COT in Hubei and Hunan; panels (c,d) show the paired difference (with COT minus without COT), with pointwise 95% intervals from 5,000 paired target-date-block bootstrap draws. Negative differences favour COT. RMSE is computed separately for each station and random seed before equal-weight regional averaging. The evaluations contain 24,451 records across 17 target dates in Hubei and 66,086 records across 51 dates in Hunan; record counts vary by lead. Gray shading marks +15 and +30 min, whose three-frame inputs include known historical imagery; all three input frames are predicted from +45 min onward. Both methods include solar geometry, and the irradiance-model parameters remain unchanged during forecasting.

## 中文图注

两区域独立训练的DyDiff外推器接真实图像训练的辐照度模型时，GHI预报误差随时效的变化。(a,b)分别为湖北和湖南不加COT、加入预测COT的RMSE；(c,d)为配对差值（加COT减不加COT），阴影为5,000次目标日期块配对bootstrap得到的逐时效95%区间，负值表示COT改善误差。先分别计算每个站点、每个随机种子的RMSE，再在区域内等权平均。湖北包含17个目标日期的24,451条记录，湖南包含51个日期的66,086条记录，各时效记录数有所不同。灰色背景标示+15和+30分钟，其三帧输入仍包含已知历史图；+45分钟起三帧全部来自预测。两组均包含太阳几何，预报时辐照度模型参数保持不变。

## Optional result paragraph / 可选结果句

The point estimates favour COT at all 16 leads in both regions (Fig. X). In Hubei, the relative RMSE reduction is 6.07% at +60 min and 2.06% at +240 min, whereas the corresponding Hunan reductions are 2.09% and 1.19%. The date-block intervals indicate greater uncertainty at some leads, particularly at longer horizons in Hubei; negative point differences throughout the forecast period therefore do not establish an improvement at every lead individually.

两区域16个时效的点估计均显示加入COT后误差降低（图X）。湖北在+60和+240分钟的相对RMSE降幅分别为6.07%和2.06%，湖南对应为2.09%和1.19%。部分时效的日期块区间较宽，湖北长时效尤其如此；因此，全时效点估计为负并不等于每个时效的改善均已得到独立确认。
