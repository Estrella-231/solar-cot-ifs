# 10.54%结果的CPP时间、太阳几何与来源判定

2026-11-01。只读核对既有train/validation资产，未读取test/buffer、未训练或重新选模。

## 确定结论

77.285→69.142 W/m²（10.536%）是可复现的**正确UTC CPP监督、原错误太阳几何**配置下的同期结果。它不是10月21日完整太阳几何修复后的同期结果。10月20日审计确认了CPP时间修复，不能将该结论扩展为太阳公式也已修正。

## 现存服务器来源

在zjnu-hpc读取现存元数据与文件哈希，见REMOTE_1054_SOURCE_20261101.json。以下链与10月20日归档记录一致：

- R：`hubei_cloud_probability_20261020/R_correctutc_attempt3/best.pt`，SHA256 `8ad4dbd1328325b778308ca4f8a6398dc476043ad41230ed094b9ae4855877c9`。
- R监督：`R_pack_correctutc_attempt2`。构建代码只替换正确UTC的CPP target/mask，直接复制原`x_norm.npy`和原`norm.json`；其几何未重建。
- COT：`cot_bank_correctutc_attempt3/BANK_REPORT.json`记录上述R；GHI输入为`ghi_fourarm_pack_attempt5`；三个GHI seed42/43/44的元数据仍与归档哈希一致。
- GHI包明确记录`geometry.npy`在保留索引上与原包逐行相同。现存数组SHA256为`fd781295dc3f0e56d07bc41351808e6ff69a99506fc35021f5198f117c953503`。
- 保存评分位于`refine-logs/hubei_cloud_probability_20261020/results_fourarm_attempt2/`；当时独立复算及完整来源核对另见`audits/hubei_cpp_time_impact_20261020/`。结果不是仅从图片估读。

## 实际几何复算

本轮读取该现存GHI包的25,899条train/val行和几何数组，核对数组SHA。按站点配准坐标和`UTC=BJT−8h`，逐行比较中心像元的SOZ：

| 划分 | 行数 | 相对正确NOAA中位绝对差 | 最大绝对差 | 相对缺少180/π的旧公式最大差 |
|---|---:|---:|---:|---:|
| train | 23,704 | 0.783846° | 3.275613° | 0.00003354° |
| validation | 2,195 | 0.743077° | 3.258841° | 0.00002887° |

旧公式为`EOT=4Q`，正确公式为`EOT=4(180/π)Q`分钟，Q是三角表达式的弧度结果。逐行结果明确匹配缺少弧度到角度转换的旧公式。独立保存的两个AGRI原始样本也与该旧公式匹配；相应GHI几何裁块和样本cosSOZ/cosRAA的比较差为0。

审计代码：audit_1054_remote.py、complete_unified_checks.py；完整数值见AUDIT_1054_GEOMETRY.json及LOCAL_GEOMETRY_1054_RECHECK.json。这里复算的是输入几何，不是修复后重训的GHI成绩。

## 论文处理

- 10.54%与同源39.57%厚云分层移入统一主稿的补充材料S4，标明原几何配置；撤出摘要、主结果及结论的定稿数字。
- 湖南3.870%原单帧重评分亦留S4，与湖南修正R三帧预报1.737%分别报告。
- 不用修复后的未来预报95.703→95.609代替尚缺的修复后同期比较。两者任务不同。
- 主稿仍保留独立湖南DyDiff完整验证及固定COT替换诊断；湖北既有四臂与来源替换的负结果保留补充材料S5。
- 不为补上缺项编造结果，也不以本次写作任务自动启动训练。
