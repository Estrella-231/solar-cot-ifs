# COT研究结果总册

## P001 当前证据总览：哪些问题已回答，哪些仍待验证？

本页意义：按科学问题定位材料；各区域与模型设置分开，不将不同实验的百分比拼接。

评价范围：截至2026-10-19的文件核对；服务器快照14:13 BJT

| 问题 | 当前完成结果 | 可支持的判断 |
| --- | --- | --- |
| 同期COT价值 | 湖南约3.02%；湖北约4.49% RMSE降低 | 当前紧凑估计器的验证结果有增益 |
| 增益是否只因容量 | 容量对照未复现全部收益 | 超过本实验增加头部容量的收益 |
| 显式COT是否优于学习特征 | 同R隐藏表示、强AGRI残差差异区间跨零 | 未识别显式COT独有优势 |
| 何种云厚收益较大 | 湖南CPP厚云子组约22.45% | 条件性收益；不是总体或所有厚云 |
| COT未来预报 | 完整3,140序列、46个CPP日期；优于持续性 | 存在COT预报技巧，CPP仍为参考 |
| 未来GHI净增益 | 湖北匹配预报无明确改善；湖南新pilot区间跨零 | 暂不能声称COT改善未来GHI |
| 10月19日监督改进 | 混合辅助及CPP-only均未通过预定筛选 | 厚云CSI小改进伴随虚警增加 |
| 最新在途 | 湖南五折OOF输入正在重建 | 尚无OOF重训后的GHI最终成绩 |
| 独立测试 | 当前test/buffer仍关闭 | 本册不包含独立test成绩 |

来源：

## P002 阅读目录：按科学问题查找 1

本页意义：页码直接对应本PDF；每页一个表或图。最新辅助监督、资产与权重目录位于后半册。

评价范围：完整结果＋历史探索＋工程复现

| 实验 | 要回答的问题 | 本册页码 |
| --- | --- | --- |
| E01 | 湖南同期估计：COT能否改善紧凑GHI模型？ | 8–11 |
| E02 | 湖北独立区域：同期COT收益能否重复？ | 12–16 |
| E03 | 反演质量：COT对CPP参考的误差有多大？ | 17–17 |
| E04 | 厚度分层：GHI收益主要出现在哪些云厚？ | 18–21 |
| E05 | 太阳资源：高太阳高度与日辐照量是否改善？ | 22–22 |
| E06 | 容量与反演特征：收益是否超过普通辅助表示？ | 23–24 |
| E07 | 湖北同R特征对照：显式COT是否优于隐藏特征？ | 25–26 |
| E08 | 湖北三站共同时间：站点缺测会改变结论吗？ | 27–28 |
| E09 | 湖北条件差异：月份和太阳高度有何影响？ | 29–30 |
| E10 | 更强AGRI残差：COT仍有额外收益吗？ | 31–32 |
| E11 | 有符号残差：早期修正结果与后续统一对照 | 33–33 |

来源：PAGE_INDEX.csv

## P003 阅读目录：按科学问题查找 2

本页意义：页码直接对应本PDF；每页一个表或图。最新辅助监督、资产与权重目录位于后半册。

评价范围：完整结果＋历史探索＋工程复现

| 实验 | 要回答的问题 | 本册页码 |
| --- | --- | --- |
| E12 | 三帧COT：历史变化是否真的被利用？ | 34–40 |
| E13 | R太阳几何修正：COT误差是否降低？ | 41–44 |
| E14 | 修正后的R进入GHI：上游改善能否传递？ | 45–45 |
| E15 | 高GHI低估：增大高透过损失的收益与代价 | 46–49 |
| E16 | 真实云事件：COT为什么有时把GHI调错方向？ | 50–63 |
| E17 | 云高与云影：空间代表性是否有可用证据？ | 64–65 |
| E18 | 输入扰动：GHI模型使用了哪些COT信息？ | 66–66 |
| E19 | 晴空屏蔽oracle：COT变准是否让GHI更准？ | 67–68 |
| E20 | 可部署保守判云：重训GHI头后是否受益？ | 69–69 |
| E21 | 门控退化归因：误清云能解释多少？ | 70–74 |
| E22 | 站点GHI输入范围：小邻域是否足够？ | 75–75 |

来源：PAGE_INDEX.csv

## P004 阅读目录：按科学问题查找 3

本页意义：页码直接对应本PDF；每页一个表或图。最新辅助监督、资产与权重目录位于后半册。

评价范围：完整结果＋历史探索＋工程复现

| 实验 | 要回答的问题 | 本册页码 |
| --- | --- | --- |
| E23 | 最新COT预报：固定SimVP与等预算监督对照 | 76–77 |
| E24 | 直接COT幅值损失：厚云误差能否修复？ | 78–79 |
| E25 | 独立厚云事件：位置、峰值和到达时间 | 80–80 |
| E26 | 离线位置对齐：错位还是厚度演变？ | 81–81 |
| E27 | 时间变化：预测COT轨迹是否过平？ | 82–82 |
| E28 | 固定GHI头2×2替换：图像与COT误差如何传递？ | 83–84 |
| E29 | 历史COT输运：与Cloud-FNO文献如何比较？ | 85–86 |
| E30 | 历史COT预报GHI：修正选模流程后的配对结果 | 87–90 |
| E31 | 湖北未来GHI：预测COT是否带来净增益？ | 91–92 |
| E32 | 湖南最新未来GHI：完整预测COT与收缩输入 | 93–95 |
| E33 | 监督来源：CPP与R伪标签是否等价？ | 96–98 |

来源：PAGE_INDEX.csv

## P005 阅读目录：按科学问题查找 4

本页意义：页码直接对应本PDF；每页一个表或图。最新辅助监督、资产与权重目录位于后半册。

评价范围：完整结果＋历史探索＋工程复现

| 实验 | 要回答的问题 | 本册页码 |
| --- | --- | --- |
| E34 | 云与厚云检出：FAR、POD、CSI分别说明什么？ | 99–100 |
| E35 | 数据划分：每个月的训练和验证覆盖如何？ | 101–102 |
| E36 | 历史探索：空间COT初筛 | 103–103 |
| E37 | 历史探索：空间排列控制 | 104–104 |
| E38 | 历史探索：AFNO固定GHI头替换 | 105–106 |
| E39 | 历史探索：输运与厚度残差 | 107–110 |
| E40 | 历史探索：完整历史与重复末帧 | 111–112 |
| E41 | 历史探索：低COT惩罚消融 | 113–113 |
| E42 | 历史探索：历史信息探针 | 114–115 |
| E43 | 历史探索：竺家残差容量对照 | 116–116 |
| E44 | 历史探索：旧天气分层 | 117–117 |

来源：PAGE_INDEX.csv

## P006 研究范围：五站、两个独立区域与日期覆盖

本页意义：说明数据范围；两个区域分别训练、分别评价，不称零样本迁移。

评价范围：研究设计示意；不作为实验成绩

![](figures/overview_fig1_study_regions_coverage.png)

来源：F:\CODE_Classify\辐照度预报\solar-cot-ifs\paper\figures\fig1_study_regions_coverage.png

## P007 方法关系：同期COT增量与未来应用如何区分？

本页意义：图区分观测AGRI反演COT与预测AGRI反演COT，避免混淆同期和预报。

评价范围：研究设计示意；不作为实验成绩

![](figures/overview_fig2_cot_estimation_forecast_workflow.png)

来源：F:\CODE_Classify\辐照度预报\solar-cot-ifs\paper\figures\fig2_cot_estimation_forecast_workflow.png

## P008 湖南同期估计：COT能否改善紧凑GHI模型？｜Result

本页意义：三种子两站等权RMSE约降低3.02%；评价的是目标时刻真实AGRI辅助的同期估计。

评价范围：验证集；同期GHI估计；已完成

| Seed | A | AC | AP | AC−A | AC−AP |
| --- | --- | --- | --- | --- | --- |
| 42 | 122.933 | 119.765 | 122.721 | −3.168 | −2.956 |
| 43 | 125.217 | 119.533 | 125.113 | −5.684 | −5.580 |
| 44 | 123.791 | 121.422 | 123.631 | −2.369 | −2.209 |
| Three-seed mean | 123.980 | 120.240 | 123.822 | −3.740 | −3.582 |

来源：experiments/monthly_R_station16_20261013_v1/results/GHI_HEAD_THREE_SEED_RESULTS_20261013.md

## P009 湖南同期估计：COT能否改善紧凑GHI模型？｜Result

本页意义：三种子两站等权RMSE约降低3.02%；评价的是目标时刻真实AGRI辅助的同期估计。

评价范围：验证集；同期GHI估计；已完成

| BJT season | Sili n | Zhujia n | Mean AC−A across seeds (W m⁻²) | Seed-wise station-equal AC−A |
| --- | --- | --- | --- | --- |
| DJF | 340 | 501 | −4.428 | −3.008, −7.614, −2.661 |
| MAM | 712 | 787 | −5.271 | −5.685, −6.314, −3.813 |
| JJA | 726 | 731 | −2.271 | −1.685, −4.198, −0.931 |
| SON | 350 | 332 | −6.734 | −5.304, −9.641, −5.259 |

来源：experiments/monthly_R_station16_20261013_v1/results/GHI_HEAD_THREE_SEED_RESULTS_20261013.md

## P010 湖南同期估计：COT能否改善紧凑GHI模型？｜Paired uncertainty

本页意义：三种子两站等权RMSE约降低3.02%；评价的是目标时刻真实AGRI辅助的同期估计。

评价范围：验证集；同期GHI估计；已完成

| Paired difference | Mean | BJT date-cluster 95% CI | Consecutive 2-day block 95% CI |
| --- | --- | --- | --- |
| AC−A | −3.740 | [−6.180, −1.597] | [−6.174, −1.483] |
| AC−AP | −3.582 | [−5.977, −1.479] | [−5.973, −1.375] |
| AP−A | −0.158 | [−0.233, −0.092] | [−0.223, −0.095] |

来源：experiments/monthly_R_station16_20261013_v1/results/GHI_HEAD_THREE_SEED_RESULTS_20261013.md

## P011 湖南同期估计：COT能否改善紧凑GHI模型？｜配套图 3

本页意义：三种子两站等权RMSE约降低3.02%；评价的是目标时刻真实AGRI辅助的同期估计。

评价范围：验证集；同期GHI估计；已完成

![](figures/E01_003.png)

来源：paper/figures/fig4_cot_incremental_value.png

## P012 湖北独立区域：同期COT收益能否重复？｜结果

本页意义：采用最新归一化重载及2026-10-18复算口径：三站等权RMSE降低约4.49%，与湖南分开报告。

评价范围：湖北三站；22个验证目标日期；三种子；已完成

| 站点 | A 无COT | AC 加COT | AP 容量对照 |
| --- | --- | --- | --- |
| 牛首 | 76.65 | 75.00 | 76.14 |
| 王寨 | 66.07 | 64.27 | 65.27 |
| 米庄 | 80.28 | 73.70 | 80.10 |
| 三站等权均值 | 74.33 | 70.99 | 73.84 |

来源：refine-logs/hubei_cot_independent_20261014/SAME_TIME_THREE_SEED_VALIDATION_20261017.md

## P013 湖北独立区域：同期COT收益能否重复？｜结果

本页意义：采用最新归一化重载及2026-10-18复算口径：三站等权RMSE降低约4.49%，与湖南分开报告。

评价范围：湖北三站；22个验证目标日期；三种子；已完成

| 比较 | RMSE差 | 配对日期bootstrap 95%区间 | 连续两日块敏感性95%区间 |
| --- | --- | --- | --- |
| AC − A | −3.34 | [−5.80, −1.12] | [−6.14, −0.73] |
| AP − A | −0.49 | [−0.80, −0.26] | [−0.85, −0.25] |
| AC − AP | −2.85 | [−5.15, −0.72] | [−5.50, −0.35] |

来源：refine-logs/hubei_cot_independent_20261014/SAME_TIME_THREE_SEED_VALIDATION_20261017.md

## P014 湖北独立区域：同期COT收益能否重复？｜配套图 4

本页意义：采用最新归一化重载及2026-10-18复算口径：三站等权RMSE降低约4.49%，与湖南分开报告。

评价范围：湖北三站；22个验证目标日期；三种子；已完成

![](figures/E02_004.png)

来源：refine-logs/hubei_cot_independent_20261014/validation_cases_normreload_20261017/Mizhuang_validation_cases.png

## P015 湖北独立区域：同期COT收益能否重复？｜配套图 5

本页意义：采用最新归一化重载及2026-10-18复算口径：三站等权RMSE降低约4.49%，与湖南分开报告。

评价范围：湖北三站；22个验证目标日期；三种子；已完成

![](figures/E02_005.png)

来源：refine-logs/hubei_cot_independent_20261014/validation_cases_normreload_20261017/Niushou_validation_cases.png

## P016 湖北独立区域：同期COT收益能否重复？｜配套图 6

本页意义：采用最新归一化重载及2026-10-18复算口径：三站等权RMSE降低约4.49%，与湖南分开报告。

评价范围：湖北三站；22个验证目标日期；三种子；已完成

![](figures/E02_006.png)

来源：refine-logs/hubei_cot_independent_20261014/validation_cases_normreload_20261017/Wangzhai_validation_cases.png

## P017 反演质量：COT对CPP参考的误差有多大？｜配套图 7

本页意义：先量化输入COT的参考一致性及厚云偏差，再解释下游价值；CPP不是独立物理真值。

评价范围：湖南与湖北各自验证；空间与样本范围不同；已完成

![](figures/E03_007.png)

来源：paper/figures/fig3_retrieval_quality.png

## P018 厚度分层：GHI收益主要出现在哪些云厚？｜all_validation

本页意义：湖南CPP参考厚云子组RMSE降幅22.45%；其余组较小且区间跨零。分层收益不代表总体收益。

评价范围：湖南同期；中心像元CPP参考分箱；不是预测云况；已完成

| COT bin | Sili / Zhujia rows | BJT dates | A RMSE | AC RMSE | AP RMSE | AC−A [95% CI] W/m² | RMSE reduction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 2128 / 2351 | 52 | 123.980 | 120.240 | 123.822 | -3.740 [-6.180, -1.597] | +3.02% |

来源：refine-logs/cot_value_first_20261013/results/cot_thickness_story_20261014/COT_THICKNESS_GHI_VALIDATION_20261014.md

## P019 厚度分层：GHI收益主要出现在哪些云厚？｜CPP_reference_centre

本页意义：湖南CPP参考厚云子组RMSE降幅22.45%；其余组较小且区间跨零。分层收益不代表总体收益。

评价范围：湖南同期；中心像元CPP参考分箱；不是预测云况；已完成

| COT bin | Sili / Zhujia rows | BJT dates | A RMSE | AC RMSE | AP RMSE | AC−A [95% CI] W/m² | RMSE reduction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| supported | 610 / 674 | 46 | 138.369 | 135.540 | 138.183 | -2.828 [-6.130, -0.260] | +2.04% |
| nearzero_[0,0.1) | 143 / 321 | 31 | 159.165 | 156.812 | 158.964 | -2.353 [-6.238, +1.157] | +1.48% |
| thin_[0.1,5) | 71 / 110 | 29 | 165.869 | 163.045 | 165.816 | -2.823 [-6.814, +1.268] | +1.70% |
| moderate_[5,30) | 316 / 150 | 36 | 122.509 | 120.430 | 122.288 | -2.079 [-7.179, +3.030] | +1.70% |
| thick_[30,100] | 80 / 93 | 25 | 74.929 | 58.105 | 74.687 | -16.824 [-21.862, -12.359] | +22.45% |

来源：refine-logs/cot_value_first_20261013/results/cot_thickness_story_20261014/COT_THICKNESS_GHI_VALIDATION_20261014.md

## P020 厚度分层：GHI收益主要出现在哪些云厚？｜R_retrieved_centre

本页意义：湖南CPP参考厚云子组RMSE降幅22.45%；其余组较小且区间跨零。分层收益不代表总体收益。

评价范围：湖南同期；中心像元CPP参考分箱；不是预测云况；已完成

| COT bin | Sili / Zhujia rows | BJT dates | A RMSE | AC RMSE | AP RMSE | AC−A [95% CI] W/m² | RMSE reduction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| supported | 2128 / 2351 | 52 | 123.980 | 120.240 | 123.822 | -3.740 [-6.180, -1.597] | +3.02% |
| nearzero_[0,0.1) | 356 / 734 | 37 | 171.543 | 170.980 | 171.429 | -0.564 [-5.068, +2.755] | +0.33% |
| thin_[0.1,5) | 258 / 353 | 41 | 164.395 | 162.751 | 164.299 | -1.644 [-4.997, +1.437] | +1.00% |
| moderate_[5,30) | 1048 / 767 | 51 | 103.665 | 97.350 | 103.403 | -6.316 [-9.526, -3.033] | +6.09% |
| thick_[30,100] | 466 / 497 | 42 | 61.734 | 46.721 | 61.514 | -15.014 [-18.976, -10.937] | +24.32% |

来源：refine-logs/cot_value_first_20261013/results/cot_thickness_story_20261014/COT_THICKNESS_GHI_VALIDATION_20261014.md

## P021 厚度分层：GHI收益主要出现在哪些云厚？｜配套图 8

本页意义：湖南CPP参考厚云子组RMSE降幅22.45%；其余组较小且区间跨零。分层收益不代表总体收益。

评价范围：湖南同期；中心像元CPP参考分箱；不是预测云况；已完成

![](figures/E04_008.png)

来源：paper/figures/fig5_hunan_cpp_cot_strata.png

## P022 太阳资源：高太阳高度与日辐照量是否改善？｜Results

本页意义：高太阳高度和完整日照日期的误差用于检查实际资源估计意义；MAE降低与日偏差更负可以并存。

评价范围：湖南验证；高太阳高度及完整日期敏感性；已完成

| Validation quantity | AGRI baseline | AGRI + COT | Paired difference (AC − A), 95% date-block interval |
| --- | --- | --- | --- |
| GHI RMSE, solar altitude ≥45° (W m⁻²) | 154.111 | 148.645 | −5.465 [−8.990, −2.152] |
| Daily irradiation MAE, complete days (kWh m⁻²) | 0.5491 | 0.5032 | −0.0459 [−0.0861, −0.0067] |
| Daily irradiation bias, complete days (kWh m⁻²) | −0.0164 | −0.0561 | — |

来源：paper/analyses/solar_resource_validation_20261017/REPORT.md

## P023 容量与反演特征：收益是否超过普通辅助表示？｜Results

本页意义：区分COT辅助表示、额外头部容量和反演器隐藏特征；不能把全部收益唯一归于显式物理值。

评价范围：验证集；同期GHI估计；已完成

| Head | Sili RMSE | Zhujia RMSE | Station-equal RMSE |
| --- | --- | --- | --- |
| A | 135.156 | 112.805 | 123.980 |
| AC | 132.368 | 108.113 | 120.240 |
| AP | 135.073 | 112.571 | 123.822 |
| AL | 132.665 | 106.950 | 119.807 |

来源：refine-logs/cot_value_first_20261013/results/T18_RAL_RESULTS_20261014.md

## P024 容量与反演特征：收益是否超过普通辅助表示？｜Results

本页意义：区分COT辅助表示、额外头部容量和反演器隐藏特征；不能把全部收益唯一归于显式物理值。

评价范围：验证集；同期GHI估计；已完成

| Contrast | Δ RMSE (W m⁻²) | 95% BJT-date-block CI |
| --- | --- | --- |
| AL − A | −4.173 | [−7.137, −1.390] |
| AL − AC | −0.433 | [−1.476, +0.677] |
| AL − AP | −4.015 | [−6.914, −1.265] |
| AC − A | −3.740 | [−6.074, −1.647] |

来源：refine-logs/cot_value_first_20261013/results/T18_RAL_RESULTS_20261014.md

## P025 湖北同R特征对照：显式COT是否优于隐藏特征？｜湖北 AL 结果独立复算（验证集）

本页意义：采用重载后的隐藏特征对照；COT与隐藏特征差异需按共同样本和区间判断，不能引用被取代的旧汇总。

评价范围：湖北同期；最新归一化重载；已完成

| 组别 | RMSE (W/m²) | MAE (W/m²) | Bias (W/m²) | 平均站点/种子 R² |
| --- | --- | --- | --- | --- |
| A | 74.333 | 53.609 | 19.647 | 0.9340 |
| AC | 70.993 | 50.914 | 12.882 | 0.9400 |
| AP | 73.839 | 53.040 | 18.660 | 0.9348 |
| AL | 71.090 | 50.987 | 12.828 | 0.9399 |

来源：refine-logs/hubei_cot_independent_20261014/audit_receipts/al_normreload_20261017/AL_RESULT_AUDIT_20261017.md

## P026 湖北同R特征对照：显式COT是否优于隐藏特征？｜湖北 AL 结果独立复算（验证集）

本页意义：采用重载后的隐藏特征对照；COT与隐藏特征差异需按共同样本和区间判断，不能引用被取代的旧汇总。

评价范围：湖北同期；最新归一化重载；已完成

| 对照差（前者−后者，RMSE；负值有利前者） | 点差 W/m² | 单日 bootstrap 95% CI | 连续两日块 95% CI |
| --- | --- | --- | --- |
| AC − A | -3.340 | [-5.780, -1.095] | [-6.049, -0.781] |
| AC − AP | -2.846 | [-5.124, -0.690] | [-5.427, -0.384] |
| AL − A | -3.244 | [-5.957, -0.651] | [-6.311, -0.412] |
| AL − AP | -2.749 | [-5.249, -0.226] | [-5.613, 0.035] |
| AC − AL | -0.097 | [-1.410, 1.219] | [-1.394, 1.387] |

来源：refine-logs/hubei_cot_independent_20261014/audit_receipts/al_normreload_20261017/AL_RESULT_AUDIT_20261017.md

## P027 湖北三站共同时间：站点缺测会改变结论吗？｜三站共同目标时刻敏感性（validation only）

本页意义：统一三站目标时刻后检验表示差异，避免不同站点样本构成造成误解。

评价范围：湖北同期；三站486个共同目标时刻；已完成

| 方法 | 共同支持 RMSE (W/m²) |
| --- | --- |
| A | 76.075 |
| AC | 72.867 |
| AP | 75.551 |
| AL | 72.785 |

来源：refine-logs/hubei_cot_independent_20261014/audit_receipts/al_normreload_20261017/THREE_STATION_COMMON_TIME_SENSITIVITY_20261017.md

## P028 湖北三站共同时间：站点缺测会改变结论吗？｜三站共同目标时刻敏感性（validation only）

本页意义：统一三站目标时刻后检验表示差异，避免不同站点样本构成造成误解。

评价范围：湖北同期；三站486个共同目标时刻；已完成

| 配对差（前者−后者，RMSE） | 点差 (W/m²) | 连续两日块95% CI (W/m²) |
| --- | --- | --- |
| AC − A | -3.207 | [-6.281, -0.615] |
| AC − AP | -2.683 | [-5.613, -0.205] |
| AL − A | -3.289 | [-7.013, -0.120] |
| AL − AP | -2.766 | [-6.258, 0.232] |
| AC − AL | 0.082 | [-1.490, 1.713] |

来源：refine-logs/hubei_cot_independent_20261014/audit_receipts/al_normreload_20261017/THREE_STATION_COMMON_TIME_SENSITIVITY_20261017.md

## P029 湖北条件差异：月份和太阳高度有何影响？｜太阳高度分层

本页意义：展示验证增益的条件性；月份和太阳角分层不能替代独立晴空、阴天分类。

评价范围：湖北最新重载验证；描述性条件分层；已完成

| 太阳高度 | 牛首 n | 王寨 n | 米庄 n | AC−A RMSE (W/m²) | AC−AP RMSE (W/m²) |
| --- | --- | --- | --- | --- | --- |
| <0° | 9 | 7 | 9 | −0.01 | −0.02 |
| 0–15° | 169 | 154 | 162 | −3.91 | −3.44 |
| 15–30° | 209 | 190 | 201 | −5.77 | −4.97 |
| 30–45° | 220 | 202 | 218 | −6.43 | −5.97 |
| 45–60° | 156 | 141 | 136 | +2.49 | +2.86 |
| 60–90° | 93 | 90 | 90 | −0.28 | +0.14 |

来源：refine-logs/hubei_cot_independent_20261014/HUBEI_VALIDATION_SUBGROUPS_NORMRELOAD_20261017.md

## P030 湖北条件差异：月份和太阳高度有何影响？｜月份分层

本页意义：展示验证增益的条件性；月份和太阳角分层不能替代独立晴空、阴天分类。

评价范围：湖北最新重载验证；描述性条件分层；已完成

| BJT月份 | 牛首 n | 王寨 n | 米庄 n | AC−A RMSE (W/m²) | AC−AP RMSE (W/m²) |
| --- | --- | --- | --- | --- | --- |
| 2025-08 | 85 | 77 | 82 | −0.86 | −0.57 |
| 2025-09 | 81 | 78 | 79 | +0.41 | +0.52 |
| 2025-10 | 75 | 70 | 70 | −10.71 | −8.97 |
| 2025-11 | 73 | 66 | 66 | −3.96 | −3.69 |
| 2025-12 | 66 | 64 | 63 | +0.73 | +1.23 |
| 2026-01 | 65 | 60 | 64 | −1.11 | −0.27 |
| 2026-02 | 71 | 35 | 73 | −8.19 | −8.00 |
| 2026-03 | 74 | 74 | 73 | −2.12 | −2.04 |
| 2026-04 | 88 | 83 | 83 | −2.59 | −1.98 |
| 2026-05 | 88 | 87 | 82 | +0.45 | +0.72 |
| 2026-06 | 90 | 90 | 81 | −1.43 | −0.78 |

来源：refine-logs/hubei_cot_independent_20261014/HUBEI_VALIDATION_SUBGROUPS_NORMRELOAD_20261017.md

## P031 更强AGRI残差：COT仍有额外收益吗？｜Result

本页意义：统一输入残差模型中，AGRI残差RMSE111.277、AGRI+COT113.419，差异区间跨零；COT收益依赖比较模型。

评价范围：验证集；同期GHI估计；已完成

| Validation method | Equal-station RMSE (W/m²) | Change from A | Paired date-block 95% CI for change |
| --- | --- | --- | --- |
| Frozen A | 123.979 | — | — |
| A + signed AGRI residual | 111.277 | −12.702 | [−17.635, −7.978] |
| A + signed AGRI+COT residual | 113.419 | −10.561 | [−16.624, −4.716] |
| AGRI+COT minus AGRI | — | +2.142 | [−0.688, +4.745] |

来源：experiments/monthly_R_station16_20261013_v1/crossfit_signed_residual/UNIFIED_FEATURE_RESIDUAL_RESULTS_20261014.md

## P032 更强AGRI残差：COT仍有额外收益吗？｜配套图 9

本页意义：统一输入残差模型中，AGRI残差RMSE111.277、AGRI+COT113.419，差异区间跨零；COT收益依赖比较模型。

评价范围：验证集；同期GHI估计；已完成

![](figures/E10_009.png)

来源：docs/组会原始材料_20261016/10_残差修正/残差修正效果.png

## P033 有符号残差：早期修正结果与后续统一对照｜Main result

本页意义：保留早期残差探索作为演进材料；最终增量判断以统一输入残差结果为准。

评价范围：验证集；同期GHI估计；历史探索；已被统一输入对照补充

| Selected validation method | Equal-station mean RMSE (W/m²) | Change from A |
| --- | --- | --- |
| Frozen A | 123.979 | — |
| A + signed AGRI residual (v2) | 123.845 | −0.134 |
| A + signed AGRI+COT residual (v2) | 123.748 | −0.231 |

来源：experiments/monthly_R_station16_20261013_v1/crossfit_signed_residual/OOF_SIGNED_RESIDUAL_RESULTS_20261014.md

## P034 三帧COT：历史变化是否真的被利用？｜2. 太阳几何：发现可复现的单位错误

本页意义：三帧3×3输入确实影响GHI，但总体RMSE点改善仅0.24%，日期区间跨零。

评价范围：湖南三帧离线区间反演；4,263条验证；已完成

| 对独立NREL SPA的名义UTC参考 | 原太阳角最大绝对差 | 修正后最大绝对差 |
| --- | --- | --- |
| 天顶角 | 3.0897° | 0.0101° |
| 方位角 | 5.8915° | 0.0215° |

来源：results/cot_interval_history_20261015/REPORT.md

## P035 三帧COT：历史变化是否真的被利用？｜4. 独立FP64结果

本页意义：三帧3×3输入确实影响GHI，但总体RMSE点改善仅0.24%，日期区间跨零。

评价范围：湖南三帧离线区间反演；4,263条验证；已完成

| 范围 | A历史 | AGRI+COT历史 | ΔRMSE | 日期簇95%区间 |
| --- | --- | --- | --- | --- |
| 四里 | 124.222 | 123.901 | -0.322 | [-2.286, +1.003] |
| 竺家 | 92.425 | 92.230 | -0.195 | [-3.537, +3.187] |
| 两站等权 | 108.323 | 108.065 | -0.258 | [-2.149, +1.571] |
| 仅评分排除可疑零值（4,213条） | 88.888 | 88.383 | -0.505 | [-2.357, +1.439] |

来源：results/cot_interval_history_20261015/REPORT.md

## P036 三帧COT：历史变化是否真的被利用？｜配套图 10

本页意义：三帧3×3输入确实影响GHI，但总体RMSE点改善仅0.24%，日期区间跨零。

评价范围：湖南三帧离线区间反演；4,263条验证；已完成

![](figures/E12_010.png)

来源：results/cot_interval_history_20261015/audit/solar_geometry_correction.png

## P037 三帧COT：历史变化是否真的被利用？｜配套图 11

本页意义：三帧3×3输入确实影响GHI，但总体RMSE点改善仅0.24%，日期区间跨零。

评价范围：湖南三帧离线区间反演；4,263条验证；已完成

![](figures/E12_011.png)

来源：results/cot_interval_history_20261015/figures/case0_sili.png

## P038 三帧COT：历史变化是否真的被利用？｜配套图 12

本页意义：三帧3×3输入确实影响GHI，但总体RMSE点改善仅0.24%，日期区间跨零。

评价范围：湖南三帧离线区间反演；4,263条验证；已完成

![](figures/E12_012.png)

来源：results/cot_interval_history_20261015/figures/case1_sili.png

## P039 三帧COT：历史变化是否真的被利用？｜配套图 13

本页意义：三帧3×3输入确实影响GHI，但总体RMSE点改善仅0.24%，日期区间跨零。

评价范围：湖南三帧离线区间反演；4,263条验证；已完成

![](figures/E12_013.png)

来源：results/cot_interval_history_20261015/figures/case2_zhujia.png

## P040 三帧COT：历史变化是否真的被利用？｜配套图 14

本页意义：三帧3×3输入确实影响GHI，但总体RMSE点改善仅0.24%，日期区间跨零。

评价范围：湖南三帧离线区间反演；4,263条验证；已完成

![](figures/E12_014.png)

来源：results/cot_interval_history_20261015/figures/case3_zhujia.png

## P041 R太阳几何修正：COT误差是否降低？｜Question and result

本页意义：修正太阳几何后单独评价R；反演指标改善不自动代表GHI改善。

评价范围：湖南R验证；几何单因素；已完成

| Comparison | Sili ΔRMSE | Zhujia ΔRMSE | Equal-station ΔRMSE | 95% date-cluster CI |
| --- | --- | --- | --- | --- |
| Corrected retrain − old-formula retrain | +0.305 | −0.133 | +0.086 | [−0.282, +0.438] |

来源：experiments/r_solar_geometry_corrected_20261015/RESULT_REPORT.md

## P042 R太阳几何修正：COT误差是否降低？｜Validation metrics

本页意义：修正太阳几何后单独评价R；反演指标改善不自动代表GHI改善。

评价范围：湖南R验证；几何单因素；已完成

| Model | RMSE | MAE | Bias |
| --- | --- | --- | --- |
| Archived original R checkpoint | 9.018 | 4.615 | +0.346 |
| Fresh seed-42 retrain, original geometry formula | 8.790 | 4.713 | −0.395 |
| Fresh seed-42 retrain, corrected geometry | 8.878 | 4.657 | −0.702 |

来源：experiments/r_solar_geometry_corrected_20261015/RESULT_REPORT.md

## P043 R太阳几何修正：COT误差是否降低？｜By station

本页意义：修正太阳几何后单独评价R；反演指标改善不自动代表GHI改善。

评价范围：湖南R验证；几何单因素；已完成

| Station | Old-formula RMSE | Corrected RMSE | Corrected − old |
| --- | --- | --- | --- |
| Sili | 8.771 | 9.077 | +0.305 |
| Zhujia | 8.808 | 8.675 | −0.133 |

来源：experiments/r_solar_geometry_corrected_20261015/RESULT_REPORT.md

## P044 R太阳几何修正：COT误差是否降低？｜By CPP reference COT category

本页意义：修正太阳几何后单独评价R；反演指标改善不自动代表GHI改善。

评价范围：湖南R验证；几何单因素；已完成

| CPP reference category | Pixels | Old RMSE | Corrected RMSE | Old bias | Corrected bias |
| --- | --- | --- | --- | --- | --- |
| Near-zero [0, 0.1) | 126,856 | 8.805 | 9.513 | +3.146 | +3.297 |
| Thin [0.1, 5) | 49,850 | 1.642 | 1.558 | +0.014 | −0.477 |
| Moderate [5, 30) | 125,359 | 5.756 | 5.426 | +0.718 | −0.112 |
| Thick [30, 100] | 50,376 | 16.131 | 15.745 | −12.484 | −12.465 |

来源：experiments/r_solar_geometry_corrected_20261015/RESULT_REPORT.md

## P045 修正后的R进入GHI：上游改善能否传递？｜Three paired GHI arms

本页意义：在GHI端检查几何修正的实际作用，避免只凭R误差判断下游收益。

评价范围：验证集；同期GHI估计；已完成

| Arm or paired contrast | Validation RMSE (W m⁻²) | ΔRMSE (W m⁻²) | Relative change | 95% date-cluster interval |
| --- | --- | --- | --- | --- |
| AGRI only | 108.236 | — | — | — |
| AGRI + old-formula R-COT | 107.818 | −0.418 vs AGRI only | −0.39% | [−2.236, +1.136] |
| AGRI + corrected-geometry R-COT | 107.784 | −0.452 vs AGRI only | −0.42% | [−2.294, +1.101] |
| Corrected-geometry R-COT − old-formula R-COT | — | −0.034 | −0.03% | [−0.365, +0.244] |

来源：experiments/r_geometry_downstream_ghi_20261016/REPORT.md

## P046 高GHI低估：增大高透过损失的收益与代价｜配对结果

本页意义：高GHI低估减轻，但低、中kt误差增加；总体未达到替换要求，高值改善不能归为COT贡献。

评价范围：验证集；同期GHI估计；已完成

| 共同验证组 | N | AGRI原损失 | AGRI新损失 | Δ | AGRI+COT原损失 | AGRI+COT新损失 | Δ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 全验证 | 4,263 | 108.236 | 110.500 | +2.264 | 107.784 | 110.004 | +2.220 |
| 高透过：clear≥100、kt≥0.9 | 748 | 124.047 | 106.234 | −17.813 | 128.375 | 108.171 | −20.204 |
| 实测GHI≥800 | 219 | 166.056 | 142.832 | −23.224 | 175.070 | 148.322 | −26.748 |
| 高透过且中心R-COT≥3 | 88 | 192.933 | 172.227 | −20.706 | 204.045 | 182.395 | −21.649 |
| 低kt：clear≥100、kt<0.3 | 1,392 | 112.800 | 118.174 | +5.374 | 109.159 | 113.950 | +4.791 |
| 中间kt：clear≥100、0.3≤kt<0.9 | 1,669 | 100.498 | 109.514 | +9.016 | 98.692 | 108.577 | +9.886 |

来源：experiments/high_kt_loss_20261016/REPORT.md

## P047 高GHI低估：增大高透过损失的收益与代价｜原失败案例是否修好

本页意义：高GHI低估减轻，但低、中kt误差增加；总体未达到替换要求，高值改善不能归为COT贡献。

评价范围：验证集；同期GHI估计；已完成

| 站点、BJT | 实测 | 原损失AC | 新损失AC |
| --- | --- | --- | --- |
| 四里2025-08-22 12:15 | 1191.3 | 699.2 | 749.9 |
| 四里2025-08-21 16:45 | 547.0 | 213.8 | 222.8 |
| 四里2024-07-09 12:00 | 1085.7 | 759.4 | 816.1 |
| 竺家2024-05-29 10:45 | 866.3 | 519.0 | 564.2 |
| 竺家2026-06-24 13:30 | 592.4 | 243.1 | 259.7 |
| 竺家2025-07-09 16:30 | 493.4 | 297.0 | 326.4 |

来源：experiments/high_kt_loss_20261016/REPORT.md

## P048 高GHI低估：增大高透过损失的收益与代价｜配套图 15

本页意义：高GHI低估减轻，但低、中kt误差增加；总体未达到替换要求，高值改善不能归为COT贡献。

评价范围：验证集；同期GHI估计；已完成

![](figures/E15_015.png)

来源：experiments/high_kt_loss_20261016/figures/loss_tradeoff.png

## P049 高GHI低估：增大高透过损失的收益与代价｜配套图 16

本页意义：高GHI低估减轻，但低、中kt误差增加；总体未达到替换要求，高值改善不能归为COT贡献。

评价范围：验证集；同期GHI估计；已完成

![](figures/E15_016.png)

来源：experiments/high_kt_loss_20261016/figures/six_real_cloud_loss_comparison.png

## P050 真实云事件：COT为什么有时把GHI调错方向？｜数据与选样

本页意义：并列改善与失败案例，检查真实AGRI、COT和GHI；案例用于解释现象，不能估计总体收益。

评价范围：湖南同期；固定失败和匹配案例诊断；已完成

| 完整验证集的输出变化 | A已经低估 | A已经高估 |
| --- | --- | --- |
| AC相对A朝正确方向变化，三个seed逐样本汇总 | 50.8% | 67.5% |
| AC相对A实际降低平方误差 | 47.0% | 61.4% |

来源：experiments/cot_underestimate_forensics_20261016/REPORT.md

## P051 真实云事件：COT为什么有时把GHI调错方向？｜六个失败案例：真实云图、COT和辐照度

本页意义：并列改善与失败案例，检查真实AGRI、COT和GHI；案例用于解释现象，不能估计总体收益。

评价范围：湖南同期；固定失败和匹配案例诊断；已完成

| 站点、北京时间 | 实测GHI | A | AC | 当前中心R-COT | 当前中心CPP参考 | 后续5分钟均值−目标15分钟均值 |
| --- | --- | --- | --- | --- | --- | --- |
| 四里 2025-08-22 12:15 | 1191.3 | 751.3 | 699.2 | 5.64 | 5.16 | −29.3 |
| 四里 2025-08-21 16:45 | 547.0 | 258.3 | 213.8 | 8.79 | 11.16 | −308.0 |
| 四里 2024-07-09 12:00 | 1085.7 | 800.7 | 759.4 | 0.01 | 0（有效参考） | −44.7 |
| 竺家 2024-05-29 10:45 | 866.3 | 572.9 | 519.0 | 2.60 | 缺失 | +7.7 |
| 竺家 2026-06-24 13:30 | 592.4 | 280.6 | 243.1 | 19.15 | 缺失 | −145.4 |
| 竺家 2025-07-09 16:30 | 493.4 | 371.4 | 297.0 | 4.87 | 5.77 | +18.9 |

来源：experiments/cot_underestimate_forensics_20261016/REPORT.md

## P052 真实云事件：COT为什么有时把GHI调错方向？｜云高、视差与云影：当前证据边界

本页意义：并列改善与失败案例，检查真实AGRI、COT和GHI；案例用于解释现象，不能估计总体收益。

评价范围：湖南同期；固定失败和匹配案例诊断；已完成

| 站点、北京时间 | 最近影区中心距离，主质量掩膜 | 本轮AC降低平方误差的seed数 |
| --- | --- | --- |
| 四里 2024-04-19 12:15 | 6.43 km；纳入低质量像元变为2.72 km | 3/3 |
| 四里 2024-06-11 14:30 | 1.85 km | 1/3 |
| 竺家 2024-05-29 11:45 | 2.44 km | 3/3 |
| 竺家 2024-04-20 14:30 | 1.65 km | 0/3 |

来源：experiments/cot_underestimate_forensics_20261016/REPORT.md

## P053 真实云事件：COT为什么有时把GHI调错方向？｜据此决定下一步

本页意义：并列改善与失败案例，检查真实AGRI、COT和GHI；案例用于解释现象，不能估计总体收益。

评价范围：湖南同期；固定失败和匹配案例诊断；已完成

| 路径 | 本轮决定 | 依据与最小后续证据 |
| --- | --- | --- |
| 时间对应 | 不移动标签或整段平移输入；保留扫描时间代表性问题 | 键、区间均值和保存输入均核对通过；实际扫描晚约3分钟，部分峰值案例确有快速变化。若做时间修订，只能另立同样本单因素对照，明确真实扫描时刻与区间支持。 |
| 云影位置 | 不自动平移COT | 现有四例不能分离改善/恶化；需先补六个新失败及匹配正确案例的有效CTH与质量敏感性，再判断局部投影是否有一致方向。 |
| GHI映射 | 接续已执行的统一输入残差修正，保持问题未解决状态 | 已把旧残差预测与本轮4,263条记录、六个失败锚点精确连接，见持续问题记录。旧修正有效但未普遍消除低估；不能重提泛化检查或重复同结构残差。后续需明确相对已完成方案唯一新增因素，并同时验收高值上调及COT增量，当前不改3×3、三帧或模型。 |

来源：experiments/cot_underestimate_forensics_20261016/REPORT.md

## P054 真实云事件：COT为什么有时把GHI调错方向？｜配套图 17

本页意义：并列改善与失败案例，检查真实AGRI、COT和GHI；案例用于解释现象，不能估计总体收益。

评价范围：湖南同期；固定失败和匹配案例诊断；已完成

![](figures/E16_017.png)

来源：experiments/cot_underestimate_forensics_20261016/figures/existing_CTH_current_model_context.png

## P055 真实云事件：COT为什么有时把GHI调错方向？｜配套图 18

本页意义：并列改善与失败案例，检查真实AGRI、COT和GHI；案例用于解释现象，不能估计总体收益。

评价范围：湖南同期；固定失败和匹配案例诊断；已完成

![](figures/E16_018.png)

来源：experiments/cot_underestimate_forensics_20261016/figures/local_COT_sensitivity.png

## P056 真实云事件：COT为什么有时把GHI调错方向？｜配套图 19

本页意义：并列改善与失败案例，检查真实AGRI、COT和GHI；案例用于解释现象，不能估计总体收益。

评价范围：湖南同期；固定失败和匹配案例诊断；已完成

![](figures/E16_019.png)

来源：experiments/cot_underestimate_forensics_20261016/figures/pair_1_sili.png

## P057 真实云事件：COT为什么有时把GHI调错方向？｜配套图 20

本页意义：并列改善与失败案例，检查真实AGRI、COT和GHI；案例用于解释现象，不能估计总体收益。

评价范围：湖南同期；固定失败和匹配案例诊断；已完成

![](figures/E16_020.png)

来源：experiments/cot_underestimate_forensics_20261016/figures/pair_2_sili.png

## P058 真实云事件：COT为什么有时把GHI调错方向？｜配套图 21

本页意义：并列改善与失败案例，检查真实AGRI、COT和GHI；案例用于解释现象，不能估计总体收益。

评价范围：湖南同期；固定失败和匹配案例诊断；已完成

![](figures/E16_021.png)

来源：experiments/cot_underestimate_forensics_20261016/figures/pair_3_sili.png

## P059 真实云事件：COT为什么有时把GHI调错方向？｜配套图 22

本页意义：并列改善与失败案例，检查真实AGRI、COT和GHI；案例用于解释现象，不能估计总体收益。

评价范围：湖南同期；固定失败和匹配案例诊断；已完成

![](figures/E16_022.png)

来源：experiments/cot_underestimate_forensics_20261016/figures/pair_4_zhujia.png

## P060 真实云事件：COT为什么有时把GHI调错方向？｜配套图 23

本页意义：并列改善与失败案例，检查真实AGRI、COT和GHI；案例用于解释现象，不能估计总体收益。

评价范围：湖南同期；固定失败和匹配案例诊断；已完成

![](figures/E16_023.png)

来源：experiments/cot_underestimate_forensics_20261016/figures/pair_5_zhujia.png

## P061 真实云事件：COT为什么有时把GHI调错方向？｜配套图 24

本页意义：并列改善与失败案例，检查真实AGRI、COT和GHI；案例用于解释现象，不能估计总体收益。

评价范围：湖南同期；固定失败和匹配案例诊断；已完成

![](figures/E16_024.png)

来源：experiments/cot_underestimate_forensics_20261016/figures/pair_6_zhujia.png

## P062 真实云事件：COT为什么有时把GHI调错方向？｜配套图 25

本页意义：并列改善与失败案例，检查真实AGRI、COT和GHI；案例用于解释现象，不能估计总体收益。

评价范围：湖南同期；固定失败和匹配案例诊断；已完成

![](figures/E16_025.png)

来源：experiments/cot_underestimate_forensics_20261016/figures/R_CPP_failure_comparison.png

## P063 真实云事件：COT为什么有时把GHI调错方向？｜配套图 26

本页意义：并列改善与失败案例，检查真实AGRI、COT和GHI；案例用于解释现象，不能估计总体收益。

评价范围：湖南同期；固定失败和匹配案例诊断；已完成

![](figures/E16_026.png)

来源：paper/figures/fig6_sili_high_change_event.png

## P064 云高与云影：空间代表性是否有可用证据？｜样本与比较口径

本页意义：利用可追溯CPP云高作视差与云影离线诊断；可视化不等于已证明生产模型完成校正。

评价范围：湖南少量真实案例；离线物理诊断；已完成

| 站点、时次（UTC） | AC相对A | 实测/晴空GHI（W/m²） | 站点CTH；质量/云类 | 视差中位数 / P90（km） | 最近影区中心距站（km） | 4 km覆盖 | 质量1敏感性最近距（km） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 四里，2024-04-19 04:15 | 改善，3/3种子；ΔSSE均值 −54,824 W²m⁻⁴ | 608 / 949 | 6.97 km；Q1 / 云 | 2.99 / 8.12 | 6.43 | 否 | 2.72（变为覆盖，但该像元质量较差） |
| 四里，2024-06-11 06:30 | 变差，3/3种子；ΔSSE均值 +17,481 W²m⁻⁴ | 809 / 840 | 2.70 km；Q2 / 云 | 2.93 / 5.10 | 1.85 | 是 | 1.88 |
| 竺家，2024-05-29 03:45 | 改善，3/3种子；ΔSSE均值 −83,278 W²m⁻⁴ | 887 / 933 | 7.34 km；Q2 / 云 | 5.38 / 8.42 | 2.44 | 是 | 2.32 |
| 竺家，2024-04-20 06:30 | 变差，3/3种子；ΔSSE均值 +25,754 W²m⁻⁴ | 190 / 829 | 8.66 km；Q3 / 云 | 6.52 / 7.38 | 1.65 | 是 | 1.60 |

来源：experiments/monthly_R_station16_20261013/cot_shadow_projection_20261014/REPORT.md

## P065 云高与云影：空间代表性是否有可用证据？｜配套图 27

本页意义：利用可追溯CPP云高作视差与云影离线诊断；可视化不等于已证明生产模型完成校正。

评价范围：湖南少量真实案例；离线物理诊断；已完成

![](figures/E17_027.png)

来源：experiments/monthly_R_station16_20261013/cot_shadow_projection_20261014/CTH_PARALLAX_SHADOW_VALIDATION_20261014.png

## P066 输入扰动：GHI模型使用了哪些COT信息？｜Results

本页意义：固定模型输入替换揭示依赖性；替换后的分布变化不能等同公平重训后的COT净收益。

评价范围：湖南同期；固定模型诊断；已完成

| Intervention | Mean RMSE change vs baseline (W m⁻²) | Range across 6 station×seed checks | Mean absolute prediction change (W m⁻²) |
| --- | --- | --- | --- |
| Physical-zero COT | +89.900 | +32.970 to +122.018 | 120.490 |
| Center-only COT | +73.012 | +25.218 to +102.818 | 106.789 |
| Neighbor permutation, center retained | +0.027 | +0.006 to +0.064 | 0.383 |
| Same-station/month COT mismatch | +37.126 | +10.600 to +79.499 | 56.180 |

来源：experiments/monthly_R_station16_20261013_v1/results/COT_INPUT_DEPENDENCE_VALIDATION_20261013.md

## P067 晴空屏蔽oracle：COT变准是否让GHI更准？｜主要发现

本页意义：参考掩膜离线清零可改变COT误差，却未必改善固定GHI头；oracle不是可部署成绩。

评价范围：湖南train/val；不可部署oracle诊断；已完成

| 站点 | CLP晴空像元 | CLP云像元 | 晴空COT RMSE | 云区COT RMSE | 晴空SSE占比 |
| --- | --- | --- | --- | --- | --- |
| 四里 | 46,915 | 131,104 | 13.02 | 8.15 | 47.8% |
| 竺家 | 71,568 | 106,582 | 9.41 | 7.71 | 50.0% |
| 两站等权RMSE均值 | — | — | 11.21 | 7.93 | — |

来源：refine-logs/cot_mask_necessity_20261014/M0_M1_RESULTS.md

## P068 晴空屏蔽oracle：COT变准是否让GHI更准？｜参考晴空门控：固定GHI头

本页意义：参考掩膜离线清零可改变COT误差，却未必改善固定GHI头；oracle不是可部署成绩。

评价范围：湖南train/val；不可部署oracle诊断；已完成

| 配对比较（W/m²） | 均值差 | 95% BJT日期配对bootstrap CI |
| --- | --- | --- |
| Oracle晴空屏蔽AC − 原AC | +9.41 | [+2.20, +21.31] |
| Oracle晴空屏蔽AC − A | +6.60 | [−0.94, +17.42] |

来源：refine-logs/cot_mask_necessity_20261014/M0_M1_RESULTS.md

## P069 可部署保守判云：重训GHI头后是否受益？｜验证结果

本页意义：门控COT与原COT各自训练匹配头；seed42未显示净收益，停止无条件扩种子。

评价范围：湖南验证；seed42门控配对；已完成

| 站点 | raw RMSE | gate RMSE | Δ RMSE | raw MAE | gate MAE | Δ MAE | gate bias − raw bias |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 四里（n=2,128） | 132.639 | 132.716 | +0.077 | 73.454 | 73.527 | +0.072 | +0.098 |
| 竺家（n=2,351） | 106.325 | 106.902 | +0.577 | 72.907 | 73.431 | +0.524 | +0.550 |
| 站点等权 RMSE | 119.482 | 119.809 | +0.327 | — | — | — | — |

来源：results/paired_gate_ghi_seed42_20261015_attempt1/REPORT.md

## P070 门控退化归因：误清云能解释多少？｜逐时刻统计

本页意义：检查被清零与未改变的局部COT记录；误差变化不能全部归于误清云。

评价范围：湖南固定模型事后诊断；已完成

| 项目 | 四里 | 竺家 |
| --- | --- | --- |
| 全 GHI 验证样本 | 2,128 | 2,351 |
| 有 CPP 参考的共同 GHI 样本 | 615 | 676 |
| 中心3×3出现误清云的时刻 | 13（8个日期） | 17（9个日期） |
| 中心3×3被清零的 CPP 有云像元数 | 38 | 56 |
| 上述像元中原 R COT≥0.1 的数量 | 14 | 2 |
| 全16×16被清零的 CPP 有云像元数 | 2,080 | 2,899 |
| 全 GHI 样本中中心3×3物理 COT 没有任何改变的时刻 | 1,666 | 1,642 |

来源：results/gate_ghi_forensics_20261015/REPORT.md

## P071 门控退化归因：误清云能解释多少？｜固定门控头的恢复诊断

本页意义：检查被清零与未改变的局部COT记录；误差变化不能全部归于误清云。

评价范围：湖南固定模型事后诊断；已完成

| 输入处理 | 四里，N=615 | 竺家，N=676 |
| --- | --- | --- |
| 实际门控 COT | 146.9321 | 123.1988 |
| 仅恢复误清有云像元的原 R 值 | 146.9226 | 123.1825 |
| 相对实际门控的变化 | −0.0095 | −0.0163 |
| 仅恢复 CPP 晴空位置的原 R 值 | 146.7273 | 121.8643 |
| 相对实际门控的变化 | −0.2048 | −1.3344 |
| 恢复全部原 R 值 | 146.7184 | 121.8481 |

来源：results/gate_ghi_forensics_20261015/REPORT.md

## P072 门控退化归因：误清云能解释多少？｜配套图 28

本页意义：检查被清零与未改变的局部COT记录；误差变化不能全部归于误清云。

评价范围：湖南固定模型事后诊断；已完成

![](figures/E21_028.png)

来源：results/gate_ghi_forensics_20261015/figures/sili_focus_cn.png

## P073 门控退化归因：误清云能解释多少？｜配套图 29

本页意义：检查被清零与未改变的局部COT记录；误差变化不能全部归于误清云。

评价范围：湖南固定模型事后诊断；已完成

![](figures/E21_029.png)

来源：results/gate_ghi_forensics_20261015/figures/zhujia_focus_cn.png

## P074 门控退化归因：误清云能解释多少？｜配套图 30

本页意义：检查被清零与未改变的局部COT记录；误差变化不能全部归于误清云。

评价范围：湖南固定模型事后诊断；已完成

![](figures/E21_030.png)

来源：results/gate_ghi_forensics_20261015/pdf_page1_render.png

## P075 站点GHI输入范围：小邻域是否足够？｜关键证据摘录

本页意义：空间范围消融解释选择小邻域的依据；旧划分实验不能当成新月度主成绩。

评价范围：旧划分预报GHI尺寸消融；历史对照；历史对照

| 内容 | 原报告数值与解释 |
| --- | --- |
| 核心判断 | 空间范围消融解释选择小邻域的依据；旧划分实验不能当成新月度主成绩。 |

来源：docs/FORECAST_GHI_WINDOW_ABLATION_20261004.md

## P076 最新COT预报：固定SimVP与等预算监督对照｜Results

本页意义：3,140验证序列、46个CPP日期的同掩膜COT评价；直接预测COT候选没有超过固定SimVP→R。

评价范围：湖南未来+15～+240分钟；单种子筛选；已完成

| Forecast source | Full 16×16 patch RMSE | Station-centred 3×3 RMSE |
| --- | --- | --- |
| Existing SimVP-forecast-AGRI → frozen R | 10.953 | 10.664 |
| Direct COT, R-derived training targets (P-R; best epoch 3) | 12.512 | 12.275 |
| Direct COT, CPP/R hybrid training targets (P-Hybrid; best epoch 9) | 12.407 | 12.163 |

来源：refine-logs/cot_retrieved_forecast_20261013/FULLTRAIN_RESULTS_20261017.md

## P077 最新COT预报：固定SimVP与等预算监督对照｜配套图 31

本页意义：3,140验证序列、46个CPP日期的同掩膜COT评价；直接预测COT候选没有超过固定SimVP→R。

评价范围：湖南未来+15～+240分钟；单种子筛选；已完成

![](figures/E23_031.png)

来源：paper/figures/fig7_forecast_cot_supervision.png

## P078 直接COT幅值损失：厚云误差能否修复？｜What the matched evaluation shows

本页意义：相同CPP像元上检查幅值损失；局部厚云误差小改动没有转化为总体COT优势。

评价范围：湖南未来COT；同掩膜幅值单因素；已完成

| CPP reference stratum | Original SimVP→R, 16×16 | P-Hybrid, 16×16 | Difference | Original SimVP→R, local 3×3 | P-Hybrid, local 3×3 | Difference |
| --- | --- | --- | --- | --- | --- | --- |
| All | 10.953 | 12.407 | +1.454 | 10.664 | 12.162 | +1.498 |
| COT < 1 | 10.955 | 8.988 | −1.967 | 10.355 | 8.540 | −1.815 |
| 1–5 | 5.223 | 5.140 | −0.084 | 6.481 | 5.469 | −1.011 |
| 5–30 | 8.721 | 9.594 | +0.873 | 8.670 | 9.724 | +1.053 |
| ≥30 | 18.263 | 25.668 | +7.405 | 17.911 | 25.511 | +7.600 |

来源：refine-logs/cot_retrieved_forecast_20261013/results/SAME_SUPPORT_AMPLITUDE_DIAGNOSIS_20261017.md

## P079 直接COT幅值损失：厚云误差能否修复？｜Fixed physical-amplitude loss follow-up

本页意义：相同CPP像元上检查幅值损失；局部厚云误差小改动没有转化为总体COT优势。

评价范围：湖南未来COT；同掩膜幅值单因素；已完成

| Candidate | Full 16×16 RMSE | Local 3×3 RMSE |
| --- | --- | --- |
| Original SimVP→R | 10.953 | 10.664 |
| P-Hybrid base | 12.407 | 12.162 |
| P-Hybrid + physical-amplitude term | 12.473 | 12.209 |

来源：refine-logs/cot_retrieved_forecast_20261013/results/SAME_SUPPORT_AMPLITUDE_DIAGNOSIS_20261017.md

## P080 独立厚云事件：位置、峰值和到达时间｜Same-event results

本页意义：七个厚云事件集中在三个日期；用事件显示云团检出与峰值问题，不能作独立总体推断。

评价范围：湖南验证；七事件描述性分解；已完成

| Station | Method | Events | Mean event local-3×3 RMSE | Mean station-pixel peak error | Pooled thick CSI |
| --- | --- | --- | --- | --- | --- |
| Sili | SimVP forecast AGRI→R | 4 | 19.53 | −25.92 | 0.149 |
| Sili | Observed future AGRI→R (hindsight) | 4 | 6.87 | −3.71 | 0.814 |
| Sili | Direct COT, R-target | 4 | 24.22 | −31.66 | 0.065 |
| Sili | Direct COT, CPP/R hybrid target | 4 | 24.47 | −30.37 | 0.120 |
| Sili | Direct COT, hybrid plus amplitude term | 4 | 24.42 | −28.88 | 0.272 |
| Zhujia | SimVP forecast AGRI→R | 3 | 13.48 | −13.52 | 0.000 |
| Zhujia | Observed future AGRI→R (hindsight) | 3 | 7.38 | +5.36 | 0.617 |
| Zhujia | Direct COT, R-target | 3 | 25.26 | −26.71 | 0.000 |
| Zhujia | Direct COT, CPP/R hybrid target | 3 | 26.79 | −28.12 | 0.000 |
| Zhujia | Direct COT, hybrid plus amplitude term | 3 | 26.76 | −28.03 | 0.000 |

来源：refine-logs/cot_retrieved_forecast_20261013/results/event_cohort_decomposition_20261017/EVENT_COHORT_DECOMPOSITION_20261017.md

## P081 离线位置对齐：错位还是厚度演变？｜Event-equal scores

本页意义：使用未来参考作位置对齐仅用于归因；对齐后误差不作为预报成绩。

评价范围：湖南未来COT；不可部署离线诊断；已完成

| Method | Event-equal patch RMSE | Event-equal local 3×3 RMSE | Pooled thick CSI | Event-equal centroid shift (px; both detect) | Event-equal local peak error |
| --- | --- | --- | --- | --- | --- |
| SimVP_to_R | 8.197 | 9.467 | 0.006 | 3.797 | -17.815 |
| P_Hybrid | 8.025 | 9.119 | 0.000 | — | -17.942 |
| P_Hybrid_amp01 | 8.181 | 9.219 | 0.000 | — | -17.952 |
| future_observed_AGRI_to_R_oracle | 3.526 | 2.293 | 0.572 | 2.042 | -3.374 |
| Persistence_R | 7.799 | 8.560 | 0.102 | 8.836 | -19.210 |

来源：refine-logs/cot_retrieved_forecast_20261013/results/position_amplitude_events_20261017/EVENT_POSITION_AMPLITUDE_DIAGNOSTIC_20261017.md

## P082 时间变化：预测COT轨迹是否过平？｜配套图 32

本页意义：与真实未来AGRI经同R得到的代理轨迹比较，预测变化相关但幅度衰减；代理不是物理真值。

评价范围：湖南52个目标日期；R到R代理对照；已完成

![](figures/E27_032.png)

来源：refine-logs/cot_value_first_20261013/results/cot_temporal_change_audit_20261017/cot_temporal_change_scatter.png

## P083 固定GHI头2×2替换：图像与COT误差如何传递？｜全验证窗结果

本页意义：分别替换预测AGRI与预测COT，定量诊断观测输入头面对预测输入的误差；不是预测输入重训成绩。

评价范围：湖南2,817序列、52日期；离线固定头归因；已完成

| 输入组合 | GHI RMSE (W m⁻²) |
| --- | --- |
| TT：真实 AGRI + 同期 R-COT | 118.14 |
| FT：预测 AGRI + 同期 R-COT | 126.46 |
| TF：真实 AGRI + 预测 COT | 125.42 |
| FF：预测 AGRI + 预测 COT | 139.39 |

来源：refine-logs/cot_value_first_20261013/results/fixed_head_2x2_allval_agri_cot_attribution_20261014_v1/ALLVAL_FIXED_HEAD_ATTRIBUTION_20261014.md

## P084 固定GHI头2×2替换：图像与COT误差如何传递？｜全验证窗结果

本页意义：分别替换预测AGRI与预测COT，定量诊断观测输入头面对预测输入的误差；不是预测输入重训成绩。

评价范围：湖南2,817序列、52日期；离线固定头归因；已完成

| 配对替换（新−基准） | ΔRMSE (W m⁻²) | BJT 日期块 bootstrap 95% CI |
| --- | --- | --- |
| AGRI：FT−TT（同期 COT） | +8.32 | [+3.01, +14.14] |
| COT：TF−TT（真实 AGRI） | +7.28 | [+4.52, +10.54] |
| COT：FF−FT（预测 AGRI） | +12.93 | [+9.73, +16.58] |
| AGRI：FF−TF（预测 COT） | +13.97 | [+7.84, +20.69] |
| 联合：FF−TT | +21.26 | [+14.22, +29.14] |

来源：refine-logs/cot_value_first_20261013/results/fixed_head_2x2_allval_agri_cot_attribution_20261014_v1/ALLVAL_FIXED_HEAD_ATTRIBUTION_20261014.md

## P085 历史COT输运：与Cloud-FNO文献如何比较？｜全云况物理 COT RMSE

本页意义：保留站点像元的1～4小时误差；区域、分辨率及参考不同，只作跨研究描述，不排名。

评价范围：旧验证集历史输运；文献描述性参照；已完成

| 范围/方法 | +1 h | +2 h | +3 h | +4 h |
| --- | --- | --- | --- | --- |
| paper_image / Cloud-FNO | 6.750 | 8.090 | 8.930 | 9.610 |
| patch / Persistence | 7.663 | 9.969 | 11.175 | 11.987 |
| patch / Transport (selected) | 7.419 | 9.799 | 11.179 | 11.988 |
| patch / 8-frame hybrid (candidate) | 8.428 | 9.869 | 10.969 | 11.019 |
| station / Persistence | 6.633 | 8.515 | 10.242 | 10.477 |
| station / Transport (selected) | 6.151 | 8.909 | 10.646 | 11.133 |
| station / 8-frame hybrid (candidate) | 7.155 | 9.037 | 9.622 | 9.497 |

来源：results/history_cot_cloud_fno_20261015/REPORT.md

## P086 历史COT输运：与Cloud-FNO文献如何比较？｜配套图 33

本页意义：保留站点像元的1～4小时误差；区域、分辨率及参考不同，只作跨研究描述，不排名。

评价范围：旧验证集历史输运；文献描述性参照；已完成

![](figures/E29_033.png)

来源：results/history_cot_cloud_fno_20261015/hourly_comparison.png

## P087 历史COT预报GHI：修正选模流程后的配对结果｜Overall result (all 16 leads pooled)

本页意义：使用日期留出选模后比较AGRI、历史COT输运及混合输入；该结果不能与同期3%混合。

评价范围：湖南旧COT输运；attempt3日期留出；已完成

| Arm | No-COT RMSE | COT RMSE | Delta (COT - A), W/m^2 | RMSE change (positive = worse) | Date-block 95% CI for delta |
| --- | --- | --- | --- | --- | --- |
| History-transport COT | 158.76 | 167.22 | +8.46 | +5.33% | [+2.55, +15.49] |
| History + hybrid COT | 158.76 | 166.79 | +8.04 | +5.06% | [+2.35, +14.76] |

来源：refine-logs/history_cot_forecast_ghi_20261015/attempt3_results.md

## P088 历史COT预报GHI：修正选模流程后的配对结果｜Selected leads

本页意义：使用日期留出选模后比较AGRI、历史COT输运及混合输入；该结果不能与同期3%混合。

评价范围：湖南旧COT输运；attempt3日期留出；已完成

| Lead | Transport COT delta | Transport RMSE change | Hybrid COT delta | Hybrid RMSE change |
| --- | --- | --- | --- | --- |
| +1 h | -0.82 | -0.53% | -1.83 | -1.18% |
| +2 h | +5.87 | +3.83% | +14.74 | +9.60% |
| +3 h | +6.63 | +4.35% | +11.65 | +7.65% |
| +4 h | +20.48 | +11.51% | +11.31 | +6.36% |

来源：refine-logs/history_cot_forecast_ghi_20261015/attempt3_results.md

## P089 历史COT预报GHI：修正选模流程后的配对结果｜配套图 34

本页意义：使用日期留出选模后比较AGRI、历史COT输运及混合输入；该结果不能与同期3%混合。

评价范围：湖南旧COT输运；attempt3日期留出；已完成

![](figures/E30_034.png)

来源：refine-logs/history_cot_forecast_ghi_20261015/forecast_ghi_cot_bar_attempt3_zh.png

## P090 历史COT预报GHI：修正选模流程后的配对结果｜配套图 35

本页意义：使用日期留出选模后比较AGRI、历史COT输运及混合输入；该结果不能与同期3%混合。

评价范围：湖南旧COT输运；attempt3日期留出；已完成

![](figures/E30_035.png)

来源：refine-logs/history_cot_forecast_ghi_20261015/forecast_ghi_rmse_by_lead_attempt3_zh.png

## P091 湖北未来GHI：预测COT是否带来净增益？｜汇总结果

本页意义：预测AGRI基线77.150、加入预测COT78.155 W/m²；差异区间跨零，未显示预报净增益。

评价范围：湖北三站；22验证日期；最新归一化重载；已完成

| 站点 | FA | FAC | FAC−FA |
| --- | --- | --- | --- |
| 牛首 | 76.49 | 77.27 | +0.78 |
| 王寨 | 72.11 | 73.85 | +1.74 |
| 米庄 | 82.85 | 83.35 | +0.49 |

来源：refine-logs/hubei_cot_independent_20261014/FORECAST_GHI_NORMRELOAD_THREE_SEED_VALIDATION_20261017.md

## P092 湖北未来GHI：预测COT是否带来净增益？｜配套图 36

本页意义：预测AGRI基线77.150、加入预测COT78.155 W/m²；差异区间跨零，未显示预报净增益。

评价范围：湖北三站；22验证日期；最新归一化重载；已完成

![](figures/E31_036.png)

来源：refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_three_seed_normreload_20261017/figures/validation_per_lead_FA_FAC.png

## P093 湖南最新未来GHI：完整预测COT与收缩输入｜比较回答的问题

本页意义：最新三种子screen中总体差异很小且区间跨零；训练forecast为in-sample，五折OOF复验尚未完成。

评价范围：2026-10-19湖南未来GHI筛选；非最终预报结论；已完成

| 组别 | 两站等权 GHI RMSE (W/m²) | 相对 AH | 相对 AC |
| --- | --- | --- | --- |
| A：无 COT | 129.323 | -0.073 | +0.154 |
| AH：仅历史 COT | 129.396 | — | +0.227 |
| AC：历史 + 完整预测 COT | 129.169 | -0.227 | — |
| AC50：历史 + 50% 未来 COT 异常 | 129.168 | -0.229 | -0.001 |

来源：refine-logs/cot_aware_simvp_20261017/FUTURE_GHI_COT_PILOT_DIAGNOSIS_20261019.md

## P094 湖南最新未来GHI：完整预测COT与收缩输入｜分层线索

本页意义：最新三种子screen中总体差异很小且区间跨零；训练forecast为in-sample，五折OOF复验尚未完成。

评价范围：2026-10-19湖南未来GHI筛选；非最终预报结论；已完成

| 站点 | 预测 COT 层 | AC−AH：+15–60 min | +75–120 min | +135–240 min |
| --- | --- | --- | --- | --- |
| 四里 | <1 | -1.77 | -1.36 | -1.16 |
| 四里 | 1–10 | +0.48 | -0.33 | -3.73 |
| 四里 | 10–30 | +0.24 | -0.71 | +0.71 |
| 四里 | ≥30 | -1.74 | -2.91 | +1.35 |
| 竺家 | <1 | -1.52 | +1.51 | +1.77 |
| 竺家 | 1–10 | +1.95 | +0.93 | +0.28 |
| 竺家 | 10–30 | -2.99 | -1.71 | -0.23 |
| 竺家 | ≥30 | -1.97 | -1.60 | -1.53 |

来源：refine-logs/cot_aware_simvp_20261017/FUTURE_GHI_COT_PILOT_DIAGNOSIS_20261019.md

## P095 湖南最新未来GHI：完整预测COT与收缩输入｜固定头对齐敏感度审计

本页意义：最新三种子screen中总体差异很小且区间跨零；训练forecast为in-sample，五折OOF复验尚未完成。

评价范围：2026-10-19湖南未来GHI筛选；非最终预报结论；已完成

| 固定头 | 替换的未来COT | 全时效 RMSE 差（W/m²） | 95%日期块区间 |
| --- | --- | --- | --- |
| AC | 置零 | +31.945 | [20.844, 43.052] |
| AC | 分层内置换 | +22.493 | [14.583, 32.099] |
| AC50 | 置零 | +15.185 | [7.914, 22.854] |
| AC50 | 分层内置换 | +12.975 | [8.004, 19.110] |

来源：refine-logs/cot_aware_simvp_20261017/FUTURE_GHI_COT_PILOT_DIAGNOSIS_20261019.md

## P096 监督来源：CPP与R伪标签是否等价？｜共同有效像元上的来源差异

本页意义：共同均值接近掩盖近零虚警与厚云压缩；R回退区与CPP覆盖区的云况构成也不同。

评价范围：2026-10-19验证抽样1,024窗口；参考差异诊断；已完成

| 站点 | 像元数 | CPP均值 | AGRI→R均值 | R−CPP偏差 | MAE | RMSE | CPP厚云比例≥30 | R厚云比例≥30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 四里 | 822,840 | 12.49 | 12.77 | +0.28 | 5.16 | 9.37 | 12.79% | 11.90% |
| 竺家 | 820,872 | 9.49 | 9.73 | +0.24 | 3.90 | 8.53 | 13.15% | 13.09% |
| 合并像元 | 1,643,712 | 10.99 | 11.25 | +0.26 | 4.53 | 8.96 | 12.97% | 12.50% |

来源：refine-logs/cot_aware_simvp_20261017/results/label_source_disagreement_summary.md

## P097 监督来源：CPP与R伪标签是否等价？｜CPP覆盖区与R回退区的分布

本页意义：共同均值接近掩盖近零虚警与厚云压缩；R回退区与CPP覆盖区的云况构成也不同。

评价范围：2026-10-19验证抽样1,024窗口；参考差异诊断；已完成

| 站点 | 区域 | R目标均值 | 中位数 | 厚云比例≥30 | 近零比例<1 |
| --- | --- | --- | --- | --- | --- |
| 四里 | CPP有效覆盖区 | 12.77 | 9.24 | 11.90% | 26.54% |
| 四里 | R回退区 | 16.92 | 13.94 | 22.42% | 24.65% |
| 竺家 | CPP有效覆盖区 | 9.73 | 2.27 | 13.09% | 45.72% |
| 竺家 | R回退区 | 16.27 | 11.50 | 25.07% | 29.71% |

来源：refine-logs/cot_aware_simvp_20261017/results/label_source_disagreement_summary.md

## P098 监督来源：CPP与R伪标签是否等价？｜按CPP参考云厚分层的R偏差

本页意义：共同均值接近掩盖近零虚警与厚云压缩；R回退区与CPP覆盖区的云况构成也不同。

评价范围：2026-10-19验证抽样1,024窗口；参考差异诊断；已完成

| CPP参考COT区间 | 像元数 | CPP均值 | R均值 | R−CPP偏差 | MAE | RMSE | R落入近零COT<1 | R仍判厚云≥30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <1 | 650,079 | 0.036 | 3.524 | +3.489 | 3.526 | 10.437 | 81.87% | 5.04% |
| 1–<10 | 406,583 | 5.406 | 5.337 | −0.069 | 2.002 | 2.905 | 14.51% | 0.06% |
| 10–<30 | 373,872 | 18.068 | 18.383 | +0.314 | 5.243 | 6.876 | 0.51% | 9.96% |
| ≥30 | 213,178 | 42.636 | 33.617 | −9.020 | 11.191 | 13.703 | 0.26% | 63.42% |

来源：refine-logs/cot_aware_simvp_20261017/results/label_source_disagreement_summary.md

## P099 云与厚云检出：FAR、POD、CSI分别说明什么？｜结果

本页意义：回归误差与云检出应同时报告：POD看漏报，FAR看虚警，CSI综合命中、漏报与虚警；分母与阈值按原报告。

评价范围：湖南反演验证；CPP-COT阈值识别；已完成

| 站点/汇总 | 识别对象 | 有效像元 | 参考正例 | TP / FP / FN | POD | Precision | FAR | CSI | 频率偏差 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 合并 | COT ≥ 0.1 | 356,169 | 227,922 | 218,215 / 32,343 / 9,707 | 0.957 | 0.871 | 0.129 | 0.838 | 1.099 |
| 四里 | COT ≥ 0.1 | 178,019 | 130,172 | 127,965 / 15,226 / 2,207 | 0.983 | 0.894 | 0.106 | 0.880 | 1.100 |
| 竺家 | COT ≥ 0.1 | 178,150 | 97,750 | 90,250 / 17,117 / 7,500 | 0.923 | 0.841 | 0.159 | 0.786 | 1.098 |
| 合并 | COT ≥ 30 | 356,169 | 50,924 | 34,499 / 17,526 / 16,425 | 0.677 | 0.663 | 0.337 | 0.504 | 1.022 |
| 四里 | COT ≥ 30 | 178,019 | 26,211 | 16,410 / 11,525 / 9,801 | 0.626 | 0.587 | 0.413 | 0.435 | 1.066 |
| 竺家 | COT ≥ 30 | 178,150 | 24,713 | 18,089 / 6,001 / 6,624 | 0.732 | 0.751 | 0.249 | 0.589 | 0.975 |

来源：refine-logs/cot_value_first_20261013/COT_CLOUD_DETECTION_VALIDATION_20261013.md

## P100 云与厚云检出：FAR、POD、CSI分别说明什么？｜配套图 37

本页意义：回归误差与云检出应同时报告：POD看漏报，FAR看虚警，CSI综合命中、漏报与虚警；分母与阈值按原报告。

评价范围：湖南反演验证；CPP-COT阈值识别；已完成

![](figures/E34_037.png)

来源：docs/组会原始材料_20261016/02_厚度分层/COT云与厚云检出指标_R验证.png

## P101 数据划分：每个月的训练和验证覆盖如何？｜配套图 38

本页意义：明确连续日期留出与缓冲，解释四季覆盖；这页是结果评价的样本定义，不是模型技能。

评价范围：新月度日期划分；test仅路径和日期元数据；已完成

![](figures/E35_038.png)

来源：experiments/monthly_block_20261002/manifests_v1/monthly_split_counts_bar_zh.png

## P102 数据划分：每个月的训练和验证覆盖如何？｜配套图 39

本页意义：明确连续日期留出与缓冲，解释四季覆盖；这页是结果评价的样本定义，不是模型技能。

评价范围：新月度日期划分；test仅路径和日期元数据；已完成

![](figures/E35_039.png)

来源：experiments/monthly_block_20261002/manifests_v1/monthly_split_timeline_zh.png

## P103 历史探索：空间COT初筛｜原始结果

本页意义：早期空间图输入只带来很小点改善，属于旧数据和旧头部探索。

评价范围：历史划分/模型；不进入当前主成绩；历史探索

| 组 | 最佳epoch | Sili RMSE | Zhujia RMSE | 等权RMSE | 相对 A_sp |
| --- | --- | --- | --- | --- | --- |
| A_sp | 7 | 164.0630 | 133.7177 | 148.8904 | 0 |
| AC_stat_sp | 7 | 164.4489 | 135.2100 | 149.8294 | +0.9391 W/m²（+0.631%） |
| AC_map | 7 | 165.0922 | 132.2524 | 148.6723 | delta=-0.2181 W/m²（-0.146%） |

来源：docs/SPATIAL_B1_PILOT_RESULTS_20260917.md

## P104 历史探索：空间排列控制｜Results

本页意义：检查空间置换与容量因素；不得据早期头部结论否定全部空间表示。

评价范围：历史划分/模型；不进入当前主成绩；历史探索

| group | best epoch | Sili RMSE (W/m²) | Zhujia RMSE (W/m²) | equal-station RMSE (W/m²) |
| --- | --- | --- | --- | --- |
| AC_map (B1) | 7 | 165.0922 | 132.2524 | 148.6723 |
| AC_shuffle | 7 | 165.0939 | 132.2829 | 148.6884 |
| A_proxy | 7 | 165.3506 | 132.6464 | 148.9985 |
| A_sp (B1) | 7 | 164.0630 | 133.7177 | 148.8904 |

来源：docs/SPATIAL_B2_CONTROLS_RESULTS_20260917.md

## P105 历史探索：AFNO固定GHI头替换｜AFNO COT paired frozen-GHI-head comparison (2026-09-23)

本页意义：替换外推器输入用于检查分布变化；不等于各自匹配训练后的优劣。

评价范围：历史划分/模型；不进入当前主成绩；历史探索

| Station | Head | Rows | SimVP RMSE | AFNO RMSE | Δ (AFNO−SimVP) | 95% day-bootstrap CI |
| --- | --- | --- | --- | --- | --- | --- |
| Sili | AC_F | 49,806 | 166.401 | 166.307 | -0.094 | [-0.511, +0.371] |
| Sili | AC_ST | 49,806 | 165.364 | 165.401 | +0.037 | [-0.249, +0.333] |
| Zhujia | AC_F | 50,043 | 140.691 | 140.550 | -0.141 | [-0.343, +0.043] |
| Zhujia | AC_ST | 50,043 | 137.423 | 137.185 | -0.238 | [-0.772, +0.235] |

来源：results/afno_fixed_ghi_head_swap_20260923/README.md

## P106 历史探索：AFNO固定GHI头替换｜配套图 40

本页意义：替换外推器输入用于检查分布变化；不等于各自匹配训练后的优劣。

评价范围：历史划分/模型；不进入当前主成绩；历史探索

![](figures/E38_040.png)

来源：results/afno_fixed_ghi_head_swap_20260923/paired_rmse_delta.png

## P107 历史探索：输运与厚度残差｜Best validation checkpoint

本页意义：旧输运与学习厚度残差比较，不与月度新划分合并。

评价范围：历史划分/模型；不进入当前主成绩；历史探索

| Method | All-valid COT RMSE | All-valid MAE | COT≥30 RMSE | COT≥30 misses predicted <10 | COT≥30 IoU | Clear false-COT rate |
| --- | --- | --- | --- | --- | --- | --- |
| SimVP→R | 9.666 | 5.537 | 27.700 | 387,738 | 0.1806 | 0.0438 |
| Persistence | 9.609 | 5.288 | 27.078 | 346,485 | 0.2259 | 0.0567 |
| Seven-pair transport | 9.520 | 5.287 | 26.504 | 334,790 | 0.2411 | 0.0581 |
| Learned transport + thickness | 9.056 | 4.900 | 28.279 | 371,192 | 0.1664 | 0.0311 |

来源：docs/SILI_TRANSPORT_THICKNESS_PILOT_RESULTS_20260928.md

## P108 历史探索：输运与厚度残差｜Sili station-pixel validation audit (2026-09-28)

本页意义：旧输运与学习厚度残差比较，不与月度新划分合并。

评价范围：历史划分/模型；不进入当前主成绩；历史探索

| CPP COT regime | n | Transport RMSE | Learned RMSE | Learned minus transport RMSE, 95% paired day-bootstrap CI |
| --- | --- | --- | --- | --- |
| all valid station leads | 5,664 | 8.738 | 8.049 | -0.689 [-1.589, 0.106] |
| 0–5 | 3,530 | 4.654 | 3.187 | -1.466 [-2.447, -0.607] |
| 5–10 | 830 | 10.177 | 7.868 | -2.309 [-5.705, 0.888] |
| 10–30 | 1,100 | 11.317 | 10.015 | -1.302 [-3.095, 0.687] |
| 30–60 | 186 | 22.993 | 26.051 | +3.058 [+0.781, +4.977] |
| 60–100 | 18 | 41.385 | 48.971 | +7.586 [+4.149, +11.999] |

来源：docs/SILI_TRANSPORT_THICKNESS_PILOT_RESULTS_20260928.md

## P109 历史探索：输运与厚度残差｜Full local-64 validation audit completed (2026-09-28, supersedes submission snapshot above)

本页意义：旧输运与学习厚度残差比较，不与月度新划分合并。

评价范围：历史划分/模型；不进入当前主成绩；历史探索

| CPP COT regime | n | Transport RMSE | Learned RMSE | Transport bias | Learned bias |
| --- | --- | --- | --- | --- | --- |
| 0–5 | 11,359,870 | 4.766 | 3.029 | +1.418 | +0.793 |
| 5–10 | 3,719,564 | 7.847 | 6.339 | +0.835 | +0.113 |
| 10–30 | 4,855,014 | 10.873 | 10.212 | −2.522 | −3.692 |
| 30–60 | 1,116,315 | 23.853 | 25.502 | −16.754 | −20.972 |
| 60–100 | 102,488 | 46.441 | 49.260 | −39.973 | −46.307 |
| ≥30 | 1,218,803 | 26.504 | 28.279 | −18.707 | −23.102 |

来源：docs/SILI_TRANSPORT_THICKNESS_PILOT_RESULTS_20260928.md

## P110 历史探索：输运与厚度残差｜Low-COT penalty ablation completed (2026-09-28)

本页意义：旧输运与学习厚度残差比较，不与月度新划分合并。

评价范围：历史划分/模型；不进入当前主成绩；历史探索

| Candidate | Extra low-COT penalty | Epoch | Thick COT RMSE | All-valid COT RMSE | Thick COT IoU |
| --- | --- | --- | --- | --- | --- |
| Step-0 transport | n/a | 0 | 26.504 | 9.520 | 0.2411 |
| Best learned epoch under penalty-on arm | on | 1 | 28.279 | 9.056 | 0.1664 |
| Best learned epoch under penalty-off arm | off | 3 | 26.856 | 9.353 | 0.2307 |

来源：docs/SILI_TRANSPORT_THICKNESS_PILOT_RESULTS_20260928.md

## P111 历史探索：完整历史与重复末帧｜结果

本页意义：历史通道堆叠未获得稳定收益；并非显式云轨迹建模的否定证据。

评价范围：历史划分/模型；不进入当前主成绩；历史探索

| 候选（patch厚云指标选出的验证快照） | Patch全域RMSE | Patch COT≥30 RMSE | Patch厚云漏报率 | Patch晴空误报率 | 站点全域RMSE | 站点COT≥30 RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| Step-0纯输运（两组均选中） | 9.520 | 26.504 | 27.47% | 5.81% | 8.738 | 25.163 |
| Repeat-last最佳学习头（update 160） | 9.440 | 27.209 | 28.01% | 3.52% | 8.192 | 27.606 |
| Full-history最佳学习头（update 96） | 9.603 | 27.143 | 27.10% | 3.88% | 8.325 | 27.609 |

来源：docs/SILI_FULL_COT_HISTORY_PILOT_RESULTS_20260928.md

## P112 历史探索：完整历史与重复末帧｜配套图 41

本页意义：历史通道堆叠未获得稳定收益；并非显式云轨迹建模的否定证据。

评价范围：历史划分/模型；不进入当前主成绩；历史探索

![](figures/E40_041.png)

来源：results/regional_station_cot_20260927/sili_full_vs_repeat_last_history_20260928_v5/history_control_validation.png

## P113 历史探索：低COT惩罚消融｜Outcome

本页意义：关闭低COT惩罚后仍未超过输运，说明该项不足以解释全部退化。

评价范围：历史划分/模型；不进入当前主成绩；历史探索

| Candidate on fixed validation bank | Thick COT RMSE | All-regime RMSE | Thick misses (<10) | COT≥30 IoU | False-COT rate (CPP COT<5) |
| --- | --- | --- | --- | --- | --- |
| Step-0 transport, shared baseline | 26.504 | 9.520 | 334,790 | 0.2411 | 0.05814 |
| Control, best learned epoch 1 | 28.279 | 9.056 | 371,192 | 0.1664 | 0.03107 |
| Penalty-off, best learned epoch 3 | 26.856 | 9.353 | 328,427 | 0.2307 | 0.03694 |

来源：docs/SILI_LOW_COT_PENALTY_ABLATION_RESULTS_20260928.md

## P114 历史探索：历史信息探针｜Results

本页意义：固定模型输入替换检查历史是否被使用；属于依赖性诊断。

评价范围：历史划分/模型；不进入当前主成绩；历史探索

| Validation scope / metric | Recent two frames | All eight frames | Full-history minus recent |
| --- | --- | --- | --- |
| Lattice all-COT RMSE | 8.942 | 8.936 | −0.007 |
| Lattice COT≥30 RMSE | 28.991 | 28.948 | −0.042 |
| Lattice COT≥30 IoU | 0.1502 | 0.1490 | −0.0012 |
| Lattice thick miss rate (prediction<10) | 33.33% | 32.98% | −0.35 pp |
| Lattice false-COT rate (CPP<5, prediction≥10) | 2.594% | 2.655% | +0.061 pp |
| Exact station pixel all-COT RMSE | 7.644 | 7.626 | −0.017 |
| Exact station pixel COT≥30 RMSE | 29.788 | 29.656 | −0.132 |
| Exact station thick records | 204 | 204 | same rows |

来源：docs/SILI_HISTORY_INFORMATION_PROBE_RESULTS_20260928.md

## P115 历史探索：历史信息探针｜配套图 42

本页意义：固定模型输入替换检查历史是否被使用；属于依赖性诊断。

评价范围：历史划分/模型；不进入当前主成绩；历史探索

![](figures/E42_042.png)

来源：results/regional_station_cot_20260927/sili_history_information_probe_20260928_v2/history_information_probe.png

## P116 历史探索：竺家残差容量对照｜Results

本页意义：历史残差容量匹配结果用于追溯路线，不替代统一输入新对照。

评价范围：历史划分/模型；不进入当前主成绩；历史探索

| Candidate selected on validation | Best step | RMSE (W m⁻²) | MAE (W m⁻²) | Bias (W m⁻²) |
| --- | --- | --- | --- | --- |
| Frozen no-COT anchor | 0 | 127.2954 | 93.0143 | −11.7035 |
| AGRI-only residual | 276 | 126.3232 | 92.0802 | −18.7875 |
| AGRI+COT residual | 0 (fallback) | 127.2954 | 93.0143 | −11.7035 |

来源：docs/ZHUJIA_RESIDUAL_CAPACITY_CONTROL_RESULTS_20260927.md

## P117 历史探索：旧天气分层｜排除项

本页意义：保留天气与站点差异的早期描述；不同样本和模型不能直接纵向归因。

评价范围：历史划分/模型；不进入当前主成绩；历史探索

| 检查 | 证据 | 判断 |
| --- | --- | --- |
| 天气分类混淆 | 四里预测晴天3046行中，2415行参考晴天、524行参考多云、107行参考阴天；只取参考晴天后QC1仍为+13.11 W/m²且3/3 seed退化 | 不是主要原因 |
| 晨昏小晴空辐照度 | clear-sky GHI<20仅156行，delta +0.24；200–600为+10.64，≥600为+15.00 | 不是晨昏小分母驱动 |
| 单一时效 | +15到+240分钟的16个时效全部退化，均值范围+8.48至+14.99 | 不是个别lead |
| 少数异常样本 | 每个seed中71.3%–75.2%的行平方误差增大；最坏5%只解释净增量的21.9%–34.3% | 退化分布广泛 |
| COT极端尾部 | 四个COT特征各自按四分位切分后，QC1所有16个分箱均为3/3 seed退化 | 不是仅由离群尾部造成 |
| 源缺测填晴 | 天气参考标签缺测从不当晴天；四维COT输入来自冻结R预测且全部有限，不使用源CPP缺测填充值 | 未发现该错误 |
| 站点像元位置 | 既有独立空间审计确认四里、竺家最近站点像元均为patch [8,8]；四里中心距站点约1.81 km | 未发现索引错误 |

来源：docs/STATION_WEATHER_COT_DIAGNOSIS_20260912.md

## P118 10月19日最终复算：CPP/R混合辅助监督能否保住云厚？｜站点中心3×3

本页意义：同一CPP有效像元：总体RMSE未超过原始基线；CSI小改善与近零虚警增大并存，候选未通过预定筛选。

评价范围：3,140验证序列；46个CPP目标日期；seed42；COT无量纲；站点中心3×3

| 方法 | COT RMSE | 厚云CSI（≥30） | 近零虚警率（参考<1、预测≥1） |
| --- | --- | --- | --- |
| 原始SimVP→R | 10.664161 | 0.422642 | 0.178334 |
| 等预算MSE微调→R | 10.750112 | 0.430586 | 0.180439 |
| CPP/R混合辅助监督→R | 10.740627 | 0.432519 | 0.181590 |

来源：sources/remote_20261019/mixed_independent_cot_audit.json

## P119 10月19日不确定性：CPP/R混合辅助监督的RMSE差是否明确？｜站点中心3×3

本页意义：区间跨零，未证实COT RMSE净改善；厚云CSI与虚警应同时检查，不能仅挑有利指标。

评价范围：同样本、同日期配对10,000次bootstrap；不是独立test

| 比较 | ΔRMSE（候选−对照） | 配对目标日期95%区间 |
| --- | --- | --- |
| 候选−原始SimVP | +0.076466 | [-0.095319, +0.246425] |
| 候选−等预算MSE | -0.009485 | [-0.020354, -0.000953] |

来源：sources/remote_20261019/mixed_independent_cot_audit.json

## P120 10月19日最终复算：CPP/R混合辅助监督能否保住云厚？｜完整16×16

本页意义：同一CPP有效像元：总体RMSE未超过原始基线；CSI小改善与近零虚警增大并存，候选未通过预定筛选。

评价范围：3,140验证序列；46个CPP目标日期；seed42；COT无量纲；完整16×16

| 方法 | COT RMSE | 厚云CSI（≥30） | 近零虚警率（参考<1、预测≥1） |
| --- | --- | --- | --- |
| 原始SimVP→R | 10.952694 | 0.413997 | 0.195983 |
| 等预算MSE微调→R | 11.038903 | 0.423894 | 0.199224 |
| CPP/R混合辅助监督→R | 11.028750 | 0.425167 | 0.200309 |

来源：sources/remote_20261019/mixed_independent_cot_audit.json

## P121 10月19日不确定性：CPP/R混合辅助监督的RMSE差是否明确？｜完整16×16

本页意义：区间跨零，未证实COT RMSE净改善；厚云CSI与虚警应同时检查，不能仅挑有利指标。

评价范围：同样本、同日期配对10,000次bootstrap；不是独立test

| 比较 | ΔRMSE（候选−对照） | 配对目标日期95%区间 |
| --- | --- | --- |
| 候选−原始SimVP | +0.076056 | [-0.096210, +0.245052] |
| 候选−等预算MSE | -0.010154 | [-0.020472, -0.002107] |

来源：sources/remote_20261019/mixed_independent_cot_audit.json

## P122 10月19日最终复算：CPP-only辅助监督能否保住云厚？｜站点中心3×3

本页意义：同一CPP有效像元：总体RMSE未超过原始基线；CSI小改善与近零虚警增大并存，候选未通过预定筛选。

评价范围：3,140验证序列；46个CPP目标日期；seed42；COT无量纲；站点中心3×3

| 方法 | COT RMSE | 厚云CSI（≥30） | 近零虚警率（参考<1、预测≥1） |
| --- | --- | --- | --- |
| 原始SimVP→R | 10.664161 | 0.422642 | 0.178334 |
| 等预算MSE微调→R | 10.750111 | 0.430586 | 0.180439 |
| CPP-only辅助监督→R | 10.743569 | 0.433654 | 0.182165 |

来源：sources/remote_20261019/cpp_only_independent_cot_audit.json

## P123 10月19日不确定性：CPP-only辅助监督的RMSE差是否明确？｜站点中心3×3

本页意义：区间跨零，未证实COT RMSE净改善；厚云CSI与虚警应同时检查，不能仅挑有利指标。

评价范围：同样本、同日期配对10,000次bootstrap；不是独立test

| 比较 | ΔRMSE（候选−对照） | 配对目标日期95%区间 |
| --- | --- | --- |
| 候选−原始SimVP | +0.079409 | [-0.105936, +0.260568] |
| 候选−等预算MSE | -0.006542 | [-0.030278, +0.012902] |

来源：sources/remote_20261019/cpp_only_independent_cot_audit.json

## P124 10月19日最终复算：CPP-only辅助监督能否保住云厚？｜完整16×16

本页意义：同一CPP有效像元：总体RMSE未超过原始基线；CSI小改善与近零虚警增大并存，候选未通过预定筛选。

评价范围：3,140验证序列；46个CPP目标日期；seed42；COT无量纲；完整16×16

| 方法 | COT RMSE | 厚云CSI（≥30） | 近零虚警率（参考<1、预测≥1） |
| --- | --- | --- | --- |
| 原始SimVP→R | 10.952694 | 0.413999 | 0.195983 |
| 等预算MSE微调→R | 11.038903 | 0.423892 | 0.199224 |
| CPP-only辅助监督→R | 11.031193 | 0.426772 | 0.200659 |

来源：sources/remote_20261019/cpp_only_independent_cot_audit.json

## P125 10月19日不确定性：CPP-only辅助监督的RMSE差是否明确？｜完整16×16

本页意义：区间跨零，未证实COT RMSE净改善；厚云CSI与虚警应同时检查，不能仅挑有利指标。

评价范围：同样本、同日期配对10,000次bootstrap；不是独立test

| 比较 | ΔRMSE（候选−对照） | 配对目标日期95%区间 |
| --- | --- | --- |
| 候选−原始SimVP | +0.078499 | [-0.106713, +0.259042] |
| 候选−等预算MSE | -0.007710 | [-0.031091, +0.011432] |

来源：sources/remote_20261019/cpp_only_independent_cot_audit.json

## P126 湖北最新完整预报对照：COT与容量分支都有效吗？

本页意义：包含10月18日补齐的容量臂；三组比较均未显示明确未来GHI改善。

评价范围：湖北三站、三种子、22目标日期；最新归一化重载；不是test

| 未来GHI输入 | 三站三种子等权RMSE（W/m²） | 性质 |
| --- | --- | --- |
| 预测AGRI | 77.150230 | 同一validation选模期评价 |
| 预测AGRI＋预测COT | 78.155269 | 同一validation选模期评价 |
| 预测AGRI＋容量对照 | 78.979810 | 同一validation选模期评价 |

来源：refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_fap_three_seed_20261018/forecast_fap_three_seed_validation_recheck.json

## P127 湖北未来GHI不确定性：预报收益尚未得到支持

本页意义：区间跨零既不证明改善，也不证明三种方法等效；保留全部正负结果。

评价范围：10,000次配对目标日期bootstrap；Δ为前者减后者

| 配对比较 | ΔRMSE（W/m²） | 日期块95%区间 |
| --- | --- | --- |
| FAC − FA | +1.005039 | [-0.721941, +3.779833] |
| FAP − FA | +1.829580 | [-0.517109, +4.778163] |
| FAP − FAC | +0.824541 | [-2.657295, +4.402145] |

来源：refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_fap_three_seed_20261018/forecast_fap_three_seed_validation_recheck.json

## P128 最新服务器权重：COT辅助与未来GHI筛选 1

本页意义：本次通过SSH读取文件目录和大小；step4000用于已完成辅助监督，OOF过程权重不当作最终模型。

评价范围：共同根目录见REMOTE_CHECKPOINT_INDEX.tsv绝对路径；本次SSH目录快照

| 远端权重（共同根目录之后） | MiB | 远端修改时间 |
| --- | --- | --- |
| experiments/cot_aware_simvp_20261017/results/seed42_cot_aux_cpp_only_v1/step_0000.pt | 42.0 | 2026-10-19T04:43:18 |
| experiments/cot_aware_simvp_20261017/results/seed42_cot_aux_cpp_only_v1/step_2000.pt | 125.8 | 2026-10-19T05:18:47 |
| experiments/cot_aware_simvp_20261017/results/seed42_cot_aux_cpp_only_v1/step_4000.pt | 125.8 | 2026-10-19T05:53:32 |
| experiments/cot_aware_simvp_20261017/results/seed42_cot_aux_v1/step_0000.pt | 42.0 | 2026-10-19T00:46:38 |
| experiments/cot_aware_simvp_20261017/results/seed42_cot_aux_v1/step_2000.pt | 125.8 | 2026-10-19T03:18:50 |
| experiments/cot_aware_simvp_20261017/results/seed42_cot_aux_v1/step_4000.pt | 125.8 | 2026-10-19T04:12:47 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_source_heads_v1/zhujia/seed42/A/best.pt | 0.2 | 2026-10-19T10:02:14 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_source_heads_v1/zhujia/seed42/AH/best.pt | 0.2 | 2026-10-19T10:02:36 |

来源：REMOTE_CHECKPOINT_INDEX.tsv

## P129 最新服务器权重：COT辅助与未来GHI筛选 2

本页意义：本次通过SSH读取文件目录和大小；step4000用于已完成辅助监督，OOF过程权重不当作最终模型。

评价范围：共同根目录见REMOTE_CHECKPOINT_INDEX.tsv绝对路径；本次SSH目录快照

| 远端权重（共同根目录之后） | MiB | 远端修改时间 |
| --- | --- | --- |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_source_heads_v1/zhujia/seed42/AC/best.pt | 0.2 | 2026-10-19T10:03:00 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_source_heads_v1/zhujia/seed44/AH/best.pt | 0.2 | 2026-10-19T10:04:50 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_source_heads_v1/zhujia/seed44/AC/best.pt | 0.2 | 2026-10-19T10:05:11 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_source_heads_v1/zhujia/seed44/A/best.pt | 0.2 | 2026-10-19T10:04:31 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_source_heads_v1/zhujia/seed43/AH/best.pt | 0.2 | 2026-10-19T10:03:43 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_source_heads_v1/zhujia/seed43/AC/best.pt | 0.2 | 2026-10-19T10:04:03 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_source_heads_v1/zhujia/seed43/A/best.pt | 0.2 | 2026-10-19T10:03:18 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_source_heads_v1/sili/seed43/A/best.pt | 0.2 | 2026-10-19T10:00:03 |

来源：REMOTE_CHECKPOINT_INDEX.tsv

## P130 最新服务器权重：COT辅助与未来GHI筛选 3

本页意义：本次通过SSH读取文件目录和大小；step4000用于已完成辅助监督，OOF过程权重不当作最终模型。

评价范围：共同根目录见REMOTE_CHECKPOINT_INDEX.tsv绝对路径；本次SSH目录快照

| 远端权重（共同根目录之后） | MiB | 远端修改时间 |
| --- | --- | --- |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_source_heads_v1/sili/seed43/AH/best.pt | 0.2 | 2026-10-19T10:00:27 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_source_heads_v1/sili/seed43/AC/best.pt | 0.2 | 2026-10-19T10:00:46 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_source_heads_v1/sili/seed42/A/best.pt | 0.2 | 2026-10-19T09:59:04 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_source_heads_v1/sili/seed42/AC/best.pt | 0.2 | 2026-10-19T09:59:43 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_source_heads_v1/sili/seed42/AH/best.pt | 0.2 | 2026-10-19T09:59:25 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_source_heads_v1/sili/seed44/A/best.pt | 0.2 | 2026-10-19T10:01:07 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_source_heads_v1/sili/seed44/AH/best.pt | 0.2 | 2026-10-19T10:01:29 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_source_heads_v1/sili/seed44/AC/best.pt | 0.2 | 2026-10-19T10:01:48 |

来源：REMOTE_CHECKPOINT_INDEX.tsv

## P131 最新服务器权重：COT辅助与未来GHI筛选 4

本页意义：本次通过SSH读取文件目录和大小；step4000用于已完成辅助监督，OOF过程权重不当作最终模型。

评价范围：共同根目录见REMOTE_CHECKPOINT_INDEX.tsv绝对路径；本次SSH目录快照

| 远端权重（共同根目录之后） | MiB | 远端修改时间 |
| --- | --- | --- |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_shrink_v1/sili/seed43/A/best.pt | 0.2 | 2026-10-19T10:12:01 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_shrink_v1/sili/seed43/AH/best.pt | 0.2 | 2026-10-19T10:12:20 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_shrink_v1/sili/seed43/AC/best.pt | 0.2 | 2026-10-19T10:12:35 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_shrink_v1/sili/seed43/AC50/best.pt | 0.2 | 2026-10-19T10:12:54 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_shrink_v1/sili/seed42/AH/best.pt | 0.2 | 2026-10-19T10:11:05 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_shrink_v1/sili/seed42/AC/best.pt | 0.2 | 2026-10-19T10:11:23 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_shrink_v1/sili/seed42/AC50/best.pt | 0.2 | 2026-10-19T10:11:41 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_shrink_v1/sili/seed42/A/best.pt | 0.2 | 2026-10-19T10:10:43 |

来源：REMOTE_CHECKPOINT_INDEX.tsv

## P132 最新服务器权重：COT辅助与未来GHI筛选 5

本页意义：本次通过SSH读取文件目录和大小；step4000用于已完成辅助监督，OOF过程权重不当作最终模型。

评价范围：共同根目录见REMOTE_CHECKPOINT_INDEX.tsv绝对路径；本次SSH目录快照

| 远端权重（共同根目录之后） | MiB | 远端修改时间 |
| --- | --- | --- |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_shrink_v1/sili/seed44/AC/best.pt | 0.2 | 2026-10-19T10:13:58 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_shrink_v1/sili/seed44/A/best.pt | 0.2 | 2026-10-19T10:13:16 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_shrink_v1/sili/seed44/AH/best.pt | 0.2 | 2026-10-19T10:13:36 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_shrink_v1/sili/seed44/AC50/best.pt | 0.2 | 2026-10-19T10:14:16 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_shrink_v1/zhujia/seed42/AH/best.pt | 0.2 | 2026-10-19T10:15:29 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_shrink_v1/zhujia/seed42/AC/best.pt | 0.2 | 2026-10-19T10:15:53 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_shrink_v1/zhujia/seed42/AC50/best.pt | 0.2 | 2026-10-19T10:16:11 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_shrink_v1/zhujia/seed42/A/best.pt | 0.2 | 2026-10-19T10:15:05 |

来源：REMOTE_CHECKPOINT_INDEX.tsv

## P133 最新服务器权重：COT辅助与未来GHI筛选 6

本页意义：本次通过SSH读取文件目录和大小；step4000用于已完成辅助监督，OOF过程权重不当作最终模型。

评价范围：共同根目录见REMOTE_CHECKPOINT_INDEX.tsv绝对路径；本次SSH目录快照

| 远端权重（共同根目录之后） | MiB | 远端修改时间 |
| --- | --- | --- |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_shrink_v1/zhujia/seed44/AH/best.pt | 0.2 | 2026-10-19T10:18:27 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_shrink_v1/zhujia/seed44/AC/best.pt | 0.2 | 2026-10-19T10:18:46 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_shrink_v1/zhujia/seed44/AC50/best.pt | 0.2 | 2026-10-19T10:19:11 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_shrink_v1/zhujia/seed44/A/best.pt | 0.2 | 2026-10-19T10:18:11 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_shrink_v1/zhujia/seed43/AH/best.pt | 0.2 | 2026-10-19T10:16:55 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_shrink_v1/zhujia/seed43/AC/best.pt | 0.2 | 2026-10-19T10:17:20 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_shrink_v1/zhujia/seed43/AC50/best.pt | 0.2 | 2026-10-19T10:17:45 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_cot_shrink_v1/zhujia/seed43/A/best.pt | 0.2 | 2026-10-19T10:16:36 |

来源：REMOTE_CHECKPOINT_INDEX.tsv

## P134 最新服务器权重：COT辅助与未来GHI筛选 7

本页意义：本次通过SSH读取文件目录和大小；step4000用于已完成辅助监督，OOF过程权重不当作最终模型。

评价范围：共同根目录见REMOTE_CHECKPOINT_INDEX.tsv绝对路径；本次SSH目录快照

| 远端权重（共同根目录之后） | MiB | 远端修改时间 |
| --- | --- | --- |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_oof_training_v1/fold_0/profile_reload.pt | 42.0 | 2026-10-19T12:15:39 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_oof_training_v1/fold_0/training/last.pt | 125.8 | 2026-10-19T12:56:51 |
| refine-logs/cot_aware_simvp_20261017/results/future_ghi_oof_training_v1/fold_0/training/resume_from_epoch001.pt | 125.8 | 2026-10-19T12:56:51 |

来源：REMOTE_CHECKPOINT_INDEX.tsv

## P135 复现导航：如何从一页结果找到数据、代码和权重？

本页意义：图表页标注来源编号；本册sources保留报告快照，code_snapshot保留实际代码。checkpoint只编目，不复制庞大权重。

评价范围：本次目录盘点，不等同重新完成所有训练

| 文件/目录 | 内容 | 如何使用 |
| --- | --- | --- |
| PAGE_INDEX.csv | 逐页标题、作用、范围、原图/报告路径 | 按页定位原始证据 |
| SOURCE_INDEX.csv 与 sources/ | 本册采用报告及SHA-256、报告快照 | 核对数字和原有评价范围 |
| CHECKPOINT_INDEX.csv | 307个本地权重文件：绝对路径、大小、关联实验 | 找到实际文件；best/last/epoch不能混用 |
| CODE_INDEX.csv 与 code_snapshot/ | 314个训练、评价、绘图或执行代码快照及SHA-256 | 按报告和实验编号定位复现程序 |
| DOCUMENTED_ASSET_REFERENCES.csv | 报告登记的代码/远程模型位置 | 标明只登记、未实时逐项远程核验 |
| INVALID_RESULTS.csv | 作废CPP配对图所在目录 | 不用于科学成绩 |
| main.tex / build_result_book.py | 可编辑LaTeX与结果册生成器 | 使用XeLaTeX编译；一图或一表一页 |

来源：

## P136 结果来源索引：科学问题对应哪个报告？

本页意义：所有路径以solar-cot-ifs仓库根目录为基准；全文报告快照另存sources。

评价范围：来源目录

| 编号 | 科学问题/实验 | 来源报告（仓库相对路径） |
| --- | --- | --- |
| E01 | 湖南同期估计：COT能否改善紧凑GHI模型？ | experiments/monthly_R_station16_20261013_v1/results/GHI_HEAD_THREE_SEED_RESULTS_20261013.md |
| E02 | 湖北独立区域：同期COT收益能否重复？ | refine-logs/hubei_cot_independent_20261014/SAME_TIME_THREE_SEED_VALIDATION_20261017.md |
| E03 | 反演质量：COT对CPP参考的误差有多大？ | paper/analyses/cot_retrieval_quality_20261017/REPORT.md |
| E04 | 厚度分层：GHI收益主要出现在哪些云厚？ | refine-logs/cot_value_first_20261013/results/cot_thickness_story_20261014/COT_THICKNESS_GHI_VALIDATION_20261014.md |
| E05 | 太阳资源：高太阳高度与日辐照量是否改善？ | paper/analyses/solar_resource_validation_20261017/REPORT.md |
| E06 | 容量与反演特征：收益是否超过普通辅助表示？ | refine-logs/cot_value_first_20261013/results/T18_RAL_RESULTS_20261014.md |
| E07 | 湖北同R特征对照：显式COT是否优于隐藏特征？ | refine-logs/hubei_cot_independent_20261014/audit_receipts/al_normreload_20261017/AL_RESULT_AUDIT_20261017.md |
| E08 | 湖北三站共同时间：站点缺测会改变结论吗？ | refine-logs/hubei_cot_independent_20261014/audit_receipts/al_normreload_20261017/THREE_STATION_COMMON_TIME_SENSITIVITY_20261017.md |
| E09 | 湖北条件差异：月份和太阳高度有何影响？ | refine-logs/hubei_cot_independent_20261014/HUBEI_VALIDATION_SUBGROUPS_NORMRELOAD_20261017.md |

来源：

## P137 结果来源索引：科学问题对应哪个报告？

本页意义：所有路径以solar-cot-ifs仓库根目录为基准；全文报告快照另存sources。

评价范围：来源目录

| 编号 | 科学问题/实验 | 来源报告（仓库相对路径） |
| --- | --- | --- |
| E10 | 更强AGRI残差：COT仍有额外收益吗？ | experiments/monthly_R_station16_20261013_v1/crossfit_signed_residual/UNIFIED_FEATURE_RESIDUAL_RESULTS_20261014.md |
| E11 | 有符号残差：早期修正结果与后续统一对照 | experiments/monthly_R_station16_20261013_v1/crossfit_signed_residual/OOF_SIGNED_RESIDUAL_RESULTS_20261014.md |
| E12 | 三帧COT：历史变化是否真的被利用？ | results/cot_interval_history_20261015/REPORT.md |
| E13 | R太阳几何修正：COT误差是否降低？ | experiments/r_solar_geometry_corrected_20261015/RESULT_REPORT.md |
| E14 | 修正后的R进入GHI：上游改善能否传递？ | experiments/r_geometry_downstream_ghi_20261016/REPORT.md |
| E15 | 高GHI低估：增大高透过损失的收益与代价 | experiments/high_kt_loss_20261016/REPORT.md |
| E16 | 真实云事件：COT为什么有时把GHI调错方向？ | experiments/cot_underestimate_forensics_20261016/REPORT.md |
| E17 | 云高与云影：空间代表性是否有可用证据？ | experiments/monthly_R_station16_20261013/cot_shadow_projection_20261014/REPORT.md |
| E18 | 输入扰动：GHI模型使用了哪些COT信息？ | experiments/monthly_R_station16_20261013_v1/results/COT_INPUT_DEPENDENCE_VALIDATION_20261013.md |

来源：

## P138 结果来源索引：科学问题对应哪个报告？

本页意义：所有路径以solar-cot-ifs仓库根目录为基准；全文报告快照另存sources。

评价范围：来源目录

| 编号 | 科学问题/实验 | 来源报告（仓库相对路径） |
| --- | --- | --- |
| E19 | 晴空屏蔽oracle：COT变准是否让GHI更准？ | refine-logs/cot_mask_necessity_20261014/M0_M1_RESULTS.md |
| E20 | 可部署保守判云：重训GHI头后是否受益？ | results/paired_gate_ghi_seed42_20261015_attempt1/REPORT.md |
| E21 | 门控退化归因：误清云能解释多少？ | results/gate_ghi_forensics_20261015/REPORT.md |
| E22 | 站点GHI输入范围：小邻域是否足够？ | docs/FORECAST_GHI_WINDOW_ABLATION_20261004.md |
| E23 | 最新COT预报：固定SimVP与等预算监督对照 | refine-logs/cot_retrieved_forecast_20261013/FULLTRAIN_RESULTS_20261017.md |
| E24 | 直接COT幅值损失：厚云误差能否修复？ | refine-logs/cot_retrieved_forecast_20261013/results/SAME_SUPPORT_AMPLITUDE_DIAGNOSIS_20261017.md |
| E25 | 独立厚云事件：位置、峰值和到达时间 | refine-logs/cot_retrieved_forecast_20261013/results/event_cohort_decomposition_20261017/EVENT_COHORT_DECOMPOSITION_20261017.md |
| E26 | 离线位置对齐：错位还是厚度演变？ | refine-logs/cot_retrieved_forecast_20261013/results/position_amplitude_events_20261017/EVENT_POSITION_AMPLITUDE_DIAGNOSTIC_20261017.md |
| E27 | 时间变化：预测COT轨迹是否过平？ | refine-logs/cot_value_first_20261013/results/cot_temporal_change_audit_20261017/RESULTS.md |

来源：

## P139 结果来源索引：科学问题对应哪个报告？

本页意义：所有路径以solar-cot-ifs仓库根目录为基准；全文报告快照另存sources。

评价范围：来源目录

| 编号 | 科学问题/实验 | 来源报告（仓库相对路径） |
| --- | --- | --- |
| E28 | 固定GHI头2×2替换：图像与COT误差如何传递？ | refine-logs/cot_value_first_20261013/results/fixed_head_2x2_allval_agri_cot_attribution_20261014_v1/ALLVAL_FIXED_HEAD_ATTRIBUTION_20261014.md |
| E29 | 历史COT输运：与Cloud-FNO文献如何比较？ | results/history_cot_cloud_fno_20261015/REPORT.md |
| E30 | 历史COT预报GHI：修正选模流程后的配对结果 | refine-logs/history_cot_forecast_ghi_20261015/attempt3_results.md |
| E31 | 湖北未来GHI：预测COT是否带来净增益？ | refine-logs/hubei_cot_independent_20261014/FORECAST_GHI_NORMRELOAD_THREE_SEED_VALIDATION_20261017.md |
| E32 | 湖南最新未来GHI：完整预测COT与收缩输入 | refine-logs/cot_aware_simvp_20261017/FUTURE_GHI_COT_PILOT_DIAGNOSIS_20261019.md |
| E33 | 监督来源：CPP与R伪标签是否等价？ | refine-logs/cot_aware_simvp_20261017/results/label_source_disagreement_summary.md |
| E34 | 云与厚云检出：FAR、POD、CSI分别说明什么？ | refine-logs/cot_value_first_20261013/COT_CLOUD_DETECTION_VALIDATION_20261013.md |
| E35 | 数据划分：每个月的训练和验证覆盖如何？ | docs/MONTHLY_BLOCK_SPLIT_20261002.md |
| E36 | 历史探索：空间COT初筛 | docs/SPATIAL_B1_PILOT_RESULTS_20260917.md |

来源：

## P140 结果来源索引：科学问题对应哪个报告？

本页意义：所有路径以solar-cot-ifs仓库根目录为基准；全文报告快照另存sources。

评价范围：来源目录

| 编号 | 科学问题/实验 | 来源报告（仓库相对路径） |
| --- | --- | --- |
| E37 | 历史探索：空间排列控制 | docs/SPATIAL_B2_CONTROLS_RESULTS_20260917.md |
| E38 | 历史探索：AFNO固定GHI头替换 | results/afno_fixed_ghi_head_swap_20260923/README.md |
| E39 | 历史探索：输运与厚度残差 | docs/SILI_TRANSPORT_THICKNESS_PILOT_RESULTS_20260928.md |
| E40 | 历史探索：完整历史与重复末帧 | docs/SILI_FULL_COT_HISTORY_PILOT_RESULTS_20260928.md |
| E41 | 历史探索：低COT惩罚消融 | docs/SILI_LOW_COT_PENALTY_ABLATION_RESULTS_20260928.md |
| E42 | 历史探索：历史信息探针 | docs/SILI_HISTORY_INFORMATION_PROBE_RESULTS_20260928.md |
| E43 | 历史探索：竺家残差容量对照 | docs/ZHUJIA_RESIDUAL_CAPACITY_CONTROL_RESULTS_20260927.md |
| E44 | 历史探索：旧天气分层 | docs/STATION_WEATHER_COT_DIAGNOSIS_20260912.md |

来源：

## P141 代码与checkpoint：湖南同期估计：COT能否改善紧凑GHI模型？

本页意义：将该实验的结果、实现和现存权重连到一起；示例路径不是自动选模指令。

评价范围：工程复现附录

| 项目 | 位置/数量 | 说明 |
| --- | --- | --- |
| 报告 | experiments/monthly_R_station16_20261013_v1/results/GHI_HEAD_THREE_SEED_RESULTS_20261013.md | 报告快照见sources/E01 |
| 本地权重 | 43个；C230, C231, C232, C233, C234, C235, C236, C237 | 完整绝对路径见CHECKPOINT_INDEX.csv |
| 权重示例 C230 | experiments/monthly_R_station16_20261013_v1/results/seed42_v1/best.pt | 5.4 MiB；本地存在 |
| 权重示例 C231 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_zhujia_20261014_v1/seed44/zhujia/AGRI_best.pt | 0.2 MiB；本地存在 |
| 权重示例 C232 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_zhujia_20261014_v1/seed44/zhujia/AGRI_COT_best.pt | 0.2 MiB；本地存在 |
| 代码 | experiments/monthly_R_station16_20261013_v1/crossfit_signed_residual/analyze_oof_signed_residual.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | experiments/monthly_R_station16_20261013_v1/analyze_signed_residual_direction.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | experiments/monthly_R_station16_20261013_v1/audit_ghi_heads_three_seed.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | experiments/monthly_R_station16_20261013_v1/scripts/audit_monthly_R_pack.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |

来源：experiments/monthly_R_station16_20261013_v1/results/GHI_HEAD_THREE_SEED_RESULTS_20261013.md

## P142 代码与checkpoint：湖北独立区域：同期COT收益能否重复？

本页意义：将该实验的结果、实现和现存权重连到一起；示例路径不是自动选模指令。

评价范围：工程复现附录

| 项目 | 位置/数量 | 说明 |
| --- | --- | --- |
| 报告 | refine-logs/hubei_cot_independent_20261014/SAME_TIME_THREE_SEED_VALIDATION_20261017.md | 报告快照见sources/E02 |
| 本地权重 | 145个；C042, C043, C044, C045, C046, C047, C048, C049 | 完整绝对路径见CHECKPOINT_INDEX.csv |
| 权重示例 C042 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed43_44_independent_recheck/seed43/AC_best.pt | 0.1 MiB；本地存在 |
| 权重示例 C043 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed43_44_independent_recheck/seed43/AP_best.pt | 0.1 MiB；本地存在 |
| 权重示例 C044 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed43_44_independent_recheck/seed43/A_best.pt | 0.1 MiB；本地存在 |
| 代码 | refine-logs/hubei_cot_independent_20261014/tooling_20261017/analyze_same_time_normreload_subgroups_20261017.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | refine-logs/hubei_cot_independent_20261014/tooling_20261017/audit_al_coldstart_normreload_20261017.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | refine-logs/hubei_cot_independent_20261014/tooling_20261017/audit_al_normreload_crossrun_20261017.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | refine-logs/hubei_cot_independent_20261014/al_20261017/audit_al_results.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |

来源：refine-logs/hubei_cot_independent_20261014/SAME_TIME_THREE_SEED_VALIDATION_20261017.md

## P143 代码与checkpoint：更强AGRI残差：COT仍有额外收益吗？

本页意义：将该实验的结果、实现和现存权重连到一起；示例路径不是自动选模指令。

评价范围：工程复现附录

| 项目 | 位置/数量 | 说明 |
| --- | --- | --- |
| 报告 | experiments/monthly_R_station16_20261013_v1/crossfit_signed_residual/UNIFIED_FEATURE_RESIDUAL_RESULTS_20261014.md | 报告快照见sources/E10 |
| 本地权重 | 0个； | 完整绝对路径见CHECKPOINT_INDEX.csv |
| 代码 | experiments/monthly_R_station16_20261013_v1/crossfit_signed_residual/analyze_oof_signed_residual.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | experiments/monthly_R_station16_20261013_v1/crossfit_signed_residual/train_signed_residual_oof.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | experiments/monthly_R_station16_20261013_v1/crossfit_signed_residual/train_signed_residual_oof_v2.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | experiments/monthly_R_station16_20261013_v1/crossfit_signed_residual/train_signed_residual_oof_v3.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |

来源：experiments/monthly_R_station16_20261013_v1/crossfit_signed_residual/UNIFIED_FEATURE_RESIDUAL_RESULTS_20261014.md

## P144 代码与checkpoint：三帧COT：历史变化是否真的被利用？

本页意义：将该实验的结果、实现和现存权重连到一起；示例路径不是自动选模指令。

评价范围：工程复现附录

| 项目 | 位置/数量 | 说明 |
| --- | --- | --- |
| 报告 | results/cot_interval_history_20261015/REPORT.md | 报告快照见sources/E12 |
| 本地权重 | 12个；C030, C031, C032, C033, C034, C035, C036, C037 | 完整绝对路径见CHECKPOINT_INDEX.csv |
| 权重示例 C030 | results/cot_interval_history_20261015/results_attempt1/seed44/zhujia/AC_history/best.pt | 0.2 MiB；本地存在 |
| 权重示例 C031 | results/cot_interval_history_20261015/results_attempt1/seed44/zhujia/A_history/best.pt | 0.2 MiB；本地存在 |
| 权重示例 C032 | results/cot_interval_history_20261015/results_attempt1/seed44/sili/AC_history/best.pt | 0.2 MiB；本地存在 |
| 代码 | refine-logs/cot_interval_history_20261015/analyze_paired.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | refine-logs/cot_interval_history_20261015/audit_and_prepare.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | refine-logs/cot_interval_history_20261015/audit_geometry.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | refine-logs/cot_interval_history_20261015/audit_workbook.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |

来源：results/cot_interval_history_20261015/REPORT.md

## P145 代码与checkpoint：R太阳几何修正：COT误差是否降低？

本页意义：将该实验的结果、实现和现存权重连到一起；示例路径不是自动选模指令。

评价范围：工程复现附录

| 项目 | 位置/数量 | 说明 |
| --- | --- | --- |
| 报告 | experiments/r_solar_geometry_corrected_20261015/RESULT_REPORT.md | 报告快照见sources/E13 |
| 本地权重 | 0个； | 完整绝对路径见CHECKPOINT_INDEX.csv |
| 代码 | experiments/r_solar_geometry_corrected_20261015/train_paired_r.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | experiments/r_solar_geometry_corrected_20261015/build_paired_geometry_pack.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | experiments/r_solar_geometry_corrected_20261015/run_pair.pbs | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | experiments/r_solar_geometry_corrected_20261015/score_paired_r.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |

来源：experiments/r_solar_geometry_corrected_20261015/RESULT_REPORT.md

## P146 代码与checkpoint：修正后的R进入GHI：上游改善能否传递？

本页意义：将该实验的结果、实现和现存权重连到一起；示例路径不是自动选模指令。

评价范围：工程复现附录

| 项目 | 位置/数量 | 说明 |
| --- | --- | --- |
| 报告 | experiments/r_geometry_downstream_ghi_20261016/REPORT.md | 报告快照见sources/E14 |
| 本地权重 | 0个； | 完整绝对路径见CHECKPOINT_INDEX.csv |
| 代码 | experiments/r_geometry_downstream_ghi_20261016/audit_r_bank_vs_saved_predictions.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | experiments/r_geometry_downstream_ghi_20261016/audit_r_input_reconstruction.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | experiments/r_geometry_downstream_ghi_20261016/plot_r_real_cloud_cases.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | experiments/r_geometry_downstream_ghi_20261016/plot_three_arm_ghi.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |

来源：experiments/r_geometry_downstream_ghi_20261016/REPORT.md

## P147 代码与checkpoint：高GHI低估：增大高透过损失的收益与代价

本页意义：将该实验的结果、实现和现存权重连到一起；示例路径不是自动选模指令。

评价范围：工程复现附录

| 项目 | 位置/数量 | 说明 |
| --- | --- | --- |
| 报告 | experiments/high_kt_loss_20261016/REPORT.md | 报告快照见sources/E15 |
| 本地权重 | 24个；C275, C276, C277, C278, C279, C280, C281, C282 | 完整绝对路径见CHECKPOINT_INDEX.csv |
| 权重示例 C275 | experiments/high_kt_loss_20261016/runs/formal_v1/seed44/zhujia/AC_bright2x/best.pt | 0.2 MiB；本地存在 |
| 权重示例 C276 | experiments/high_kt_loss_20261016/runs/formal_v1/seed44/zhujia/AC_uniform/best.pt | 0.2 MiB；本地存在 |
| 权重示例 C277 | experiments/high_kt_loss_20261016/runs/formal_v1/seed44/zhujia/A_bright2x/best.pt | 0.2 MiB；本地存在 |
| 代码 | experiments/high_kt_loss_20261016/train_loss_pair.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | experiments/high_kt_loss_20261016/verify_and_plot.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | experiments/high_kt_loss_20261016/archive_results.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | experiments/high_kt_loss_20261016/loss_witness.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |

来源：experiments/high_kt_loss_20261016/REPORT.md

## P148 代码与checkpoint：真实云事件：COT为什么有时把GHI调错方向？

本页意义：将该实验的结果、实现和现存权重连到一起；示例路径不是自动选模指令。

评价范围：工程复现附录

| 项目 | 位置/数量 | 说明 |
| --- | --- | --- |
| 报告 | experiments/cot_underestimate_forensics_20261016/REPORT.md | 报告快照见sources/E16 |
| 本地权重 | 0个； | 完整绝对路径见CHECKPOINT_INDEX.csv |
| 代码 | experiments/cot_underestimate_forensics_20261016/analyze_select.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | experiments/cot_underestimate_forensics_20261016/audit_artifacts.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | experiments/cot_underestimate_forensics_20261016/audit_repository_evolution.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | experiments/cot_underestimate_forensics_20261016/plot_cases.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |

来源：experiments/cot_underestimate_forensics_20261016/REPORT.md

## P149 代码与checkpoint：晴空屏蔽oracle：COT变准是否让GHI更准？

本页意义：将该实验的结果、实现和现存权重连到一起；示例路径不是自动选模指令。

评价范围：工程复现附录

| 项目 | 位置/数量 | 说明 |
| --- | --- | --- |
| 报告 | refine-logs/cot_mask_necessity_20261014/M0_M1_RESULTS.md | 报告快照见sources/E19 |
| 本地权重 | 0个； | 完整绝对路径见CHECKPOINT_INDEX.csv |
| 代码 | refine-logs/cot_mask_necessity_20261014/analyze_conditional_cot.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | refine-logs/cot_mask_necessity_20261014/plot_m03_cases.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | refine-logs/cot_mask_necessity_20261014/decompose_ghi_gate_effect.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | refine-logs/cot_mask_necessity_20261014/fixed_ghi_oracle_clear_gate.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |

来源：refine-logs/cot_mask_necessity_20261014/M0_M1_RESULTS.md

## P150 代码与checkpoint：可部署保守判云：重训GHI头后是否受益？

本页意义：将该实验的结果、实现和现存权重连到一起；示例路径不是自动选模指令。

评价范围：工程复现附录

| 项目 | 位置/数量 | 说明 |
| --- | --- | --- |
| 报告 | results/paired_gate_ghi_seed42_20261015_attempt1/REPORT.md | 报告快照见sources/E20 |
| 本地权重 | 9个；C012, C013, C014, C015, C224, C225, C226, C227 | 完整绝对路径见CHECKPOINT_INDEX.csv |
| 权重示例 C012 | results/paired_gate_ghi_seed42_20261015_attempt1/zhujia/gate/best.pt | 0.1 MiB；本地存在 |
| 权重示例 C013 | results/paired_gate_ghi_seed42_20261015_attempt1/zhujia/raw/best.pt | 0.1 MiB；本地存在 |
| 权重示例 C014 | results/paired_gate_ghi_seed42_20261015_attempt1/sili/gate/best.pt | 0.1 MiB；本地存在 |
| 代码 | refine-logs/cot_cloud_gate_retrain_20261014/analyze_gate_clear_physics.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | refine-logs/cot_cloud_gate_retrain_20261014/analyze_gate_point_ghi_correspondence.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | refine-logs/cot_cloud_gate_retrain_20261014/analyze_paired_ghi_gate_seed42.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | refine-logs/cot_cloud_gate_retrain_20261014/audit_cloud_gate_seed42.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |

来源：results/paired_gate_ghi_seed42_20261015_attempt1/REPORT.md

## P151 代码与checkpoint：站点GHI输入范围：小邻域是否足够？

本页意义：将该实验的结果、实现和现存权重连到一起；示例路径不是自动选模指令。

评价范围：工程复现附录

| 项目 | 位置/数量 | 说明 |
| --- | --- | --- |
| 报告 | docs/FORECAST_GHI_WINDOW_ABLATION_20261004.md | 报告快照见sources/E22 |
| 本地权重 | 0个； | 完整绝对路径见CHECKPOINT_INDEX.csv |
| 代码 | experiments/forecast_ghi_window_20261004_v1/train_forecast_ghi_window_20261004.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | experiments/forecast_ghi_window_20261004_v1/run.pbs | 已复制code_snapshot；SHA见CODE_INDEX.csv |

来源：docs/FORECAST_GHI_WINDOW_ABLATION_20261004.md

## P152 代码与checkpoint：最新COT预报：固定SimVP与等预算监督对照

本页意义：将该实验的结果、实现和现存权重连到一起；示例路径不是自动选模指令。

评价范围：工程复现附录

| 项目 | 位置/数量 | 说明 |
| --- | --- | --- |
| 报告 | refine-logs/cot_retrieved_forecast_20261013/FULLTRAIN_RESULTS_20261017.md | 报告快照见sources/E23 |
| 本地权重 | 1个；C223 | 完整绝对路径见CHECKPOINT_INDEX.csv |
| 权重示例 C223 | refine-logs/cot_retrieved_forecast_20261013/results/amplitude_loss_20261017_seed42_b1024_e12/P_Hybrid_best.pt | 8.8 MiB；本地存在 |
| 代码 | refine-logs/cot_retrieved_forecast_20261013/analyze_fulltrain_vs_simvp_same_support_20261017.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | refine-logs/cot_retrieved_forecast_20261013/analyze_fullval_cot_event_cohort_20261017.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | refine-logs/cot_retrieved_forecast_20261013/analyze_selected_event_position_amplitude_20261017.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | refine-logs/cot_retrieved_forecast_20261013/audit_retrieved_cot_full_bank_20261017.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |

来源：refine-logs/cot_retrieved_forecast_20261013/FULLTRAIN_RESULTS_20261017.md

## P153 代码与checkpoint：历史COT预报GHI：修正选模流程后的配对结果

本页意义：将该实验的结果、实现和现存权重连到一起；示例路径不是自动选模指令。

评价范围：工程复现附录

| 项目 | 位置/数量 | 说明 |
| --- | --- | --- |
| 报告 | refine-logs/history_cot_forecast_ghi_20261015/attempt3_results.md | 报告快照见sources/E30 |
| 本地权重 | 36个；C187, C188, C189, C190, C191, C192, C193, C194 | 完整绝对路径见CHECKPOINT_INDEX.csv |
| 权重示例 C187 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/A_seed42/best.pt | 0.2 MiB；本地存在 |
| 权重示例 C189 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/A_seed43/best.pt | 0.2 MiB；本地存在 |
| 权重示例 C191 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/A_seed44/best.pt | 0.2 MiB；本地存在 |
| 代码 | refine-logs/history_cot_forecast_ghi_20261015/audit_source_manifests.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | refine-logs/history_cot_forecast_ghi_20261015/plot_attempt3_score.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | refine-logs/history_cot_forecast_ghi_20261015/diagnosis_20261015/plot_diagnosis.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | refine-logs/history_cot_forecast_ghi_20261015/plot_forecast_ghi_cot_bar_attempt3_zh.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |

来源：refine-logs/history_cot_forecast_ghi_20261015/attempt3_results.md

## P154 代码与checkpoint：湖南最新未来GHI：完整预测COT与收缩输入

本页意义：将该实验的结果、实现和现存权重连到一起；示例路径不是自动选模指令。

评价范围：工程复现附录

| 项目 | 位置/数量 | 说明 |
| --- | --- | --- |
| 报告 | refine-logs/cot_aware_simvp_20261017/FUTURE_GHI_COT_PILOT_DIAGNOSIS_20261019.md | 报告快照见sources/E32 |
| 本地权重 | 0个； | 完整绝对路径见CHECKPOINT_INDEX.csv |
| 代码 | refine-logs/cot_aware_simvp_20261017/audit_cot_aux_validation.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | refine-logs/cot_aware_simvp_20261017/audit_forecast_cot_bank_distribution.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | refine-logs/cot_aware_simvp_20261017/audit_future_cot_input_shift.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |
| 代码 | refine-logs/cot_aware_simvp_20261017/audit_future_ghi_a_ac.py | 已复制code_snapshot；SHA见CODE_INDEX.csv |

来源：refine-logs/cot_aware_simvp_20261017/FUTURE_GHI_COT_PILOT_DIAGNOSIS_20261019.md

## P155 完整checkpoint目录：1–10 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C001 | results/zhujia_ghi_objective_residual_20260928_v3/formal/agri_residual_best.pt | 2.8 | 历史/共享；见原目录 |
| C002 | results/zhujia_ghi_objective_residual_20260928_v3/formal/cot_residual_best.pt | 2.8 | 历史/共享；见原目录 |
| C003 | results/regional_station_cot_20260927/sili_transport_thickness_pilot/best.pt | 0.3 | 历史/共享；见原目录 |
| C004 | results/regional_station_cot_20260927/sili_lowcot_penalty_ablation_seed42_20260928_v1/penalty_off_seed42/best.pt | 0.3 | 历史/共享；见原目录 |
| C005 | results/regional_station_cot_20260927/sili_lowcot_penalty_ablation_seed42_20260928_v1/penalty_on_seed42/best.pt | 0.3 | 历史/共享；见原目录 |
| C006 | results/regional_station_cot_20260927/sili_lowcot_penalty_ablation_20260928_v1/penalty_off_seed42/best.pt | 0.3 | 历史/共享；见原目录 |
| C007 | results/regional_station_cot_20260927/sili_lowcot_penalty_ablation_20260928_v1/penalty_on_seed42/best.pt | 0.3 | 历史/共享；见原目录 |
| C008 | results/regional_station_cot_20260927/sili_full_vs_repeat_last_history_20260928_v5/full_history/best.pt | 0.3 | 历史/共享；见原目录 |
| C009 | results/regional_station_cot_20260927/sili_full_vs_repeat_last_history_20260928_v5/full_history/best_learned.pt | 0.3 | 历史/共享；见原目录 |
| C010 | results/regional_station_cot_20260927/sili_full_vs_repeat_last_history_20260928_v5/repeat_last_control/best.pt | 0.3 | 历史/共享；见原目录 |

来源：

## P156 完整checkpoint目录：11–20 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C011 | results/regional_station_cot_20260927/sili_full_vs_repeat_last_history_20260928_v5/repeat_last_control/best_learned.pt | 0.3 | 历史/共享；见原目录 |
| C012 | results/paired_gate_ghi_seed42_20261015_attempt1/zhujia/gate/best.pt | 0.1 | E20 |
| C013 | results/paired_gate_ghi_seed42_20261015_attempt1/zhujia/raw/best.pt | 0.1 | E20 |
| C014 | results/paired_gate_ghi_seed42_20261015_attempt1/sili/gate/best.pt | 0.1 | E20 |
| C015 | results/paired_gate_ghi_seed42_20261015_attempt1/sili/raw/best.pt | 0.1 | E20 |
| C016 | results/cot_real_agri_staged_hurdle_seed42_20260924/stage1_classifier/best.pt | 0.0 | 历史/共享；见原目录 |
| C017 | results/cot_real_agri_staged_hurdle_seed42_20260924/stage1_classifier/initial.pt | 0.0 | 历史/共享；见原目录 |
| C018 | results/cot_real_agri_staged_hurdle_seed42_20260924/stage2_conditional_regressors/best.pt | 0.0 | 历史/共享；见原目录 |
| C019 | results/cot_real_agri_staged_hurdle_seed42_20260924/stage2_conditional_regressors/initial.pt | 0.0 | 历史/共享；见原目录 |
| C020 | results/cot_real_agri_lowcot_weighted_seed42_20260923/log_control/best.pt | 0.5 | 历史/共享；见原目录 |

来源：

## P157 完整checkpoint目录：21–30 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C021 | results/cot_real_agri_lowcot_weighted_seed42_20260923/low_cot_weighted/best.pt | 0.5 | 历史/共享；见原目录 |
| C022 | results/cot_real_agri_hurdle_v2_seed42_20260924/hurdle_classifier_two_regressors/best.pt | 0.5 | 历史/共享；见原目录 |
| C023 | results/cot_real_agri_hurdle_v2_seed42_20260924/hurdle_classifier_two_regressors/initial.pt | 0.5 | 历史/共享；见原目录 |
| C024 | results/cot_real_agri_hurdle_v2_seed42_20260924/single_head_control/best.pt | 0.5 | 历史/共享；见原目录 |
| C025 | results/cot_real_agri_hurdle_v2_seed42_20260924/single_head_control/initial.pt | 0.5 | 历史/共享；见原目录 |
| C026 | results/cot_real_agri_hurdle_seed42_20260924/hurdle_classifier_two_regressors/best.pt | 0.5 | 历史/共享；见原目录 |
| C027 | results/cot_real_agri_hurdle_seed42_20260924/hurdle_classifier_two_regressors/initial.pt | 0.5 | 历史/共享；见原目录 |
| C028 | results/cot_real_agri_hurdle_seed42_20260924/single_head_control/best.pt | 0.5 | 历史/共享；见原目录 |
| C029 | results/cot_real_agri_hurdle_seed42_20260924/single_head_control/initial.pt | 0.5 | 历史/共享；见原目录 |
| C030 | results/cot_interval_history_20261015/results_attempt1/seed44/zhujia/AC_history/best.pt | 0.2 | E12 |

来源：

## P158 完整checkpoint目录：31–40 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C031 | results/cot_interval_history_20261015/results_attempt1/seed44/zhujia/A_history/best.pt | 0.2 | E12 |
| C032 | results/cot_interval_history_20261015/results_attempt1/seed44/sili/AC_history/best.pt | 0.2 | E12 |
| C033 | results/cot_interval_history_20261015/results_attempt1/seed44/sili/A_history/best.pt | 0.2 | E12 |
| C034 | results/cot_interval_history_20261015/results_attempt1/seed43/zhujia/AC_history/best.pt | 0.2 | E12 |
| C035 | results/cot_interval_history_20261015/results_attempt1/seed43/zhujia/A_history/best.pt | 0.2 | E12 |
| C036 | results/cot_interval_history_20261015/results_attempt1/seed43/sili/AC_history/best.pt | 0.2 | E12 |
| C037 | results/cot_interval_history_20261015/results_attempt1/seed43/sili/A_history/best.pt | 0.2 | E12 |
| C038 | results/cot_interval_history_20261015/results_attempt1/seed42/zhujia/AC_history/best.pt | 0.2 | E12 |
| C039 | results/cot_interval_history_20261015/results_attempt1/seed42/zhujia/A_history/best.pt | 0.2 | E12 |
| C040 | results/cot_interval_history_20261015/results_attempt1/seed42/sili/AC_history/best.pt | 0.2 | E12 |

来源：

## P159 完整checkpoint目录：41–50 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C041 | results/cot_interval_history_20261015/results_attempt1/seed42/sili/A_history/best.pt | 0.2 | E12 |
| C042 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed43_44_independent_recheck/seed43/AC_best.pt | 0.1 | E02 |
| C043 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed43_44_independent_recheck/seed43/AP_best.pt | 0.1 | E02 |
| C044 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed43_44_independent_recheck/seed43/A_best.pt | 0.1 | E02 |
| C045 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed43_44_independent_recheck/seed44/AC_best.pt | 0.1 | E02 |
| C046 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed43_44_independent_recheck/seed44/AP_best.pt | 0.1 | E02 |
| C047 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed43_44_independent_recheck/seed44/A_best.pt | 0.1 | E02 |
| C048 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed43_44_independent_recheck/seed44/Mizhuang/AC_best.pt | 0.1 | E02 |
| C049 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed43_44_independent_recheck/seed44/Mizhuang/AP_best.pt | 0.1 | E02 |
| C050 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed43_44_independent_recheck/seed44/Mizhuang/A_best.pt | 0.1 | E02 |

来源：

## P160 完整checkpoint目录：51–60 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C051 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed43_44_independent_recheck/seed44/Wangzhai/AC_best.pt | 0.1 | E02 |
| C052 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed43_44_independent_recheck/seed44/Wangzhai/AP_best.pt | 0.1 | E02 |
| C053 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed43_44_independent_recheck/seed44/Wangzhai/A_best.pt | 0.1 | E02 |
| C054 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed43_44_independent_recheck/seed43/Mizhuang/AC_best.pt | 0.1 | E02 |
| C055 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed43_44_independent_recheck/seed43/Mizhuang/AP_best.pt | 0.1 | E02 |
| C056 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed43_44_independent_recheck/seed43/Mizhuang/A_best.pt | 0.1 | E02 |
| C057 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed43_44_independent_recheck/seed43/Wangzhai/AC_best.pt | 0.1 | E02 |
| C058 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed43_44_independent_recheck/seed43/Wangzhai/AP_best.pt | 0.1 | E02 |
| C059 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed43_44_independent_recheck/seed43/Wangzhai/A_best.pt | 0.1 | E02 |
| C060 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed42_independent_recheck/Mizhuang/AC_best.pt | 0.1 | E02 |

来源：

## P161 完整checkpoint目录：61–70 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C061 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed42_independent_recheck/Mizhuang/AP_best.pt | 0.1 | E02 |
| C062 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed42_independent_recheck/Mizhuang/A_best.pt | 0.1 | E02 |
| C063 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed42_independent_recheck/Niushou/AC_best.pt | 0.1 | E02 |
| C064 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed42_independent_recheck/Niushou/AP_best.pt | 0.1 | E02 |
| C065 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed42_independent_recheck/Niushou/A_best.pt | 0.1 | E02 |
| C066 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed42_independent_recheck/Wangzhai/AC_best.pt | 0.1 | E02 |
| C067 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed42_independent_recheck/Wangzhai/AP_best.pt | 0.1 | E02 |
| C068 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_seed42_independent_recheck/Wangzhai/A_best.pt | 0.1 | E02 |
| C069 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed44_normreload_v1/Mizhuang/AC_best.pt | 0.1 | E02 |
| C070 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed44_normreload_v1/Mizhuang/AP_best.pt | 0.1 | E02 |

来源：

## P162 完整checkpoint目录：71–80 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C071 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed44_normreload_v1/Mizhuang/A_best.pt | 0.1 | E02 |
| C072 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed44_normreload_v1/Niushou/AC_best.pt | 0.1 | E02 |
| C073 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed44_normreload_v1/Niushou/AP_best.pt | 0.1 | E02 |
| C074 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed44_normreload_v1/Niushou/A_best.pt | 0.1 | E02 |
| C075 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed44_normreload_v1/Wangzhai/AC_best.pt | 0.1 | E02 |
| C076 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed44_normreload_v1/Wangzhai/AP_best.pt | 0.1 | E02 |
| C077 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed44_normreload_v1/Wangzhai/A_best.pt | 0.1 | E02 |
| C078 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed43_normreload_v1/Mizhuang/AC_best.pt | 0.1 | E02 |
| C079 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed43_normreload_v1/Mizhuang/AP_best.pt | 0.1 | E02 |
| C080 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed43_normreload_v1/Mizhuang/A_best.pt | 0.1 | E02 |

来源：

## P163 完整checkpoint目录：81–90 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C081 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed43_normreload_v1/Niushou/AC_best.pt | 0.1 | E02 |
| C082 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed43_normreload_v1/Niushou/AP_best.pt | 0.1 | E02 |
| C083 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed43_normreload_v1/Niushou/A_best.pt | 0.1 | E02 |
| C084 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed43_normreload_v1/Wangzhai/AC_best.pt | 0.1 | E02 |
| C085 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed43_normreload_v1/Wangzhai/AP_best.pt | 0.1 | E02 |
| C086 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed43_normreload_v1/Wangzhai/A_best.pt | 0.1 | E02 |
| C087 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed42_normreload_v1/Mizhuang/AC_best.pt | 0.1 | E02 |
| C088 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed42_normreload_v1/Mizhuang/AP_best.pt | 0.1 | E02 |
| C089 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed42_normreload_v1/Mizhuang/A_best.pt | 0.1 | E02 |
| C090 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed42_normreload_v1/Niushou/AC_best.pt | 0.1 | E02 |

来源：

## P164 完整checkpoint目录：91–100 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C091 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed42_normreload_v1/Niushou/AP_best.pt | 0.1 | E02 |
| C092 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed42_normreload_v1/Niushou/A_best.pt | 0.1 | E02 |
| C093 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed42_normreload_v1/Wangzhai/AC_best.pt | 0.1 | E02 |
| C094 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed42_normreload_v1/Wangzhai/AP_best.pt | 0.1 | E02 |
| C095 | refine-logs/hubei_cot_independent_20261014/audit_receipts/same_time_normreload_runs_20261017/same_time_seed42_normreload_v1/Wangzhai/A_best.pt | 0.1 | E02 |
| C096 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_seed44_20261017/forecast_FA_FAC_seed44/Mizhuang/FAC_best.pt | 0.1 | E02 |
| C097 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_seed44_20261017/forecast_FA_FAC_seed44/Mizhuang/FA_best.pt | 0.1 | E02 |
| C098 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_seed44_20261017/forecast_FA_FAC_seed44/Niushou/FAC_best.pt | 0.1 | E02 |
| C099 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_seed44_20261017/forecast_FA_FAC_seed44/Niushou/FA_best.pt | 0.1 | E02 |
| C100 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_seed44_20261017/forecast_FA_FAC_seed44/Wangzhai/FAC_best.pt | 0.1 | E02 |

来源：

## P165 完整checkpoint目录：101–110 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C101 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_seed44_20261017/forecast_FA_FAC_seed44/Wangzhai/FA_best.pt | 0.1 | E02 |
| C102 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_seed43_20261017/forecast_FA_FAC_seed43/Mizhuang/FAC_best.pt | 0.1 | E02 |
| C103 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_seed43_20261017/forecast_FA_FAC_seed43/Mizhuang/FA_best.pt | 0.1 | E02 |
| C104 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_seed43_20261017/forecast_FA_FAC_seed43/Niushou/FAC_best.pt | 0.1 | E02 |
| C105 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_seed43_20261017/forecast_FA_FAC_seed43/Niushou/FA_best.pt | 0.1 | E02 |
| C106 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_seed43_20261017/forecast_FA_FAC_seed43/Wangzhai/FAC_best.pt | 0.1 | E02 |
| C107 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_seed43_20261017/forecast_FA_FAC_seed43/Wangzhai/FA_best.pt | 0.1 | E02 |
| C108 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_seed42_20261017/forecast_FA_FAC_seed42/Mizhuang/FAC_best.pt | 0.1 | E02 |
| C109 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_seed42_20261017/forecast_FA_FAC_seed42/Mizhuang/FA_best.pt | 0.1 | E02 |
| C110 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_seed42_20261017/forecast_FA_FAC_seed42/Niushou/FAC_best.pt | 0.1 | E02 |

来源：

## P166 完整checkpoint目录：111–120 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C111 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_seed42_20261017/forecast_FA_FAC_seed42/Niushou/FA_best.pt | 0.1 | E02 |
| C112 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_seed42_20261017/forecast_FA_FAC_seed42/Wangzhai/FAC_best.pt | 0.1 | E02 |
| C113 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_seed42_20261017/forecast_FA_FAC_seed42/Wangzhai/FA_best.pt | 0.1 | E02 |
| C114 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_normreload_seed44_20261017/run/Mizhuang/FAC_best.pt | 0.1 | E02 |
| C115 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_normreload_seed44_20261017/run/Mizhuang/FA_best.pt | 0.1 | E02 |
| C116 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_normreload_seed44_20261017/run/Niushou/FAC_best.pt | 0.1 | E02 |
| C117 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_normreload_seed44_20261017/run/Niushou/FA_best.pt | 0.1 | E02 |
| C118 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_normreload_seed44_20261017/run/Wangzhai/FAC_best.pt | 0.1 | E02 |
| C119 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_normreload_seed44_20261017/run/Wangzhai/FA_best.pt | 0.1 | E02 |
| C120 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_normreload_seed43_20261017/run/Mizhuang/FAC_best.pt | 0.1 | E02 |

来源：

## P167 完整checkpoint目录：121–130 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C121 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_normreload_seed43_20261017/run/Mizhuang/FA_best.pt | 0.1 | E02 |
| C122 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_normreload_seed43_20261017/run/Niushou/FAC_best.pt | 0.1 | E02 |
| C123 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_normreload_seed43_20261017/run/Niushou/FA_best.pt | 0.1 | E02 |
| C124 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_normreload_seed43_20261017/run/Wangzhai/FAC_best.pt | 0.1 | E02 |
| C125 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_normreload_seed43_20261017/run/Wangzhai/FA_best.pt | 0.1 | E02 |
| C126 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_normreload_seed42_20261017/run/Mizhuang/FAC_best.pt | 0.1 | E02 |
| C127 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_normreload_seed42_20261017/run/Mizhuang/FA_best.pt | 0.1 | E02 |
| C128 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_normreload_seed42_20261017/run/Niushou/FAC_best.pt | 0.1 | E02 |
| C129 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_normreload_seed42_20261017/run/Niushou/FA_best.pt | 0.1 | E02 |
| C130 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_normreload_seed42_20261017/run/Wangzhai/FAC_best.pt | 0.1 | E02 |

来源：

## P168 完整checkpoint目录：131–140 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C131 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_normreload_seed42_20261017/run/Wangzhai/FA_best.pt | 0.1 | E02 |
| C132 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_fap_seed44_20261018/forecast_FAP_seed44_normreload_20261018/Mizhuang/FAP_best.pt | 0.1 | E02 |
| C133 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_fap_seed44_20261018/forecast_FAP_seed44_normreload_20261018/Niushou/FAP_best.pt | 0.1 | E02 |
| C134 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_fap_seed44_20261018/forecast_FAP_seed44_normreload_20261018/Wangzhai/FAP_best.pt | 0.1 | E02 |
| C135 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_fap_seed43_20261018/forecast_FAP_seed43_normreload_20261018/Mizhuang/FAP_best.pt | 0.1 | E02 |
| C136 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_fap_seed43_20261018/forecast_FAP_seed43_normreload_20261018/Niushou/FAP_best.pt | 0.1 | E02 |
| C137 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_fap_seed43_20261018/forecast_FAP_seed43_normreload_20261018/Wangzhai/FAP_best.pt | 0.1 | E02 |
| C138 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_fap_seed42_20261018/forecast_FAP_seed42_normreload_20261018/Mizhuang/FAP_best.pt | 0.1 | E02 |
| C139 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_fap_seed42_20261018/forecast_FAP_seed42_normreload_20261018/Niushou/FAP_best.pt | 0.1 | E02 |
| C140 | refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_fap_seed42_20261018/forecast_FAP_seed42_normreload_20261018/Wangzhai/FAP_best.pt | 0.1 | E02 |

来源：

## P169 完整checkpoint目录：141–150 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C141 | refine-logs/hubei_cot_independent_20261014/audit_receipts/al_normreload_raw_20261017/runs_maskfix_v2/seed44/Mizhuang/AL_best.pt | 0.1 | E02 |
| C142 | refine-logs/hubei_cot_independent_20261014/audit_receipts/al_normreload_raw_20261017/runs_maskfix_v2/seed44/Niushou/AL_best.pt | 0.1 | E02 |
| C143 | refine-logs/hubei_cot_independent_20261014/audit_receipts/al_normreload_raw_20261017/runs_maskfix_v2/seed44/Wangzhai/AL_best.pt | 0.1 | E02 |
| C144 | refine-logs/hubei_cot_independent_20261014/audit_receipts/al_normreload_raw_20261017/runs_maskfix_v2/seed43/Mizhuang/AL_best.pt | 0.1 | E02 |
| C145 | refine-logs/hubei_cot_independent_20261014/audit_receipts/al_normreload_raw_20261017/runs_maskfix_v2/seed43/Niushou/AL_best.pt | 0.1 | E02 |
| C146 | refine-logs/hubei_cot_independent_20261014/audit_receipts/al_normreload_raw_20261017/runs_maskfix_v2/seed43/Wangzhai/AL_best.pt | 0.1 | E02 |
| C147 | refine-logs/hubei_cot_independent_20261014/audit_receipts/al_normreload_raw_20261017/runs_maskfix_v2/seed42/Mizhuang/AL_best.pt | 0.1 | E02 |
| C148 | refine-logs/hubei_cot_independent_20261014/audit_receipts/al_normreload_raw_20261017/runs_maskfix_v2/seed42/Niushou/AL_best.pt | 0.1 | E02 |
| C149 | refine-logs/hubei_cot_independent_20261014/audit_receipts/al_normreload_raw_20261017/runs_maskfix_v2/seed42/Wangzhai/AL_best.pt | 0.1 | E02 |
| C150 | refine-logs/hubei_cot_independent_20261014/al_20261017/frozen_R_20261016/best.pt | 1.4 | E02 |

来源：

## P170 完整checkpoint目录：151–160 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C151 | refine-logs/hubei_cot_independent_20261014/al_20261017/runs_maskfix_v2/seed44/Mizhuang/AL_best.pt | 0.1 | E02 |
| C152 | refine-logs/hubei_cot_independent_20261014/al_20261017/runs_maskfix_v2/seed44/Niushou/AL_best.pt | 0.1 | E02 |
| C153 | refine-logs/hubei_cot_independent_20261014/al_20261017/runs_maskfix_v2/seed44/Wangzhai/AL_best.pt | 0.1 | E02 |
| C154 | refine-logs/hubei_cot_independent_20261014/al_20261017/runs_maskfix_v2/seed43/Mizhuang/AL_best.pt | 0.1 | E02 |
| C155 | refine-logs/hubei_cot_independent_20261014/al_20261017/runs_maskfix_v2/seed43/Niushou/AL_best.pt | 0.1 | E02 |
| C156 | refine-logs/hubei_cot_independent_20261014/al_20261017/runs_maskfix_v2/seed43/Wangzhai/AL_best.pt | 0.1 | E02 |
| C157 | refine-logs/hubei_cot_independent_20261014/al_20261017/runs_maskfix_v2/seed42/Mizhuang/AL_best.pt | 0.1 | E02 |
| C158 | refine-logs/hubei_cot_independent_20261014/al_20261017/runs_maskfix_v2/seed42/Niushou/AL_best.pt | 0.1 | E02 |
| C159 | refine-logs/hubei_cot_independent_20261014/al_20261017/runs_maskfix_v2/seed42/Wangzhai/AL_best.pt | 0.1 | E02 |
| C160 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed44/Mizhuang/AC_best.pt | 0.1 | E02 |

来源：

## P171 完整checkpoint目录：161–170 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C161 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed44/Mizhuang/AP_best.pt | 0.1 | E02 |
| C162 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed44/Mizhuang/A_best.pt | 0.1 | E02 |
| C163 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed44/Niushou/AC_best.pt | 0.1 | E02 |
| C164 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed44/Niushou/AP_best.pt | 0.1 | E02 |
| C165 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed44/Niushou/A_best.pt | 0.1 | E02 |
| C166 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed44/Wangzhai/AC_best.pt | 0.1 | E02 |
| C167 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed44/Wangzhai/AP_best.pt | 0.1 | E02 |
| C168 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed44/Wangzhai/A_best.pt | 0.1 | E02 |
| C169 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed43/Mizhuang/AC_best.pt | 0.1 | E02 |
| C170 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed43/Mizhuang/AP_best.pt | 0.1 | E02 |

来源：

## P172 完整checkpoint目录：171–180 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C171 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed43/Mizhuang/A_best.pt | 0.1 | E02 |
| C172 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed43/Niushou/AC_best.pt | 0.1 | E02 |
| C173 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed43/Niushou/AP_best.pt | 0.1 | E02 |
| C174 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed43/Niushou/A_best.pt | 0.1 | E02 |
| C175 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed43/Wangzhai/AC_best.pt | 0.1 | E02 |
| C176 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed43/Wangzhai/AP_best.pt | 0.1 | E02 |
| C177 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed43/Wangzhai/A_best.pt | 0.1 | E02 |
| C178 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed42/Mizhuang/AC_best.pt | 0.1 | E02 |
| C179 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed42/Mizhuang/AP_best.pt | 0.1 | E02 |
| C180 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed42/Mizhuang/A_best.pt | 0.1 | E02 |

来源：

## P173 完整checkpoint目录：181–190 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C181 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed42/Niushou/AC_best.pt | 0.1 | E02 |
| C182 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed42/Niushou/AP_best.pt | 0.1 | E02 |
| C183 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed42/Niushou/A_best.pt | 0.1 | E02 |
| C184 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed42/Wangzhai/AC_best.pt | 0.1 | E02 |
| C185 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed42/Wangzhai/AP_best.pt | 0.1 | E02 |
| C186 | refine-logs/hubei_cot_independent_20261014/al_20261017/baseline_runs/seed42/Wangzhai/A_best.pt | 0.1 | E02 |
| C187 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/A_seed42/best.pt | 0.2 | E30 |
| C188 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/A_seed42/last.pt | 0.2 | E30 |
| C189 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/A_seed43/best.pt | 0.2 | E30 |
| C190 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/A_seed43/last.pt | 0.2 | E30 |

来源：

## P174 完整checkpoint目录：191–200 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C191 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/A_seed44/best.pt | 0.2 | E30 |
| C192 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/A_seed44/last.pt | 0.2 | E30 |
| C193 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/hybrid_seed42/best.pt | 0.2 | E30 |
| C194 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/hybrid_seed42/last.pt | 0.2 | E30 |
| C195 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/hybrid_seed43/best.pt | 0.2 | E30 |
| C196 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/hybrid_seed43/last.pt | 0.2 | E30 |
| C197 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/hybrid_seed44/best.pt | 0.2 | E30 |
| C198 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/hybrid_seed44/last.pt | 0.2 | E30 |
| C199 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/transport_seed42/best.pt | 0.2 | E30 |
| C200 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/transport_seed42/last.pt | 0.2 | E30 |

来源：

## P175 完整checkpoint目录：201–210 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C201 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/transport_seed43/best.pt | 0.2 | E30 |
| C202 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/transport_seed43/last.pt | 0.2 | E30 |
| C203 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/transport_seed44/best.pt | 0.2 | E30 |
| C204 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/transport_seed44/last.pt | 0.2 | E30 |
| C205 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/results_attempt1/A_seed42/best.pt | 0.2 | E30 |
| C206 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/results_attempt1/A_seed42/last.pt | 0.2 | E30 |
| C207 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/results_attempt1/A_seed43/best.pt | 0.2 | E30 |
| C208 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/results_attempt1/A_seed43/last.pt | 0.2 | E30 |
| C209 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/results_attempt1/A_seed44/best.pt | 0.2 | E30 |
| C210 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/results_attempt1/A_seed44/last.pt | 0.2 | E30 |

来源：

## P176 完整checkpoint目录：211–220 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C211 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/results_attempt1/hybrid_seed42/best.pt | 0.2 | E30 |
| C212 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/results_attempt1/hybrid_seed42/last.pt | 0.2 | E30 |
| C213 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/results_attempt1/hybrid_seed43/best.pt | 0.2 | E30 |
| C214 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/results_attempt1/hybrid_seed43/last.pt | 0.2 | E30 |
| C215 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/results_attempt1/hybrid_seed44/best.pt | 0.2 | E30 |
| C216 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/results_attempt1/hybrid_seed44/last.pt | 0.2 | E30 |
| C217 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/results_attempt1/transport_seed42/best.pt | 0.2 | E30 |
| C218 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/results_attempt1/transport_seed42/last.pt | 0.2 | E30 |
| C219 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/results_attempt1/transport_seed43/best.pt | 0.2 | E30 |
| C220 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/results_attempt1/transport_seed43/last.pt | 0.2 | E30 |

来源：

## P177 完整checkpoint目录：221–230 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C221 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/results_attempt1/transport_seed44/best.pt | 0.2 | E30 |
| C222 | refine-logs/history_cot_forecast_ghi_20261015/results_attempt3/results_attempt1/transport_seed44/last.pt | 0.2 | E30 |
| C223 | refine-logs/cot_retrieved_forecast_20261013/results/amplitude_loss_20261017_seed42_b1024_e12/P_Hybrid_best.pt | 8.8 | E23 |
| C224 | refine-logs/cot_cloud_gate_retrain_20261014/remote_result_c1_attempt3/fold0_last.pt | 0.1 | E20 |
| C225 | refine-logs/cot_cloud_gate_retrain_20261014/remote_result_c1_attempt3/fold1_last.pt | 0.1 | E20 |
| C226 | refine-logs/cot_cloud_gate_retrain_20261014/remote_result_c1_attempt3/fold2_last.pt | 0.1 | E20 |
| C227 | refine-logs/cot_cloud_gate_retrain_20261014/remote_result_c1_attempt3/full_train_last.pt | 0.2 | E20 |
| C228 | refine-logs/cot_cloud_gate_retrain_20261014/remote_result_retry1/best.pt | 0.0 | E20 |
| C229 | experiments/audit_sili_pilot_snapshot_20260928/best.pt | 0.3 | 历史/共享；见原目录 |
| C230 | experiments/monthly_R_station16_20261013_v1/results/seed42_v1/best.pt | 5.4 | E01 |

来源：

## P178 完整checkpoint目录：231–240 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C231 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_zhujia_20261014_v1/seed44/zhujia/AGRI_best.pt | 0.2 | E01 |
| C232 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_zhujia_20261014_v1/seed44/zhujia/AGRI_COT_best.pt | 0.2 | E01 |
| C233 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_zhujia_20261014_v1/seed43/zhujia/AGRI_best.pt | 0.2 | E01 |
| C234 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_zhujia_20261014_v1/seed43/zhujia/AGRI_COT_best.pt | 0.2 | E01 |
| C235 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_zhujia_20261014_v1/seed42/zhujia/AGRI_best.pt | 0.2 | E01 |
| C236 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_zhujia_20261014_v1/seed42/zhujia/AGRI_COT_best.pt | 0.2 | E01 |
| C237 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_oof_v3_unified_20261014_v1/seed44/sili/AGRI_best.pt | 0.1 | E01 |
| C238 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_oof_v3_unified_20261014_v1/seed44/sili/AGRI_COT_best.pt | 0.1 | E01 |
| C239 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_oof_v3_unified_20261014_v1/seed44/zhujia/AGRI_best.pt | 0.1 | E01 |
| C240 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_oof_v3_unified_20261014_v1/seed44/zhujia/AGRI_COT_best.pt | 0.1 | E01 |

来源：

## P179 完整checkpoint目录：241–250 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C241 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_oof_v3_unified_20261014_v1/seed43/sili/AGRI_best.pt | 0.1 | E01 |
| C242 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_oof_v3_unified_20261014_v1/seed43/sili/AGRI_COT_best.pt | 0.1 | E01 |
| C243 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_oof_v3_unified_20261014_v1/seed43/zhujia/AGRI_best.pt | 0.1 | E01 |
| C244 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_oof_v3_unified_20261014_v1/seed43/zhujia/AGRI_COT_best.pt | 0.1 | E01 |
| C245 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_oof_v3_unified_20261014_v1/seed42/sili/AGRI_best.pt | 0.1 | E01 |
| C246 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_oof_v3_unified_20261014_v1/seed42/sili/AGRI_COT_best.pt | 0.1 | E01 |
| C247 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_oof_v3_unified_20261014_v1/seed42/zhujia/AGRI_best.pt | 0.1 | E01 |
| C248 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_oof_v3_unified_20261014_v1/seed42/zhujia/AGRI_COT_best.pt | 0.1 | E01 |
| C249 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_direction_20261014_v3/seed44/sili/AGRI_best.pt | 0.2 | E01 |
| C250 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_direction_20261014_v3/seed44/sili/AGRI_COT_best.pt | 0.2 | E01 |

来源：

## P180 完整checkpoint目录：251–260 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C251 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_direction_20261014_v3/seed43/sili/AGRI_best.pt | 0.2 | E01 |
| C252 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_direction_20261014_v3/seed43/sili/AGRI_COT_best.pt | 0.2 | E01 |
| C253 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_direction_20261014_v3/seed42/sili/AGRI_best.pt | 0.2 | E01 |
| C254 | experiments/monthly_R_station16_20261013_v1/results/signed_residual_direction_20261014_v3/seed42/sili/AGRI_COT_best.pt | 0.2 | E01 |
| C255 | experiments/monthly_R_station16_20261013_v1/results/ghi_heads_new_r_seed44_v1/sili/AC_best.pt | 0.1 | E01 |
| C256 | experiments/monthly_R_station16_20261013_v1/results/ghi_heads_new_r_seed44_v1/sili/AP_best.pt | 0.1 | E01 |
| C257 | experiments/monthly_R_station16_20261013_v1/results/ghi_heads_new_r_seed44_v1/sili/A_best.pt | 0.1 | E01 |
| C258 | experiments/monthly_R_station16_20261013_v1/results/ghi_heads_new_r_seed44_v1/zhujia/AC_best.pt | 0.1 | E01 |
| C259 | experiments/monthly_R_station16_20261013_v1/results/ghi_heads_new_r_seed44_v1/zhujia/AP_best.pt | 0.1 | E01 |
| C260 | experiments/monthly_R_station16_20261013_v1/results/ghi_heads_new_r_seed44_v1/zhujia/A_best.pt | 0.1 | E01 |

来源：

## P181 完整checkpoint目录：261–270 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C261 | experiments/monthly_R_station16_20261013_v1/results/ghi_heads_new_r_seed43_v1/sili/AC_best.pt | 0.1 | E01 |
| C262 | experiments/monthly_R_station16_20261013_v1/results/ghi_heads_new_r_seed43_v1/sili/AP_best.pt | 0.1 | E01 |
| C263 | experiments/monthly_R_station16_20261013_v1/results/ghi_heads_new_r_seed43_v1/sili/A_best.pt | 0.1 | E01 |
| C264 | experiments/monthly_R_station16_20261013_v1/results/ghi_heads_new_r_seed43_v1/zhujia/AC_best.pt | 0.1 | E01 |
| C265 | experiments/monthly_R_station16_20261013_v1/results/ghi_heads_new_r_seed43_v1/zhujia/AP_best.pt | 0.1 | E01 |
| C266 | experiments/monthly_R_station16_20261013_v1/results/ghi_heads_new_r_seed43_v1/zhujia/A_best.pt | 0.1 | E01 |
| C267 | experiments/monthly_R_station16_20261013_v1/results/ghi_heads_new_r_seed42_v1/sili/AC_best.pt | 0.1 | E01 |
| C268 | experiments/monthly_R_station16_20261013_v1/results/ghi_heads_new_r_seed42_v1/sili/AP_best.pt | 0.1 | E01 |
| C269 | experiments/monthly_R_station16_20261013_v1/results/ghi_heads_new_r_seed42_v1/sili/A_best.pt | 0.1 | E01 |
| C270 | experiments/monthly_R_station16_20261013_v1/results/ghi_heads_new_r_seed42_v1/zhujia/AC_best.pt | 0.1 | E01 |

来源：

## P182 完整checkpoint目录：271–280 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C271 | experiments/monthly_R_station16_20261013_v1/results/ghi_heads_new_r_seed42_v1/zhujia/AP_best.pt | 0.1 | E01 |
| C272 | experiments/monthly_R_station16_20261013_v1/results/ghi_heads_new_r_seed42_v1/zhujia/A_best.pt | 0.1 | E01 |
| C273 | experiments/hubei_r_runtime_gate_20261015_v1/runs/hubei_R_station16_fixedv2_v1_seed42/best.pt | 1.4 | 历史/共享；见原目录 |
| C274 | experiments/hubei_r_runtime_gate_20261015_v1/runs/hubei_R_station16_fixedv2_v1_seed42/last.pt | 1.4 | 历史/共享；见原目录 |
| C275 | experiments/high_kt_loss_20261016/runs/formal_v1/seed44/zhujia/AC_bright2x/best.pt | 0.2 | E15 |
| C276 | experiments/high_kt_loss_20261016/runs/formal_v1/seed44/zhujia/AC_uniform/best.pt | 0.2 | E15 |
| C277 | experiments/high_kt_loss_20261016/runs/formal_v1/seed44/zhujia/A_bright2x/best.pt | 0.2 | E15 |
| C278 | experiments/high_kt_loss_20261016/runs/formal_v1/seed44/zhujia/A_uniform/best.pt | 0.2 | E15 |
| C279 | experiments/high_kt_loss_20261016/runs/formal_v1/seed44/sili/AC_bright2x/best.pt | 0.2 | E15 |
| C280 | experiments/high_kt_loss_20261016/runs/formal_v1/seed44/sili/AC_uniform/best.pt | 0.2 | E15 |

来源：

## P183 完整checkpoint目录：281–290 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C281 | experiments/high_kt_loss_20261016/runs/formal_v1/seed44/sili/A_bright2x/best.pt | 0.2 | E15 |
| C282 | experiments/high_kt_loss_20261016/runs/formal_v1/seed44/sili/A_uniform/best.pt | 0.2 | E15 |
| C283 | experiments/high_kt_loss_20261016/runs/formal_v1/seed43/zhujia/AC_bright2x/best.pt | 0.2 | E15 |
| C284 | experiments/high_kt_loss_20261016/runs/formal_v1/seed43/zhujia/AC_uniform/best.pt | 0.2 | E15 |
| C285 | experiments/high_kt_loss_20261016/runs/formal_v1/seed43/zhujia/A_bright2x/best.pt | 0.2 | E15 |
| C286 | experiments/high_kt_loss_20261016/runs/formal_v1/seed43/zhujia/A_uniform/best.pt | 0.2 | E15 |
| C287 | experiments/high_kt_loss_20261016/runs/formal_v1/seed43/sili/AC_bright2x/best.pt | 0.2 | E15 |
| C288 | experiments/high_kt_loss_20261016/runs/formal_v1/seed43/sili/AC_uniform/best.pt | 0.2 | E15 |
| C289 | experiments/high_kt_loss_20261016/runs/formal_v1/seed43/sili/A_bright2x/best.pt | 0.2 | E15 |
| C290 | experiments/high_kt_loss_20261016/runs/formal_v1/seed43/sili/A_uniform/best.pt | 0.2 | E15 |

来源：

## P184 完整checkpoint目录：291–300 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C291 | experiments/high_kt_loss_20261016/runs/formal_v1/seed42/zhujia/AC_bright2x/best.pt | 0.2 | E15 |
| C292 | experiments/high_kt_loss_20261016/runs/formal_v1/seed42/zhujia/AC_uniform/best.pt | 0.2 | E15 |
| C293 | experiments/high_kt_loss_20261016/runs/formal_v1/seed42/zhujia/A_bright2x/best.pt | 0.2 | E15 |
| C294 | experiments/high_kt_loss_20261016/runs/formal_v1/seed42/zhujia/A_uniform/best.pt | 0.2 | E15 |
| C295 | experiments/high_kt_loss_20261016/runs/formal_v1/seed42/sili/AC_bright2x/best.pt | 0.2 | E15 |
| C296 | experiments/high_kt_loss_20261016/runs/formal_v1/seed42/sili/AC_uniform/best.pt | 0.2 | E15 |
| C297 | experiments/high_kt_loss_20261016/runs/formal_v1/seed42/sili/A_bright2x/best.pt | 0.2 | E15 |
| C298 | experiments/high_kt_loss_20261016/runs/formal_v1/seed42/sili/A_uniform/best.pt | 0.2 | E15 |
| C299 | experiments/direct_cot_monthly_20261005_v1/m1_smoke/run_output/smoke_last.pt | 8.8 | 历史/共享；见原目录 |
| C300 | audits/cpp_producer/FY4B_CPP_V2_Portable_20260917/__MACOSX/FY4B_CPP_V2_Portable/Model/CER/._FY4B_CER_best.pth | 0.0 | 历史/共享；见原目录 |

来源：

## P185 完整checkpoint目录：301–307 / 307

本页意义：便于定位已有权重；不将本地存在误写成重新完成重载验收。

评价范围：本次实际文件盘点；绝对路径保存在CSV

| 编号 | 本地权重（仓库相对路径） | 大小MiB | 关联实验 |
| --- | --- | --- | --- |
| C301 | audits/cpp_producer/FY4B_CPP_V2_Portable_20260917/__MACOSX/FY4B_CPP_V2_Portable/Model/CLP/._FY4B_CLP-IR_best.pth | 0.0 | 历史/共享；见原目录 |
| C302 | audits/cpp_producer/FY4B_CPP_V2_Portable_20260917/__MACOSX/FY4B_CPP_V2_Portable/Model/COT/._FY4B_COT_best.pth | 0.0 | 历史/共享；见原目录 |
| C303 | audits/cpp_producer/FY4B_CPP_V2_Portable_20260917/__MACOSX/FY4B_CPP_V2_Portable/Model/CTH/._FY4B_CTH_best.pth | 0.0 | 历史/共享；见原目录 |
| C304 | audits/cpp_producer/FY4B_CPP_V2_Portable_20260917/FY4B_CPP_V2_Portable/Model/CER/FY4B_CER_best.pth | 15.5 | 历史/共享；见原目录 |
| C305 | audits/cpp_producer/FY4B_CPP_V2_Portable_20260917/FY4B_CPP_V2_Portable/Model/CLP/FY4B_CLP-IR_best.pth | 15.5 | 历史/共享；见原目录 |
| C306 | audits/cpp_producer/FY4B_CPP_V2_Portable_20260917/FY4B_CPP_V2_Portable/Model/COT/FY4B_COT_best.pth | 15.5 | 历史/共享；见原目录 |
| C307 | audits/cpp_producer/FY4B_CPP_V2_Portable_20260917/FY4B_CPP_V2_Portable/Model/CTH/FY4B_CTH_best.pth | 15.5 | 历史/共享；见原目录 |

来源：

## P186 作废材料：哪些CPP评价图不能再引用？

本页意义：旧CPP局部/全局索引错配的图已作废；旧29.13/23.04等数值不作为结果收录。

评价范围：工程更正记录；不与修复后分数混算

| 材料 | 处理 | 当前采用来源 |
| --- | --- | --- |
| cot_forecast_eval旧v1–v5及相关bank/个例旧图 | INVALID_CPP_JOIN目录标记；不收录原错误图 | paper/figures/fig7_forecast_cot_summary_20261017.json |
| 旧8日期271窗口修正值 | 有界历史诊断；不冒充完整验证 | 最新3,140序列、46CPP日期同掩膜评价 |
| 湖北旧未保存归一化run汇总 | 保留历史；不作为最新主表 | 2026-10-17 normreload与10-18独立复算 |
| 合成COT示意场 | 仅用于解释结构 | 本册优先收录真实数据图 |
| 未完成OOF训练/预测 | 无最终成绩可填 | 单列在途状态，不预写收益 |

来源：