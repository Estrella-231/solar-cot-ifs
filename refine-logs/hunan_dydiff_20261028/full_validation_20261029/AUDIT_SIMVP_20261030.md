# SimVP+COT完整验证增益独立审计

日期：2026-10-30。审计者：独立子agent `/root/audit_simvp_full_gain`；保存数组数值复核由其子agent `/root/audit_simvp_full_gain/numeric_check` 独立完成，未导入原评分函数。审计按 `experiment-audit` 执行；`review_independence: same-family`，`acceptance_status: provisional`。具体模型标识未由本次审计接口公开，故不推定模型名称。

**总体：WARN。确定性算分、共同样本、主预报输入及现有权重来源检查PASS；目前未发现使1.633%增益失效的实现错误。结果仍是重复使用验证集上的冻结头回放，不是独立测试。** 发现并修复本地合并完成记录的分站数与耗时元数据错误，未改模型、样本、预测或评分结果；服务器保持只读。

## 1. 独立复算结果：PASS

每个站点、seed先计算自己的RMSE，再将六组等权平均；没有先平均三种子的预测。以下单位为W/m²，差值为M10−M00。

| 范围 | 无COT | 加预测COT | 降幅 | 差值 | 原目标日bootstrap 95%区间 |
|---|---:|---:|---:|---:|---|
| 全部冻结窗口的有效GHI记录 | 127.421176274 | 125.340128428 | 1.633204% | −2.081047846 | [−3.714465349,−0.561349539] |
| 预定无湖北训练日期重叠子组 | 125.142223330 | 122.468108156 | 2.136861% | −2.674115174 | [−4.430037651,−0.890490678] |

六个站点×seed均为点改善。四里完整验证的三个差值为−3.070619、−2.862133、−2.631408；竺家为−1.763264、−0.801982、−1.356881。**竺家的单站及各seed日期区间跨零，不能表述为“两站均显著”。**

原评分代码按同一索引配对两臂与truth，`score_saved.py:58–78`；原始RMSE为误差平方均值的平方根，`score_saved.py:9–12`。降幅分母为无COT RMSE，未用模型输出的最大值、均值或范围做分数缩放。独立输出：`audit_simvp_numeric_20261030/independent_numeric_audit.json`。

## 2. 标签、时间和共同样本：PASS，覆盖范围WARN

- GHI来源为冻结站点标签包的`ghi`，晴空量来自同一行`clear`，不是R或SimVP输出：`evaluate_full_gpu.py:62,130–135,169,199`。重新与原头`validation_inputs.npz`按`pack_index`连接，63480行全部匹配；晴空值完全相同，GHI最大差仅0.000040694 W/m²，来自CSV双精度与原训练float32保存差异，见`audit_simvp_sources_20261030/label_join_recheck.json`。本次没有重新从原始工作簿生产标签。
- 63480条保留记录＋37000条排除记录＝3140窗口×2站×16时效；排除为36232条缺冻结有效GHI标签、768条既有四里无效日。全部3140窗口均有处理，但**只有2728窗口贡献有效GHI，4171个唯一station/target**，不是3140窗口全部有完整GHI标签。
- canonical `(station,init,target)`重复为0；lead均为+15…+240分钟，时间差和BJT日期错误为0。相同station/target最多跨16个起报窗口重复，其truth与clear完全一致。BJT下init与target跨日为0，不能按UTC日误分日期块。
- 六组M00/M10均在共同站点索引上有限，未因某臂误差或缺失选择不同样本；既定四里2024-09-05全日排除一致。
- 当前GHI范围含早晚低太阳高度：全部目标按标签使用的NOAA公式计算为SZA<90°，其中2013行SZA>85°、5653行SZA>80°。这与R/CPP严格日间支持不是同一掩膜，**不能把GHI全表写成SOZ<80°样本**。clear最小1.516726，GHI=0有585行、可疑持续零标志246行；这些非既定排除行保留，未为了提高增益删除。

标签包构建按`main_mask`和正晴空量选行：`../../../experiments/monthly_R_station16_20261013_v1/prepare_ghi_head_pack.py:52–68`；三帧共同时间支持再按同split/同BJT日期连接，`../../cot_interval_history_20261015/audit_and_prepare.py:117–126`。因此“完整验证”应解释为全部冻结卫星验证窗口中的共同有效GHI记录，不能称全年所有时刻或所有原始GHI验证记录。

## 3. 主预报输入和无COT臂：PASS

- SimVP只消费八张历史AGRI；未来只加入已知太阳几何。未来AGRI整段由SimVP输出替换，未来COT由同一修正R作用于预测AGRI得到：`evaluate_full_gpu.py:105–118`。逐时效取目标前30分钟、前15分钟和目标时刻三帧，前两时效含已知历史帧，+45…240分钟三帧全部为预测场：`evaluate_full_gpu.py:133–154`。
- 未来预测槽的源QA被覆盖为预测自身有限值，未让真实未来有效性成为头输入，`evaluate_full_gpu.py:138–144`。真实未来R响应只放在明确命名的oracle分支；CPP只写COT评价参考，未进入GHI主输入：`evaluate_full_gpu.py:122–129,150–154`。原16窗口的canary检查可作为已有补充证据，但本次未声称重新执行全3140窗口canary。
- M00训练、验证和回放均明确将COT槽15清零；两臂p_cloud槽16均为0。来源训练代码`../../cloud_probability_input_20261020/train_four_arm.py:430–434,449–453,473–477`与服务器实际文件SHA一致；回放代码`evaluate_full_gpu.py:185,195–197`。
- 只读实际加载12个冻结头checkpoint，六个M00的COT/p_cloud第一层权重均逐值为0；六个M10的COT权重非0而p_cloud权重均0。12个checkpoint SHA与完成记录一致；详见`audit_simvp_sources_20261030/remote_provenance.json:58–215`。因此无COT臂是独立训练的M00，并非对训练好的M10临时置零后冒充基线。

## 4. 模型和归一化来源：PASS至已检查层级

- 实际SimVP checkpoint metadata为湖南、随机初始化、新月度split、seed42、训练35320/验证3140、history8/target16、geometry24；其checkpoint、归一化、模型代码SHA与训练记录一致。保存最优checkpoint的`epoch`字段为10，不能与旧湖北16通道权重混同。
- R为修正太阳几何的`7273a21d…`，归一化为`a1fcb466…`；R训练代码仅以train行共同CPP像元拟合均值/标准差：`../../../experiments/r_solar_geometry_corrected_20261015/train_paired_r.py:91–107`。R输出本来就是log1p(COT)，头仅做训练统计标准化，没有再次log1p，`evaluate_full_gpu.py:146`及`train_four_arm.py:176–180`。
- 头为20261020同期真实三帧、3×3的42529参数FourArmIntervalHead，固定30epoch、batch1792、配对初始化/样本顺序/优化器；头按原验证GHI RMSE选最优epoch，`train_four_arm.py:400–465`。四个分片的R/S/pack/头setup/脚本来源完全一致，实际远端五个执行脚本SHA在修复合并器前均与本地相同。
- 本审计没有重建CPP生产链、原始工作簿或全部归一化拟合载荷；该上游部分只能确认已存来源、执行代码和hash，不能称独立认证的COT物理真值。

## 5. 日期相关性和统计边界：WARN，敏感性未推翻点增益

`score_saved.py:13–25`实际重采样单位是单个目标BJT日期；所有方法/站点/seed共用同一日期权重，已保留同日重叠起报相关性，未把63480行当作独立事件。其`np.maximum(denom,1)`会在空组重采样时隐藏空组，但本次所有重采样分母均非零，没有影响本次区间。

为检查相邻两天相关性，直接读取原头已保存的26块`date_blocks.csv`（SHA `7cbbc841…`），沿用原`score_four_arm.py:147–175`的共同整数抽样及seed20261020，5000次；子组保留原26块，含没有子组样本的块。**这是事后敏感性，不替换主表。**

| 范围 | 原冻结两日块95%差值区间 |
|---|---|
| 全表 | [−3.870198910,−0.405658217] |
| 无湖北训练日期重叠 | [−4.764495821,−0.728160654] |

全表仅保留clear≥50/≥100时，改善分别为1.635587%/1.625355%，两日块区间仍不跨零。因此当前收益不是低光照记录独自造成。完整敏感性结果见`audit_simvp_numeric_20261030/frozen_blocks_and_clear_sensitivity.json`。

**最主要的科学风险仍是验证集重复使用与选模。** R、SimVP、GHI头均曾用该验证划分选模，bootstrap不校正长期方法搜索和checkpoint选择偏差。原头是同期真实图训练，回放预报场还存在输入分布差异。名义参数量相同也不隔离纯物理作用：M00的COT输入权重没有有效输入，R又把16×16上下文转为中心3×3 COT。只能称派生表示的条件增量，不能称COT独立观测价值、纯物理因果或部署提升。

## 6. 确定缺陷、死代码和修复

**发现并修复：合并收据继承rank0元数据。** 原`merge_shards.py:27`直接展开`receipts[0]`而没有汇总`stations`、重定义耗时，导致总GHI_rows=63480但站点数仅7652/8268；实际CSV及评分是30516/32964。原文件备份为`audit_simvp_sources_20261030/merge_shards_before_fix.py`和`completion_before_fix.json`。

本地合并器现从实际合并行计算站点数并与分片计数交叉核对；耗时定义为最大评价分片时长652.346508秒，明确排除生成、排队和合并评分。已修当前本地completion。`METADATA_FIX.json`保存前后hash；ghi_rows、ghi_predictions与ghi_scores的SHA均未改变。此错误影响元数据可信度，**不产生1.633%增益**。远端原始收据保留。

其他非本次分数缺陷：`evaluate_full_gpu.py:3,157,200,203`遗留CPU措辞，但实际device及completion为GPU_FP32；`score_extended.py:10`的`sub.mkdir()`使原目录重评分不可直接幂等运行；`generate_full.py:44,62–64`计算的AGRI sums未输出评分结果，不应宣称存在完整AGRI量化结果。这些未改变本次GHI对照，未扩大修复范围。

评分函数确实由`score_extended.py:6,18`调用，保存结果可由原数组独立复现；没有phantom GHI指标。

## 7. 与旧负结果的关系及允许结论

早期四统计量、输运/残差和湖北其他头的负结果不等同于当前“湖南SimVP→修正R→同期真实三帧头”合同。旧四统计配置不支持普遍COT无效，见`../../../docs/COT_PAIR_RESULT_TO_CLAIM_20260908.md:25,45–52`。本次与原16窗口使用同一批头，原16窗口SimVP已是3.301%点改善、区间跨零；扩至全表后点改善较小、日期更多。不能把转为区间不跨零表述成SimVP突然训练变好，也不能把差异归因于某一项修复。

允许结论：**在当前湖南冻结验证窗口的共同有效GHI记录中，冻结SimVP→修正R与现有配对头产生1.633%的macro RMSE点改善；原逐日及冻结两日块敏感性区间均不跨零。** 这不是独立test、不是每站均显著，不证明预测COT优于真实COT或任何单项方法改动的因果效果。历史合同对照另见`HISTORICAL_COMPARISON_20261030.md`。

评价类型：GHI为real_gt（站点实测标签）；CPP为检索参考；R(真实未来AGRI)为同模型响应代理；oracle仅离线诊断。test载荷未读取，没有训练或提交新任务。
