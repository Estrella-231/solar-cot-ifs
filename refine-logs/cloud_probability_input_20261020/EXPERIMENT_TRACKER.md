# 云状态条件输入任务表

2026-10-20。判云分类器太阳几何与三帧输入对齐已核验；PBS `215069.tc6000` 已在浙师大 node22/A800 GPU4 完成24/24配对训练，保存预测已FP64独立评分。C1（已有COT上增加判云）未通过冻结判据；B3二值对照不启动。B2总体/分层/高GHI核心数值完成，真实连续事件图尚待整理。test未读取。

| 任务 | 回答的问题/产物 | 状态 |
|---|---|---|
| P00 历史核查 | `HISTORY_AUDIT_20261020.md`；不重复硬门控、oracle、hurdle | COMPLETE |
| P01 文献与方法选择 | `LITERATURE_METHOD_DECISION_20261020.md`；保留COT＋独立判云通道作为待验证方法 | COMPLETE |
| P02 旧门控新观测日复算 | `OLD_GATE_DAYCLEAN_20261020.json`；等站差+.3344 W/m²，未证实改善 | COMPLETE_FIXED_WEIGHT_RESCORE |
| P03 资产与键覆盖 | `ASSET_AVAILABILITY_20261020.json`；54,895/54,895键匹配，数组载荷未复核 | COMPLETE_METADATA；SSH结束255已注明 |
| P04 概率接入三帧包 | 精确站点-帧连接、中心3×3、OOF来源及4,226条有效val | PASS_PREFLIGHT; runtime join in PBS |
| P05 17槽公共头 | 同初始化、同初始函数、通道梯度、42,529参数 | CPU_FORWARD_PASS; GPU_WITNESS_IN_JOB |
| P06 四组两站三seed | 24次小头训练；30epoch，同1792有效batch；保存best/last/曲线/逐样本预测 | COMPLETE `215069.tc6000` |
| P07 配对统计与实图 | 总体FP64、日期块CI、CPP/CLP分层、高GHI诊断及配对差图 | CORE_NUMBERS_COMPLETE; 连续真实事件图待整理 |
| P08 二值/连续对照 | 仅在P06/P07支持时追加6次；不乘COT | CLOSED_NOT_RUN：C1未支持 |
| P09 湖北与预报 | 独立来源、因果未来判云，分别定义后续实验 | CONDITIONAL_NOT_STARTED |

详细配置见 `EXPERIMENT_PLAN.md`。训练后的正负结果都需保存；不以不断改阈值或扩网络代替本轮判断。

## 执行记录 2026-10-20

- 太阳几何：分类器训练张量与旧辅助文件几何逐像元一致（最大绝对差0）；同一批96个UTC时刻/192个站点裁块与修正几何相比，`cosSOZ`平均绝对差0.01308、`cosRAA`为0.04026，日掩膜差异765像元。故本轮只把现有冻结分类器得分作为**旧几何版本的独立连续输入**，没有把修正几何强行送入旧权重；不得把本轮称作修正几何判云实验。
- 键对齐：54,895/54,895个所需站点-帧键覆盖；三帧固定为UTC `T−30/T−15/T`，站点行列、split、AGRI路径及验证集分类器来源均匹配，缺键/错split/错站点/错赋值均为0。判云使用其站点16×16数组的`[7:10,7:10]`中心3×3，与AGRI、修正R输出对齐。
- 观测日：按权威规则仅排除四里北京时间2024-09-05的37条验证目标；训练不删行，清理后验证4226条（四里1999、竺家2227）。
- 实现：新增17槽四臂公共头训练器及PBS脚本；本地`py_compile`、远端CPU前向均通过，参数42,529。GPU内会先做通道梯度检查和一批更新的smoke，再运行全部24次拟定训练。
- 提交前`qstat -u slfu`为1个未结束作业；提交后为2个。PBS `215069.tc6000` 当前状态R，`node22`、1张A800（GPU4），墙钟2小时；日志 `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/cloud_probability_input_20261020/logs/four_arm_attempt1.log`，输出目录 `.../results/four_arm_paired_v1_attempt1/`。
- 上述为提交时的阶段记录；最终训练和评分结论见下方“结果更新 2026-10-20”。

## 结果更新 2026-10-20

- PBS `215069.tc6000` 完成：浙师大 node22，一张A800 GPU4；24/24 arm×station×seed均完成，30 epoch；本地镜像目录`results/four_arm_paired_v1_attempt1/`，远端输出`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/cloud_probability_input_20261020/results/four_arm_paired_v1_attempt1/`，日志`.../logs/four_arm_attempt1.log`。全量哈希及完成记录审计通过。
- 等站等seed FP64 RMSE：AGRI+geometry 89.485、+COT 88.264、+cloud score 88.186、+COT+cloud score 87.795 W/m²。主比较M11−M10为−0.469 W/m²（0.531%），两日块95%CI [−1.007,+0.047]；Sili平均差+0.074、Zhujia−1.012 W/m²。CI跨0且一站退化，C1未证实；C2也不支持，按预案不启动B3。
- COT分层仅有1,265/4,226行CPP/CLP共同有效。判云得分版本与分类器旧太阳几何绑定，不能把此结果称为修正几何分类器实验。近零/中等厚度区间CI跨0；薄云组M11−M10为+1.644 W/m²，修正后区间[-0.573,+3.207]跨0；厚云组为−1.285 W/m²，区间[-2.138,-0.425]，但只169行/6个共同支持块、多个分层未作校正，只是待独立验证的探索信号。
- GHI≥800 W/m²有219条独立目标行；M11−M10等站等seed RMSE差−1.511 W/m²，日期块区间[-5.500,+2.032]跨0。657个站点-seed配对中SSE总量下降1.57%，399对改善、258对变差；不能据此称高辐照度稳定受益。
- B2 JSON/CSV、图件和结果解释见`B2_RESULTS_20261020.md`及`results/four_arm_paired_v1_attempt1/figures/`。总体分层区间经修正后与主评分逐数一致；所有结果为同时间重复使用的验证分析，不是独立test。
- PBS完成后账户仍有既有作业`215046.tc6000`运行；未终止他人作业。未改论文、未读取test、未启动额外训练。
- 薄云专项复核：独立重算180条，六配对五退化一改善；仅7条中心判云得分<0.5且该组SSE反而改善，不能把退化简单归因于漏判晴空。两臂更新数一致；报告见THIN_CLOUD_DIAGNOSIS_20261020.md。未新增训练。
- 用户新增授权：湖北三站独立复核云状态输入，不再以湖南C1通过为前置；计划见../hubei_cloud_probability_20261020/EXPERIMENT_PLAN.md。学院远端已核查R/GHI包，湖北专属判云来源继续追溯，未提交训练。
