# COT评估执行清单：Yi × Sun v1

日期：2026-10-14（BJT）；最近状态核验：bank完成后FP64复算。
本表为COT评估子任务清单，整体工程进度仍由EXPERIMENT_TRACKER.md记录。
本轮无新增训练、无test访问。验证集同期反演复算已完成；远端预测bank检查为只读。

| ID | 关联主任务 | 工作 | 状态 | 验收 |
|---|---|---|---|---|
| E01 | T04/T19 | 时间、空间、CPP有效mask与新S/R来源落为matching_contract | COMPLETE_FOR_CPP_SUPPORTED_VAL_COHORT | 新月度SimVP checkpoint、训练metadata、split/归一化hash、S/R哈希和test_payloads_read=0已核实；完整bank有271个CPP支持val窗口、16lead时间键逐窗15min连续；仅8个窗口有两站全部16lead支持。未覆盖时段保持缺测。 |
| E02 | T21 | FP64指标、逆变换、NaN、FS和厚云计数单元检查 | COMPLETE | 合成手算RMSE/MAE/bias、NaN排除、一次FP64 expm1、FS及厚云固定阈值计数已检查；保存预测独立复算完成。 |
| E03 | T04/T21 | 保存的新月度R验证预测分尺度重算 | COMPLETE_VALIDATION_ONLY | 中心像元/3×3/16×16及覆盖重算；独立结果见COT_RETRIEVAL_VALIDATION_RESULTS_20261013.md |
| E04 | T19/T20 | 新合同val的S→R/持续性/已有输运预测表 | COMPLETE_CPP_SUPPORTED_VAL_COHORT; TRANSPORT_NOT_AVAILABLE | PBS 213858 exit_status=0；271个支持窗口、两站×16lead，数组hash与profile一致，test_payloads_read=0。100窗profile为100/100有限、端到端34.57s（2.89窗/s），batch8/6workers完整bank271窗35.47s（7.64窗/s），峰值分配/预留6.32/8.30 GiB。全bank计550,400共同CPP像元；严格完整双站16lead仅8窗。检查当前实验目录无可用既有transport bank，按计划不另开发。工程回放非精度结论。 |
| E05 | T21 | 16lead主要/所有方法公共域/固定16lead评价 | COMPLETE_VALIDATION_ONLY_CPP_SUPPORTED_COHORT | 三尺度中心/3×3/16×16及16lead逐时效FP64 RMSE/MAE/bias/FS、覆盖/支持数已落盘；唯一支持8个目标日期，未生成区域结果。主3×3逐lead宏RMSE：S→R 25.538、Persistence-R 28.252，差-2.714 COT；CPP是检索参考。结果见`results/cot_forecast_eval_20261014_v3/`。 |
| E06 | T22 | 厚云混淆、条件SSIM、配对日期bootstrap | COMPLETE_VALIDATION_DIAGNOSTIC | COT≥30预报混淆计数已完成；S→R 3×3 CSI=0.082，Persistence-R=0.189。1日BJT日期块2000次CI主区间`[-6.476,0.493]`跨0；连续2日块敏感性也跨0。仅8个独立日期，区间只作描述。训练CPP参考15,414个patch/3,921,824像元给出q99.9=74.0397；11×11 SSIM仅在完整mask窗口计算，共77,400个中心（理论候选24.8%），S→R 0.2356、持续性0.2468。结构指标未显示优势。细节见`results/cot_forecast_eval_20261014_v5/`。 |
| E07 | T22/T16 | 同色标例图与文献可比性表 | COMPLETE_BOUNDED_VALIDATION_DIAGNOSTIC | M03六个验证案例图及Cloud-FNO协议表已完成；本轮补充Yi预报评价与Sun检索时空匹配原则对照表`results/COT_REFERENCE_COMPARISON_20261014.md`，并追加机械规则选出的2025-08-21双站COT逐lead诊断曲线`results/cot_forecast_case_20261014_v1/`。该单例不代表总体，CPP仍是检索参考。Sun文献页码核正为16347–16361。既有产物见`results/cot_forecast_eval_20261014_v3/COT_FORECAST_LITERATURE_PROTOCOL_20261014.md`与`../cot_mask_necessity_20261014/M03_CASES_20261014.md`。test未读取。|
| E08 | T29 | 接入既有GHI对照结论 | CONDITIONAL | 按主计划B1/B2→B4，不新开训练臂 |
| E09 | T30/T31 | 冻结后正式test | NOT_STARTED；当前关闭 | 主计划许可；本计划不构成test授权 |

2026-10-14补充：训练尺度SSIM不支持S→R结构优势。M03六例、Cloud-FNO协议、Yi/Sun匹配矩阵及固定规则案例曲线已完成。本轮基于冻结验证bank，对全部8个完整共同支持窗口计算固定3×3 COT≥30首达与峰值；8窗全在2025-08-21且相互重叠，四里SimVP→R/持续性各1/8命中（命中较CPP早105 min），竺家均0/8，CPP厚云参考8/8。无CI、不视为独立云事件；细节见`results/cot_forecast_case_20261014_v1/COT_FORECAST_ARRIVAL_COHORT_DIAGNOSTIC_20261014.md`。T17的GHI预报头仍gated，不启动B4训练；test关闭。

区域COT仅有合规区域S/R产物时加入E05/E06；没有时明确NOT_APPLICABLE。外部MODIS交叉匹配是可选项，不是当前任务前提。2026-10-14现场核验：PBS213858已成功完成，完整CPP支持val bank回传并经profile hash、逐窗键、shape、finite和mask核验；本地FP64评估完成。原生forecast evaluator为另一统计聚合口径，已保留作 pooled-over-leads 补充，主结论按计划逐lead后等权汇总；1日bootstrap为主，2日块为敏感性。只有8个目标日期，且厚云CSI低于持续性，不能声称稳定优越。新test仍关闭；PBS账户另有213850运行，不与其并行扩卡。 |
