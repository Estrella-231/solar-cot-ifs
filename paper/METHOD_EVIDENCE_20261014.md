# 中英论文方法与结果来源记录

日期：2026-10-14。范围：本地实现、已保存的train/validation产物、论文写作；无服务器训练、无test载荷读取。本文件是工程文档，正文不收录其中的路径、历史实现名称或编译环境问题。下列路径均相对于仓库根目录。

## 正文已写入的方法

| 正文位置 | 核实内容 | 代码/产物来源 |
|---|---|---|
| 3.1–3.3 | 湖南两站、13 AGRI通道、月度2天val/2天test及缓冲、同期唯一键 | docs/MONTHLY_BLOCK_SPLIT_20261002.md；refine-logs/cot_value_first_20261013/EXPERIMENT_PLAN.md |
| 3.2 | 三个5分钟区间末记录合成15分钟GHI；晴空在区间中点取样；kt不cap | refine-logs/cot_value_first_20261013/join_trainval_ghi_clear_sky.py:67–93 |
| 4.1 | 16通道16×16；两级32/64/128 U-Net；1通道softplus log1p COT；CPP有效支持；Huber及训练参数 | experiments/monthly_R_station16_20261013_v1/scripts/train_station16_R.py；results/seed42_v1/run_metadata.json（同实验目录） |
| 4.1 | R-COT中心[7:10,7:10]；train-only归一化；55,733帧bank连接后50,458/4,479 train/val | experiments/monthly_R_station16_20261013_v1/prepare_ghi_head_pack.py；主tracker T06/T07 |
| 4.2 | A公共15→32→32逐像元编码；AC的独立1→4 COT支路；AP的32→4公共特征支路；展平后64→32→1 | experiments/monthly_R_station16_20261013_v1/head_candidate_v0.py:12–63 |
| 4.2 | AL从冻结R dec1取32×3×3特征；独立train-only特征标准化；32→4投影 | experiments/monthly_R_station16_20261013/head_r_al_candidate.py；refine-logs/cot_value_first_20261013/T18_RAL_CONTRACT_20261014.md |
| 4.3 | 物理GHI/1000的MSE，AdamW、30轮、batch2000、配对初始化和样本顺序、3 seeds | experiments/monthly_R_station16_20261013_v1/train_ghi_heads.py；各站/seed run_metadata.json；experiments/monthly_R_station16_20261013/train_r_al_head.py |
| 4.4 | 256×256、8→16共享两段SimVP、24时刻geometry、目标几何投影、12轮新月度训练 | experiments/monthly_block_20261002/source_snapshot/triad_models.py:176–202；scripts/train_monthly_simvp_20261002.py；refine-logs/cot_value_first_20261013/results/forecast_contract.json |
| 5.1 | 先种子内分站RMSE后等权平均、主CI共享52个BJT目标日期块 | 主分析FP64输出；refine-logs/cot_value_first_20261013/results/ci_method_reconciliation_20261014/ |
| 5.2/6.2 | 中心CPP既有阈值分层；固定验证预测；描述性日期块CI | refine-logs/cot_value_first_20261013/results/cot_thickness_story_20261014/ |

当前H代码文件名含candidate_v0且旧docstring提到原型，但正式train脚本与run_metadata确实使用这份实现；不能据文件名误写成仍未训练。当前H没有时间序列输入、没有四层因果3D卷积、没有GAP，也不是把COT直接作为AGRI第14通道送入同一编码器。R读16×16，因此AC的COT含有更大反演感受野；AP只控制GHI头容量，AL同时复用R视场与CPP监督。

## 已写入的结果及范围

- 同期主结果：A/AC/AP/AL=123.980/120.240/123.822/119.807 W m⁻²。AC−A=−3.740 W m⁻²，3.02%。共享日期主CI [−6.180,−1.597]；AL−AC的[−1.476,+0.677]为站点分别重采样敏感性区间，正文已区分。
- 总体与容量结果源：experiments/monthly_R_station16_20261013_v1/results/GHI_HEAD_THREE_SEED_RESULTS_20261013.md。
- AL结果源：refine-logs/cot_value_first_20261013/results/T18_RAL_RESULTS_20261014.md。
- 厚度分析源：refine-logs/cot_value_first_20261013/results/cot_thickness_story_20261014/COT_THICKNESS_GHI_VALIDATION_20261014.md；同目录summarize_thickness.py、thickness_metrics.json和thickness_summary.csv包含各站各seed、输入哈希和配对CI。CPP有效中心子集1,284行/46日，不是CLP云况分析的1,291行。
- 厚云CPP组173行/25日，A/AC=74.929/58.105，降低22.45%；六个站点×seed均改善。其他CPP层点估计1.48%/1.70%/1.70%，CI跨0。该结果是同期GHI反演误差下降，不是COT反演自身精度、独立test或预报收益。
- 稀疏CPP预测诊断源：refine-logs/cot_value_first_20261013/results/cot_forecast_eval_20261014_v3/。3×3 COT RMSE25.538/28.252，差−2.714、CI跨0；厚云CSI0.082/0.189，仅8个CPP支持日期。
- 全验证固定头输入替换源：refine-logs/cot_value_first_20261013/results/fixed_head_2x2_allval_agri_cot_attribution_20261014_v1/。这是已训练同期AC头的输入替换，不是匹配forecast-head收益。

## 待填项与不应提前确定的内容

1. 湖北独立区域实验：计划为refine-logs/hubei_cot_independent_20261014/EXPERIMENT_PLAN.md；尚无可写入的训练结果。计划“新月度日期映射”和R段落“旧train/oldval重建”措辞仍需在执行时统一，正文暂不规定湖北R的最终日期归属；湖南结果不变。三站观测原生时间语义也需核对。
2. 真正的预测输入GHI头：已写固定S/R及拟进行的匹配比较，未把原同期头架构、预算默认为已完成forecast训练。既有B4停止状态未由写作改变。
3. 文献综述与独立测试：正文明确占位；没有用不同文献RMSE当本站点性能上限；test未打开。
4. 归因：若预报增益较小，外推误差是候选机制，须与R误差和H分布适应区分，不能先写定唯一原因。

## 双语与编译核对

中英文稿均为8节、10个公式，并同步了2张表、现有图1、厚度数据及未完成项。只读协作核查确认方法与代码吻合、主要数据和分层数值一致、厚云六组合均有改善。该核查是有界写作复核，不冒称实验级盲审。

对当前打开的paper/main.tex调用内置compile_latex_document，结果为compile-failed，日志仅有“Unable to find standard directories for platform”。编译器在初始化阶段失败，没有进入TeX语法/引用/版式诊断。保持源文件及原编辑器，不另编译PDF、不安装TeX；整稿渲染尚未验证。

## 2026-10-16 manuscript update

- Replaced the previous forecast-COT CPP metrics because their reference keys failed the independent source-to-target audit; no forecast-COT skill or thick-cloud detection claim is retained from that cohort. The corrected 271-window sample is not treated as a replacement full-validation result.
- Added the original Hunan R validation against its correctly keyed CPP support: 356,169 valid pixels, RMSE 9.074, MAE 4.646, bias +0.279, Pearson r 0.821. CPP is described as a retrieval reference, not independent physical truth.
- Retained the same-time overall GHI comparison and the synchronous CPP-thickness subgroup because they come from a separate station/target keyed cohort; do not conflate them with the forecast-bank CPP audit.
- Matched forecast-input GHI skill remains unestablished. English and Chinese manuscripts were edited in parallel; the built-in LaTeX editor compiled the updated source successfully.
- Added the already completed all-validation fixed-head input-replacement diagnostic to Results: 2,817 sequences / 52 target dates; site-equal RMSE TT/FT/TF/FF = 118.14/126.46/125.42/139.39 W m-2; forecast-COT replacement increments +7.28 [4.52,10.54] and +12.93 [9.73,16.58]. It is explicitly an offline sensitivity result from a head trained on observed inputs, not forecast skill. Source: refine-logs/cot_value_first_20261013/results/fixed_head_2x2_allval_agri_cot_attribution_20261014_v1/ALLVAL_FIXED_HEAD_ATTRIBUTION_20261014.md.

## 2026-10-16 method-figure update

- Replaced the Figure 2 placeholder in both manuscripts with an editable, reproducible workflow figure (`figures/plot_fig2_workflow.py`, PDF/PNG/SVG outputs).
- The diagram distinguishes the 16×16 COT retrieval support from the central 3×3 GHI support; it separates synchronous observed-AGRI estimation from SimVP forecast-AGRI → frozen-R forecast-COT application.
- The forecast lane is labelled as an evaluation task and does not imply a verified COT forecast skill or matched forecast-input GHI gain. Figure is factual method documentation, not a result plot.

## 2026-10-17 overall-value figure

- Replaced the Figure 4 placeholder in both manuscripts with a point-and-interval figure, not a bar chart. Panel (a) plots three-seed mean RMSE for each of the five stations; panel (b) plots region-equal paired RMSE reductions with 95% validation date-block intervals. The two regional cohorts remain separate because their ground-label sampling differs.
- Hunan station values are arithmetic means of the three seed-specific station RMSEs in `experiments/monthly_R_station16_20261013_v1/results/ghi_heads_three_seed_validation_audit.json`. Hubei station values and regional paired intervals are from `refine-logs/hubei_cot_independent_20261014/PROGRESS_20261016.md`. The cohort sizes are 4,479 (Hunan) and 2,456 (Hubei); paired gains are 3.740 [1.597, 6.180] and 2.932 [0.939, 5.070] W m-2, respectively.
- Source summary tables and plotting code are `paper/figures/fig4_station_increment_data.csv`, `paper/figures/fig4_regional_paired_gain.csv`, and `paper/figures/plot_fig4_cot_increment.py`; PDF, PNG, and SVG exports share the same values. Station panel values are descriptive three-seed means; confidence intervals are regional paired estimates. No station-specific interval or cross-region pooled effect is implied.

## 2026-10-17 forecast-COT full-validation result

The source-keyed CPP comparison now covers all 3,140 validation sequences and the complete +15 to +240 minute lead range. CPP has valid support on 46 BJT target dates (5,224,460 16x16 patch pixels; 183,564 local 3x3 pixels); no test payload was read. For the station-and-lead-macro local 3x3 score, original SimVP RMSE/CSI is 10.664/0.423; equal-budget MSE fine-tune 10.750/0.431; temporal-change 10.756/0.430; spatial-gradient 10.759/0.428; and last-frame persistence 20.487/0.306. Paired date-block intervals for candidate-minus-original RMSE cross zero, so none of the loss modifications is supported as an improvement. Original SimVP is better than last-frame persistence (persistence-minus-original +9.823, 95% CI [+7.722,+12.154]).

The per-lead figure is a COT-only result figure; it does not claim matched GHI forecast skill. CPP is a retrieval reference, not independent physical truth. The three fine-tuning candidates did not pass the predeclared screen. Final evidence is in `refine-logs/cot_value_first_20261013/results/simvp_supervision_cot_allval_20261017_v2/scored_final/summary_fp64.json` and `cot_metrics_by_station_lead_scale.csv`; figure inputs and exports are versioned under `paper/figures/`.
