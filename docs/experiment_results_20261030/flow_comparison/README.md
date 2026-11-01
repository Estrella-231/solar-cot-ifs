# 湖南COT预报：正负结果处理流程对照

主预览：`flow_comparison_preview.png`；打印版：`flow_comparison_preview.pdf`。由内置图像生成工具制作，提示词见`GENERATION_PROMPT.md`。准确流程另存为FigureSpec JSON及可编辑`flow_comparison.svg`，原生渲染器已验证通过。图中只有已保存结果数字和示意流程，没有虚构的观测图像。

## 如何读图

- 左侧绑定湖南`agri_direct_cot_20261020`的B2：U-only上游，H日期的预测输入训练单帧3×3 GHI头，V日期评价。S_U与R_U并非右侧的同一份权重。SimVP→R和P-XC是各自配对训练/评分的不同COT输入臂，不是同时拼接两种COT。
- 右侧绑定10月29日完整湖南冻结验证：全train湖南SimVP与全train几何修正版R，GHI头在真实同期三帧输入上训练，然后冻结接预测场回放。三帧为目标前30、15分钟及目标时刻；+15/+30分钟仍有已知历史帧，+45至+240分钟三帧均为预测图。
- 两侧都比较独立训练的有/无COT配对头，而非把一个训练好的有COT头临时清零来冒充无COT基线。每个实验内部共用该实验的AGRI、太阳几何、标签和评价行；两实验之间并非同权重、同样本。
- 图中左侧选择的是与正结果区域相同、且同时包含SimVP→R及直接预测COT的较早负结果。湖北残差、四里输运、历史COT新夏季等其他负结果属于另外的设置，未伪装为这条左侧流程；详见上级`SUMMARY.md`。

## 核实的差异

| 项目 | 较早负结果 | 最新正结果 |
|---|---|---|
| 上游训练范围 | U子集；固定fold1 SimVP与U-only R_U | 全train湖南SimVP与修正R |
| GHI训练域 | H日期上的预测AGRI/预测COT | 真实同期AGRI/反演COT |
| GHI时间支持 | 目标时刻1帧 | 目标前30、15分钟及目标时刻3帧 |
| 空间支持 | 每帧站点3×3 | 每帧站点3×3 |
| 推理 | 预测AGRI与预测COT进入单帧头 | 预测AGRI及对应R-COT进入冻结三帧头 |
| 共同评价行 | 66,086/seed | 63,480/seed，51目标日 |
| 无COT→SimVP-R COT | 135.329→136.663 W/m² | 127.421→125.340 W/m² |
| P-XC直接COT | 136.698 W/m² | 未包含此P-XC臂 |

两侧太阳几何均采用各自已修正方案，不应把左侧画成“几何错误”。R权重、train-only归一化和上游样本范围仍不同。多个因素同时变化，尚未通过单因素实验确定正收益由哪项导致。

## 图注

**中文：** 湖南两项预测COT实验的训练与推理流程。左侧在U子集训练上游模型，以H日期预测输入训练单帧站点GHI头；未来COT分别来自SimVP预测图的R_U反演或P-XC直接预测。右侧使用全train湖南SimVP和修正R，并将真实同期三帧训练的配对GHI头冻结后用于预报回放。右侧总体RMSE降低1.633%，日期块差值95%区间不跨零；左侧两种预测COT臂均未获总体收益。每项实验内部有无COT同样本、同标签，实验之间的上游权重、训练域、帧数和评价样本不同。因此本图展示已核实的流程差异，不将收益归因于任何单一改变。两者均是重复使用的开发验证，非独立测试。

**English:** Training and inference workflows for two Hunan forecast-COT experiments. The earlier experiment fits single-frame, station-local GHI heads on forecast inputs from U-trained upstream models, with COT supplied either by SimVP-to-R_U retrieval or by the direct P-XC predictor. The recent experiment replays paired GHI heads trained on three observed frames and then frozen, using full-train Hunan SimVP and a corrected retriever. Its macro RMSE decreases by 1.633%, with a paired target-date interval excluding zero. Both earlier COT arms lack overall benefit. Within-experiment comparisons share observations and evaluation rows, whereas upstream weights, head-training domain, temporal support and evaluation cohorts differ between experiments. The figure documents these differences without attributing the gain to a single factor. Both are reused development-validation analyses, not independent tests.

来源与SHA见`SOURCES.json`。核心来源为新实验`AUDIT_SIMVP_20261030.md`、`HISTORICAL_COMPARISON_20261030.md`及旧实验`EXPERIMENT_PLAN.md`和保存评分JSON。
