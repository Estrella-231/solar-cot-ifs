# 当前使用与历史记录索引

规则更新入口：[研究规则纠错机制及20项审计](rule_correction_20261102/REPORT.md)。当前项目AGENTS已收敛为现行规则；历史全文保存在该审计目录，不能继续作为执行指令。

新增条件分析：[两省预报厚云分层及四个真实AGRI/COT/GHI案例](../refine-logs/forecast_thick_cloud_events_20261102/RESULTS.md)。厚云点降幅湖北15.45%、湖南2.90%，两者日期块区间均跨零；与总体预报4.90%/1.70%及同期厚云35.51%分开。连续参考段不等于独立云团，已纳入主文§5.6/图9–10与补充S7；新窗口接续见[交接入口](../paper/HANDOFF_NEXT_WINDOW_20261103.md)。

核对日期：2026-11-02。这里区分“当前选用”“已完成对照”“负结果”“失效来源”和“运行故障”。文件夹日期较新或存在 best.pt，都不自动表示正在使用。机器可读索引见 [ASSET_STATUS.json](ASSET_STATUS.json)。

## 现在使用什么

| 用途 | 当前资产和结果 | 权威入口 |
|---|---|---|
| 湖南未来云图 | 湖南独立训练 DyDiff，epoch150 EMA，25步、单成员；SimVP保留为基线 | [路线及服务器位置](GHI_HEAD_ROUTE_20261102.md) |
| 湖南COT反演 | 修正太阳几何的冻结R；不使用存在CPP时间错配的旧来源 | [当前模型选择](../configs/hunan_active_ghi_heads.json)及路线文档 |
| 湖南GHI预报 | 真实同期三帧训练的既有GHI头，冻结后接预测AGRI/COT；两站、三seed、有无COT共12个头。不用预测图重训或微调头 | [选用权重与哈希](../configs/hunan_active_ghi_heads.json) |
| 当前湖南预报评价 | 66,086条验证记录、51个目标日期；122.588→120.502 W/m²，下降1.702%；四里2.470%、竺家0.765%，竺家单站区间跨零 | [结果与评分口径](../refine-logs/fulltrain_forecast_cot_20261101/observed_trained_heads_evaluation_v1/RESULTS.md) |
| 湖北同期COT价值 | 正确太阳几何、真实三帧；三站三seed，1,905条共同验证记录、20个日期；74.406→68.629 W/m²，下降7.763%；米庄单站区间跨零 | [完成记录](../refine-logs/hubei_same_time_geometry_20261102/EXPERIMENT_TRACKER.md)、[结果](../refine-logs/hubei_same_time_geometry_20261102/RESULTS.md) |
| 论文 | 唯一英文主稿main.tex、对应main_zh.md及编译main.pdf；已完成本轮结果整合；完整四组合与最终留出评价仍未完成，尚非投稿定稿 | [论文版本入口](../paper/VERSION_INDEX.md) |
| 执行规则 | 普通级联默认全train，可复用已核实上游；OOF可选；外层划分、因果输入与共同评价行不变 | [现行规则](execution_simplification_20261030/CURRENT_POLICY.md)、[AGENTS](../AGENTS.md) |

上述分数均为validation结果，不是独立test成绩。两区域独立训练，不称零样本迁移。平均分数按站点和seed等权，不能用合并样本RMSE替换。

## 哪些保留为对照，哪些不能继续当主结果

| 状态 | 记录 | 如何使用 |
|---|---|---|
| 有效历史参照 | 湖南原63,480条冻结头评价：DyDiff 124.958→122.788，1.737%；SimVP 127.421→125.340，1.633% | 与对应样本、输入替换图共同使用；不与66,086条直接当成同一批数据。最新产物在共同63,480条上为1.733% |
| 已完成、未选用 | 全train预测图训练GHI头：112.447→111.766，0.605%，总体差值区间跨零 | [结果](../refine-logs/fulltrain_forecast_cot_20261101/RESULTS.md)。绝对误差更低，不代表当前用户选用；与旧头还存在训练域、归一化、更新数差异，不能仅归因于数据量 |
| 有效负结果 | 早期静态统计COT、历史/输运修正、真实历史直接未来COT等探索，部分未显示可靠GHI收益 | 保留各自样本和原判定；不据此否定其他输入、头或区域上的COT价值。[直接未来COT记录](../refine-logs/agri_direct_cot_20261020/EXPERIMENT_TRACKER.md) |
| 被替代配置 | 湖北UTC修正但太阳几何尚未修正的10.54%；湖南较早单帧及其重评分 | 仅作带来源限定的历史/补充比较，不能替代当前正确几何的同期结果 |
| 不可支撑当前结论 | 湖北CPP监督有8小时时间错配的旧R及其4.49%；包含已确认无效四里日期的旧3.02% | 保存审计与修复记录，不当作当前有效提升。纠正后的分数须单独标来源与样本 |
| 停止或故障 | PBS 214946本地盘保护停止；217013为保存进度后迁移；湖北217109为训练前cuDNN冲突 | 是运行事件，不能叫模型科学负结果。后续成功attempt另列，不覆盖失败产物 |
| 待完成 | 独立test、短临IFS额外收益、正文覆盖图/部分数据表及误差分解 | 未完成不等于失败；不自动启动训练或恢复旧定时 |

其他未在本表逐项核定的日期目录统一视为“历史保留、未确认为当前选用”，不能因没有进入主路线就标成失败。完整日期目录导航见[EXPERIMENT_DIRECTORY_STATUS_20261102.json](EXPERIMENT_DIRECTORY_STATUS_20261102.json)。[共同样本原因分析](../refine-logs/fulltrain_forecast_cot_20261101/common_sample_diagnosis/REPORT.md)解释旧头与预测图训练头的差异。

## 目录与版本怎么读

- `configs/`：当前选择及配置。`hunan_active_ghi_heads.json`是湖南当前GHI头入口。
- `paper/`：唯一主稿与必要图；`drafts/`、`planning_history/`、`revision_records/`是历史版本或编辑记录，不是并行有效主稿。
- `refine-logs/`：按日期保留的方案、代码、attempt和结果。旧方案中的“下一步”“必须先OOF”等指令须服从当前AGENTS和本索引。
- `docs/`：交接、来源、审计和政策；路径、哈希、故障沿革放这里，不搬入正文。
- `scripts/`、`experiments/`：实现与复现实验入口。运行旧脚本前按其配置确认区域、输入和日期，不能仅凭文件名选模型。
- `results/`、`figures/`、`audits/`：结果摘要、科学图和诊断证据；合成示意不是实际模型输出。

Git保存main分支的提交历史，日期目录和提交记录共同保留多个阶段；截至本次整理，未建立release/tag版本体系。本次不删历史、不重写提交。

## GitHub与服务器/本地各保存什么

GitHub同步代码、配置、方案、指标摘要、来源记录、必要日期清单、论文双语正文及自有图/PDF。当前完整权重、逐样本预测数组和原始数据保留在服务器/本地；位置见路线与各实验tracker。仓库历史中已经存在部分小权重和诊断数组，这次不删除，也不能笼统宣称“仓库从无数据或权重”。

本次不新增原始HDF/Excel、训练bank和pack数值载荷、checkpoint、逐样本大表/数组、第三方整篇论文PDF、依赖安装包或编译缓存。文献条目、阅读笔记和必要出处保留。未上传文件没有被删除；发布清单及排除统计见 [repository_publish_receipt_20261102.json](repository_publish_receipt_20261102.json)。
