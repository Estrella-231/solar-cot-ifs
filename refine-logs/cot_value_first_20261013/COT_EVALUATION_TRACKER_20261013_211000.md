# COT评估执行清单：Yi × Sun v1

日期：2026-10-13（BJT）。
本表为COT评估子任务清单，整体工程进度仍由EXPERIMENT_TRACKER.md记录。
计划已制定不等于任务执行完成；无新增训练、无test访问、无服务器状态轮询。

| ID | 关联主任务 | 工作 | 状态 | 验收 |
|---|---|---|---|---|
| E01 | T04/T19 | 时间、空间、CPP有效mask与新S/R来源落为matching_contract | PLANNED | 复用已有未变化审计；记录实际范围/时间语义 |
| E02 | T21 | FP64指标、逆变换、NaN、FS和厚云计数单元检查 | PLANNED | 合成输入对手算；3×3不强算SSIM |
| E03 | T04/T21 | 保存的新月度R验证预测分尺度重算 | PLANNED；已有预测可复用 | 唯一target像元/3×3/16×16误差及覆盖 |
| E04 | T19/T20 | 新合同val的S→R/持续性/已有输运预测表 | PLANNED | 因果历史输入；方法逐样本键一致 |
| E05 | T21 | 16lead主要/所有方法公共域/固定16lead评价 | PLANNED | RMSE/MAE/bias/FS，像元和日期支持数 |
| E06 | T22 | 厚云混淆、条件SSIM、配对日期bootstrap | PLANNED | 完整有效域混淆计数；2000日期抽样 |
| E07 | T22/T16 | 同色标例图与文献可比性表 | PLANNED | 预注册与事后图分开；不跨论文数字排序 |
| E08 | T29 | 接入既有GHI对照结论 | CONDITIONAL | 按主计划B1/B2→B4，不新开训练臂 |
| E09 | T30/T31 | 冻结后正式test | NOT_STARTED；当前关闭 | 主计划许可；本计划不构成test授权 |

区域COT仅有合规区域S/R产物时加入E05/E06；没有时明确NOT_APPLICABLE。外部MODIS交叉匹配是可选项，不是当前任务前提。
