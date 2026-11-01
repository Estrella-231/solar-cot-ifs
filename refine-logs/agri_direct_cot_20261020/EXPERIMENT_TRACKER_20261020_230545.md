# 真实历史 AGRI → 未来 COT：执行记录

更新：2026-10-20。权威方案：[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)。

## 当前状态

**PLAN_READY / NOT_STARTED**：完成方案及本地证据核对。没有提交新训练、没有恢复已停止任务或自动化、没有修改论文正文、没有打开test/buffer载荷。

目标：以同架构P-C/P-XC隔离原始光谱价值，再以单帧3×3 GHI头检验未来COT收益。不能用真实未来诊断或COT精度代替可部署GHI结果。

## 工作项

| 块 | 状态 | 已知/待完成 | 完成证据 |
|---|---|---|---|
| B0a 既有fold1 S与U/H/V | LOCAL_RECORD_REVIEWED | 本地记录epoch12、SHA13e120c5…；服务器实际文件、完整日期清单/归一化待复核 | 尚无本轮现场核验 |
| B0b U-only几何修正R | NOT_STARTED | 使用修正实现、冻结归档CPP，随机初始化，固定50轮 | 无权重 |
| B0c 缓存与因果性 | NOT_STARTED | 全24帧清单、QA、未来载荷隔离、train-only归一化、实际profile | 无新缓存 |
| B1 P-C/P-XC | NOT_STARTED | 同网络/同样本/同预算，seed42，各12轮 | 无预测/评分 |
| B2 单帧GHI四输入对照 | NOT_STARTED | H训练、V评价；两站×4输入×3头seed，各30轮 | 无GHI收益结论 |
| B3 容量/稳定性或机制修正 | CONDITIONAL | B2后按方案只选择一个因素 | 未触发 |
| B4 湖北独立重复 | CONDITIONAL | 湖北独立U/H/V及正确UTC来源；共享云场去重 | 未触发 |

## 下一次实际执行顺序

1. 只读核验正式fold1 `epoch_012.pt`、归一化、U/H帧清单和代码；不能误用停止的DropPath候选。
2. 冻结新U/H/V资格清单、CPP内容版本和单帧GHI接口，记录缺失原因及覆盖。
3. 完成局地数据/profile及小样本数值检查，报告真实GPU/IO成本，再按固定预算运行B0→B1→B2。

如果fold1资产缺失或不合规，不恢复旧五折；先给出一次独立S_U训练的成本修订。U/H没有内部1天缓冲，筛选结果不得称严格内部连续块交叉验证。

## 预定评分与证据边界

- 主GHI差值：联合外推COT−无COT；联合外推COT−SimVP反演COT。
- 两站分别、全16时效和+15～60min；原始2天验证块配对bootstrap、逐验证块删除及自然季度敏感性。不伪造连续5天验证样本。
- COT对CPP、对R教师分别评价；厚云识别/到达/幅值与总RMSE一起解释。
- 头种子42/43/44不等于P的多seed；P稳定性需后续真实补训。
- 不用固定头替换冒充重训，不删除负结果，不将validation改称test。

## 更新记录

- 2026-10-20：新建方案。历史COT-only失败与原始AGRI直接预测未完成的证据已区分；设计一次S_U复用＋U-only R/P＋H训练GHI的低成本链条。所有性能结果待执行。

