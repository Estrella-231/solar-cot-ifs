# Solar COT / GHI：当前入口

**先看[当前使用与历史记录索引](docs/REPOSITORY_INDEX.md)**：区分正在选用的模型、已完成但未采用的对照、科学负结果、失效来源、运行故障与待完成事项。机器索引见[ASSET_STATUS.json](docs/ASSET_STATUS.json)，唯一[论文版本入口](paper/VERSION_INDEX.md)。历史记录保留，不把所有旧方案当作现行指令。

## 2026-11-02 用户确定：真实图训练GHI头

当前主路线使用真实同期AGRI三帧训练的既有GHI头，冻结后接预测AGRI及预测COT进行预报。不再用预测图训练或微调GHI头；不得把这类变化隐含在“全train重训”中。此决定覆盖此前“两条训练路径均可默认推进”的表述。train/validation/test外层日期清单与其他数据规则不变。

Details: docs/GHI_HEAD_ROUTE_20261102.md (repository root).

---

# 第二篇：SimVP + COT + IFS 辐照度短临预报

## 最新结果与当前路线（2026-11-02）

当前固定湖南DyDiff第150轮EMA与修正R，使用**真实同期云图训练的既有GHI头**，冻结后接预测云图/COT评价，不用预测图重训或微调GHI头。详见[路线与服务器资产](docs/GHI_HEAD_ROUTE_20261102.md)、[当前旧头选择清单](configs/hunan_active_ghi_heads.json)。

本轮生成完成33,253个train/val序列，预测域GHI重训对照与旧头回放均已完成；未使用test。旧头接本轮预测产物，完整66,086条验证记录上两站三种子等权RMSE为122.588→120.502 W/m²，降幅**1.702%**；日期配对95%区间[-3.527,-0.598]。四里改善2.470%，竺家0.765%，竺家单站区间跨零。与原63,480条记录相同样本上的降幅为1.733%，接近原1.737%。[完整结果](refine-logs/fulltrain_forecast_cot_20261101/observed_trained_heads_evaluation_v1/RESULTS.md)

预测图训练头仅保留为完成的对照：其完整验证降幅0.605%，共同63,480条上0.616%，总体区间跨零；它的绝对GHI误差更低，但不是当前用户选定路线。两种训练域并非仅数据量不同，不能将差异归因于取消U/H。[共同样本与原因诊断](refine-logs/fulltrain_forecast_cot_20261101/common_sample_diagnosis/REPORT.md)

PBS217013为保留进度的四卡阶段、217014为短时性能测试、217015为八卡续跑与后续头部训练，均已结束；不是正在运行的作业。最新工程记录见[实验tracker](refine-logs/fulltrain_forecast_cot_20261101/EXPERIMENT_TRACKER.md)。GitHub保存代码、方案、结果和复核收据；数据、权重、逐样本预测留原服务器/本地归档，不随本次推送上传。

## 执行入口（2026-10-30修订，头部路线以2026-11-02为准）

当前研究覆盖湖北三站和湖南两站，两区域独立训练与评价。先读[当前执行规则](AGENTS.md)、[简化执行方式](docs/execution_simplification_20261030/CURRENT_POLICY.md)及[结果总表](docs/experiment_results_20261030/SUMMARY.md)。下方9月、10月初的进度是历史记录，不能作为当前任务的前置条件。

- 普通级联默认复用已核实的全train上游模型和输出，不再强制U/H、五折OOF、U-only R或多层内部日期留出。OOF保留为可选实验。
- 有预测和有效地面标签即可评价有无COT的GHI差异；COT/CPP辅助指标、个别站点或时效未改善，不单独阻止开发。validation可以开发和选模，独立test继续封存。
- 真实同期图像训练的冻结GHI头，可以接实际预测图像评价预报；不强制重新训练。报告须注明头的训练域、帧数及验证范围。
- 来源、时间、空间、单位、标签及数值接口检查通过后复用；未改变的部分不重复全量审计。长任务按现行规则优先多卡，不再单卡先行。

最新完整湖南SimVP冻结头验证：无COT **127.421**、加预测COT **125.340 W/m²**，RMSE下降 **1.633%**；目标日期块差值95%区间为[−3.714,−0.561] W/m²。这是开发验证结果，竺家单站区间仍跨零；旧负结果保留，不能笼统写成“预测COT一律无效”。来源见[完整结果](refine-logs/hunan_dydiff_20261028/full_validation_20261029/RESULTS.md)。

湖北现有普通预测数据包入口已允许全train预测；以下为命令格式（路径替换为实际已构建数据包和新输出目录，预算沿用当前实验方案）：

```bash
python scripts/train_hubei_ghi_heads.py --pack PACK_DIR --output NEW_RUN_DIR --input-mode forecast
```

复现特定五折设计时才额外指定`--require-oof --oof-fold-count 5`。示例不自动启动训练，也不代表应复跑已完成实验。执行交接见[AI_EXECUTION_HANDOFF.md](docs/AI_EXECUTION_HANDOFF.md)；八项修复状态见[IMPLEMENTATION.md](docs/execution_simplification_20261030/IMPLEMENTATION.md)。

## 历史背景与进度

**2026-10-03 详细交接入口：** [原始数据、预处理、划分、S/R/H逐层结构、训练与评价](docs/END_TO_END_HANDOFF_20261003.md)。新月度划分SimVP已完成12轮训练，test未读取；区域R与GHI时空头的现有实现仍属旧划分探索资产，新合同全链尚待重建。下方带旧日期的状态保留为历史记录。

**2026-09-17 方向更新：** 固定湖南 SimVP 和 COT 反演器，将完整16×16 COT空间场作为GHI网络的输入通道。先验证空间图相对无COT及四统计量的收益，再做容量/监督/空间排列归因与IFS增量。方案已更新，空间头尚未实现或训练，来源门禁仍保留。见[方向把控](docs/COT_SPATIAL_DIRECTION_20260917.md)、[实验计划](refine-logs/EXPERIMENT_PLAN.md)、[英文正文](paper/main.tex)、[中文对照](paper/main_zh.md)。以下为既有工程背景和历史结果。

本工程于 2026-09-05 独立建立。核心问题：固定 SimVP 外推后，显式云光学厚度表示是否改善湖南站点的 +15～+240 min GHI，以及 IFS 是否提供互补收益。

2026-09-08最新状态：湖南SimVP和修正CPP监督下的COT反演器已训练并冻结，train/val外推缓存及标签pack已验收。original/QC1两训练分支共8次A/AC探索pilot完成；全部预测重算和4个选定模型的独立CPU抽样重载通过。两分支加入COT后的验证RMSE均升高，尚不支持COT增益。当前进入无收益诊断，test未使用，IFS因果来源仍待补齐。见[完整结果](docs/HEAD_PAIRED_PILOT_RESULTS_20260908.md)、[最新状态](docs/STATUS_20260908.md)和[每小时推进清单](docs/HOURLY_NEXT_ACTIONS_20260908.md)。

补充诊断已完成：同 872 配对的 CPP-reference COT RMSE 从真实输入 4.29 升至 SimVP 输入 10.13，独立复算通过；这不构成 GHI 收益或退化机制证明。见[配对诊断与全时效图](docs/COT_PAIR_DIAGNOSTIC_RESULTS_20260908.md)。

## 独立克隆

```bash
git clone https://github.com/Estrella-231/solar-cot-ifs.git
cd solar-cot-ifs
```

先读 `AGENTS.md`、`docs/UPSTREAM_PROJECT_RULES.md` 和 `IRRADIANCE_ERROR_PREVENTION.md`。本地工程包含论文方案、审查记录、双语 LaTeX/Markdown 骨架及scripts中的训练/审计程序；数据、checkpoint及ARIS技能安装不包含在内。远端GitHub未随本轮自动推送，克隆内容以已发布提交为准。工程文档中的服务器资产应按登记的SHA和最新审计核验，不要求克隆到相同本地目录。

## 阅读入口

1. `refine-logs/FINAL_PROPOSAL.md`：论文定位、方法、待验证主张。
2. `refine-logs/EXPERIMENT_PLAN.md`：四格消融、COT 核心证据、评价和执行门禁。
3. `refine-logs/EXPERIMENT_TRACKER.md`：任务依赖与状态。
4. `paper/PAPER_PLAN.md`：章节、图表和写作顺序。
5. `docs/ASSET_REUSE_AUDIT.md`：第一篇可复用资产、已知不合规权重和待查项。
6. `docs/AI_EXECUTION_HANDOFF.md`：下一位 AI 的执行步骤与完成标准。
7. `docs/LITERATURE_POSITIONING.md`：最近邻与创新边界。
8. `refine-logs/REVIEW_SUMMARY.md`：ARIS 独立审查及修订记录。

第一篇在相邻 `solar-energy` 中。本篇使用相同数据合同与合规基线以减少成本，但科学问题、下游训练、结果和正文独立。研究结论待实验决定。

## 历史工程状态（2026-09-12）

最新、可复核的工程和服务器状态见 [PROJECT_STATUS_20260912.md](docs/PROJECT_STATUS_20260912.md)。当前已完成湖南冻结 SimVP/COT 缓存、A/AC 多种子验证、天气分层和固定 AC 的反事实诊断；没有使用 test。四维 COT 直接拼接未得到稳定收益，IFS 因果合同未通过，所以 AN/ACN 尚未运行。

## 2026-09-05 推进记录
已完成服务器P200首轮审计和COT候选时间重划，详见 `docs/P200_AUDIT_20260905.md`。已部署工程根 `/home/Data_Pool_3/wangyc/irradiance_paper/solar_cot_ifs_20260905`。8项时间/IFS合同测试通过，独立CPU复跑通过。上游权重与数据合同仍有明确缺口，因此尚无GPU训练或第二篇精度结果。

CPP质量来源已追通并发现旧裁剪偏移，18份修正样例已核验；详见 [CPP来源与空间审计](docs/CPP_QUALITY_PROVENANCE_20260905.md) 和 [COT合同](configs/cot_contract.json)。当前共14项合同测试通过，全量参考缓存待重建。

CPP修复任务已启动（18例门禁后全量），最新执行说明：docs/CPP_REPAIR_RUNBOOK_20260905.md。全量完成情况以服务器status.json为准。
