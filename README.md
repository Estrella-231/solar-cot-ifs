# 第二篇：SimVP + COT + IFS 辐照度短临预报

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

## 当前工程状态（2026-09-12）

最新、可复核的工程和服务器状态见 [PROJECT_STATUS_20260912.md](docs/PROJECT_STATUS_20260912.md)。当前已完成湖南冻结 SimVP/COT 缓存、A/AC 多种子验证、天气分层和固定 AC 的反事实诊断；没有使用 test。四维 COT 直接拼接未得到稳定收益，IFS 因果合同未通过，所以 AN/ACN 尚未运行。

## 2026-09-05 推进记录
已完成服务器P200首轮审计和COT候选时间重划，详见 `docs/P200_AUDIT_20260905.md`。已部署工程根 `/home/Data_Pool_3/wangyc/irradiance_paper/solar_cot_ifs_20260905`。8项时间/IFS合同测试通过，独立CPU复跑通过。上游权重与数据合同仍有明确缺口，因此尚无GPU训练或第二篇精度结果。

CPP质量来源已追通并发现旧裁剪偏移，18份修正样例已核验；详见 [CPP来源与空间审计](docs/CPP_QUALITY_PROVENANCE_20260905.md) 和 [COT合同](configs/cot_contract.json)。当前共14项合同测试通过，全量参考缓存待重建。

CPP修复任务已启动（18例门禁后全量），最新执行说明：docs/CPP_REPAIR_RUNBOOK_20260905.md。全量完成情况以服务器status.json为准。
