# 第二篇：SimVP + COT + IFS 辐照度短临预报

本工程于 2026-09-05 独立建立。核心问题：固定 SimVP 外推后，显式云光学厚度表示是否改善湖南站点的 +15～+240 min GHI，以及 IFS 是否提供互补收益。

当前交付是研究立项、消融设计和双语写作骨架；尚未启动新的服务器训练，尚无第二篇实验结果。

## 独立克隆

```bash
git clone https://github.com/Estrella-231/solar-cot-ifs.git
cd solar-cot-ifs
```

先读 `AGENTS.md`、`docs/UPSTREAM_PROJECT_RULES.md` 和 `IRRADIANCE_ERROR_PREVENTION.md`。仓库包含论文方案、审查记录和双语 LaTeX/Markdown 骨架；数据、checkpoint、训练程序及 ARIS 技能安装不包含在内。工程文档中的服务器路径是待核验的资产位置，历史本地路径用于溯源，不要求克隆到相同目录。

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
