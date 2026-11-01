# 八项修复落实

日期：2026-10-30。用户批准后实施。下表中的“完成”指当前普通实验的默认执行方式和已检查的通用入口，不表示重跑了所有旧实验，也不表示删除所有历史OOF脚本。

| 原审计项 | 当前处置 | 状态与依据 |
|---|---|---|
| 1 默认五折OOF | 普通全train预测包可训练GHI；`--require-oof`和`--oof-fold-count`仅作显式复现选项 | 代码完成：`scripts/hubei_forecast_pack_gate.py`、`scripts/train_hubei_ghi_heads.py`；13项门禁测试通过 |
| 2 U/H及额外U-only R | 普通任务默认全部train，复用现有合规S/R/输出；历史U/H设计不作新任务前提 | 规则与入口完成：工作区/仓库AGENTS、README、CLAUDE、直接COT计划及tracker；没有把训练内级联一概称为泄漏 |
| 3 多层内部日期拆分 | 取消默认F/C及F_fit/F_select/C_fit/C_select，保留原外层日期清单 | 规则与入口完成：当前政策、主计划、湖北增量计划、执行交接；旧实验划分和分数保留 |
| 4 COT/CPP门槛阻断GHI | 有有效预测与地面标签即可直接做配对GHI评价，COT辅助结论单列 | 规则与入口完成：价值优先计划、方向文件、旧GHI门文件和当前交接；不改旧结果的原判定 |
| 5 多判据作为继续研究许可 | 主要比较与站点/时效/云况分析分工，取消默认“所有项都改善”才可开发 | 规则与入口完成：主计划、当前政策、README和交接；严重退化仍完整报告 |
| 6 必须等新观测 | validation允许继续开发和选模，新日期用于独立确认 | 规则与入口完成：直接COT tracker及当前交接；test/buffer仍关闭 |
| 7 必须重训GHI头 | 冻结头接实际预测图和预测域重训均可评价，注明训练域与帧数 | 规则与入口完成：方向文件、价值优先计划、README和交接；保留最新湖南1.633%验证正结果及旧负结果 |
| 8 单卡先行 | 长任务按现行规则优先有效的多卡配置，无8卡总并行上限，PBS≤3 | 工作区/仓库规则及交接完成；本次没有新增长作业，不把规则更新称为已测得加速 |

## 代码验证

- `scripts/test_hubei_forecast_pack_gate.py`：13项通过，覆盖普通非OOF、可选OOF、原五折复现及未来观测/test隔离。
- `scripts/test_hubei_forecast_ghi_heads_contract_20261016.py`：通过；A/FA/AP/FAP数值与形状、AC/FAC所需COT输入检查通过。
- `scripts/train_hubei_ghi_heads.py --help`：可选OOF参数显示正确。

湖北现有预测包构建器已允许不指定`--oof-train-bank`，因此无需为普通路线另写数据构建器。显式五折合并器和已完成实验脚本保留原设计，用于复现，不作为普通新任务入口。没有改网络、损失、全局batch、训练预算、归一化、评价行、旧预测或分数。

## 入口与归档

已更新工作区`../AGENTS.md`、仓库`AGENTS.md`、README、CLAUDE、AI交接和PIPELINE_SUMMARY。首轮十份计划的优先说明继续有效；原文件见`before/`，后续代码及入口原文见`runtime_fix_before/`。首轮`CHANGES.json`描述当时仅加文档前言的阶段，后续落实单独记录，不回写首轮记录。

实时服务器状态和远端复制范围以`RUNTIME_IMPLEMENTATION.json`及`REMOTE_SYNC.json`为准。未同步的旧运行目录不应直接拿来启动普通新实验；既有作业保持原脚本，以免破坏复现。

本次不提交训练、不取消作业、不恢复定时、不打开test/buffer、不修改论文正文，也不改全局memory。
