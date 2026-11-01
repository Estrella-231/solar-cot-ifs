# 湖北独立实验 test-freeze 审计

**日期：** 2026-10-17  
**审阅方式：** 新鲜同系列 reviewer agent，只读复核；结果为 provisional，不代表跨模型家族审查。  
**范围：** 湖北同刻 A/AC/AP 验证、forecast-GHI FA/FAC 验证、论文主张与 held-out test 放行准备。审计未解码 test/buffer 载荷，也未启动训练。

## 总结：FAIL / NO-GO

保留独立 test 与 buffer 关闭。现有 validation 可用于有限的验证集描述，但谱系、模型重载复现和一次性 test 评价程序尚未冻结。

## A. 真值来源：PASS with WARN

GHI 真值来自站点观测工作簿构造的 `ghi15_raw`，模型链只生成预测输入，不生成标签。操作单位沿用旧工程 W/m² 约定，但仪器校准及供数平台字段字典没有独立确认。证据：`scripts/build_hubei_forecast_ghi_heads_pack_20261016.py:196-217`、`refine-logs/hubei_cot_independent_20261014/audit_receipts/forecast_ghi_label_binding_20261017/forecast_ghi_label_binding_audit_v2.json`。

## B. 指标与共同支持：验证 PASS，test freeze FAIL

FA/FAC 使用相同站点、键、真值和 clear-sky，计算原始 W/m² RMSE/MAE；已保存验证预测可复算。训练器此前只在内存中持有每站 train-only normalization，checkpoint 未序列化精确值或其哈希；S 的 OOF 汇总也缺少可逐折核验的完整训练日期/索引排除证明。证据：`scripts/evaluate_hubei_forecast_heads_validation_20261016.py:51-59,74-126`；`scripts/train_hubei_ghi_heads.py` 的归一化和 checkpoint 输出段；`scripts/merge_hubei_forecast_ghi_oof_train_bank_20261017.py:64-112,706-739`。

## C. 结果与论文谱系：FAIL

当前同刻正式重建包为 `e8c5…`，A/AC/AP 验证 RMSE 为 74.361233 / 71.429292 / 73.894113 W/m²；正文的 AC 71.428 来自旧包 `7990846…`。两代结果不可混用。论文还保留了“湖北区间不可用”“forecast-GHI 尚未评价”等过时陈述。证据：`audit_receipts/same_time_three_seeds_recheck/independent_three_seed_validation_audit.json:20-69`、`audit_receipts/cot_pack_source_rebuild_20261017_v2/pack_report.json`、`paper/main.tex:25,227,233,273,294-298`、`paper/main_zh.md:5,232`。

## D. 分析脚本和产物冻结：FAIL

forecast-GHI 三种子复算为 FA 77.249908、FAC 78.580996、差值 +1.331088 W/m²；FAC 未显示验证收益，日期块区间跨 0。论文分析所用 10,000 次 bootstrap、连续两日敏感性与 CSV 汇总由另一段分析逻辑生成，未在当前冻结脚本清单中绑定。单种子审计复算了保存预测，但没有实际加载 checkpoint 重跑推理。证据：`audit_receipts/forecast_ghi_three_seed_validation_20261017/three_seed_validation_summary.json:112-139,3021-3026`、`scripts/evaluate_hubei_forecast_heads_validation_20261016.py:66-68,128-177`、`scripts/analyze_hubei_forecast_ghi_three_seed_validation_20261017.py`。

## E. 范围与语言：WARN

湖北结果来自三处相邻站点、22 个验证日、三种子；应写为“单独训练的湖北验证组”，不要让“独立验证队列”被误解成独立 test 或多个独立气候区。forecast FAC 的验证指标高于 FA，若后续 test 用于确认该负/不确定结果，须事先冻结这个目的并完整报告方向，不得按 test 调参。

## F. 评价类型：PASS

同刻 A/AC/AP 和 forecast FA/FAC 都是对真实站点观测的 validation。前者是同刻估计验证，后者是预测输入链验证；目前两者都不是 held-out independent test。

## Test/buffer 访问边界

本轮审计只检查元数据及已有收据，没有读取当前候选 test/buffer 数值。现有收据显示 `test_values_read=0`、`buffer_values_read=0`；此前 2025-08-01 有七条值被接触，已从新 split 排除。因此只能说当前冻结候选载荷未解码，不能说历史上相关字节从未接触。

## 放行前动作

1. 将 `e8c5…` 设为唯一同刻结果来源；旧 `7990846…` 只读标为 superseded。
2. 训练输出精确 normalization 文件和哈希，并对 checkpoint reload 做推理一致性验证。
3. 将 10,000 次单日/两日 bootstrap、逐时效 CSV 与绘图的实际分析脚本纳入 provenance；统一 metrics 和 summary 来源。
4. 补齐 S 各 fold 的训练 manifest，证明 OOF 日期不进入对应 S/R 训练；不能只验证输出 fold 标签。
5. 预先冻结预测头参评 arms（当前明确 FA/FAC；FAP 是否纳入需在 test 前决定）、三个 seed、checkpoint、共同支持、正负结果报告格式。
6. 增建 fail-closed test pack/inference/evaluator；仅使用 synthetic fixture 和 validation 做 dry-run，不触碰真实 test/buffer。
7. 同步更新 tracker、PAPER_PLAN、claim audit、英文 LaTeX 与中文 Markdown，再做独立只读复核后，才讨论一次性解封 test。

## 对目前论文结论的影响

- 同刻湖北 A/AC/AP：仅可引用新 `e8c5…` validation 结果，旧 pack 数字不得沿用。
- Forecast FA/FAC：验证上 FAC 未优于 FA；不支持“COT 提升未来 GHI 预报”的结论。
- 目前不允许写湖北独立 test 结论，也不允许把同刻验证称为预报技能。

## 输入文件 SHA-256

见同目录 `EXPERIMENT_AUDIT.json`。Reviewer 原始结论摘要保存在 `REVIEWER_TRACE.md`；该存档是本地审计留痕，不是平台级可验证模型 trace。
