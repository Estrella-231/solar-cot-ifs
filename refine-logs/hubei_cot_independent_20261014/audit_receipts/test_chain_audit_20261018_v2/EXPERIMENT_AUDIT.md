# 湖北冻结独立测试链审计（2026-10-18 v2）

**总体结论：FAIL / NO-GO，禁止实际独立test评分。** 本轮为新鲜只读审查；未读取真实test/buffer的GHI、AGRI、CPP/COT或预测/模型载荷。

**审阅者：** fresh same-family Codex 子代理（gpt-6.1-sol，read-only，provisional）。本结论不代表数据解封授权。

## A. 真值来源：FAIL

评分器已要求单独的站点真值包，并将真值与预测包分开；但它只验证真值manifest状态、站点、路径与文件哈希，没有把原始工作簿哈希、标签构建器哈希、右闭15分钟窗口合同、来源接触记录绑定到真值包（`scripts/evaluate_hubei_forecast_heads_frozen_test.py:245-281`）。元数据清单仍写明 `test_truth_pack_created=false`、`test_evaluation_authorized=false`，且仪器量纲元数据未核实（`audit_receipts/test_truth_source_inventory_20261018/source_inventory.json:774-783`）。4–5月工作簿摘要来自历史进度记录的固定哈希，而非train/val来源报告独立绑定（同文件 `:98-106`）。

**最小下一步：** 实现受发布门禁保护的真值构建器，将冻结split、原始来源、构建器、单位/标签合同和逐文件输出哈希写入manifest；先用合成端到端夹具验证。

## B. 模型与推理来源：FAIL

`model_manifest_sha256` 目前只通过64位十六进制字符串格式检查（`evaluate_hubei_forecast_heads_frozen_test.py:83-87`），随后被复制到ledger/report（`:322`, `:506`）；没有读取模型清单，也没有校验实际checkpoint、归一化、训练器、冻结S/R与推理入口/回执。三seed验证复算虽记录pack/trainer摘要，但不是test推理回执（`audit_receipts/forecast_fap_three_seed_20261018/forecast_fap_three_seed_validation_recheck.json:22-25,396-402`）。

**最小下一步：** 生成逐模型文件的路径/SHA清单，绑定FA/FAC/FAP九组站点×种子实际checkpoint、归一化、训练器、S/R来源和推理源码；完成受控推理与完整receipt后再审。

## C. 一次性账本：WARN

账本使用 `O_CREAT|O_EXCL` 原子创建、文件fsync，且占用后不因异常删除（`scripts/hubei_test_ledger.py:30-74`）；评分器在真值哈希和预测载荷读取前占用ledger（`evaluate_hubei_forecast_heads_frozen_test.py:307-340`）。但ledger路径仍由调用方提供的split路径派生；复制同一split至另一路径会产生另一份ledger，理论上绕开“一次性”限制。当前目录不存在该split对应的ledger；这只能说明当前未见该文件，不能证明历史上从未访问test。

**最小下一步：** 将ledger锚定到一个固定的实验release身份/唯一规范位置，并把所有依赖模块源码哈希纳入release receipt。

## D. 软件拒绝门禁及对照支持：PASS（仅合成范围）

评分器在任何预测/真值载荷打开前拒绝非合成scope（`evaluate_hubei_forecast_heads_frozen_test.py:68-75`）；预测包禁止内嵌真值（`:156-159`）；truth文件哈希在账本占用后核验（`:277-281,328`）；标准时效键和精确支持得到校验（`:176-198,342-366`）。代码强制三站×三seed×FA/FAC/FAP，并检查预期键哈希、seed/arm间真值与晴空值及三站共同支持（`:35-38,109-114,336-366,418-495`）。这是软件合同证据，不是独立真值或科学test证据。

## E. 指标合同：WARN

RMSE/MAE/bias/R²、nRMSE/nMAE、晴空敏感性、逐lead、三站共同支持和日期簇bootstrap函数均存在（`scripts/hubei_forecast_metrics.py:26-128,176-270`）。尚未实现预注册的连续两日块敏感性（`EXPERIMENT_PLAN.md:82`），也未覆盖计划中的月份/太阳高度/云况分层（`:84`）。bootstrap重复次数和随机种子由CLI调用者提供，未冻结在receipt中（`evaluate_hubei_forecast_heads_frozen_test.py:298-304,548-549`）。

**最小下一步：** 按计划实现两日块敏感性及可支持的分层报告，或在读取test前正式修订并冻结指标合同；将bootstrap配置绑定至release receipt。

## F. 论文主张范围：PASS（验证范围内）

正文中FA/FAC/FAP结果被限定为validation，RMSE为77.150/78.155/78.980 W/m²，三组区间跨零且明确无独立test（`paper/main.tex:183,297-302`; `paper/main_zh.md:181`）。验证收据声明test/buffer读取为0（`audit_receipts/forecast_fap_three_seed_20261018/forecast_fap_three_seed_validation_recheck.json:2-25`）。当前来源清单声明未解码真实test辐照度值（`source_inventory.json:774-784`）。这些记录限制了论文陈述，但不能独立证明全部历史访问链。

## G. 旧审计状态与哈希记录：WARN

`test_chain_audit_20261018/EXPERIMENT_AUDIT.md:7-9` 中关于“真值同包、没有ledger、缺少主要指标”的判断已被当前源码部分修复，不能再作为当前状态使用。当前评分器SHA-256为 `6839bdba23114005748841810166a6d42a8ac44cb0f9005de8cbbcbe32e55ecb`；新测试脚本SHA与执行表记录一致。当前执行表此前记录的评分器SHA少字符，已更正。

## 结论与放行状态

真实test仍 **FAIL / NO-GO**。最少仍需：独立真值构建及源字段合同、实际模型/归一化/训练器/推理文件SHA绑定、受门禁控制的推理receipt、固定一次性ledger与冻结指标参数、两日块敏感性，以及新的独立审计。test与buffer载荷保持关闭；不得以本审计或合成测试代替数据发布授权。
