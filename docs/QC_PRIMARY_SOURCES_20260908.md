# 源观测恒定值质量控制：一次来源核查

核查日期：2026-09-08。范围：为 A/AC pilot 的源 GHI 质量处置提供可引用依据；本记录不修改标签、样本、损失或训练配置。

**结论：检出的正值长恒定段有必要调查，但本次核查未找到“5 min GHI 连续正值恒定 60 min 即自动删除”的 BSRN 或 SERI-QC 标准。若采用此阈值，必须称为本研究预先定义、经证据复核的源观测 QC 规则，不能借标准名称掩盖自定阈值。**

## 1. BSRN 2026 最佳实践（已读取全文）

Knap 等，*BSRN Data Quality Best Practices*, v1.0，2026。AWI 官方资源页将其列为 WRMC 社区指导文档：[官方资源页](https://bsrn.awi.de/resources/quality-control-resources/)；[DOI](https://doi.org/10.5281/zenodo.21717394)；[PDF](https://zenodo.org/records/21717394/files/BSRN_DataQuality_BestPractices_v1.0_20260728.pdf)。

- 第 2 页 §1.1：时间分辨率假设为 1 Hz 采样形成的 **1 min 平均**。本项目 5 min 区间均值不属于直接原样套用范围。
- 第 4 页 §2.1：建议从时间序列中检查尖峰、骤降、平线和不对称日变化；**没有给出平线的 60 min 阈值**。
- 第 5 页 §2.2：自动旗标应引发调查；极少见值可能是云增强等真实现象，不能见旗标即删除。
- 第 21 页 §4.2：调查并注释旗标，保留原始数值；结合维护日志解释问题。
- 第 21–22 页 §4.4：只有已知原因、物理上合理的修正方法和明确区间才可修正；原因未明时保留原值并加旗标/注释；发布更正需新版本及处置记录。

证据限制：全文检索 `flat / stuck / constant / 60 / minute` 并阅读相关节，没有找到连续 60 min 正值平线的自动剔除规则。该结论仅指本次检查的文档，不声称所有辐射 QC 文献均无类似研究阈值。

获取记录：web PDF 解码失败后，经正常公开 HTTPS 下载取得 PDF，使用 `pdftotext -layout` 读取；大小 3,412,612 字节，SHA-256 `4261dc584b476f4ea1189adf7ed8e76f0cacb952306412c1a376796e002ed948`。副本位于本次执行系统临时目录，不作为工程长期依赖。

## 2. BSRN 推荐 QC v2.0：物理界限与平线检测是两回事

[Long 与 Dutton 原始推荐文件](https://bsrn.awi.de/fileadmin/user_upload/bsrn.awi.de/Publications/BSRN_recommended_QC_tests_V2.pdf)共 3 页，规定物理可能界限、极少见界限和分量比较。GHI 物理上限为 `Sa × 1.5 × max(cos(SZA), 0)^1.2 + 100 W m^-2`，下限为 `-4 W m^-2`；它不包含 60 min 正值恒定测试。

应用时需验证太阳位置、时间和单位，说明由 1 min 指导到 5 min 区间的适配；不能用本项目 `GHI / clear_sky_GHI` 的极大值本身冒充 BSRN 物理界限检验。[AWI 质量检查说明](https://bsrn.awi.de/data/quality-checks/)也将 QC 定义为可按用户目的适配的后处理。

## 3. SERI-QC 与可配置平线工具

[NREL/NLR 官方 SERI-QC 说明](https://www.nlr.gov/grid/solar-resource/seri-qc)：对逐小时 GHI、DHI、DNI 赋质量旗标，旗标可用于用户定义的筛选，但不修改原值。其 `Kt` 是 GHI/大气外水平辐照度，**不是本项目 GHI/晴空 GHI**。该页面没有 5 min、60 min 正值平线标准。

[PVAnalytics `stale_values_diff` 官方文档](https://pvanalytics.readthedocs.io/en/stable/generated/pvanalytics.quality.gaps.stale_values_diff.html)用可配置的连续样本数和数值容差标记恒定段，并明确可标记整窗、除首点外的尾部或窗末；默认窗长是 6 个值，**不是 60 min**。[`stale_values_round`](https://pvanalytics.readthedocs.io/en/stable/generated/pvanalytics.quality.gaps.stale_values_round.html)则可按小数精度判断重复值。这支持恒定段检测作为工程手段，不能为任意持续时间赋予官方标准地位。

## 4. 对当前处置的建议（本研究判断，非来源原文）

1. 原始工作簿和未筛选标签保持不变；另存源记录旗标、段起止、重复值、记录数、相邻时间间隔、原因证据、派生 15 min 标签影响以及审查结论。
2. 同一规则覆盖训练/验证全范围，A 与 AC 使用同一 QC cohort；旗标发现与模型性能无关。不能按某日效果临时删整天，也不以裁剪 kt 代替查明源问题。
3. 明确“60 min”的数学定义：12 个连续、互不重叠的 5 min **区间**覆盖 60 min，但首末**时间戳跨度**仅 55 min；若规则要求首末跨度至少 60 min，应至少 13 个值。必须在代码、审计和叙述中使用同一口径。
4. 本项目是源记录的离线标签 QC，可使用标签区间的完整证据；旗标不是业务推理输入，也不代表在线能提前知道传感器故障。
5. 若只有平线而无进一步证据，应保留“待复核”状态；对已核查为无效的具体源段，可建立可追溯的新 QC 标签版本，明确这是研究处置而非完成官方 BSRN 认证。
