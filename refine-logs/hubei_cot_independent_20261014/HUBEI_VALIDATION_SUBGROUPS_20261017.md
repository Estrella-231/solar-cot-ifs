# 湖北三站同期验证分层复核（2026-10-17）

> 历史快照说明：本文使用旧的同期A/AC/AP训练输出。当前归一化重载复现的分层结果以[新合同报告](HUBEI_VALIDATION_SUBGROUPS_NORMRELOAD_20261017.md)为准；不得混合两批预测。

## 范围与可复核来源

本记录仅分析新冻结 pack 上 A/AC/AP 三个 seed（42/43/44）的固定 validation 保存预测。它们是用于 best-epoch 选择的验证样本；分层用于描述异质性，不是独立确认性 test，也不代表未来 GHI 预报技能。没有读取 test/buffer 载荷。

- 复算脚本：`tooling_20261017/audit_same_time_month_solar_subgroups.py`
- 机器可读结果：`audit_receipts/same_time_three_seeds_recheck/month_solar_subgroups_validation.json`
- 冻结 pack SHA-256：`e8c5b2dc587e8a938bba43fc564be43aec777788aca223c054341f75dd09a515`
- 父级三 seed 独立验证审计 SHA-256：`4c2005b458c4514a3b17aea32a28c2693139926384d17666feee833c0d0573de`
- rows 与 geometry SHA-256 分别为 `43f7bc09892fb31307a75534b0a0c449ecb0ba4056b715af1b6bfd16042f46db`、`b6a4b4d010fe005706dbdd6f55e827201f1bad89a6235433ff92b0741163ec9b`；每个 seed/站点/方法的逐样本预测文件SHA也写入JSON。

RMSE差按每个seed分别计算后等权平均；负值表示 AC 相对对照误差较低。太阳高度由同一 validation 目标时刻几何通道 `cosSOZ` 计算，固定分箱 `[−90,0)、[0,15)、[15,30)、[30,45)、[45,60)、[60,90.0001)`。修订前脚本漏掉了太阳高度小于0°的样本；本版把它们单独列出并验证所有预测键均落入一个分箱：Niushou 856/856（其中9条低于0°），Wangzhai 784/784（7条），Mizhuang 816/816（9条），未分箱为0。

## 太阳高度分层

| 站点 | 太阳高度 | n | AC−A RMSE（W/m²） | AC−AP RMSE（W/m²） |
|---|---:|---:|---:|---:|
| Niushou | 0–15° | 169 | −3.50 | −2.85 |
| Niushou | 15–30° | 209 | −3.07 | −2.27 |
| Niushou | 30–45° | 220 | +0.57 | +0.72 |
| Niushou | 45–60° | 156 | −0.41 | −0.07 |
| Niushou | 60–90° | 93 | −5.29 | −4.23 |
| Wangzhai | 0–15° | 154 | −2.91 | −2.34 |
| Wangzhai | 15–30° | 190 | −4.09 | −2.99 |
| Wangzhai | 30–45° | 202 | −2.48 | −1.72 |
| Wangzhai | 45–60° | 141 | +1.90 | +2.48 |
| Wangzhai | 60–90° | 90 | −0.28 | +0.04 |
| Mizhuang | 0–15° | 162 | −6.88 | −6.57 |
| Mizhuang | 15–30° | 201 | −9.79 | −9.24 |
| Mizhuang | 30–45° | 218 | −13.02 | −13.02 |
| Mizhuang | 45–60° | 136 | +8.10 | +8.34 |
| Mizhuang | 60–90° | 90 | +1.89 | +1.92 |

低于0°组每站只有7–9条、观测 GHI 近零，只作为分箱完整性记录，不作技能解释。有效太阳高度各组的差异并不一致：Mizhuang 在15–45°组的 AC 点估计低于 A/AP，但45–90°转为较高；Wangzhai在45–60°也转为较高；Niushou各组的差异较小且有正有负。各组没有另外作显著性检验，不能据这些validation子集宣称太阳高度条件下稳定增益。

## 月份分层和天气条件限制

Validation 日期块为22个日期，逐月支持稀疏。AC−A 的月别点估计正负混合：Niushou为8/11个月低于A，Wangzhai 8/11，Mizhuang 6/11；若干月变化幅度较大且每月支持有限。因此不写季节性或全年稳定改善结论。

本轮共同 GHI pack 不含独立 CLP/CPP 云覆盖标签；可见的 `cpp_16_fixed_v2` 参考数据时间覆盖截至2026-03-02，也未证明它提供与预注册阈值兼容的独立云覆盖分类变量。故本报告**不生成晴/多云/厚云分类**，不从 GHI、预测误差或模型生成 COT 倒推天气类别。若无法找到覆盖对应 validation 日期、字段含义和阈值口径都可核验的 CLP/CPP，则正式记录为“云况分层缺项”。

## 后续

旧版 `validation_cases_b2_20261016_v2/FIGURE_AUDIT.json` 的27个预测文件SHA与本次冻结pack三seed新预测全部不匹配，因此已判为不适用于本次数值结果，原文件保留不覆盖。

依原先冻结、未查看模型误差的24案例清单，已从本次带键预测包重绘GHI折线图：`validation_cases_b2_20261017_rebuilt_v2/`。图中是站点观测与A/AC/AP三seed均值，阴影为跨seed样本标准差（不是置信区间）；目标15分钟缺测仍断开。新的`FIGURE_AUDIT.json`绑定本次父级审计、pack和所有预测文件SHA，确认test/buffer读取为0。旧2026-10-17初版因标题与图例重叠已标记为superseded，v2已人工检查布局。

该交付**仅为GHI折线示例图**，没有同位AGRI/COT图像，也没有独立云况分类；不能作为完整天气案例分析或论文最终图。若云况参考仍缺，不补造天气标签。test/buffer继续关闭，除非预定冻结与一次性test流程另行满足。
