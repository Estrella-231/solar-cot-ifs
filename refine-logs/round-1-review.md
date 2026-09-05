## 独立方法审查（同家族审稿，结论仅 provisional）

**Problem Anchor：保留。** 方案始终围绕“固定 SimVP 后，显式 COT 表示能否改善湖南 +15～+240 min 站点辐照度预测，并与因果可用 IFS 互补”，没有滑向 diffusion、骨干改造或 D+1～D+3。把 COT 明确界定为同源 AGRI 的物理归纳偏置，而非新增观测信息，是方案最稳健的部分。

| 维度 | 权重 | 分数 |
|---|---:|---:|
| Problem Fidelity | 15% | 9.0 |
| Method Specificity | 25% | 7.5 |
| Contribution Quality | 25% | 7.0 |
| Frontier Appropriateness | 15% | 9.0 |
| Feasibility | 10% | 8.0 |
| Validation Focus | 5% | 8.5 |
| Venue Readiness | 5% | 7.0 |

**加权分：7.90/10**
**CALIBRATION: none**
**Verdict：REVISE**

**GAP：** 当前方案已是清晰、低成本的应用遥感表示研究，但距离 READY 的主要差距不在模型复杂度，而在主张隔离仍不够严格：COT 显式表示、CPP 辅助监督和新增容量尚未形成完全同预算对照；IFS 的业务因果链也只写成门禁，尚未落实到唯一可执行的数据选择规则。最近邻全文差异未闭合，因此目前能支持“受控实证研究”，尚不能稳健支持“独特方法贡献”。

最少必要修订如下：

1. **CRITICAL：补齐 COT 表示与监督预算控制。** `AN-aux` 命名和接口含混。应定义一个与 AC/ACN 使用同一 R 架构、同一 CPP 样本、训练轮数和四维瓶颈的 `COT-supervised latent` 对照，仅把显式 `log(1+COT)` 统计量替换为等维隐变量；分别比较 AC 与该对照、ACN 与其 IFS 版本。否则只能证明“用了 CPP 监督的额外分支有效”，不能归因于显式 COT 表示。

2. **CRITICAL：把 IFS 因果性写成确定算法。** 对每个 `init_time` 选择 `release_time≤init_time` 的最新周期，记录 cycle、step、valid time、单位，并明确累计 `ssrd` 的差分和目标 15 min 窗口对齐。`legacy` 有效时刻匹配不得进入因果主表。`ssrd` 与其晴空指数基本同源，建议只保留一个规范化量，避免伪多变量增益。

3. **IMPORTANT：固定 CPP/COT 质量合同。** 预先写明 CPP 质量位、太阳/观测角阈值、缺测和异常 COT 处理；R 的架构、掩膜损失、输出范围及训练域也需确定。CPP 仅是检索参考，R–CPP 误差只能在有效检索子集报告，而下游 A/AC/AN/ACN 必须保持同一完整 cohort，不能因 COT 可用性改变测试样本。

4. **IMPORTANT：闭合新颖性与两篇独立性。** 全文核验最近邻后，把贡献收窄为“固定外推误差域中，显式 COT 物理瓶颈及其与因果 IFS 的受控增量”，避免把 COT 使用本身称为创新。第二篇应使用独立结果目录、主表和文字；第一篇的 diffusion/transport 主张和结果不得复用为第二篇证据。

可进一步精简：核心只保留四格、容量/监督匹配对照和必要的 R 域偏移诊断；C-only、N-only、置换实验可降为补充材料。无需加入注意力、门控、diffusion 或其他“前沿”组件，当前克制路线更符合应用遥感问题与低成本约束。固定湖南 SimVP 合规权重、CPP 覆盖和因果 IFS 三项资产审计通过后，方案可进入实验计划。
