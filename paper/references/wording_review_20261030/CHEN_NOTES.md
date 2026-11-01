# Chen两篇论文的词语与句式核对

2026-10-30。目的为改进当前COT论文语言，不是复述其结果或认定这些期刊采用统一写法。以下页码均为PDF印刷页码；两篇均取得17页全文。引号内才是原文短语，其余为原创分析和建议。

## 来源与阅读范围

- Chen et al. (2023), *Applied Energy* 352,121979，[DOI](https://doi.org/10.1016/j.apenergy.2023.121979)，[作者公开全文](https://www.li-realab.info/publication/solar-forecast/chen-2023-resourcing/chen-2023-resourcing.pdf)。复读摘要、§3.1–3.3（pp.6–9）、§4（pp.10–11）及§5（pp.11–12）。本地文件沿用前次阅读目录。
- Chen et al. (2024), *Energy Conversion and Management* 313,118598，[DOI](https://doi.org/10.1016/j.enconman.2024.118598)，[作者公开全文](https://www.li-realab.info/publication/solar-forecast/chen-2024-dlchain/chen-2024-DLchain.pdf)。复读摘要、§1.2（p.3）、§3.1（p.9）、§3.3–4（p.12）、§4–5（p.15）；其余方法、图表页用于确认上下文。

## 2023：如何写同期估计与条件性增益

| 位置 | 原词或表达动作 | 对当前稿的启示 |
|---|---|---|
| 摘要；§3.1，p.7 | “irradiance estimation” | 本文同期GHI统一用估计，COT模块用反演；不同任务不靠同一个prediction词包办。 |
| §3.1，p.7 | “is reduced from” | 先给指标起止值，再给相对降幅，避免只说性能提升。 |
| §3.2，p.7 | “clear and cloudy sky conditions” | 用明确条件限定比较；本文按参考COT分层，不能把薄厚分层改名成天气分类。 |
| §4，p.10 | “possible reason” | 观察判断直接写；提出解释时单独标明推断，不把解释与测量结果混为一句。 |
| §4，p.11 | “shows potential” | 展望用能力方向，不把尚未执行的后续算法写成既定收益。 |

其结果段常先说某指标的总体方向，接数值及图表，再指出偏差或条件中的不同方向。我们可以学这种信息顺序，不必重复其句首套语。

特别注意：p.7将归一化RMSE从25.72%降至20.57%的差写成5.15%。这是归一化指标的百分点差；不能照抄为本项目相对RMSE降幅。本文须按自身公式计算并清楚区分。

## 2024：如何写物理量中间表示与预报

| 位置 | 原词或表达动作 | 对当前稿的启示 |
|---|---|---|
| §1.2，p.3 | “model chain” | 按模块的物理输入、输出说明顺序，不用任务目录或资产名解释流程。 |
| §3.1，p.9 | “forecast horizon” | 逐时效比较直接写时效，不用工程坐标/交付术语。本文图轴仍明确相对历史末帧。 |
| §3.1，p.9 | “lower nRMSE” | 总体误差较低并不自动表示偏差也小；各指标的结论分开写。 |
| §4，p.12 | “could be due to” | 机制动词与证据程度对应；该文对训练目标与偏差的解释使用推测语气。 |
| §4，p.15 | “is related to” | 把云检测、位置和演变困难连接到预报误差，再提出后续研究，不需要插入工程进度。 |

该文先检验从图像得到的物理量是否适合预报，再用条件分布和辐照度突变评价解释不足；它没有提供本项目COT误差的因果分解。可借鉴中间表示的表述方法，不能借其结论替代我们的输入替换结果。

其若干原句语法不够简练，例如机制短语后直接接that的写法。因此学习的是主语、比较对象、指标和解释之间的关系，不能逐字套用。

## 本项目可采用的原创句式（非文献引语）

| 用途 | 英文建议 | 中文对应 |
|---|---|---|
| 方法比较 | We assess the contribution of retrieved COT by training irradiance models with and without this input. | 分别训练加入和不加入反演COT的辐照度模型，评价COT的作用。 |
| 量化结果占位 | Adding COT reduced GHI RMSE from [x] to [y] W m⁻², corresponding to a relative reduction of [z]%. | 加入COT后，GHI RMSE由[x]降至[y] W/m²，相对降幅为[z]%。 |
| 条件差异占位 | The reduction was largest for [condition], whereas [other condition] showed [result]. | [条件]下的降幅最大，而[另一条件]表现为[结果]。 |
| 位移机制 | Errors in cloud position shift the timing of attenuation at the station. | 云位置误差改变云遮蔽站点的时刻。 |
| 结构机制 | Spatial smoothing reduces local contrasts in retrieved COT. | 空间平滑削弱反演COT的局地对比。 |
| 机制结果占位 | Replacing forecast-derived COT with observation-derived COT changed GHI RMSE by [x] under otherwise identical inputs. | 保持其余输入相同，用真实图像反演的COT替换预测图像反演的COT，使GHI RMSE变化[x]。 |

结果句只有在回填时才确定方向；机制句需与正文输入替换和实际云事件对照。这里的比较术语、数值占位和原创例句用于写作，不改变实验方案。

## 阅读资料保留

新下载PDF及提取文本仅本地阅读，由本目录`.gitignore`排除；公开仓库只保留来源、获取记录和原创笔记。英文原文摘录合计：2023篇14词，2024篇13词；未复制完整段落。
