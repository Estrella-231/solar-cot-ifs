# 论文用词与句式核对：从文献原文到本稿修改

日期：2026-10-30。任务是核对具体词语、句式和论述方式，并修改现有双语稿。不是新增实验，也不是把同领域论文的写法当成统一规范。

## 1. 阅读范围与定位

| 论文 | 本轮核读范围 | 重点 |
|---|---|---|
| Chen et al. (2023), Applied Energy, 352, 121979 | 已有17页全文；摘要、§3.1–3.3、§4–5重新核读。详见[逐条笔记](CHEN_NOTES.md) | 同期估计的任务名、指标比较、晴空指数条件、机制解释 |
| Chen et al. (2024), Energy Conversion and Management, 313, 118598 | 新取得作者公开17页全文；§1.2、§3.1、§3.3–5重点核读。详见[逐条笔记](CHEN_NOTES.md) | 中间物理量、链式预报、误差随时效变化和机制措辞 |
| Papachristopoulou et al. (2024), AMT, 17, 1851–1877 | 网页全文；§2.1–2.3、§3.1.1、§3.1.3–3.1.4、§3.2.1 | 现时估计与未来预报、偏差与总误差、物理原因 |
| Zhang et al. (2026), GRL, 53, e2025GL119553 | 核对出版社返回的§2.2.2、§2.3、§3.1–3.3、§4正文段落；未连续通读完整PDF | 连续量与分类评价、参考产品一致性、时效和结构模糊 |
| Liao & Coimbra (2024), Renewable Energy, 232, 121055 | 出版社可检索摘要和Highlights；全文请求403，未按全文归纳 | 有无云物理信息的直接比较、方法命名 |

原始来源：[Chen2023作者全文](https://www.li-realab.info/publication/solar-forecast/chen-2023-resourcing/chen-2023-resourcing.pdf)、[Chen2024作者全文](https://www.li-realab.info/publication/solar-forecast/chen-2024-dlchain/chen-2024-DLchain.pdf)、[AMT全文](https://amt.copernicus.org/articles/17/1851/2024/)、[GRL出版社正文](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025GL119553)、[SCOPE-CNN出版社摘要](https://www.sciencedirect.com/science/article/pii/S0960148124011236)。

## 2. 原词如何与论述作用对应

Chen两篇的短引语及准确页码单列于[CHEN_NOTES.md](CHEN_NOTES.md)，这里不重复摘录。下表引号内为原文短语，其余为原创概括。

| 来源与位置 | 原词 | 可借鉴的表达方式 |
|---|---|---|
| AMT §2.1–2.2、§3开头 | “modeled GHI”；“forecasted GHI” | 通过输入时态和目标时刻区分估计与预报。该文也用retrieval描述辐照度估计，不能宣称retrieval只可修饰COT。 |
| AMT §3.1.3 | “can be attributed to” | 先提供误差诊断，再解释来源；不能仅由总体RMSE改变推导原因。 |
| AMT §3.1.1、§3.1.4 | “overestimates” | 有符号偏差用高估/低估；RMSE用误差增加/降低，两个结论分开。 |
| GRL §2.2.2–2.3、§3.1 | “CLP identification” | 相态分类与COT连续量预测分别命名和评价。 |
| GRL §3.2 | “higher consistency” | 与卫星产品比较可描述一致性，不能直接称真实云物理误差。 |
| GRL §3.1、§3.3 | “As the lead time increases” | 时效趋势先用明确指标描述，再讨论可能机制。 |

AMT摘录共9词，GRL共10词。只采用可核对的短语，不复制段落。SCOPE-CNN摘要直接比较有无云物理信息的模型；本稿学习这种明确比较对象的方式，不照搬其中的效果形容词或推断其全文细节。

## 3. 本稿固定的术语

| 含义 | 本稿采用 | 避免的含糊写法 |
|---|---|---|
| 根据目标区间末观测云图估计区间平均GHI | GHI estimation from observed imagery；基于观测云图的GHI估计 | synchronous若未定义容易含糊；instantaneous GHI会改变15分钟均值含义 |
| AGRI到云光学厚度 | COT retrieval / COT retrieval model；COT反演／反演模型 | 单独的retriever重复过多，或把COT当独立观测 |
| 历史云图产生未来图像 | cloud-image prediction；涉及具体外推过程时用extrapolation | 底图作为英文直译basemap；对所有步骤只写prediction |
| GHI模块 | irradiance model；首次说明parameters remain fixed | frozen head在摘要、结果和图注中反复出现 |
| 增益 | reduction in GHI RMSE、relative RMSE reduction | accuracy提高百分比；representation value作为结果的唯一表述 |
| 配对评价 | paired comparison / paired differences / paired resampling | paired作为泛泛形容词附加到每个名词 |
| 样本一致 | the same valid samples；相同有效样本 | common support、validity support、sample keys用于一般正文 |
| 薄厚条件 | COT bins / optical-thickness ranges | 将薄厚阈值直接改称晴天、多云或阴天 |
| 物理解释 | cloud-position errors、local COT contrasts、growth and dissipation | 仅说底图不好、表示退化、性能损失，缺少物理对象 |

术语是本稿的编辑选择，不是对整个学科词义的排他规定。已保留真正需要的训练参数固定、配对样本和区间定义。

## 4. 句子怎样写

以下均为本稿原创模板，不是文献引语；结果方向及数值在回填时确定。

**研究问题。** 用输入、输出与比较对象落地，而不是抽象讨论“表示价值”：

> We compare irradiance models with and without retrieved COT while retaining the original multispectral inputs.
>
> 保留原始多光谱输入，比较加入和不加入反演COT的辐照度模型。

**方法。** 主语明确是模型，动词写实际操作，参数固定在方法中说明一次：

> The model receives all nine COT values in their original spatial arrangement.
>
> 模型接收按原空间位置排列的九个COT值。

**结果。** 先比较对象和指标，再量化幅度，再说条件差异：

> Including COT [reduced/increased] GHI RMSE from [x] to [y] W m−2, a relative change of [z]%. The largest change occurred under [condition].
>
> 加入COT后，GHI RMSE由[x]变为[y] W/m²，相对变化为[z]%；[条件]下的变化最大。

**机制。** 将实际现象、控制比较和解释写成不同句子。可先写云团到站提前/延迟及COT峰值误差，再写输入替换使GHI误差改变多少，最后解释两者的联系。位置误差用位置和时刻，平滑用局地对比，生消误差用峰值和变化幅度；不要用一个“外推不足”概括所有环节。

**因果动词。** 结果直接用increased/reduced；关系用associated with或consistent with；有针对性对照才使用contributes to或attributed to。关键是证据与动词相称，不是机械地把每句话改成may。本文结果未齐处保留占位，不预先填入主导比例。

**展望。** 说明接下来要改进的具体量：云位置、边界和光学厚度演变；说明将如何检验其对GHI的作用。删除正文中“本篇完成什么、下一篇再发什么”的项目安排语言。

## 5. 已落实到现有双语稿

已对`paper/main.tex`与`paper/main_zh.md`实施35组成对替换，共70项：摘要、引言、局地COT输入、预报应用、评价方法、误差传递、结果标题、图注和讨论。

典型修改：

| 修改前 | 修改后 |
|---|---|
| measure retention of the estimation representation in forecasting | determine whether the benefit of including COT persists in forecasting |
| physically defined coordinate for cloud attenuation | physically interpretable input to the irradiance model |
| common reference-supported rows | the same samples with valid CPP references |
| 反演表示在预报中的保留程度 | 加入COT的作用能否延续到预报 |
| 同一站点—初始化—目标键、标签、有效支持 | 相同站点、起报时刻、目标区间、观测值、有效样本 |
| 下一篇进一步研究如何降低这些上游误差 | 后续可保持反演与辐照度模型不变，检验更准确的云图预测能否进一步降低GHI预报误差 |

修改前稿、全部替换记录、公式/引用一致性检查保存在`paper/revision_records/20261030_wording_review/`。十一条公式和引用保持不变；科学范围、评价样本、数值设置未变。结果图表仍为占位，没有生成虚构数据。

## 6. 不机械模仿论文原句

Chen2023有把归一化指标的百分点差直接写成百分数的地方；Chen2024个别机制句语法并不简练。这说明正式发表的论文也需要辨别。我们借鉴的是比较对象、指标、条件、解释的组织关系；相对降幅仍按本稿定义计算，英语按准确自然的表达改写。
