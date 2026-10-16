# COT反演价值与预报应用：系统补检与论文定位

日期：2026-10-16。范围：云物理属性直接外推、COT辅助地表辐照度估计、COT参与短临辐照度预报。本文是有检索轨迹的专题综述，不宣称完成全部数据库的PRISMA穷尽检索。仅整理文献和研究定位，不改变模型合同、训练任务或测试集状态。

## 1. 结论先行

1. **云光学厚度进入辐射估计，以及先预测COT再计算辐射，均已有明确先行研究。** 本文不能以“首次引入COT”或“首次直接预测云物理属性”作为创新。
2. 与当前论文最接近的不是只有Cloud-FNO：SCOPE-CNN直接比较增加云物理特征的价值；NextSENSE2同时评价同期GHI估计与未来GHI预报；GOES-16 PredRNN与Peng的物理机器学习链路直接连接COT预测和辐射。
3. 我们最适合回答：**在已有AGRI与太阳几何的条件下，显式COT表示给站点GHI估计增加多少价值，在哪些云厚度及区域成立，以及这种价值经过固定外推后保留多少。** 这是待用实验回答的问题，不预先宣布COT在所有结构上有效。
4. 预测收益弱于同期收益可以检验是否来自外推误差，但不能直接全部归因于外推。反演误差、空间代表性、时间平均、输入分布和头部利用方式也须区分。

## 2. 纠正“全部漏检”的说法

本轮核对了 `docs/LITERATURE_POSITIONING.md`、`docs/COT_GHI_LITERATURE_BREAKTHROUGH_20260929.md`、`docs/CLOUD_FNO_BASELINE_ASSESSMENT_20261003.md` 及 `paper/references/narrative_review_20261015/README.md`。

- 09-05记录已经列出GOES-16 COT→SSI和Peng的COD→GHI论文；09-29已列HelioNet、SolarSTEPS、Cui、Miller和Wiltink。问题包括**检索命中后全文核对、对照提取和论文整合不足**，不能说现在才发现。
- Cloud-FNO有10-03用户PDF阅读与代码核验记录，不能称本轮首次读到。
- SCOPE-CNN在10-15写作参考记录中已出现，但只核对摘要、侧重组件命名；本轮将其提升为科学问题的最近邻，仍需全文核对。Papachristopoulou2024也已有10-15全文阅读及本地PDF记录，应标为已有工作重新整合，不能因为“NextSENSE”关键词未命中就判断漏检。
- 在本轮检查的docs与paper文本中，未检到Xu2024、Yu2023、Sirch2017、Takenaka2011、Heliosat-4和Al Asmar2021的对应关键词记录。这里只说明所查记录范围，不证明全部历史会话从未提过。
- Drosoglou预印本于2026-10-15发布，属于新增动态，不能归咎于早期漏检。

## 3. 文献矩阵与证据层级

F：已核对原文相关方法/结果章节，不等于逐字通读。P：出版社可检索正文片段。A：摘要/作者机构页面。M：仅元数据或引文链，不能支撑具体性能结论。日期采用期刊卷期年；DOI中的年份可能较早。下表“未核实消融”不等于原文必然没有消融。

### 3.1 云物理属性直接外推

| 文献及原始来源 | 方法、评价与结果边界 | 对本文的用途；读取层级 |
|---|---|---|
| Zhang et al. (2026), *Short-Term Forecasting of Cloud Physical Properties Based on Fourier Neural Operator Method*, GRL. [DOI:10.1029/2025GL119553](https://doi.org/10.1029/2025GL119553) | 历史CLP/CTH/CER/COT联合预测，3h输入、3h输出并递推6h；CPP及MODIS对照。没有GHI任务。 | 云物理直接外推基准；6h综合COT指标不能直接和我方局地单时效排名。F，用户PDF及既有阅读记录。 |
| Xu et al. (2024), *Deep-Learning-Based Daytime COT Retrieval and Prediction Method Using FY4A AGRI Data*, Remote Sensing16,2136. [原文](https://www.mdpi.com/2072-4292/16/12/2136) | 同时做日间COT反演和CPM预测；64×64历史COT，比较SimVP、ConvLSTM，Charbonnier与边缘损失；不同输入输出长度。 | FY-4直接COT预测先例；其60min首输出RMSE不能当4h指标；未见GHI评价。P，第5.4节及结论。 |
| Yu, Tan, Ma & Yan (2023), *Nowcast for cloud top height from Himawari-8 data based on deep learning algorithms*, Meteorological Applications. [DOI:10.1002/met.2130](https://doi.org/10.1002/met.2130) | 四川CTH过去1h→未来2h，10min步长；TrajGRU、ConvLSTM、光流和持久性，分季节。 | CTH不是COT；属性预测有季节差异，不能只报总平均。F，方法与数据。 |
| Drosoglou et al. (2026-10-15), *From cloud optical thickness to solar radiation: deep spatiotemporal forecasting with geostationary satellite imagery*, EGUsphere. [预印本](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-5862/) | SEVIRI COT→ConvLSTM-UNet，Charbonnier+SSIM；递推并评价CMF。摘要约30%改善是CMF相对持久性skill。 | 最新高度相关工作，仍审稿；未确认地面GHI验证，不能写GHI提升30%。A，PDF获取未完成。 |

### 3.2 COT及云物理属性辅助同期辐照度估计

| 文献及原始来源 | 方法、评价与结果边界 | 对本文的用途；读取层级 |
|---|---|---|
| Takenaka et al. (2011), *Estimation of solar radiation using a neural network based on radiative transfer*, JGR116,D08215. [原文](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2009JD013337) | 用RT计算训练快速神经网络求解器，输入水/冰云光学厚度、有效半径、云顶压力、太阳天顶角、地表及气体参数。 | COT→辐射神经映射的经典先行研究；不是AGRI与AGRI+COT同头消融。F，第2节。 |
| Qu et al. (2017), *Fast radiative transfer parameterisation for assessing the surface solar irradiance: The Heliosat-4 method*, Meteorologische Zeitschrift26,33–57. [作者机构全文](https://elib.dlr.de/106996/1/Qu_Fast_radiative_transfer_parameterisation_for_assessing_the_surface_solar_irradiance_The_Heliosat_4_method_87036.pdf), DOI10.1127/metz/2016/0781 | McClear+McCloud快速RT，云与大气属性→GHI及直散分量；13个BSRN站15min均值。 | 物理模型基线和标签时间口径参考；不能与我们的站点RMSE横向直接排名。F，摘要与方法说明。 |
| Al Asmar et al. (2021), *Improvement of solar irradiance modelling during cloudy-sky days using measurements*, Solar Energy230,1175–1188. [原文](https://www.sciencedirect.com/science/article/pii/S0038092X21009488), DOI10.1016/j.solener.2021.10.084 | code_saturne辐射模型结合SIRTA云观测；敏感性分析认为COD最重要，其次云量，直接辐射对云量尤其敏感。 | 支持研究COD价值的物理动机；地面云测量条件不同，不是卫星学习头的纯COT收益证明。A及引言。 |
| Liao & Coimbra (2024), *Hybrid solar irradiance nowcasting and forecasting with the SCOPE method and convolutional neural networks*, Renewable Energy232,121055. [原文入口](https://www.sciencedirect.com/science/article/pii/S0960148124011236), DOI10.1016/j.renene.2024.121055 | GOES-R、2018全年、7个SURFRAD站、5min；对比10个长波通道CNN与增加SCOPE物理特征的CNN，另做ConvLSTM预测。 | **当前最接近“云物理特征增量”问题的论文。** SCOPE不能未经核查简化为单一COT；摘要44.7与68.9W/m²不直接标为1h预测差。A，完整消融/容量/时效待全文。 |
| Letu et al. (2023), *Surface Solar Radiation Compositions Observed from Himawari-8/9 and Fengyun-4 Series*, BAMS104,E1772–E1789. [原文](https://journals.ametsoc.org/view/journals/bams/104/10/BAMS-D-22-0154.1.xml), DOI10.1175/BAMS-D-22-0154.1 | 云相态、气溶胶、气体等结合RT与DNN生成短波、PAR、UVA、UVB及直散分量。 | 与FY-4数据和物理估计直接相关；同期产品路线，不等于预测。P，输入/方法/产品段。 |
| Tana et al. (2023), *Retrieval of cloud microphysical properties from Himawari-8/AHI infrared channels and its application in surface shortwave downward radiation estimation in the sun glint region*, RSE290,113548. [DOI:10.1016/j.rse.2023.113548](https://doi.org/10.1016/j.rse.2023.113548) | 标题与作者出版记录确认反演云微物理及太阳耀斑区域短波估计应用。 | 候选精读，暂不引用具体增益或断言其COT独立消融。M，出版社403；[NASA出版目录](https://ceres.larc.nasa.gov/science/publications/)。 |
| Wiltink et al. (2025), *Evaluating parallax and shadow correction methods for global horizontal irradiance retrievals from Meteosat SEVIRI*, AMT18,3917–3936. [原文](https://amt.copernicus.org/articles/18/3917/2025/), DOI10.5194/amt-18-3917-2025 | 视差与云影校正对GHI检索的影响。 | 支持空间误差归因；不意味着我方已经完成几何校正，也不能照搬其改善比例。既有原文阅读记录，本轮保留。 |

### 3.3 云属性外推后进入辐照度预报

| 文献及原始来源 | 方法、评价与结果边界 | 对本文的用途；读取层级 |
|---|---|---|
| Sirch et al. (2017), *Cloud and DNI nowcasting with MSG/SEVIRI for the optimized operation of concentrating solar power plants*, AMT10,409–429. [原文](https://amt.copernicus.org/articles/10/409/2017/) | 分层云跟踪、光流及卷云强度变化→2h云光学厚度与DNI。 | 运动与演变都重要；DNI不是GHI，不能作我方地面GHI增量证据。A及方法片段。 |
| Kosmopoulos et al. (2020), *Short-Term Forecasting of Large-Scale Clouds Impact on Downwelling Surface Solar Irradiation*, Energies13,6555. [原文](https://www.mdpi.com/1996-1073/13/24/6555), DOI10.3390/en13246555 | NextSENSE：COT光流→快RT/CMF及辐射，15min、3h，持久性比较。 | COT先外推再辐射的明确前例；检索派生辐射参考与地面实测验证须分开。P，原文项目副本。 |
| Papachristopoulou et al. (2024), *Effects of clouds and aerosols on downwelling surface solar irradiance nowcasting and short-term forecasting*, AMT17,1851–1877. [全文](https://amt.copernicus.org/articles/17/1851/2024/) | SENSE2同期估计与NextSENSE2未来3h；COT光流+辐射映射，2017全年10站实测，讨论像元与点站代表性。 | **最适合借鉴的分层论证。** 每15min瞬时对比不等于15min区间均值；运动相对持久性的skill不等于加COT收益。F。 |
| Yi et al. (2025), *Nowcasting of Surface Solar Irradiance Based on Cloud Optical Thickness from GOES-16*, Remote Sensing17,2861. [原文](https://www.mdpi.com/2072-4292/17/16/2861), DOI10.3390/rs17162861 | 改进PredRNN预测COT→LightGBM估计SSI，1–3h；使用站点及周围COT统计。 | 最直接近期对照；未核实与我方一致的AGRI/AGRI+COT同容量消融。文献用区域均值不构成恢复我方已否决均值方案的理由。A/P，HTML429。 |
| Peng et al. (2026), *A physics-informed machine learning method for short-term solar radiation forecast with satellite-derived cloud optical depth*, Applied Energy406,127284. [原文](https://www.sciencedirect.com/science/article/pii/S0306261925020148), DOI10.1016/j.apenergy.2025.127284 | GOES-16 COD与云掩膜分别预测，结合RT训练的辐射映射得到10min–4h GHI。 | 必须正面对比；有无云掩膜不等于有无COT。物理信息来自RT映射不能未经核查写成物理约束损失。A/P，全文受限。 |
| Cui et al. (2024), *Solar radiation nowcasting based on geostationary satellite images and deep learning models*, Solar Energy282,112866. [作者机构记录](https://www.knmi.nl/kennis-en-datacentrum/publicatie/solar-radiation-nowcasting-based-on-geostationary-satellite-images-and-deep-learning-models), DOI10.1016/j.solener.2024.112866 | UNet、DGMR-SO直接辐射预测与云属性光流+CPP辐射算法比较。 | “直接预报辐射”与“属性→辐射”路线对照；不是单独COT增加值。A。 |
| Kosmopoulos et al. (2024), *Multi-Layer Cloud Motion Vector Forecasting for Solar Energy Applications*, Applied Energy353,122144. [原文入口](https://www.sciencedirect.com/science/article/pii/S0306261923015088), DOI10.1016/j.apenergy.2023.122144 | 多层云运动与快RT的候选相关工作。 | 全文受限，训练/评价日期与地面验证范围待查；暂不据二手摘要报增益。M，作者机构记录及引文链。 |

### 3.4 必须区分的邻近路线

| 文献 | 与COT的区别；作用 |
|---|---|
| Straub et al. (2024), *Satellite-Based Solar Irradiance Forecasting: Replacing Cloud Motion Vectors by Deep Learning*, Solar RRL8,2400475. [原文](https://onlinelibrary.wiley.com/doi/full/10.1002/solr.202400475) | HelioNet预测cloud index，再转辐射；CI不是COT。图像改善与辐射改善分开评价。F。 |
| Carpentieri et al. (2023), *Intraday probabilistic forecasts of surface solar radiation with cloud scale-dependent autoregressive advection*, Applied Energy351,121775. [DOI](https://doi.org/10.1016/j.apenergy.2023.121775) | SolarSTEPS外推clear-sky index，尺度分解、自回归、平流与随机扰动。概率CRPS改善不能改写成确定性GHI RMSE降幅。既有阅读记录。 |
| Straub et al. (2025), *HelioNet-IR: Combining Infrared and Visible Satellite Images for Solar Irradiance Forecasting in the Early-Morning Hours*, Solar RRL9,2500365. [原文](https://onlinelibrary.wiley.com/doi/full/10.1002/solr.202500365) | 红外为清晨预测补信息，属于输入模态增加，不能说COT增益。F，相关章节。 |

## 4. 重要遗漏到底在哪里

已有写作库还保存两项应作为直接AGRI/物理学习背景的工作：Chen et al. (2023), *Global and direct solar irradiance estimation using deep learning and selected spectral satellite images*, Applied Energy，[DOI10.1016/j.apenergy.2023.121979](https://doi.org/10.1016/j.apenergy.2023.121979)；Sasaki et al. (2024), *A Physics-Informed Machine Learning Approach utilizing Multiband Satellite Data for Solar Irradiance Estimation*，[arXiv:2407.04283](https://arxiv.org/abs/2407.04283)，按预印本引用。10-15记录已阅读全文；本轮不重复把它们报新发现，也不把具体物理约束等同于单纯增加COT输入。两项为背景补充，不计入上面21条专题矩阵。

第一层是物理辐射估计传统。Takenaka、Heliosat-4与Al Asmar说明COT影响辐射及其物理映射并不新。我们的动机应从“COT是有物理含义的表示”推进到“当多光谱云图已经在输入中时，是否仍有可测的增量”。COT由AGRI和外部产品监督得到时，不应简单声称它是一项独立观测；它可能提供更有用的表征或外部监督先验。

第二层是最接近的学习式增量对照。SCOPE-CNN已经做了云物理信息加入前后的比较，必须在相关工作中正面讨论。其多种物理特征、通道与评价合同未确认与我们一致，因此既不能称完全相同，也不能声称无人研究过。需要精读的关键不是它的降幅比我们大，而是增加了什么、是否容量匹配、同期和预报分别如何计分。

第三层是COT预报的连续谱系。Sirch、NextSENSE、NextSENSE2、GOES-16、Peng和Cloud-FNO分别覆盖物理平流、演变、深度预测与辐射映射。我们不应继续把此领域理解为“只有AGRI图像外推”和“新尝试直接COT外推”两个选项。当前第二篇不追逐外推创新，仍可把固定方法作为应用验证。

第四层是评价误差传播。COT误差小不保证GHI增益：像元与站点、瞬时与区间均值、厚云辐射敏感性、云影位置、晴空误报都能改变结果。NextSENSE2与Wiltink为这些解释提供文献依据，但具体到我方哪个因素主要起作用，仍须由已保存样本和配对实验判断。

## 5. 按当前论文方向组织正文

建议相关工作三段，而非按网络名罗列：

1. **云物理属性与GHI估计。** Takenaka→Heliosat-4/Letu→Al Asmar→SCOPE-CNN。落点：物理有效性已有研究，但本数据与学习框架中的条件增量仍需控制对照。
2. **物理属性外推与辐射预报。** Sirch/NextSENSE→Xu/Cloud-FNO→Yi/Peng。落点：COT预测质量与下游GHI必须分别量化。
3. **研究问题与本文范围。** 在两个区域独立训练，采用固定小邻域、相同AGRI与太阳几何，先研究同期反演、再评估固定外推下的保留收益；不把外推网络创新作为贡献，不称跨省零样本迁移。

可用的论文主旨草案：

> 本研究考察检索监督的云光学厚度表示在多光谱卫星输入之上对站点辐照度估计的增量价值，并分析该价值随云光学厚度、区域和输入预测误差的变化。同期估计与短临预报分别评价，以区分云物理表示的价值及其在预测链路中的保留程度。

不能预写“预报也显著提高”。只有配对预测结果支持时填入相应结论；若不支持，写清适用条件和链路限制。原始3.02%同期收益与统一残差控制下未证实额外COT收益必须各按合同呈现，不择优拼接。

## 6. 对现有实验的具体影响

| 科学问题 | 延续现有设计 | 文献提供的检查点 |
|---|---|---|
| COT对同期GHI是否有用 | 固定3×3、三帧及当前结构，A/AC同样本配对；保留已有容量及R隐特征对照 | 与SCOPE-CNN区分增加的物理量；不把额外参数或监督一并算COT独有贡献 |
| 在哪些云条件有用 | 按预定CPP厚度分组，报告各组日期/样本数、绝对RMSE和配对差；含全部自然验证样本 | Al Asmar的物理动机；CPP仅检索参考，不称独立云真值；不只选厚云正结果 |
| 能否重复成立 | 湖南两站与湖北三站独立训练、区域内时间留出 | 多站/多气候传统；不是用他文数值替代我方重复验证 |
| 外推后能保留多少 | 固定外推，真实目标AGRI/COT的同期结果与因果预测链分别评价 | NextSENSE2、Peng、Yi；共同站点/目标时间/15min区间标签，不能借真实未来云图作为预报 |
| COT预测本身怎样 | 1/2/3/4h、持久性、相同支持；另看云/厚云识别和位置 | Cloud-FNO/Xu量纲、变换、掩膜和区域不同，跨文仅参考，不作排行榜 |
| 预报收益为何弱 | 复用已有云图、COT轨迹与GHI曲线；按证据讨论来源误差 | Wiltink与NextSENSE2启发；不要为综述立即新增大模型、全图均值或无来源云高实验 |

## 7. 阅读优先级与剩余缺口

- P0：SCOPE-CNN全文——物理特征列表、COD波段定义、基础CNN输入、参数量、日期划分、44.7/68.9对应时间任务。
- P0：Peng全文——COD/掩膜融合、训练划分、RT映射、地面站支持、有无掩膜与有无COT两类对照的区别。
- P1：Yi全文——LightGBM输入、站点邻域统计、夜间COT构建、预测与标签时间口径，是否存在真正去COT对照。
- P1：NextSENSE2已可读——作为同期/预报分开评价的核心方法参考；不照搬其瞬时标签。
- P1：Xu与Cloud-FNO——外推部分比较合同与可复现性；无需据文献立即重训。
- P2：Tana、3D-CMV、最新Drosoglou预印本——补全文后再提取定量结果。不得把暂时不可读当作无相关研究。

本轮没有访问test载荷、提交GPU任务或修改论文正文。后续正文更新必须英文LaTeX与中文对照同步。
