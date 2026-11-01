# COT–GHI 云物理与卫星辐照度文献补充

核验日期：2026-10-30。已对照当前 `main.tex` 的8条参考文献，以下10篇均不重复；其中4篇发表于2024–2026年，均为正式期刊论文。本文件不修改正文，也不填入未经当前实验验证的结果。

阅读深度逐条列明。出版社检索返回的正文片段只按所读章节使用；摘要、可访问全文与已通读全文不混写。项目旧笔记仅用于发现线索，以下科学判断均重新核对原始来源。

优先整合：引言用Cano/Rigollier区分cloud index路线，用Nakajima说明COT反演基础，用Qu/Takenaka说明云光学属性映射；讨论优先用Wiltink说明位置/云影及空间匹配，用Chen说明观测质量；Cui/Sirch说明云场外推和辐射映射；Shen提供近期FY-4B相关工作。

全部10篇已用Crossref逐项核对题目、作者、年份、期刊、卷期、页码/文章号和DOI；科学判断仍以出版社或作者机构原始来源为依据。

## 1. cano1986

**A method for the determination of the global solar radiation from meteorological satellite data**

- 作者：D. Cano; J. M. Monget; M. Albuisson; H. Guillard; N. Regas; L. Wald。
- 书目：Solar Energy, 1986, 37(1), 31–39。
- DOI：[10.1016/0038-092X(86)90104-0](https://doi.org/10.1016/0038-092X(86)90104-0)。
- 主来源：[出版社或作者机构页面](https://www.sciencedirect.com/science/article/pii/0038092X86901040)。
- 出版身份：已发表同行评议期刊论文；不是预印本。
- 已读范围：出版社题录和摘要；另见SoDa保存的原论文PDF首页，未通读全文。
- 对应正文位置：引言：cloud index方法的源头。
- 可支撑句（原创，14英文词）：Early satellite irradiance methods inferred cloud influence from reflectance relative to a surface-albedo reference.
- 中文含义：早期卫星辐照度方法通过观测反射率相对地表反照率参考的变化估计云的影响。
- 适用边界：cloud coverage index不是COT；不能作为显式COT增量有效的证据。
- 来源跟踪：本次核对原始Heliosat来源。

## 2. rigollier2004

**The method Heliosat-2 for deriving shortwave solar radiation from satellite images**

- 作者：C. Rigollier; M. Lefèvre; L. Wald。
- 书目：Solar Energy, 2004, 77(2), 159–169。
- DOI：[10.1016/j.solener.2004.04.017](https://doi.org/10.1016/j.solener.2004.04.017)。
- 主来源：[出版社或作者机构页面](https://www.sciencedirect.com/science/article/pii/S0038092X04001082)。
- 出版身份：已发表同行评议期刊论文；不是预印本。
- 已读范围：出版社题录和摘要；作者机构SoDa方法页面交叉核验书目信息。
- 对应正文位置：引言：从卫星反射信息到地表辐照度的既有方法。
- 可支撑句（原创，13英文词）：Heliosat-2 combines calibrated satellite radiances with atmospheric-optics models to estimate surface solar irradiation.
- 中文含义：Heliosat-2将定标卫星辐亮度与大气光学模型结合，用于估计地表太阳辐射。
- 适用边界：这是同期辐射反演方法；不能将cloud index或irradiation直接改称COT或未来GHI。
- 来源跟踪：本次补充；与当前8篇不重复。

## 3. qu2017

**Fast radiative transfer parameterisation for assessing the surface solar irradiance: The Heliosat-4 method**

- 作者：Zhipeng Qu; Armel Oumbe; Philippe Blanc; Bella Espinar; Gerhard Gesell; Benoît Gschwind; Lars Klüser; Mireille Lefèvre; Laurent Saboret; Marion Schroedter-Homscheidt; Lucien Wald。
- 书目：Meteorologische Zeitschrift, 2017, 26(1), 33–57。
- DOI：[10.1127/metz/2016/0781](https://doi.org/10.1127/metz/2016/0781)。
- 主来源：[出版社或作者机构页面](https://elib.dlr.de/106996/)。 [原论文PDF](https://elib.dlr.de/106996/1/Qu_Fast_radiative_transfer_parameterisation_for_assessing_the_surface_solar_irradiance_The_Heliosat_4_method_87036.pdf)。
- 出版身份：已发表同行评议期刊论文；不是预印本。
- 已读范围：DLR作者机构保存的原论文PDF：摘要及§1；机构题录核验2017卷期。PDF为2016预出版排版，不是未审稿预印本。
- 对应正文位置：引言/方法依据：云光学属性与辐射传输。
- 可支撑句（原创，15英文词）：Heliosat-4 separates clear-sky irradiance from cloud extinction using radiative-transfer lookup tables and satellite-derived cloud properties.
- 中文含义：Heliosat-4以辐射传输查找表和卫星云属性分别计算晴空辐照度及云消光。
- 适用边界：物理映射先例，不证明在原始AGRI已给定时再加入COT必然改善学习模型。
- 来源跟踪：历史检索线索，本次重新核验原论文。

## 4. takenaka2011

**Estimation of solar radiation using a neural network based on radiative transfer**

- 作者：H. Takenaka; T. Y. Nakajima; A. Higurashi; A. Higuchi; T. Takamura; R. T. Pinker; T. Nakajima。
- 书目：Journal of Geophysical Research: Atmospheres, 2011, 116(D8), D08215。
- DOI：[10.1029/2009JD013337](https://doi.org/10.1029/2009JD013337)。
- 主来源：[出版社或作者机构页面](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2009JD013337)。
- 出版身份：已发表同行评议期刊论文；不是预印本。
- 已读范围：出版社检索返回的正文§2.1、§4、§5.1及图3–5、图17周边文字；非仅摘要。
- 对应正文位置：引言云物理依据；讨论COT敏感性与碎云代表性。
- 可支撑句（原创，19英文词）：Radiative-transfer emulation links retrieved cloud optical properties to surface fluxes, while unresolved broken clouds limit agreement with point measurements.
- 中文含义：辐射传输近似可将反演云光学属性映射为地表通量，但未解析的碎云会限制其与站点观测的一致性。
- 适用边界：该文特定物理算法的敏感性不能推广为本稿COT分支的因果贡献；原文COT最敏感的表述须限定算法。
- 来源跟踪：历史检索线索，本次核验正文。

## 5. sirch2017

**Cloud and DNI nowcasting with MSG/SEVIRI for the optimized operation of concentrating solar power plants**

- 作者：Tobias Sirch; Luca Bugliaro; Tobias Zinner; Matthias Möhrlein; Margarita Vazquez-Navarro。
- 书目：Atmospheric Measurement Techniques, 2017, 10(2), 409–429。
- DOI：[10.5194/amt-10-409-2017](https://doi.org/10.5194/amt-10-409-2017)。
- 主来源：[出版社或作者机构页面](https://amt.copernicus.org/articles/10/409/2017/)。
- 出版身份：已发表同行评议期刊论文；不是预印本。
- 已读范围：出版社摘要、How to cite及短摘要；本次未读取PDF正文。
- 对应正文位置：引言预报链；讨论平流与厚度变化。
- 可支撑句（原创，19英文词）：Cloud-motion forecasts can produce irradiance forecasts by advecting optical thickness, with separate treatment of cloud layers and convective thinning.
- 中文含义：云运动方法可外推光学厚度再计算辐照度，并分别处理云层运动和对流云变薄。
- 适用边界：本文预测目标是DNI，主要验证参照为MSG/SEVIRI；不能写成地面站GHI增益证据。
- 来源跟踪：历史检索线索，本次核验出版社摘要。

## 6. cui2024

**Solar radiation nowcasting based on geostationary satellite images and deep learning models**

- 作者：Yang Cui; Ping Wang; Jan Fokke Meirink; Nikolaos Ntantis; Jasper S. Wijnands。
- 书目：Solar Energy, 2024, 282, 112866。
- DOI：[10.1016/j.solener.2024.112866](https://doi.org/10.1016/j.solener.2024.112866)。
- 主来源：[出版社或作者机构页面](https://www.sciencedirect.com/science/article/pii/S0038092X24005619)。
- 出版身份：已发表同行评议期刊论文；不是预印本。
- 已读范围：出版社题录、Highlights和完整摘要；未完成全文合同审计。
- 对应正文位置：引言相关工作；讨论云场模糊和预报评价。
- 可支撑句（原创，21英文词）：Irradiance nowcasting comparisons show that model ranking depends on sky conditions and on whether numerical errors or spatial appearance are evaluated.
- 中文含义：辐照度短临比较表明，模型排序会随天空条件及数值误差或空间形态评价而变化。
- 适用边界：研究包含云属性平流、空间模糊和深度模型，不是统一头部有无COT的严格消融；荷兰结果不直接移植湖南湖北。
- 来源跟踪：历史检索线索，本次重新核验出版社摘要。

## 7. wiltink2025

**Evaluating parallax and shadow correction methods for global horizontal irradiance retrievals from Meteosat SEVIRI**

- 作者：Job I. Wiltink; Hartwig Deneke; Chiel C. van Heerwaarden; Jan Fokke Meirink。
- 书目：Atmospheric Measurement Techniques, 2025, 18(16), 3917–3936。
- DOI：[10.5194/amt-18-3917-2025](https://doi.org/10.5194/amt-18-3917-2025)。
- 主来源：[出版社或作者机构页面](https://amt.copernicus.org/articles/18/3917/2025/)。
- 出版身份：已发表同行评议期刊论文；不是预印本。
- 已读范围：出版社全文页：摘要、§1及方法问题定义；尚未逐表审查所有校正结果。
- 对应正文位置：讨论：云位置、云影与点站匹配；机制诊断动机。
- 可支撑句（原创，18英文词）：Parallax and cloud-shadow displacement affect satellite-to-station GHI comparisons, and correction effects depend on cloud regime and spatial resolution.
- 中文含义：视差和云影位移会影响卫星与站点GHI比较，其校正效果依赖云况及空间分辨率。
- 适用边界：几何错位与外推错位是不同来源；可支撑位置误差的重要性，不能证明本稿误差全部来自位移。
- 来源跟踪：本次针对空间代表性补核；已有历史文献表中可能曾列题名。

## 8. chen2025fy4b

**Validating physical and semi-empirical satellite-based irradiance retrievals using high- and low-accuracy radiometric observations in a monsoon-influenced continental climate**

- 作者：Y. Chen; D. Yang; C. Huang; H. Shi; A. R. Jensen; X. Xia; Y.-M. Saint-Drenan; C. A. Gueymard; M. J. Mayer; Y. Shen。
- 书目：Atmospheric Measurement Techniques, 2025, 18(23), 7315–7336。
- DOI：[10.5194/amt-18-7315-2025](https://doi.org/10.5194/amt-18-7315-2025)。
- 主来源：[出版社或作者机构页面](https://amt.copernicus.org/articles/18/7315/2025/)。
- 出版身份：已发表同行评议期刊论文；不是预印本。
- 已读范围：出版社全文页：摘要、§1、§2.1–2.1.4；出版社How to cite核验完整作者列表及刊期年份。
- 对应正文位置：数据与评价：站点观测质量；讨论微小收益解释。
- 可支撑句（原创，19英文词）：FY-4B irradiance validation can be materially affected by ground-radiometer quality, so small apparent model improvements require careful observation-quality assessment.
- 中文含义：地面辐射仪质量会实质影响FY-4B辐照度验证，因此解释较小模型改善时需检查观测质量。
- 适用边界：齐齐哈尔单个共址观测设计；不能直接据此判定本项目站点存在同等偏差或删除负收益样本。
- 来源跟踪：本次新增近期FY-4B原始验证研究。

## 9. shen2026fy4b

**A weather typing adaptive bias correction method for FY-4B satellite-driven solar radiation nowcasting**

- 作者：Long Shen; Yuan Yao; Yang Cui; Yang Xu; Bo Li; Linlin Li; Le Guo。
- 书目：Solar Energy, 2026, 312, 114600。
- DOI：[10.1016/j.solener.2026.114600](https://doi.org/10.1016/j.solener.2026.114600)。
- 主来源：[出版社或作者机构页面](https://www.sciencedirect.com/science/article/abs/pii/S0038092X26002884)。
- 出版身份：已发表同行评议期刊论文；不是预印本。
- 已读范围：出版社检索返回完整摘要、引言和部分结果/结论片段；直接打开出现403，不能标为全文已获取。
- 对应正文位置：引言近期FY-4B应用；讨论天气条件依赖。
- 可支撑句（原创，16英文词）：Recent FY-4B GHI nowcasting combines regional irradiance prediction with weather-dependent correction to address satellite-to-station forecast differences.
- 中文含义：近期FY-4B GHI短临研究将区域辐照度预报与天气条件相关订正结合，处理卫星到站点的预报差异。
- 适用边界：输入为卫星反演GHI场及后处理，不是COT增强AGRI的同头对照；未核验完整划分和滚动订正因果合同。
- 来源跟踪：本次新增；2026年7月已正式发表。

## 10. nakajima1990

**Determination of the Optical Thickness and Effective Particle Radius of Clouds from Reflected Solar Radiation Measurements. Part I: Theory**

- 作者：Teruyuki Nakajima; Michael D. King。
- 书目：Journal of the Atmospheric Sciences, 1990, 47(15), 1878–1893。
- DOI：`10.1175/1520-0469(1990)047<1878:DOTOTA>2.0.CO;2`。
- 主来源：[AMS出版社](https://journals.ametsoc.org/doi/abs/10.1175/1520-0469%281990%29047%3C1878%3ADOTOTA%3E2.0.CO%3B2)；[作者机构NASA NTRS](https://ntrs.nasa.gov/citations/19900061651)。
- 出版身份：已发表同行评议期刊论文；不是预印本。
- 已读范围：出版社检索返回完整摘要及题录；NASA NTRS作者机构摘要和东京大学作者库题录交叉核验；未通读全文。
- 对应正文位置：引言及方法物理背景：多光谱反射率约束COT和有效粒子半径。
- 可支撑句（原创，19英文词）：Multispectral reflectance constrains cloud optical thickness and effective particle radius, but retrieval ambiguity increases for optically thin water clouds.
- 中文含义：多光谱反射率可以约束云光学厚度和有效粒子半径，但薄水云的反演存在多解性。
- 适用边界：理论针对层状水云及给定辐射传输假设，不能证明学习型COT反演独立准确，也不能证明把同源COT加入AGRI后必然改善GHI。
- 来源跟踪：本次补充经典COT反演基础；替换与当前主线较远的FY-4B数值同化来源。

## 检索与去重记录

- 当前8篇：chen2023、liao2024、pap2024、zhang2026、ineichen、pvlibclear、simvp、simvpv2；未将它们计入新增数量。
- 关键词覆盖：Heliosat/cloud index；cloud optical properties/radiative transfer/irradiance；parallax/cloud shadow/spatial representativeness；cloud motion/optical thickness nowcasting；FY-4A/FY-4B GHI/irradiance validation/forecasting。
- 默认本地 `papers/`、`literature/` 未发现本次可用PDF；现有references和历史专题笔记用于去重与线索定位。
- arXiv定向网页回退搜索未贡献最终入选项；10篇都已正式发表，因此没有把预印本混作期刊论文。
- 所有可支撑句限制为30个英文词以内；未复制长引文，未引入本文未测得的数值。
- 声明边界：COT、cloud index和clear-sky index分开；DNI和GHI分开；同期估计和未来预报分开；其他论文的控制结果不能替代本文配对实验。

机器可读元数据见 `CLOUD_SOURCES.json`。

