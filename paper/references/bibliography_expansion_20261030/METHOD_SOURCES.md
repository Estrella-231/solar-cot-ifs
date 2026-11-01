# 必要方法来源核验与引用位置（2026-10-30）

> 用途：补齐确实用到的方法出处，不按篇数凑引用。未修改论文正文。原文可访问范围逐条注明；出处可引用不等于文献验证了本项目的实现或性能。

| 方法 | 正式出处 | DOI | 建议引用位置 |
|---|---|---|---|
| ronneberger2015unet | Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015 9351: 234–241 (2015) | [10.1007/978-3-319-24574-4_28](https://doi.org/10.1007/978-3-319-24574-4_28) | Methods / COT retrieval: first occurrence of U-Net in the retriever architecture paragraph. |
| huber1964robust | The Annals of Mathematical Statistics 35(1): 73–101 (1964) | [10.1214/aoms/1177703732](https://doi.org/10.1214/aoms/1177703732) | Methods / COT retrieval: masked Huber objective, at the name or immediately after its formula. |
| ineichen2002linke | Solar Energy 73(3): 151–157 (2002) | [10.1016/S0038-092X(02)00045-2](https://doi.org/10.1016/S0038-092X(02)00045-2) | Data and preprocessing: the sentence introducing the Ineichen–Perez clear-sky model, before the interval clear-sky-index formula. |
| holmgren2018pvlib | Journal of Open Source Software 3(29): 884 (2018) | [10.21105/joss.00884](https://doi.org/10.21105/joss.00884) | Data and preprocessing: adjacent to the actual pvlib implementation statement; cite alongside, rather than instead of, the Ineichen–Perez algorithm. |
| roberts2017cross | Ecography 40(8): 913–929 (2017) | [10.1111/ecog.02881](https://doi.org/10.1111/ecog.02881) | Data partitions / evaluation design: explain why split dates rather than randomly splitting overlapping forecast windows. |
| field2007cluster | Journal of the Royal Statistical Society: Series B (Statistical Methodology) 69(3): 369–390 (2007) | [10.1111/j.1467-9868.2007.00593.x](https://doi.org/10.1111/j.1467-9868.2007.00593.x) | Evaluation / uncertainty: date-cluster paired resampling paragraph, only for the general clustered-data bootstrap rationale. |

## 1. U-Net: Convolutional Networks for Biomedical Image Segmentation

- 作者：Olaf Ronneberger; Philipp Fischer; Thomas Brox。
- 文献类型：conference paper。
- 原始来源：[来源1](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)；[来源2](https://arxiv.org/abs/1505.04597)；[来源3](https://arxiv.org/pdf/1505.04597)。
- 本次访问范围：Publisher bibliographic record and abstract; author arXiv preprint full text (8 pages) accessible.
- 证据位置：Author preprint §2 Network Architecture, PDF pp.3–4; contracting/expanding paths and concatenation described on PDF p.4.
- 可支持的正文表述（原创草拟）：The COT retriever uses a U-Net encoder–decoder with skip connections.
- 中文对应：COT反演器采用带跳跃连接的U-Net编码器—解码器。
- 引用范围：Supports the architectural family, not our two-stage depth, widths 32/64/128, SiLU, batch normalization, softplus, 16×16 grid, or performance. Describe those as implementation choices; do not imply the original network used them.

## 2. Robust Estimation of a Location Parameter

- 作者：Peter J. Huber。
- 文献类型：journal article。
- 原始来源：[来源1](https://projecteuclid.org/euclid.aoms/1177703732)；[来源2](https://api.crossref.org/works/10.1214/aoms/1177703732)。
- 本次访问范围：Publisher DOI resolves to Project Euclid, but the service returned an access challenge. Publisher-deposited Crossref metadata was fetched successfully using requests on 2026-10-30; this is metadata verification, not a claim of full-text access.
- 证据位置：Original article bibliographic record verified; no original full-text page/equation was verified in this pass.
- 可支持的正文表述（原创草拟）：The retriever is optimized using a masked Huber loss in log-COT space.
- 中文对应：反演器在log-COT空间中采用带有效像元掩膜的Huber损失训练。
- 引用范围：Huber is the standard source for the robust loss family; the log transformation, mask, delta value, normalization, and application to COT are ours and require explicit definition. Do not claim Huber1964 proves this model is robust to CPP retrieval error. Full-text formula checking remains unverified in this pass.

## 3. A new airmass independent formulation for the Linke turbidity coefficient

- 作者：Pierre Ineichen; Richard Perez。
- 文献类型：journal article。
- 原始来源：[来源1](https://www.sciencedirect.com/science/article/abs/pii/S0038092X02000452)；[来源2](https://www.unige.ch/sysener/equipe/les-anciens-membres/ineichen)；[来源3](https://research.asrc.albany.edu/people/faculty/perez/directory/RA.html)。
- 本次访问范围：Publisher search-indexed metadata/abstract and author institutional publication records. Publisher direct retrieval later returned 403; no full-text equation audit claimed.
- 证据位置：Publisher abstract and bibliographic record: airmass-independent Linke turbidity and clear-sky global/direct-normal irradiance formulations.
- 可支持的正文表述（原创草拟）：Clear-sky irradiance is computed using the Ineichen–Perez model.
- 中文对应：晴空辐照度采用Ineichen–Perez模型计算。
- 引用范围：Supports the clear-sky model; 15-minute interval averaging, station altitude, turbidity source, time zone, geometry alignment, and the definition mean(GHI)/mean(clear-sky GHI) are study implementation details, not established by this citation.

## 4. pvlib python: a python package for modeling solar energy systems

- 作者：William F. Holmgren; Clifford W. Hansen; Mark A. Mikofski。
- 文献类型：software paper。
- 原始来源：[来源1](https://joss.theoj.org/papers/10.21105/joss.00884)；[来源2](https://joss.theoj.org/papers/10.21105/joss.00884.pdf)。
- 本次访问范围：Publisher article record and full 3-page open-access paper accessible.
- 证据位置：PDF p.1 Summary: reference implementations for solar position, clear-sky irradiance, and solar-energy modeling.
- 可支持的正文表述（原创草拟）：The clear-sky calculation is implemented in pvlib python.
- 中文对应：晴空辐照度计算通过pvlib python实现。
- 引用范围：Only state use of pvlib after code/config confirms it. The paper does not specify this project's installed version, turbidity inputs, solar-position algorithm, or interval averaging. Version/settings belong in reproducibility details.

## 5. Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure

- 作者：David R. Roberts; Volker Bahn; Simone Ciuti; Mark S. Boyce; Jane Elith; Gurutzeta Guillera-Arroita; Severin Hauenstein; José J. Lahoz-Monfort; Boris Schröder; Wilfried Thuiller; David I. Warton; Brendan A. Wintle; Florian Hartig; Carsten F. Dormann。
- 文献类型：review and methodological case studies。
- 原始来源：[来源1](https://nsojournals.onlinelibrary.wiley.com/doi/full/10.1111/ecog.02881)；[来源2](https://florianhartig.github.io/publications/roberts2017cross/)。
- 本次访问范围：Publisher full-text HTML and coauthor bibliographic page accessible. Online first 2016-12-08; final volume/issue year is 2017.
- 证据位置：Sections The problem of structured data and Cross-validation with structured data; publisher HTML discussion of strategic temporal/spatial blocks and buffered hold-out samples.
- 可支持的正文表述（原创草拟）：Date-separated partitions are used to reduce dependence between training and evaluation samples.
- 中文对应：采用按日期分离的数据划分，以降低训练与评价样本之间的依赖。
- 引用范围：Supports dependence-aware validation design, not the exact monthly 2-day holdout, 1-day buffer, complete independence of our data, or any bootstrap method. The block choice must match the prediction goal; it can change interpolation/extrapolation difficulty.

## 6. Bootstrapping clustered data

- 作者：C. A. Field; A. H. Welsh。
- 文献类型：original statistical methods article。
- 原始来源：[来源1](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2007.00593.x)；[来源2](https://academic.oup.com/jrsssb/article-abstract/69/3/369/7109361)；[来源3](https://researchportalplus.anu.edu.au/en/publications/bootstrapping-clustered-data/)；[来源4](https://www.mathstat.dal.ca/~field/wiki/doku.php?id=publications)。
- 本次访问范围：Publisher metadata and abstract plus author institutional record verified. Wiley full-text link redirects to abstract; Oxford PDF/full-text retrieval unavailable in this pass. Do not report a full-paper read or cite an unverified equation/page.
- 证据位置：Publisher abstract: theory of bootstrap methods for clustered data and model-dependent validity of variance estimates.
- 可支持的正文表述（原创草拟）：Uncertainty is summarized using a paired bootstrap that resamples complete date clusters, retaining all matched records within each selected date.
- 中文对应：使用按完整日期簇重采样的配对bootstrap汇总不确定性；每个被抽中的日期保留全部匹配记录。
- 引用范围：This is a suitable original methodological reference for clustered bootstrap, not proof that days are independent or that this particular RMSE-difference percentile interval has exact coverage. The paired implementation and date grouping are ours. Do not call it moving-block bootstrap unless overlapping contiguous blocks are actually sampled; no moving-block-bootstrap citation is proposed.

## 日期簇bootstrap的命名和可写范围

建议英文使用 **paired date-cluster bootstrap**，中文使用“按日期簇重采样的配对bootstrap”。每次重采样对两种方法抽取相同日期及相同匹配记录，再计算指标差；保留每个日期的全部相关记录。正文必须说清分簇用目标日期还是初始化日期，应与实际代码一致。

Field & Welsh (2007)可作为簇数据bootstrap的原始方法文献，但本次只核到了出版商摘要及元数据，未核验全文具体页码/公式。它不证明本项目按日分簇足以消除跨日相关，也不自动保证本项目RMSE差区间的覆盖率。不能将这一抽样称为移动块bootstrap（moving-block bootstrap）；后者须实际构造并抽取时间连续块。Roberts2017应引用于数据划分理由，不应代替bootstrap来源。

本项目需自行明确：日期簇定义、簇数、重采样次数、区间类型、配对保持方式和随机种子。这些是本方法的复现细节，不应归给上述文献。

## 元数据易错点

- Ineichen–Perez2002精确DOI是 `10.1016/S0038-092X(02)00045-2`。
- U-Net是MICCAI会议论文（LNCS9351），不是期刊论文；原始ReLU及网络深度不能直接套成我们的SiLU两级网络。
- pvlib2018的884是文章编号；以正式软件论文替代只有网页文档的引用，文档可留在工程/补充复现信息。
- Roberts文章在线首发2016，正式卷期为2017，参考文献使用2017。
- Huber的原始期刊是The Annals of Mathematical Statistics，不是The Annals of Statistics；卷35，不是53。原站本次返回服务校验，元数据用出版社提交的Crossref记录复核，未冒称全文核验。

