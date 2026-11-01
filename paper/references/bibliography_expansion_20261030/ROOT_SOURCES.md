# 补充来源与正文用途

2026-10-30。元数据保存于 ROOT_METADATA.json；完整整合书目见 MANUSCRIPT_REFERENCES.json。本轮未通读以下每篇全文，仅按注明的阅读范围使用。以下为原创概述，不是引文。

| 来源 | 原始出处与本轮阅读范围 | 正文支持内容 |
|---|---|---|
| Huang et al., 2019, RSE233,111371 | [出版条目](https://doi.org/10.1016/j.rse.2019.111371)；[作者公开PDF](https://www2.atmos.umd.edu/~zli/PDF_papers/Huang_et_al-RSE-2019.pdf)，摘要及方法分类/代表性相关段落 | 卫星地表辐照度估计的发展与方法背景；不作为本文增益证据。 |
| Yang et al., 2020, Solar Energy210,20–37 | [作者公开全文](https://personal.utdallas.edu/~jxz156730/Journals/Zhang_2020_SE_deterministic_verification.pdf)，摘要、引言与评价框架说明 | 整体误差之外还应分析预报与观测分布；不声称本文相对RMSE降幅等同该文推荐的气候/持续性组合skill score。 |
| Chen et al., 2024, ECM313,118598 | [作者公开全文](https://www.li-realab.info/publication/solar-forecast/chen-2024-dlchain/chen-2024-DLchain.pdf)，具体章节阅读沿用../wording_review_20261030/CHEN_NOTES.md | 模型链连接卫星信息与日内辐照度预报，云演变困难；不将其数值或机制当本项目结果。 |
| Song et al., 2025, GRL52,e2025GL116237 | [出版社全文页](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025GL116237)，摘要、Plain Language Summary、§2.1 | STARNet使用FY-4A时空邻域信息估计地表短波辐射。不能写成FY-4B或显式COT消融。 |
| Schuurman & Meyer, 2025, IJRS46,3331–3362 | [正式出版社PDF题录](https://www.tandfonline.com/doi/pdf/10.1080/01431161.2025.2471596)；[作者arXiv记录](https://arxiv.org/abs/2409.16316)，摘要及出版关联 | 卫星产品模拟结合地面观测微调。正式发表题名为“generalizes to other climate zones”，与预印本“generalizes well”不同，书目使用正式题名。 |
| Xia et al., 2025, Remote Sensing17,737 | [出版社页面](https://www.mdpi.com/2072-4292/17/5/737)，出版社检索返回的§3与方法范围；直接打开曾返回429 | 夜间FY-4B COT学习型反演的先例，明确单层非降水云；不将其作为白天GHI改善证据。 |
| Miller et al., 2018, Solar Energy168,102–117 | [出版社](https://www.sciencedirect.com/science/article/pii/S0038092X17310435)及[NOAA作者机构记录](https://repository.library.noaa.gov/view/noaa/49543)，摘要与引言 | 云平流、光学属性、太阳/观测几何及云影与辐照度预报的联系。没有声称本文已做视差校正。 |
| Nielsen et al., 2021, Solar Energy228,659–669 | [作者机构PDF](https://pure.au.dk/portal/files/273838938/1_s2.0_S0038092X21008306_main.pdf)，首页摘要和引言 | IrradianceNet预测的是卫星反演辐照度场，不是直接预测多光谱AGRI或COT。 |
| Nakajima & King, 1990 | [NASA作者机构摘要](https://ntrs.nasa.gov/citations/19900061651)，并与CLOUD_SOURCES交叉核验 | 层状水云COT与粒子尺度反演、薄云多解性；合并为同一条，未重复计数。 |

原稿Chen2023、Liao2024、Papachristopoulou2024、Cloud-FNO2026的元数据本轮也经Crossref重新核对。SimVP保留已验证的CVF出处。终审发现SimVPv2已正式发表于IEEE Transactions on Multimedia 27:5170–5184 (2025)，书目更新为[正式DOI](https://doi.org/10.1109/TMM.2025.3543051)；出版社登记元数据保存在SIMVPV2_PUBLISHED_METADATA.json，并以[作者主页](https://lupin1998.github.io/)交叉核验；[arXiv](https://arxiv.org/abs/2211.12509)仅作为可访问稿入口。方法来源见METHOD_SOURCES.md；其他物理与近期应用来源见CLOUD_SOURCES.md。

## 使用限制

全文不可达时只使用已核验的摘要或章节范围；Crossref核对的是书目信息，不能代替科学内容阅读。未将搜索结果的引用次数当作文末参考文献数量。未把只读到摘要的文章登记为已通读全文。未复制完整段落或第三方参考文献表。
