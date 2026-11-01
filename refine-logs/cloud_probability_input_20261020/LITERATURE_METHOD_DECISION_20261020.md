# 云掩膜怎样加入 GHI：文献依据与方法选择

日期：2026-10-20。状态：文献与代码核查完成，形成待验证方法；未启动训练，未修改论文正文。

## 1. 要回答的问题

在保留同位置、同时刻原始 COT 的条件下，增加逐像元判云信息，能否减少晴空伪云引起的 GHI 错误修正，同时保留薄云、厚云及云边缘的辐射信息？

这与“把判为晴空的 COT 全部清零”不同。后者已有实际负结果，见 [历史实验核查](HISTORY_AUDIT_20261020.md)。本次不把它重新包装为新实验。

## 2. 检索范围与阅读层级

先查本项目 `docs/COT_LITERATURE_REVIEW_20261016.md` 和已有实验。当前项目 `papers/`、`literature/` 未找到可贡献本专题的本地 PDF，也未找到配置的专用论文库；不代表用户微信或文献管理器中没有这些论文。随后检索出版方、作者机构及作者公开稿。没有使用未配置的 Zotero/Obsidian；没有把二手摘要当全文。

检索词包括 cloud mask/probability、surface solar irradiance、cloud optical depth、Heliosat-4、CLARA-A3、CPP/SICCS、satellite mask feature encoding。按 DOI 去重。以下是方法选择所需的定向综述，不声称系统检索穷尽全部工作。

| 文献、来源与阅读范围 | 实际接口 | 对本项目的启示及边界 |
|---|---|---|
| Karlsson et al., 2023, CLARA-A3，ESSD 15,4901–4926；[原文](https://essd.copernicus.org/articles/15/4901/2023/)，DOI 10.5194/essd-15-4901-2023。读 §3.1.1、§3.2.1。 | 概率云掩膜先划分晴/云。晴空用 MAGIC，有云由大气顶反射通量和 LUT 估计透过率，再算辐照度。 | 判云参与辐射计算方式选择；没有把判云概率当辐射衰减系数。其地表相关阈值不能直接迁移。 |
| Qu et al., 2017, Heliosat-4；[作者机构全文](https://elib.dlr.de/106996/1/Qu_Fast_radiative_transfer_parameterisation_for_assessing_the_surface_solar_irradiance_The_Heliosat_4_method_87036.pdf)，DOI 10.1127/metz/2016/0781。读 §2、§4.2–4.4、§8.6。 | 晴空辐射与云影响分开。APOLLO 对部分覆盖像元存在“云覆盖面积比 × 同类满云像元 COT”的有效厚度近似。 | 面积比例不等于分类概率。总 GHI 的云影响由辐射传输处理；不能把直接光的指数衰减公式直接套到总 GHI。 |
| CAMS Radiation Service User Guide 2024；[ECMWF 官方指南](https://confluence.ecmwf.int/download/attachments/266592908/CAMS2_73_2021SC2_D3.2.1_2024_UserGuide_v1.pdf?api=v2&modificationDate=1740595656584&version=1)。读 §4.1、§4.4–4.6。 | McClear/McCloud 分工；APOLLO_NG 提供判云、COT、云型。§4.6 的低概率阈值用于切换有云辐射计算。 | 硬分类有合理用途，不能笼统说硬门控错误。但它不是修改一个已训练 GHI 网络的 COT 输入，更不提供本项目阈值。 |
| Klüser et al., 2015, APOLLO_NG；[原文](https://amt.copernicus.org/articles/8/4155/2015/)，DOI 10.5194/amt-8-4155-2015。读 §2.1–2.3、§4.1。 | 多种光谱/空间检验形成判云概率；云存在性与 COT/粒径反演及其不确定性分别处理。 | 支持区分判云与厚度；本项目 sigmoid 输出是否校准仍需单独评价，不能直接称为 COT 可信度。 |
| Jain, Yadav & Dev, 2024, *Holistic and Lightweight Approach for Solar Irradiance Forecasting*, IEEE TGRS 62；DOI 10.1109/TGRS.2024.3415413。[作者机构记录](https://researchportal.hkust.edu.hk/en/publications/holistic-and-lightweight-approach-for-solar-irradiance-forecastin/)、[作者公开稿全文](https://www.researchgate.net/publication/381496430_Holistic_and_Lightweight_Approach_for_Solar_Irradiance_Forecasting)。读 §II-D、§III-E、§III-G。 | 100×100 二值卫星云掩膜经卷积自编码器压成 64 维特征，再与其他特征拼接预测 GHI。 | 是“云掩膜作为独立特征”的直接先例；不是本项目逐像元连续概率＋COT 的同一接口。其历史实测 GHI 输入不适用我们的设置，不能照搬。 |
| Deneke et al., 2021, CPP/SICCS；[原文](https://amt.copernicus.org/articles/14/5107/2021/)，DOI 10.5194/amt-14-5107-2021。读 §3.3–3.5。 | 云掩膜、云属性反演、辐射 LUT 分步处理；晴空、水云和冰云有不同计算。提高空间分辨率会改变破碎云的误报/漏报。 | 云掩膜与 COT 需在一致网格上配对；二值类别重采样不能用普通双线性插值伪造概率。 |
| Wiltink et al., 2025；[原文](https://amt.copernicus.org/articles/18/3917/2025/)，DOI 10.5194/amt-18-3917-2025。读方法部分云属性、视差与云影处理。 | CPP-SICCS 云场经几何位置处理后与地面辐射比较。 | 卫星看到的云位置不是地面阴影位置。仅加入云掩膜不等于完成云影/视差校正。 |
| Peng et al., 2026, *A physics-informed machine learning method for short-term solar radiation forecast with satellite-derived cloud optical depth*, Applied Energy 406,127284；[出版社](https://www.sciencedirect.com/science/article/abs/pii/S0306261925020148)，DOI 10.1016/j.apenergy.2025.127284。原摘要/highlights，全文未取得。 | 摘要说明分别预测 COD 与云掩膜，组合后进入全天空辐射估计。 | 与“存在性＋厚度”的分解最接近，但未读到完整融合公式，不能据摘要写成 `P×COT` 或照搬其阈值。 |
| Liao & Coimbra, 2024, SCOPE-CNN；[出版社](https://www.sciencedirect.com/science/article/pii/S0960148124011236)，DOI 10.1016/j.renene.2024.121055。原摘要，全文访问受限。 | 长波卫星图像结合 SCOPE 云物理信息进入 CNN，并与无这些物理特征的模型比较。 | 支持显式云特征的配对研究；不能引用为已经证明“独立云概率通道”有效。 |

IGARSS 2023 *Using Meteosat Cloud Masks for Solar Irradiance Nowcasting*（10.1109/IGARSS52108.2023.10283418）定位到正式会议条目，未取得原全文；其方法线由上述可读的 2024 作者稿承接，本次不另引用二手描述作技术证据。

## 3. 四个量必须分开

| 量 | 含义 | 本项目能否直接获得 |
|---|---|---|
| 判云得分 `p_cloud` | 分类器根据观测判断有云的得分；校准后才可作频率意义概率解释 | 已保存 sigmoid 输出。尚不能承诺校准良好 |
| 二值云掩膜 `m` | 按预定规则判为晴空/有云 | 可由得分或参考分类产生；来源必须标明 |
| 云覆盖率 `f_cloud` | 某像元或区域内被云覆盖的面积比例 | 不能把一个像元的分类得分直接当它 |
| COT `τ` 与云影响因子 `Kc` | 前者是光学厚度；后者为 GHI/晴空GHI，还受几何、云相及散射等影响 | R 给出前者的估计；后者是 GHI 网络输出的目标表示 |

云掩膜也不等于数据有效掩膜。CPP 无效/缺测不是晴空。Heliosat/CAMS 文中大写 `P_cloud` 在部分公式里是一组云物理参数，不是一个标量判云概率。

## 4. 方法选择

**首轮采用“云状态条件输入”：原始 COT 与判云得分作为两个独立通道，同位置、同三帧进入现有小邻域 GHI 头。**

依据是文献中的云状态/云光学性质分解、独立云掩膜特征输入先例，以及本项目已有硬清零实验。具体从二值特征改成连续得分、从单独编码改成局地通道拼接，是本项目待检验的适配，不冒充文献已经验证的结论。

- 保留连续得分，是为了让网络有机会处理判云模糊的薄云和边缘；不预先断言它比二值掩膜好。
- COT 不被得分相乘或阈值清零；GHI 也不乘掩膜，不把有云像元强制为低辐射。
- 首轮只增加输入，不同时更换 R、太阳几何、损失、骨干或增加晴/云专家分支。
- 晴/云分路计算在文献中有充分动机，但会改变网络及训练问题，留作独立后续假设，不与输入实验一起改。
- 主评价用地面 15 分钟 GHI；判云和 COT 指标只作辅助。真实未来参考掩膜不能进入预报输入。

本次可确定的是接法的语义和可复现对照；是否提高 GHI、是否改善高值偏低、能否延伸到预报，都不能从文献直接下结论。执行细节见 [详细计划](EXPERIMENT_PLAN.md)。
