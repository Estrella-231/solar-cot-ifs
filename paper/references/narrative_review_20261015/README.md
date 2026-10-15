# 相关论文叙事阅读与本项目改写依据

日期：2026-10-15。目的为写作结构调研，不是穷尽性系统综述，不能据此声称统计了“大多数论文”。选取多光谱GHI估计、物理信息辅助估计、云属性驱动预报等紧邻任务；三篇已下载全文，共49页。另两篇保留可访问的出版信息/摘要作补充，不冒充全文阅读。

## 全文阅读记录

### 1. Chen et al., 2023, Applied Energy

**Global and direct solar irradiance estimation using deep learning and selected spectral satellite images**。
[DOI](https://doi.org/10.1016/j.apenergy.2023.121979) · [作者公开PDF](https://www.li-realab.info/publication/solar-forecast/chen-2023-resourcing/chen-2023-resourcing.pdf) · [本地全文](chen_2023_spectral_irradiance.pdf)。17页，PDF内页码为定位依据。

核读位置：p.2方法总流程，p.6起Results，§3.1网络比较，§3.2晴空/有云比较，§3.3误差分析，§4讨论及§5结论。

其叙事不是“完成数据→跑网络→修复”，而是先提出多光谱信息利用问题，再用明确变体比较通道选择和注意力，随后考察不同天空条件及晴空指数区间。DNN16、DNN8、DNNa8虽用缩写，但含义对应具体差异，并在比较前定义。可借鉴的是总体比较→条件分析→解释，不是照抄其网络、采样或训练划分。我们采用更直接的AGRI baseline、AGRI + COT标签。

### 2. Papachristopoulou et al., 2024, Atmospheric Measurement Techniques

**Effects of clouds and aerosols on downwelling surface solar irradiance nowcasting and short-term forecasting**。
[期刊全文](https://amt.copernicus.org/articles/17/1851/2024/) · [本地PDF](papachristopoulou_2024_cloud_aerosol.pdf)。27页，CC BY 4.0。

核读位置：PDF p.4–8数据方法；p.9开始§3.1，先总体表现，再气溶胶、云影响和云输入偏差校正；p.16开始§3.2短时预报，与持续性比较；p.17开始按云况分析；p.20总结。

最值得借鉴的是把当前估计与未来预报分开论证，并用云相关误差解释辐照度差异；站点与卫星像元代表性也是讨论内容。其nowcasting术语和时间取样有自己的定义，我们不照搬，仍明确本项目同期估计与未来15分钟区间GHI。该文不能替我们证明“预报退化全部来自外推”。

### 3. Sasaki et al., 2024, arXiv:2407.04283

**A Physics-Informed Machine Learning Approach utilizing Multiband Satellite Data for Solar Irradiance Estimation**。
[条目](https://arxiv.org/abs/2407.04283) · [HTML全文](https://arxiv.org/html/2407.04283v1) · [本地PDF](sasaki_2024_physics_informed.pdf)。5页；这里按arXiv版本引用，不擅称期刊正式发表。

核读位置：§1研究动机，§3辐射相关约束及组合方法，§4结果；Fig.2有无辐射约束比较，Table 1站点比较。

其开篇把问题落到薄云/晴空中的地表反射混淆，再引入与该问题对应的约束，并结合场景与总体评价。可借鉴“物理困难→针对性设计→比较证据”的顺序。它具有具体物理约束，我们仅加入COT输入时不能借用physics-informed标签夸大方法性质。

## 补充条目：不计入已下载全文

- Liao & Coimbra (2024), **Hybrid solar irradiance nowcasting and forecasting with the SCOPE method and convolutional neural networks**，[出版条目](https://doi.org/10.1016/j.renene.2024.121055)。本次仅核对公开摘要：比较有无SCOPE特征及时间模型的应用；不据此推断全文各节和完整实验细节。CNN、SCOPE-CNN、SCOPE-CNN-ConvLSTM是可读组件命名实例。
- **IrradianceNet** (2021), Solar Energy 228,659–669，[出版条目](https://doi.org/10.1016/j.solener.2021.09.073)。大学PDF地址返回403，未取得全文，不纳入全文结构归纳。
- 另一个2023年cloudiness prediction大学PDF亦返回403，保留在下载清单中，不作为已读来源。

## 归纳及落实

三篇全文支持的共同写法：问题先于实现；比较服务于明确问题；整体误差之后分析物理条件；估计与预报需分别给证据；讨论解释结果而非复述开发过程。这是代表性样本的定性归纳，不是所有期刊的统一模板。完整性规则来自本项目科学需求，并非声称每篇参考都执行了相同统计方法。

落实到本文：显式COT是否提供增量→厚度/天气依赖→独立区域重复→辅助特征与反演特征对照→固定外推下的应用。对照可以排在条件分析之前以帮助解释总体收益；顺序根据最终证据调整，不能按正负结果决定隐藏哪组。

[写作要求](../../WRITING_REQUIREMENTS.md)规定措辞和信息放置；[新论文计划](../../PAPER_PLAN.md)给出问题、章节与图表。旧计划单独存档，工程状态继续由实验tracker维护。

## 文件与获取说明

`download_manifest.json`记录实际下载URL、文件大小、页数及哈希；三个PDF均能由PDF解析器打开，已提取全文用于阅读。失败下载明确保留失败状态。PDF及抽取全文仅作本地阅读；本目录.gitignore排除这些第三方全文，不把它们重新发布到公开仓库。仓库保留来源链接、获取记录和原创阅读笔记，用户可在本地打开上述PDF。
