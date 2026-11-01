# COT—GHI 瓶颈的文献调研与下一轮候选协议

日期：2026-09-29。状态：文献与实验方案建议；本轮未提交训练、未访问 test、未修改已批准的模型及运行配置。本文不是已证实的论文贡献。遵循 research-lit 技能，检索分工用于文献抽取，路线判断由主任务综合项目证据完成。

## 1. 建议与证据边界

建议暂停仅以“换一个更强的未来 AGRI 生成模型”为目的的探索。优先检验：**用真实历史卫星图及其 COT 空间序列预测站点 GHI，让真实未来 AGRI→冻结 R256 的 COT 在训练时提供空间监督；部署时不要求先生成准确的未来 AGRI 或 COT 图。** 原 SimVP 和 R256 保持冻结，现有四里及竺家基线保留。

这不是说未来云场不重要，而是避免把完整多光谱重建成功设成 GHI 研究必须跨过的前置门槛。直接预测云物理场已有文献先例；项目 9 月 25 日区域方案也已有 E_region 和后续任务指导的设想，不能把“先反演再外推”重新包装成全新方法。本次实质调整是：先做一个以 GHI 为主任务、以未来 COT 为辅助监督的受控实验，不等待一个独立 COT 场预测器先胜出。是否采用此候选路线，须与现有合同另名登记。

当前证据有两个瓶颈，不能合并成“只有底图不行”：

- 128 序列的既有 R256 GHI pilot 中，两站等权 RMSE 为零 COT 146.980、预测 COT 146.410、oracle COT 142.597 W/m²；oracle 相对零 COT 的日期配对 95% 区间为 [-10.599,+1.278]，仍跨零。四里 oracle 改善较明显，竺家整体仍退化。三个 H 分别训练，差异包含优化差异，oracle 不是严格信息上限。
- 2025-08-14 四里事件，COT 对 CPP 的 RMSE 从预测的 22.3 降至 oracle 的 4.5，但 GHI RMSE 为无 COT 119.1、预测 COT 157.3、oracle 156.7 W/m²。未来 COT 更接近参考，仍没有消除 GHI 错误。反之，7 月 9 日四里漏报云增长事件中 oracle 带来更大 GHI 改善。三例仅用于解释，不能代表总体。
- 容量匹配的八历史帧与重复末帧试验都选纯输运。已核验实现把历史帧堆进通道，再沿未来 lead 复制；没有逐历史帧运动对齐。这关闭的是该残差头，不足以否定历史轨迹信息。只增加更新数不能解决训练损失下降而厚云指标恶化的目标取舍。

证据文件：[整体 pilot](REGIONAL_COT_GHI_NEXT_20260927.md)、[COT/GHI 同事件审计](REGIONAL_GHI_COT_EVENT_AUDIT_20260928.md)、[历史对照](SILI_FULL_VS_REPEAT_LAST_HISTORY_RESULTS_20260928.md)、[实现核查](SILI_HISTORY_USE_COUNTERFACTUAL_20260928.md)。上述均为已有验证诊断，不代表本轮重新训练或服务器实时状态。

## 2. 直接相关文献

以下均链接原始出版物、作者机构或作者稿。数字只说明原论文结果，不能与湖南 pilot 跨数据比较。

| 文献与状态 | 问题、方法及结果 | 对项目的意义与限制 |
|---|---|---|
| Nils Straub, Steffen Karalus, Wiebke Herzberg, Elke Lorenz (2024), **Satellite-Based Solar Irradiance Forecasting: Replacing Cloud Motion Vectors by Deep Learning**, Solar RRL 8, 2400475；已发表。[全文](https://onlinelibrary.wiley.com/doi/10.1002/solr.202400475)，DOI 10.1002/solr.202400475 | HelioNet 直接预测云指数 CI，再计算辐照度，避免从模糊的未来卫星图重新反演 CI。hybrid 在其试验中图像 RMSE 平均改善超过 11%，辐照度约 8%。 | 最直接支持改变预测对象。CI 不等于 COT；其收益不保证迁移到湖南。 |
| Wanshan Peng, Tao He, Yichuan Ma, Yueming Zheng (2026), **A physics-informed machine learning method for short-term solar radiation forecast with satellite-derived cloud optical depth**, Applied Energy 406, 127284；已发表。[出版社](https://www.sciencedirect.com/science/article/abs/pii/S0306261925020148)，DOI 10.1016/j.apenergy.2025.127284 | 从历史 GOES-16 COD 与云掩膜预测未来场，结合辐射传输模拟得到的映射及 AOD、水汽、地表反照率、太阳天顶角计算 GHI；10 min–4 h。云掩膜缓解低 COD 高估带来的 GHI 偏差。 | 直接 COT/COD 外推已有最近邻，不能声称首次。只取得摘要/出版社片段，完整结构、站点数量、辅助数据因果性及代码未核验，不据此复制实现。 |
| Alberto Carpentieri, Doris Folini, Daniele Nerini, Seppo Pulkkinen, Martin Wild, Angela Meyer (2023), **Intraday probabilistic forecasts of surface solar radiation with cloud scale-dependent autoregressive advection**, Applied Energy 351, 121775；已发表。[出版社](https://doi.org/10.1016/j.apenergy.2023.121775)、[作者预印本](https://arxiv.org/abs/2305.07293) | SolarSTEPS 在随云移动的坐标中分解空间尺度，建模晴空指数的时间演变和随机扰动，预测 4 h。其 CRPS 相对概率平流改善约 9%。 | 支持先移除运动，再判断哪些尺度的增长/消散可预测。不是历史通道堆叠，也不是直接验证 COT→单站 GHI 的收益。正式版题名/作者顺序与预印本不同，以正式版为准。 |
| Yang Cui, Ping Wang, Jan Fokke Meirink, Nikolaos Ntantis, Jasper S. Wijnands (2024), **Solar radiation nowcasting based on geostationary satellite images and deep learning models**, Solar Energy 282, 112866；已发表。[KNMI](https://www.knmi.nl/kennis-en-datacentrum/publicatie/solar-radiation-nowcasting-based-on-geostationary-satellite-images-and-deep-learning-models)，DOI 10.1016/j.solener.2024.112866 | 比较 U-Net、DGMR-SO、云属性平流及物理辐射计算等；荷兰 0–4 h。U-Net 数值误差较低，DGMR-SO 在较长时效有更好的视觉表现。 | 更清晰的图像不等于更准确的站点 GHI；第二篇不因此引入生成模型。 |
| Steven D. Miller, Matthew A. Rogers, John M. Haynes, Manajit Sengupta, Andrew K. Heidinger (2018), **Short-term solar irradiance forecasting via satellite/model coupling**, Solar Energy 168, 102–117；已发表。[NOAA](https://repository.library.noaa.gov/view/noaa/49543)，DOI 10.1016/j.solener.2017.11.049 | 按云高选择 NWP 风，输运云对象，结合视差、太阳阴影和辐射计算。论文给出约 10 W/m² 的相对持续性改善。 | IFS 可以提供风场条件，不只在末端拼一个 GHI 值。但需可靠云高、风廓线、初始化时已经发布的数据；不能直接解决所有云生成。 |
| Job I. Wiltink, Hartwig Deneke, Chiel C. van Heerwaarden, Jan Fokke Meirink (2025), **Evaluating parallax and shadow correction methods for global horizontal irradiance retrievals from Meteosat SEVIRI**, Atmospheric Measurement Techniques 18, 3917–3936；已发表。[全文](https://amt.copernicus.org/articles/18/3917/2025/)，DOI 10.5194/amt-18-3917-2025 | 99 个地面辐射仪的检索评价中，视差与云影联合修正使 RMSE 降低 11.7 W/m²（10.8%）；单独修正可能恶化。 | 站点上空像元不必对应遮挡站点的云。需要云高、卫星/太阳位置，不能仅靠 COT 推出几何高度；这是检索研究，不是我们的预报收益保证。 |
| David Lopez-Paz, Léon Bottou, Bernhard Schölkopf, Vladimir Vapnik (2016), **Unifying distillation and privileged information**, ICLR；会议论文。[作者全文](https://leon.bottou.org/publications/pdf/iclr-2016.pdf)、[arXiv 1511.03643](https://arxiv.org/abs/1511.03643) | 训练阶段利用部署阶段不可见的信息，向仅有可用输入的学生模型传递监督。 | 支持未来真实 COT 只用于训练。教师未验证可靠前，不应直接蒸馏；辅助监督也不能创造历史中不存在的未来信息。 |
| Rickard K.A. Karlsson, Martin Willbo, Zeshan M. Hussain, Rahul G. Krishnan, David Sontag, Fredrik Johansson (2022), **Using time-series privileged information for provably efficient learning of prediction models**, AISTATS, PMLR 151:5459–5484；会议论文。[全文入口](https://proceedings.mlr.press/v151/k-a-karlsson22a.html) | 学习时使用预测起点与未来结果之间的时间序列，部署只用起点信息。特定高斯线性动态系统下证明样本效率优势。 | 与未来 COT 轨迹作监督相关；理论条件不能直接套在非线性云场上。 |

补充前沿线索：Tao Jing 等 (2026) 的 [WIP-Flow 作者页](https://www.li-realab.info/publication/solar-forecast/jing-2026-wipflow/)介绍光流、多尺度演变和 flow matching 概率辐射预测。作者标注 IEEE TSE，但本轮 DOI 独立核验未完成；仅作趋势线索，不作为第二篇启动生成模型的依据。

共同启示是预测对象应接近辐射任务，云移动与云演变应分开处理，而训练目标、视觉清晰度、COT 场误差和站点 GHI 并不是同一个指标。文献没有给出“直接预测 COT 必定优于卫星图外推”的普遍结论：云相、粒径、云高和大气条件仍影响辐射；COT 自身也存在检索误差。

概率路线回答的是云到达/消散不确定性，可能改善可靠性，却不自动改善确定性 RMSE。物理映射路线需要额外大气输入及来源审计。训练专用监督路线可以先用现有数据试验，因而这里优先级最高；这是结合项目资源与负结果作出的判断，不是文献已替我们验证的结论。

## 3. 为什么更好看的未来图可能仍无效

设 A 为 AGRI，R 为冻结反演器。当前链路是历史 A→预测未来 A→R→GHI。未来位置存在不确定性时，平方误差目标倾向折中多个可能位置，造成云边界/幅值变钝；R 又是非线性的，因此一般有 R(E[A|历史])≠E[R(A)|历史]。这是解释候选，不表示已证明当前每个模型都在输出条件均值。

预测 COT 是既有信息的变换，不是新观测。价值可以来自物理监督和表示偏置，但不能从变换本身获得未观测的未来云生成信息。真实未来 AGRI→R 的 oracle 多了未来观测，不能承诺靠蒸馏恢复其全部优势。

还需区分“数值太阳几何正确”与“云—地面作用位置正确”。输入 cosSOZ/cosRAA/day_mask 并不会自动完成视差和云影投影，cosRAA 也不保留完整方向信息。云高与太阳天顶角决定的阴影位移可达多个像元，但 COT 是光学厚度，不是云的几何高度。GHI 含直射和散射，不能直接写成 GHI=clear_sky×exp(-COT/cosSOZ)，也不能强制所有厚云只做负修正。

## 4. 推荐顺序与停止条件

### P0：把下一次训练变成可归因的试验

由当前窗口完成已有 zero/forecast/oracle H 的共同样本、分站/分时效及事件响应复核；另一个窗口继续云场位置、增长/消散归因，不重复启动 COT 残差训练。已有极端案例用于解释，整体验证仍用完整共同 cohort。

固定 H 替换 COT 可做敏感性矩阵，但替换到零或 oracle 可能离开训练分布。替换无收益不证明 oracle 无信息，替换有收益也不等于重训后的可部署收益。分别训练且预算相同的 oracle 头是补充诊断，仍不是独立物理真值。旧报告中的“据单个替换直接判断主要瓶颈”应收紧为上述边界。

检查原始 CPP 中已有 CTH/CLP 的有效性、时间对应和来源。仓库历史文档确有这些字段，但这不证明当前有合格的实时云高预测器。先在少量真实时刻做云影位置诊断；缺乏可信云高时只报告几何敏感性，不自动平移标签或将未来 CTH 用于正式预测。

### P1：优先的最小训练对照——未来 COT 辅助监督

共同输入为真实历史 AGRI 八帧、历史和可解析计算的目标太阳几何。冻结 SimVP 的输出若作为共同参考特征，两组完全相同；不重新训练 SimVP/R256，不把未来实测 GHI 或真实未来 AGRI 输入模型。

共同网络保留历史时间轴与二维空间位置，用同一容量的区域/局地编码器预测未来 16 个站点 kt，同时设空间 COT 解码分支。GHI 可以从时空特征直接读出，不必经过预测 COT 图这个唯一接口。COT 输出仍可视化并审计，不退回全图均值或 p90。

| 组别 | 唯一差异 | 回答的问题 |
|---|---|---|
| T0 | COT 辅助损失权重为 0 | 同一时空模型、同一预算下的无 COT 监督参照 |
| T1 | 开启一个预先固定的 COT 辅助损失权重 | 未来云物理监督能否改善 GHI 表示，而无须未来图像重建成功？ |

两组保留完全相同的网络、数据顺序、优化器、GHI 损失和选模口径；T0 的辅助输出存在但不产生该分支的监督梯度。参数总数相同不等于训练信号相同，研究对象正是监督差异。主比较是 T1−T0；与已有旧头的差异只算整体模型比较，不能全部归因于 COT。

未来监督首选同一冻结 R256 对真实未来 AGRI 的输出，明确称为代理标签；原始 CPP 仅在有效共同支持上独立检查，不冒充物理真值。只使用 train 的未来标签更新权重，保持既有时间 split，排除训练窗口跨进 validation/test；缺 CPP 不删除 GHI 行，辅助监督缺失只关闭该项损失。

采用现有已审定的 GHI 主目标，单独添加 COT 辅助项，避免同时换 GHI 损失、加 ramp 权重、改学习率和采样。辅助项按有效像元归一化并检查梯度尺度，预先写入一个权重，不依据验证结果连续搜索。最终选模看分站 GHI，不看训练 COT loss。未来 R256 标签误差和对厚云/近零的权衡须分层报告。

这个阶段是多任务监督，不必先引入教师蒸馏。只有 oracle 教师经独立训练内留出验证可靠，才有理由另做教师→学生对照；不能用教师对其训练样本的过拟合输出制造学生收益。

### P2：由证据选择一条后续分支

- **T1 改善 GHI，且两站稳定**：固定 T1，再比较历史 COT 完整空间序列输入与等容量零输入槽，隔离“输入表示”相对“物理监督”的额外价值。区域及局地结构都保留；不把旧四统计量恢复为输入。
- **主要是云位置/到达时刻错误**：先用历史估计的运动场将八帧对齐到同一坐标，比较对齐/不对齐，检验固定云对象的厚度演变。移动估计不能使用未来真值；任何用未来 CPP 找最佳平移的结果只作诊断。该比较应由负责云动力诊断的窗口接续，避免双重提交。
- **主要是增长/消散不可预测**：先做运动坐标内各尺度相关性的 train/validation 分离诊断。若短尺度很快失去可预测性，再以非生成式 SolarSTEPS 风格概率参照或 kt 分位数预测检验不确定性，不立即引入 diffusion。
- **长时效确实缺少动力条件**：独立增加初始化时已发布的 IFS 风/湿度等条件，检验 2–4 h 增量；需先验证实际产品变量、层次、release_time 和单位。按云高选风还额外需要可信云高。不可拿验证时刻的未来再分析替代。
- **T1 仍无收益，且代表性 oracle 也不稳定**：停止宣称 COT 改善点预报。若概率指标也无收益，第二篇应收缩为 COT 表示适用条件/负结果或其他已验证科学问题，而非继续堆叠模型。

### 预算与验收

本文件不自动启动上述实验。若实施，先冻结配对样本 manifest 和训练内调参/选择划分；不把验证集当训练集，不访问 test。GHI cohort 不因 CPP 缺失而变化；重叠窗口合并为事件或至少按日期聚类，事件定义不依据候选误差挑选。当前已多次查看的 validation 只能提供探索性结论，确认性结果仍需以后独立时间评估。

先一个配对种子作可行性筛查，只有主比较方向合理才扩为三个预设种子。两臂固定相同实际更新预算，报告吞吐、epoch、updates 与学习曲线；不再把计划上限当实际训练量。单卡 profile 通过后确定预算，资源限制仍为全账户最多三个未结束 PBS 作业，提交前实时查队列。本次不估算未经 profile 的耗时。

主终点是两站等权 GHI RMSE，另分别报告两站、16 leads、+15～45 min、严重退化事件、MAE/bias、日期聚类配对区间。COT 只作解释及辅助诊断，至少包括厚云幅值、边界/到达、近零误报、valid 覆盖；晴空标签须有可靠语义，CPP 0 不自动等于有效晴空。

预设推进门槛：单种子不称成功；多种子两站方向均不得被总体均值掩盖，短时效和严重退化无系统恶化，配对不确定性支持收益才进入独立确认。如果只对四里有效，明确限定四里，竺家保留无 COT 基线。零初始化/残差结构本身不保证训练后不退化。

## 5. 可以形成的科学问题与缺口

候选科学问题：**在卫星未来图像误差限制云物理反演时，训练期云物理轨迹监督能否提高站点短临辐照度预报；这种收益受云运动、演变和云影几何怎样的限制？**

能否成为贡献取决于 T1−T0、历史输入增量、站点差异及后续因果 IFS 对照。直接 COT 预测、物理映射和特权信息学习均已有先例，不声称首次；本轮不是系统综述或查新通过。拟用于论文时须同步英文 LaTeX 与中文对照，本轮未改论文正文。

检索边界：项目 papers/、literature/ 下未发现本地论文库，已读取旧 LITERATURE_POSITIONING.md；可用工具中没有接入 Zotero/Obsidian，本次依靠出版社、作者机构与 arXiv。arXiv helper 的宽泛查询无有效增量，原论文按题名和 DOI 再检索。Peng 全文未取得；GOES-16 COT 2025 最近邻（Remote Sensing 17(16),2861）页面访问失败，不使用未核验的方法细节。WIP-Flow DOI 独立核验待补。当前没有完成 CTH 数据质量验收或 IFS 因果放行，不能把文献存在当作本地前置条件已经满足。
