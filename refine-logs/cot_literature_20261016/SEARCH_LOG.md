# COT专题补检记录

日期：2026-10-16；交付：`docs/COT_LITERATURE_REVIEW_20261016.md`。

## 范围和程序

按三条主题检索，并追溯关键论文的原始来源及引用链：①COT/CPP/CTH直接外推；②云物理属性辅助同期GHI估计；③属性外推后辐射映射。由总控整合，两名只读研究代理分别核查直接属性外推与COT→GHI预测链。采用research-lit技能要求的独立检索分工，不涉及实验窗口或新训练。

来源为公开网页检索、出版社原文及可索引正文、作者机构存档、既有项目文献记录、用户提供Cloud-FNO PDF。没有使用Scopus/WoS付费数据库全量导出，未访问用户Zotero/Obsidian；本地标准`papers/`与`literature/`库不存在，已有`paper/references/`和docs仍纳入查重。不能称为穷尽系统综述，不提供虚构PRISMA筛选计数。

## 总控执行的查询（逐字保留）

```text
"cloud optical thickness" "solar irradiance" deep learning estimation
"cloud optical depth" "irradiance" satellite neural network retrieval
"cloud optical thickness" "irradiance" forecasting review
"cloud optical" "irradiance" "McClear" "Heliosat-4"
"cloud optical thickness" "surface solar radiation" "Himawari" machine learning
"cloud optical depth" "FY-4" "irradiance" estimation
"cloud optical depth" "irradiance" "neural" satellite estimation COT input ablation
Heliosat-4 method Qu 2017 cloud optical depth GMD 477
Takenaka 2011 Estimation solar radiation neural network radiative transfer 2009JD013337
"S0034425723000998"
"Improvement of solar irradiance modelling during cloudy-sky days using measurements"
"Fast radiative transfer parameterisation" Heliosat 2017 Qu DOI
"Retrieval of cloud microphysical properties" "sun glint" 2023
"cloud optical thickness" "FY-4A" "2136"
"surface solar" "SSRC" FY-4B algorithm
"Hybrid solar irradiance nowcasting" "SCOPE" Liao Coimbra 2024
"Retrieval of cloud microphysical properties" "113548" radiation
"Surface Solar Radiation Compositions Observed" Letu 2023
"Multi-Layer Cloud Motion Vector Forecasting" 122144
site.sciencedirect.com/science/article/pii/S0960148124011236 SCOPE 44.7
site.sciencedirect.com/science/article/pii/S0034425723000998 abstract
site.journals.ametsoc.org "Surface Solar Radiation Compositions" "Letu"
```

注意：查询中的“GMD477”为待核查线索，核查后Heliosat-4正确期刊为Meteorologische Zeitschrift，DOI10.1127/metz/2016/0781；报告不沿用错误线索。

两个只读分支另做标题/DOI/参考文献追溯；其最终返回表已合并进报告，未声称此处列尽代理所有搜索调用。主要追溯链为Cloud-FNO→Xu/Yu；NextSENSE→NextSENSE2/3D-CMV；GOES16/Peng→COT辐射映射；HelioNet/SolarSTEPS作为非COT邻近路线。

## 纳入、排除与去重

- 纳入：直接预测云物理量；显式云物理量进入辐射估计；卫星属性预报→辐射；对本研究误差解释直接重要的空间校正研究。
- 单列邻近：cloud index、clear-sky index、红外模态增量，不混称COT。
- 不纳入主表：由地面UV反演COT的反向任务；纯PV时间序列；只有云检测且无属性或辐射联系；没有可核实来源的“HelioNow”模型名。
- 按DOI去重；线上发表年与卷期年不同保留卷期年，最新预印本单独标注。报告矩阵21条（含待精读候选与3条邻近路线），不是全部初始检索命中数。

## 旧记录核对

09-05定位文档已经含GOES16/Peng；09-29包含Peng/HelioNet/SolarSTEPS/Cui/Miller/Wiltink；10-03含Cloud-FNO阅读和代码记录；10-15写作参考含SCOPE-CNN摘要，以及Papachristopoulou2024、Chen2023、Sasaki2024全文阅读记录。故这些不能报首次发现。NextSENSE关键词未命中并不代表对应题名的Papachristopoulou论文未记录，已通过全文阅读索引纠正。

搜索`docs/`和`paper/`的md/tex文件，关键词包括SCOPE、NextSENSE、2130、2136、Takenaka、Heliosat-4、Al Asmar、6555、121055、Drosoglou。关键词未命中只代表该次范围，不能证明所有历史会话从未出现。

## 访问限制与证据处理

- ScienceDirect的Peng、SCOPE、Tana及3D-CMV部分页面403；使用出版社摘要/代理已核对摘要时标A；仅元数据标M，不声称完整阅读。
- MDPI部分正文429；索引返回相关章节标P，而非全文精读。
- Drosoglou EGUsphere页面可读，PDF未完整获取；只引摘要，明确预印本及CMF指标。
- AMT NextSENSE2全文可读；明确每15min瞬时与我方区间均值不同。
- Heliosat-4使用DLR作者机构PDF，核实期刊与DOI。
- 不从二手搜索条目采纳未经原文核实的精确性能、split或消融结论。

## 完成与下一步

已交付分组文献矩阵、旧记录纠错、现有论文定位、实验问题映射、精读优先级。最重要未闭合项是SCOPE/Peng完整输入与消融合同；它们不阻止报告当前已核实的研究谱系，但限制“未有人做过同类对照”的新颖性断言。没有修改正文、模型、日期清单，未读测试载荷。
