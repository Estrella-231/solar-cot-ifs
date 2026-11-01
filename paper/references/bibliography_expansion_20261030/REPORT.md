# COT论文参考文献扩充与核对

2026-10-30。用户提出参考文献过少，要求核对可比论文实际引用规模。本轮同时完成数量核对、相关研究补充、正文整合和双语一致性检查。

## 可比论文

实际逐条计数：Chen2023 Applied Energy **57**条；Chen2024 Energy Conversion and Management **55**条；Papachristopoulou2024 AMT **77**条；Cloud-FNO2026 GRL **53**条。见[计数方法及原始出处](COUNTS.md)和[条目索引](REFERENCE_COUNTS.json)。这个小样本不能作为期刊统一数量要求。

## 稿件修改

- 英文LaTeX与中文Markdown从8条扩充至**30条不同文献**；每一条都有正文引用，没有只列在文末的条目。
- 引言补足云指数与COT的区别、云光学属性到辐射的物理映射、多光谱及邻域学习、云监督表示、短临中间变量和条件性评价；仍以显式COT的增量价值为核心。
- 方法补上U-Net、Huber、日期分块、簇重采样来源；保留本项目独立的结构和采样选择，不将原论文当作这些设置的直接证明。
- 讨论补上碎云代表性、视差/云影、云演变及地面观测质量的来源；没有写成已经完成相应校正或确证本项目误差原因。
- pvlib网页引用改为JOSS正式软件论文；补齐Ineichen–Perez DOI；SimVPv2更新为2025年IEEE TMM正式版本。Schuurman2025使用正式发表题名。
- 其余29条均有核验DOI，SimVP以CVF官方会议论文页面为原始出处。完整书目元数据见[MANUSCRIPT_REFERENCES.json](MANUSCRIPT_REFERENCES.json)。

## 核验与交付

- 30条书目、31组正文引文，中英文出现次序一致；无重复DOI、无缺失引用、无未引用条目。
- 11个公式、7个图占位、3张表及结果占位均保留，未填写新实验结论。
- 独立阅读审查修正了Yang2020引用范围、作者标点、SimVPv2出版版本以及中文global irradiance的译法。
- 现有main.tex经桌面内置编译器成功编译；对同一源文件进行本地排版检查，12页，未出现缺失引用、Overfull或Underfull警告。
- 查看全部12页缩略图及参考文献末页原尺寸图，未发现裁切、重叠或书目溢出。

修改脚本：`paper/revise_references_20261030.py`。修改前文件、逐项修改记录、检查JSON及页面渲染保存在`paper/revision_records/20261030_bibliography_expansion/`。现有编辑器与主文件保持不变。

## 文献覆盖范围

这是围绕当前正文科学问题完成的一轮定向补充，不是穷尽性系统综述，也不把30篇视为最终固定数量。结果和讨论回填后，应再按实际发现补入直接比较研究。每篇实际阅读深度见[CLOUD_SOURCES.md](CLOUD_SOURCES.md)、[METHOD_SOURCES.md](METHOD_SOURCES.md)、[ROOT_SOURCES.md](ROOT_SOURCES.md)，未将题录核验冒充全文阅读。
