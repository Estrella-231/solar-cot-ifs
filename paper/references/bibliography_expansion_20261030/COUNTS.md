# 相关论文实际参考文献条数

核对日期：2026-10-30。统计的是论文末尾参考文献表的**条目数**，包括列入该表的论文、书籍、报告、数据集、软件和网页；不是论文被引次数，也不是正文引用标记出现次数。

| 论文 | 期刊 | 主文页数／范围 | PDF总页数 | 参考条目数 | 参考文献所在PDF页 |
|---|---|---:|---:|---:|---|
| Chen et al. (2023), *Global and direct solar irradiance estimation using deep learning and selected spectral satellite images* | Applied Energy 352,121979 | 12页（1–12） | 17 | **57** | 16–17 |
| Chen et al. (2024), *Improved satellite-based intra-day solar forecasting with a chain of deep learning models* | Energy Conversion and Management 313,118598 | 15页（1–15） | 17 | **55** | 16–17 |
| Papachristopoulou et al. (2024), *Effects of clouds and aerosols on downwelling surface solar irradiance nowcasting and short-term forecasting* | Atmospheric Measurement Techniques 17,1851–1877 | 20页（PDF 1–20，期刊1851–1870） | 27 | **77** | 24–27（期刊1874–1877） |
| Zhang et al. (2026), *Short-Term Forecasting of Cloud Physical Properties Based on Fourier Neural Operator Method* | Geophysical Research Letters, DOI:10.1029/2025GL119553 | 9页（1–9；第9页也含参考文献开头） | 11 | **53** | 9–11 |

“主文页数”指从首页到结论所在最后一页的页码范围，包含该范围内的图表；不是按文字面积折算的纯文本页数。Chen2023的附录在13–16页，AMT附录自21页开始。第9页兼有主文和参考文献的Cloud-FNO不能以9+3解释为12页。

## 计数与交叉核验

1. **Chen2023：57条。** 实际PDF参考文献编号从[1]到[57]；程序逐条提取起始编号，确认连续、无重复、无缺号。第16页32条，第17页25条。
2. **Chen2024：55条。** 同样逐条核验[1]至[55]。第16页42条，第17页13条。
3. **AMT2024：77条。** 作者—年份制没有编号，采用两条独立计数：官方HTML中77个`p.ref`条目，ID连续为`bib1.bib1`至`bib1.bib77`；PDF悬挂缩进的参考条目首行独立得到77条，四页分别为17、22、25、13条。首作者顺序与HTML逐条对应。
4. **Cloud-FNO：53条。** 使用用户提供的正式11页PDF，参考文献首行横坐标为167.24 pt，续行缩进至175.01 pt；排除左侧致谢、页眉和页脚。三页首行数为2、32、19，合计53；每个条目首行均含作者和括号年份。

这四篇的参考文献为53–77条，说明此前仅保留少数关键来源的结构稿尚不是完整相关研究梳理。这个小样本不能作为期刊统一数量要求；补充文献仍应围绕实际论点和方法来源。

## 可复核来源

- Chen2023：[作者公开PDF](https://www.li-realab.info/publication/solar-forecast/chen-2023-resourcing/chen-2023-resourcing.pdf)；[DOI](https://doi.org/10.1016/j.apenergy.2023.121979)。
- Chen2024：[作者公开PDF](https://www.li-realab.info/publication/solar-forecast/chen-2024-dlchain/chen-2024-DLchain.pdf)；[DOI](https://doi.org/10.1016/j.enconman.2024.118598)。
- AMT2024：[官方全文与参考文献列表](https://amt.copernicus.org/articles/17/1851/2024/)；[官方PDF](https://amt.copernicus.org/articles/17/1851/2024/amt-17-1851-2024.pdf)。
- Cloud-FNO：[出版社入口](https://doi.org/10.1029/2025GL119553)；实际计数使用用户已提供的正式PDF，路径和SHA256见JSON。

[REFERENCE_COUNTS.json](REFERENCE_COUNTS.json)保存每条的序号、PDF页、起始坐标、首作者标识与首行哈希，以及原PDF的SHA256；AMT另有对应HTML ID。[count_references.py](count_references.py)可复算全部四篇。索引没有复制完整参考文献表。
