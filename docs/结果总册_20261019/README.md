# COT研究结果总册（2026-10-19核对版）

先读 **COT_重点结果速览.pdf**；完整材料见 **COT_全部结果与复现索引.pdf**。

本册为独立结果资料集，不替换论文正文。每页一个表或图；标题说明问题，“本页意义”说明结果用途，“评价范围”注明同期、预报、探索及样本限制。完整册开头四页按科学问题列出页码，PDF另有书签。速览保留完整册的原页码，便于回查。

## 本次核对范围

- 重新读取最新总控、湖南2026-10-19实验记录和湖北2026-10-17至18日复算文件，未用旧聊天数值代替文件证据。
- 2026-10-19直接SSH读取学院服务器队列，并取回混合COT辅助及CPP-only辅助的最终独立复算JSON。候选总体RMSE均未通过预定筛选；两种尺度分别从JSON的相应字段生成，避免将16×16与中心3×3的CSI混在一起。
- 湖北同期采用归一化重载及最新再生成汇总；预报补齐FA/FAC/FAP三臂。湖南当前五折OOF准备属于在途，尚无最终OOF-GHI成绩。
- 图表覆盖44组当前结果和历史探索；旧索引错配CPP图已排除。重复格式、重复存档、工程截图不逐张进入科学图页；全项目PNG清单见ALL_FIGURE_INDEX.csv。
- 当前均为验证/筛选/诊断材料，独立test和buffer载荷未打开。同期COT增益与未来GHI收益分别报告。

## 文件说明

| 文件 | 内容 |
|---|---|
| main.tex / main.pdf | 可编辑中文LaTeX源与直接编译版本 |
| main_zh.md | 同结构、同表格数字和同结论的中文Markdown对照 |
| COT_全部结果与复现索引.pdf | 完整册，加入PDF书签 |
| COT_重点结果速览.pdf | 从同一份编译PDF抽取的重点阅读路线，没有改写数据 |
| PAGE_INDEX.csv / page_manifest.json | 每页标题、意义、范围、原始文件和实验编号 |
| SOURCE_INDEX.csv / sources/ | 44组报告来源、SHA-256及原报告快照；最新服务器JSON另存 |
| CHECKPOINT_INDEX.csv | 本地307个权重的绝对路径、大小、本次字节SHA-256及实验关联；权重原位保留 |
| REMOTE_CHECKPOINT_INDEX.tsv | 本次SSH查到的51个最新远端权重文件、大小和mtime；不是权重下载或重新重载验收 |
| CODE_INDEX.csv / code_snapshot/ | 314个代码文件的原路径、快照路径、SHA-256；含训练、评价、绘图和PBS入口 |
| DOCUMENTED_ASSET_REFERENCES.csv | 原报告中登记的模型/代码位置；标明没有逐项远程核验 |
| ALL_FIGURE_INDEX.csv | 原工程PNG图清单和本册收录状态 |
| INVALID_RESULTS.csv | 已作废CPP图及更正说明 |
| BUILD_MANIFEST.json / LAYOUT_AUDIT.json | 页数、图表数、代码/权重数量及分页/缺字检查 |
| review/ | 全页缩略联系表与代表页面，用于检查真实渲染 |

本地权重目录的SHA-256是本次文件字节核对，不表示重新跑过全部checkpoint重载。远端权重有在途训练文件，不能自动视为最终选中模型；具体使用哪个权重，应以对应报告和训练配置为准。大型权重不重复复制到此文件夹。

## 重新生成

在本目录依次运行：

```powershell
python build_result_book.py
xelatex -interaction=nonstopmode -halt-on-error main.tex
& 'C:\Users\29728\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' audit_and_deliver.py
```

第一步只读已有结果并整理；第二步编译；第三步检查每页编号、缺字和溢出，生成书签版、速览、图索引和全页联系表。没有训练、调参或新test评分。若原工程在整理之后继续产生结果，应重新核对最新记录后再更新，不将本册快照当作服务器实时状态。
