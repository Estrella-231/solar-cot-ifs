# 当前方法视觉稿整合

按用户要求，优先使用原生 image_gen 生成模型图，以视觉效果和后续替换便利为目标。未改研究路线、实验数据或科学结论。

- 当前三帧同期模型：`paper/figures/generated_current_20261102/three_frame_estimation.png`，正文图2，第5页。
- 当前冻结预报应用与前三时效帧对齐：`paper/figures/generated_current_20261102/forecast_application.png`，正文图3，第7页。
- 生成提示词完整保存于同目录 `PROMPTS.md`；使用内置image_gen，未用CLI或程序绘图替代生成模型。预报图经一次原生图像编辑，修正输出为站点GHI、几何路径和帧对齐提示。
- 两图均为示意视觉稿，生成云纹理不代表实际观测或预报；主稿双语图注已注明。后续可在同一文件引用位置替换。
- 六张真实数据图改用 `paper/figures/visual_refresh_20261102/`，映射、原数据哈希与一致性检查见该目录。只改配色、字号、线型和排版，实际RGB与数值不变。
- 英文main.tex与中文main_zh.md同步；新增图3后所有主图引用顺延，补充图号不变。原单帧/预测图训练的旧示意图仍保留在补充材料，不作当前主方法。
- 内置编译成功。本地PDF曾短暂被占用，重试后两遍编译成功，无未定义引用和overfull，最终27页。已检查方法页和全文页面布局。检查见CHECKS.json。

本目录main_before.txt/main_zh_before.txt仅是修订前档案，不能作为新正文分支；没有新建另一份论文PDF。
