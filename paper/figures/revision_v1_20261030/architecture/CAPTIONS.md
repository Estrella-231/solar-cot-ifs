# English caption

**Implemented single-frame COT retrieval and local GHI mapping.** (a) A two-level U-Net maps 13 AGRI channels and three geometric inputs (cosine of solar zenith angle, cosine of relative azimuth, and day mask) on a 16×16 context to nonnegative log-COT. Each convolutional block contains two 3×3 convolution–batch-normalization–SiLU sequences; same-resolution encoder and decoder features are concatenated. The audited implementations use base width (w=32) in Hunan and (w=16) in Hubei. The retriever is trained against valid CPP COT reference pixels and is frozen during GHI-head training. (b) The Hunan single-COT-channel head uses co-registered 3×3 AGRI and COT fields. AGRI features and the two geometric cosines are encoded pointwise, while log-transformed, training-standardized COT is mapped to four auxiliary channels. Concatenation and an ordered flattening retain all nine spatial positions for the dense readout. A Softplus output gives a nonnegative clear-sky index, which is multiplied by interval-mean clear-sky GHI. (c) Numbered cells illustrate correspondence, not observed data; flattening follows channel, row, then column. The Hubei four-arm implementation instead reserves two auxiliary slots for standardized COT and cloud probability; inactive slots are zero, and its 2→4 convolution has no bias. Solid arrows denote inference features and the dashed link denotes retrieval supervision. Batch dimensions are omitted. These code-verified designs are distinct from the separately trained historical-COT forecast head and do not by themselves establish model skill or a completed result-to-checkpoint binding.

# 中文图注

**已实现的单帧COT反演与局地GHI映射。** (a) 两级U-Net将16×16邻域中的13个AGRI通道和三个几何输入（太阳天顶角余弦、相对方位角余弦、昼夜标记）映射为非负log-COT。每个卷积块包含两次3×3卷积—批归一化—SiLU，同分辨率编码与解码特征进行拼接。已核验实现中，湖南基础宽度为(w=32)，湖北为(w=16)。反演器以有效CPP COT像元作为参考监督，训练GHI头时权重冻结。(b) 湖南单COT通道头读取逐像元对齐的3×3 AGRI和COT场。AGRI与两个几何余弦经逐像元编码，COT经过对数变换及训练集标准化后映射为四个辅助通道。拼接与有序展平将九个空间位置保留至全连接读出。Softplus给出非负晴空指数，再乘以对应区间的平均晴空GHI。(c) 编号网格只示意位置对应，不是实测数据；展平依次遵循通道、行、列。湖北四臂实现另保留标准化COT和云概率两个辅助槽，不活动的槽置零，2→4卷积不含偏置。实线表示推理特征流，虚线表示反演训练监督；图中省略batch维。该结构与另行训练的历史COT预报头不同；核验代码设计本身不证明模型效果，也不替代结果与最终权重的完整对应。

## 评审版说明（不直接作为正式图注发表）

本图能用于讨论真实实现与方法细节。湖北几何修复后的同期完整评分来源目前尚未锁定，所以不得用该图暗示10/20四臂结果已完成几何修复重跑。湖南单帧结果也仍保留固定权重、旧几何及重复验证的限制。最终主文确认后可缩短最后一句，并只保留与最终实验对应的地区/控制；不能在图中无证据地统一两个实现。

## Alt text

A three-panel network diagram. A small U-Net uses a 16 by 16 satellite context to produce cloud optical thickness. In a separate trainable station head, aligned 3 by 3 satellite and COT inputs pass through separate pointwise encoders, are concatenated without spatial pooling, and enter a dense clear-sky-index readout. The bottom panel distinguishes Hunan's one-channel COT branch from Hubei's two-slot COT/cloud-probability control.
