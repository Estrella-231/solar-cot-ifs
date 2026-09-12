# COT空间表示对照：来源门禁与实施准备

用户已授权按来源追溯→空间表示最小对照→独立IFS审计推进。原四维统计配置的多种子稳定性未通过；不得覆盖该负结果。本文为实施准备，不宣称新实验已训练或已产生收益。

## 可以复用的资产

scripts/cache_frozen_forecast.py已在原始train/validation分片中保存cot_log1p，形状为[sequence,2站,16lead,16,16]，还保留predicted_agri_physical、geometry、cot_features及indices。服务器data/frozen_forecast_trainval_20260907_v1/complete.json存在。新输入应从既有分片提取，而非重跑SimVP/R或重新构建CPP。

实施时以现有head pack逐行键绑定(sequence,station,lead)，核验分片和source_shards哈希、行数/顺序/真值、归一化、相同99849验证行。只取train/val，不打开test。新空间输入须完整保留16×16像元位置，不能先做全局均值又称空间编码。

## 三组最小对照

1. 参数量匹配AGRI基线：增加可实际学习的AGRI分支，不能仅堆无效参数冒充容量匹配。
2. AGRI+四维COT统计：保留当前信息量，作为统计表示对照。
3. AGRI+COT空间编码：使用同一冻结R输出的16×16场。

共用AGRI主干和预测头，辅助编码器采用同结构、同输出维度；在实现前固定各组输入映射并验算全部可训练参数量。参数量匹配不等于有效容量完全一致，报告须保留此边界。当前旧四维拼接结果作为历史锚点，若新统计组结构变化需另名，不能冒充旧模型重跑。

固定已批准GHI损失/尺度、cohort、split、原始LR0.0003/QC1 LR0.001、batch2048、最多50epoch/patience8；不同时改R、loss、监督参考或筛选样本。首先固定小预算pilot，只有初步收益才按独立协议做成对种子稳定性，不自动打开test。

## 当前来源门禁

CPP源NC生产者→脚本/权重/归一化→实际AGRI输入→输出经纬度/时刻的绑定须先核实。优先获取抽查NC所属批次的最小生产包；不以目录邻接或文件名替代证据。无法获得来源时，不进入上述新训练。

IFS来源独立核对历史日期覆盖、空间范围、cycle/valid/step、SSRD差分所需前一步及发布/到达记录。当前发现的其他raw_nc年份目录仅为候选线索；不以文件mtime伪造release，不使用未来发布产品，不以补值掩盖缺步长。

工程根/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907。只在门禁通过并冻结实现后提交PBS；账户最多3个未结束任务和8GPU，优先单卡。用户未要求持续监控，保持关闭。
