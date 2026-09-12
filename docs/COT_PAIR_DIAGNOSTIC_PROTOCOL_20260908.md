# 固定目标的COT输入误差传播诊断协议

在读取所选COT预测/参考误差前冻结本协议。此项是有限validation诊断，不是新训练、正式test或GHI收益对照。旧8次A/AC结果和全部cache保持不变。

## 选择与前置条件

使用inventory的原样64个目标：每站全部R validation唯一target按UTC升序，第floor(k×(n−1)/31)、k=0..31行。selection SHA为`8a06a7f06cf3339cd8f067111d5397c173ba63868598c6904348362994b08318`。其中58目标有872个可用target/lead配对；152个缺lead及6个完全无forecast的目标全部列出，不以覆盖、误差、CPP值或天气更换目标。不改变GHI主cohort。

必须核验 `cot_pair_coordinates_20260908.json`：58目标原文件/修正sidecar/S grid patch坐标和身份绑定通过；`cot_pair_geometry_20260908_v2.json`：872配对几何和同target跨初始化几何通过。两个实际检查均为最大差0。v1检查工具的UTC字符串格式失败保留，它未访问数值；不能据此修改数据时间。若任一来源、selection、hash、空间或几何合同变化，停止并另存错误，不静默绕过。

## 两条输入，仅改变AGRI

- R(real)：已完成R pack的相同target/站点13 AGRI + 原3几何。
- R(forecast)：相同target/站点/lead的冻结S物理AGRI + cache三几何。
- 几何须再次逐值验证一致。相同冻结R权重、通道顺序、train norm用于两支；沿用原R合同，将归一化后非有限输入置0（训练均值填补），两支一致执行并记录数量，不clip或按此删样本。
- 两支均重新用CPU float32、2线程、batch32、eval/no_grad前向，不开启autocast/GPU、不训练。最后不足32的batch重复末项填满，预测后丢弃补齐项，以保持相同执行形状；真实端58输出仅算一次后复用。保存CPU/MKLDNN/runtime实际设置。
- 不用旧GPU缓存COT作为本次forecast预测，以免将数值后端与输入差异混合。207626的旧cache max0.001033超限仍保留，本诊断不关闭该门，也不单独证明TF32原因。

## 参考、指标与保存

参考为固定R pack `target.npy×100`（physical COT），先按原R训练代码以float32乘100，再转float64用于评价；沿用原 `mask.npy`并重新核验其来源及SHA。只读取58个匹配validation参考行，目标与mask在两支完全一致。原mask反映修正CPP参考范围/白天/角度等既有规则，不新加阈值、不改mask、不按预测误差筛选。先保存每一pair的两支log1p预测、COT参考、mask以及station/init/target/lead/R行/sequence键，再从保存数组独立NumPy float64重算。

报告所有2站×16 lead的n_pair、n_unique_target、有效像元数、physical COT MAE/RMSE/bias和log1p MAE/RMSE；缺配说明保留。另报58个唯一target的R(real)参考误差，避免将跨lead重复目标当成独立真值。整体汇总若重复target/lead必须注明权重，不能与唯一target汇总直接混比。不做显著性、天气挑选或模型选优，不计算GHI性能。

输出独立目录 `audits/cot_pair_diag_20260908_v1`，完整预测为`cot_pair_predictions_20260908_v1.npz`，结果为`report.json`，附SHA/protocol/source/runtime provenance。已有输出拒绝覆盖；来源或数值失败保存错误证据，不因失败更换目标。

## 允许的结论

最多说明该固定样本、同CPP参考掩膜下，外推输入相对真实输入的R检索误差如何随lead变化。R曾在此validation选模，采样也是描述性而非盲测；CPP为参考产品，其生成checkpoint/source绑定未完全闭合。不称无误差物理真值，不称部署主模型，不证明AC负收益唯一机制或普遍COT无效。结果不触发扩大网络/损失/超参数搜索；只对明确实现错误考虑一次独立命名且公平复跑的修复。
