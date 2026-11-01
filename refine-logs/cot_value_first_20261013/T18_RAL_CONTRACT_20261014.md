# T18：R-AL 冻结反演器隐特征对照合同

日期：2026-10-14 BJT。状态：**已执行完成（validation-only）**。候选合同参数与训练预算已固定；特征bank、见证、profile、六组训练和独立评价均已完成，结果见文末执行更新。本文件执行T17中被触发的表征混杂控制，不能改写已完成A/AC/AP结果。仅使用月度train/validation，test载荷关闭。

## 问题与比较

同期真实AGRI→冻结R→GHI的AC相对A有稳定验证增益，但这个增益可能来自COT显式物理图，也可能来自R的空间感受野/CPP监督特征。AL用同一冻结R的隐藏特征替代显式COT，并保持相同GHI标签和AGRI/几何主干。

比较对象：已有A、AC、AP与新增AL。AC输入3×3 `log1p(COT)`；AP输入由GHI AGRI主干生成的3×3学习特征；AL输入冻结R的最终解码特征。AL只测试“R隐藏表征是否也能提供下游信息”，不证明该信息是纯物理COT，也不把CPP参考当真值。

## 冻结层和张量

月度COT U-Net的 `dec1` 是最终1/1分辨率解码块，`head` 随后以1×1卷积映射到COT。冻结R并置于eval模式，在`dec1`输出、进入COT `head` 之前提取特征：

```text
R input: [B, 16, 16, 16]  (13 AGRI + 3 solar-geometry channels)
dec1 activation: [B, 32, 16, 16]
station crop: rows/cols [7:10, 7:10] -> [B, 32, 3, 3]
```

按 `(station,target_time_utc)` 与现有GHI pack精确连接。不得用未来CPP/CLP筛行或改特征；这属于同期真实AGRI反演合同，不是forecast输入。

## AL头和参数匹配

保留共同AGRI+geometry主干、展平读出、损失、优化器、训练行、顺序和30 epoch预算不变。AL辅助编码器采用与AP完全相同的 `Conv2d(32,4,1)+SiLU`，输出 `[B,4,3,3]` 后与32通道主干特征拼接。只有辅助分支输入源不同（R decoder feature vs AGRI trunk feature）。因此AL与AP的头部参数量相同；AC保留其 `Conv2d(1,4,1)+SiLU`，不得为了表面精确匹配另加无效参数。按现有结构推算（geometry=2、trunk=32、aux=4）：A=22,177、AC=24,489、AP/AL=24,613个参数；AP/AL比AC多124个、约0.51%，完整实现后须由目标PyTorch运行时重新打印确认。

训练只更新GHI头；SimVP、R及特征提取全冻结。每站seed42/43/44，共6个AL训练；与相同站点/seed的A、AC、AP配对。不得将AL当作替换AC的主方法；主要解释看AC−A与AL−A、AC−AP与AL−AP的三seed及日期块CI。

AL特征逐通道均值/标准差只从train split拟合，随后原样用于val。保留所有共同有效GHI样本与mask，不因CPP缺测删除行。输出feature manifest，记录R checkpoint/code、AGRI来源、样本键、train-only归一化和特征数组哈希；val只用于选模/评价，test不读取。

## 开始训练前的单批见证

实现后先在目标服务器PyTorch环境完成一个batch的前向/反向见证：特征shape、有限值、站点中心裁切位置、GHI loss梯度可到达AL辅助编码器、且R/SimVP梯度为None；验证键覆盖与输入特征缓存重载一致。再用一轮profile测单卡吞吐和显存。完成后更新本合同中的实际参数数、checkpoint/hash、profile与数据manifest；见证通过并检查全队列不超过3个未结束PBS任务后，再串行运行两站三seed。此处不提交训练。

## 停止与解释

- 若AL≈AC且两者均优于A/AP：下游增益不能归因于COT物理语义，R学习到的表征/监督可能足以解释。
- 若AC优于AL且优于AP：支持显式COT表示相对这些容量控制有额外价值，但仍限于当前R、3×3头、train/validation合同。
- 若AL/AC都无稳定差异：结论收缩为COT与R隐特征的可分辨增量不足；不扩宽网络、不搜层、不打开test。
- R-AL无论结果如何，均不能替代B3预测COT精度，也不放行B4 forecast-GHI训练；B4仍由B3结果和总计划单独决定。

执行更新（2026-10-14）：R-AL特征bank现已完成，见tracker最新T18更新。实测AL/AP均为24,613个参数；梯度与单卡profile作业213867已排队。本文档原状态和“尚未实现”表述为bank完成前记录，现应按tracker最新状态理解。正式训练仍须等待见证与profile。

执行更新（2026-10-14）：特征bank完成，真实batch梯度见证和单卡profile均通过，详见tracker最新T18更新。首版见证因匹配初始化的辅助读出列为零而在第一步观察到预期零编码器梯度；修正为首步读出更新后再检验编码器梯度，PBS213868 exit_status=0。profile显示batch=2000满足单卡稳定性并提供每epoch至少12次更新；为保持A/AC/AP训练合同，六组AL正式训练继续使用2000。六组训练PBS213869已排队。T18最终科学判断须等统一验证集预测与配对统计完成。

T18执行及最终结论已完成：见`results/T18_RAL_RESULTS_20261014.md`。最关键限制是AL和AC均来自同一冻结R/CPP监督，AL近似或略优于AC的验证结果不构成COT与R特征的等效性检验，也不构成预测GHI成绩。

