# 研究方案：固定SimVP外推下的COT表示与IFS互补

## Problem Anchor
- 底线问题：固定 SimVP 确定性外推后，显式 COT 云光学表示能否提高湖南站点 +15～+240 min 辐照度预测精度，并与初始化时刻可用的 IFS 形成互补？
- 必须解决：区分 COT 表示收益、额外监督收益、模型容量收益和 IFS 收益，检验真实外推误差下的有效性。
- 非目标：不研究 diffusion、不改造外推骨干、不扩展到 D+1～D+3、不把湖北历史结果作为湖南证据。
- 约束：湖南独立时间划分；站点 GHI 仅作监督评价；优先复用合规权重并缓存外推；新增训练以小型站点头为主。
- 成功条件：同 cohort、同外推、同训练预算对照支持 COT 增益及其适用边界；IFS 互补单独判定，负结果不隐藏。

## 方法定位与贡献边界

本篇为应用遥感的受控表示研究。检验同一预测AGRI中显式COT物理统计的归纳偏置，而不是宣称增加独立观测信息。最近邻已有COT短临和卫星/NWP耦合，使用COT本身不构成首次创新。独立贡献候选是固定学习式外推误差下，显式COT管线的可用性、误差传播及其与严格可用IFS的受控增量。最近邻全文核验未完成，所有创新主张尚待裁定。

两条主张：C1为显式COT统计管线相对无COT及预注册同CPP监督隐特征的效果；C2为COT/IFS分别增量和交互。C1不概括为优于一切隐表示或物理因果证明。C2不成立仍可报告COT结果，不隐藏IFS负收益。

## 冻结部件与最小方法

1. S：符合湖南相同split的冻结SimVP，历史8帧输入、未来16帧13波段AGRI输出，主时效+15～+240 min。优先共享已有权重，缺合规资产时先登记审计结果，再安排唯一一次基线训练。
2. R：冻结COT反演器，逐站预测AGRI 16×16 patch与目标太阳几何输入。新训练的候选R为两级U-Net、base_channels16；输入13 AGRI+cosSOZ+cosRAA+day_mask。cos量无单位，使用train统计归一化；mask保持0/1；AGRI先反归一到物理量再用R的norm。禁止未知metadata隐式追加。
3. R输出非负log1p(COT)，使用有效CPP检索参考的masked Huber(delta0.1)监督。AdamW lr1e-3/wd1e-4、最多50epoch、patience8，按val检索损失选模后冻结。CPP质量位、填充值、范围、相态、官方角度限制必须先解析写入合同；候选SOZ<80度与官方限制取更严格者，只控制R监督/检索诊断。缺测不能当零COT，不能按未来CPP有效性筛选GHI主表。若复用不同结构的合规R，以登记合同为准，不能强改checkpoint。
4. a：两层CNN(16/32通道、3×3卷积)与全局池化编码预测AGRI。c：log1p(COT)的均值、标准差、p90、中心像元共4维。g为共同太阳几何，l为共同lead embedding，站点标识全组一致。
5. n：单一IFS区间平均辐射/同区间晴空均值，不同时再输入同源SSRD。共同产品合同内选择已发布且完整覆盖16目标区间的最新周期；release降序、cycle init降序、product_id升序唯一排序，重复冲突hash报错。同cycle相邻step差分后以原生区间和目标15min窗口的重叠秒数加权；缺步标缺，不跨缺口补值。小时内等通量是假设，不是新增15min观测。
6. H：拼接a/c/n/g/l后两层64hidden MLP，softplus输出kt，不强制≤1。各组预留相同输入槽、缺因子使用标准化零，独立训练。共同有效batch先profile后冻结，AdamW lr{3e-4,1e-3}/wd1e-4、最多50epoch、patience8、相同val搜索预算。损失为站点/lead等权kt MSE，val以站点等权GHI RMSE选模。

主标签kt=mean(GHI)/mean(clear-sky GHI)，同区间回算GHI。所有头在S生成的预测AGRI域训练。上游训练内预测和验证外推的难度差需诊断；严重失配时先用train内部连续日期分块交叉拟合小子集检查，不用test调方案，不默认追加大型重训。

不引入门控、注意力、diffusion或严格COT→GHI单调约束。物理解释必须保留散射、云边增强及检索误差，不把GHI整体等同Beer定律衰减。

## 主验证与归因

四格A/AC/AN/ACN固定S/R、同cohort、同预算、3种子42/43/44。N校准IFS与训练期climatology为强参照。容量wide对照防止无COT组过弱。
AL/ALN使用与AC/ACN完全同一个冻结R的倒数层全局池化特征，经固定非训练正交投影为4维，替换显式c。投影种子11/22/33预注册，均报告，不根据val或test选择；train统计标准化。分别与AC/ACN比较，无额外CPP监督差异。该实验仅排除所测试的监督/表示替代解释，不穷尽所有隐表示。
C-only、COT置换放可选附录；真实未来AGRI/CPP只做冻结后的oracle域偏移诊断，不参与模型选择，不称严格上界。

COT层面比较R(real AGRI)/R(predicted AGRI)对同target有效CPP参考误差；GHI层面保留四格同完整cohort。分时效/云厚度/天气样本量和失败边界必须报告。
主要终点为两站各自全16lead RMSE的等权均值；配对target日期bootstrap与3/7日块敏感性。主张检验与阈值见EXPERIMENT_PLAN，主对比ACN-AN，支持对比AC-A，同监督比较另列。MSE交互I=MSE(A)-MSE(AC)-MSE(AN)+MSE(ACN)，负且CI排除0才解释为超加性。

## 资源与启动条件

P200资产/CPP/IFS/cohort→P201smoke和profile→P202四格val pilot→P203归因与最近邻→P204正式三种子→P205test。COT和IFS审核独立，IFS缺记录只阻止融合分支，不阻止A/AC。
预测cache共用一次，优先站点patch，小头训练替代重复全骨干训练。先优化一张卡再扩卡。总预算只能在实测后填：共享cache+必要上游训练+各头次数×单跑耗时+评价成本。当前没有真实GPU小时或完工ETA。

数据、论文结构、先前test暴露与两篇重复性按工程文件处理。英文/中文正文同步，证据不足不预填结论。当前可进入审计和实现，不等于权重可用、创新已证实或正式实验已放行。
