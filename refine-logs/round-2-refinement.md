# 第二篇论文初始研究方案

## Problem Anchor
- 底线问题：固定 SimVP 确定性外推后，显式 COT 云光学表示能否提高湖南站点 +15～+240 min 辐照度预测精度，并与初始化时刻可用的 IFS 形成互补？
- 必须解决：区分 COT 表示收益、额外监督收益、模型容量收益和 IFS 收益，检验真实外推误差下的有效性。
- 非目标：不研究 diffusion、不改造外推骨干、不扩展到 D+1～D+3、不把湖北历史结果作为湖南证据。
- 约束：湖南独立时间划分；站点 GHI 仅作监督评价；优先复用合规权重并缓存外推；新增训练以小型站点头为主。
- 成功条件：同 cohort、同外推、同训练预算对照支持 COT 增益及其适用边界；IFS 互补单独判定，负结果不隐藏。

## 定位与最近邻
候选标题：Cloud-Optical-Thickness Representations for FY-4B Solar Irradiance Nowcasting with a Fixed SimVP Backbone and IFS Guidance。
中文：固定 SimVP 外推下融合云光学厚度与 IFS 的 FY-4B 辐照度短临预报。
研究定位为应用遥感的受控表示研究，不把简单多源拼接称为架构创新。COT 由同一 AGRI 计算，因此不宣称增加独立观测信息，而检验显式物理监督与表示的归纳偏置。
已发现最近邻：GOES-16 COT nowcasting (Remote Sensing 2025, 17, 2861)；Miller et al. 2018 satellite/model coupling；Applied Energy 的 satellite-derived cloud optical depth 物理引导预报。需完成最近邻全文核验后才能宣称独特机制。

## 方法
历史八帧经冻结的湖南 SimVP 得到未来十六帧 13 波段 AGRI。未来太阳几何由经纬度和时间计算，不由卫星真值提供。站点 16×16 patch 必须按湖南网格取值。
冻结 COT 反演器 R 将预测 AGRI patch 与太阳几何映射为 COT 场，输入波段顺序、单位和归一化须匹配。若无合规湖南权重，单独训练一个反演器；CPP 检索值是有质量掩膜的监督参考，不称绝对真值，缺测不能当晴空零 COT。
R 首先只在湖南训练期真实 AGRI/CPP 配对训练；验证期选模后冻结。所有站点头均在 SimVP 生成的预测 AGRI 域训练，避免只在真实未来图像上训练的部署失配。上游可在全 train 拟合，但其训练内预测与 held-out 表现差距须报告；必要时在 train 内做连续日期交叉拟合诊断，不用 test 调整。

基础表示 a=Encoder(predicted AGRI patch)；COT 表示 c 为 log(1+COT) patch 的均值、标准差、p90 和中心像元；太阳几何 g、lead embedding l 所有模型共有。IFS 表示 n 初版仅使用可用周期的 ssrd 区间平均通量及其晴空指数，减少多变量搜索。检索缺测率只作统一 QC，不让仅 COT 组得到额外云掩膜信息。
预测头 H 为两层 64 hidden 单元 MLP，输出非负 kt（softplus，不设 kt≤1 硬截断）；共同 AGRI encoder 为两层 3×3 CNN（16/32 通道）+全局池化。四组使用相同输入槽和模型参数，缺少的 c/n 用训练均值对应的标准化零占位，单独训练；另设充分调参的无 COT 头防止占位对照成为弱基线。
最小机制采用拼接 H(a,c,n,g,l)。只有简单拼接能证明 COT 价值才考虑复杂融合；不预设注意力、置信门控或严格 COT→GHI 单调规律。COT 不是可直接代入 Beer 定律的 GHI 全量衰减因子，散射与云边增强需要保留。
目标 kt=mean(GHI)/mean(clear-sky GHI)，损失为逐站/lead 等权的 kt MSE，验证选择 GHI RMSE。GHI=pred_kt×同窗口 clear-sky mean。

## 两个待验证主张
C1：在固定外推和 IFS 条件下，COT 物理表示对站点预测有稳定增益，且不能完全由额外参数或任意辅助监督解释。
C2：IFS 与 COT 在时效/天气上的互补可以通过相同样本四格对照估计；交互不显著则只宣称各自增量，不宣称协同。

## 核心实验
四格：A=AGRI；AC=AGRI+COT；AN=AGRI+IFS；ACN=AGRI+COT+IFS；太阳几何与lead全部相同。
对照：N=IFS校准头、C=COT+几何头、A-wide=容量控制、AN-aux=与COT监督相同但仅使用隐特征的辅助监督头、ACN-permute=测试时同lead/太阳高度层内COT置换（仅干预诊断，不当重训基线）。
诊断：R(real AGRI) vs R(SimVP AGRI) 的参考 COT 误差与下游误差，真实未来/参考CPP只用于单独标识的 oracle 诊断，不进主表；分薄/中/厚云、稳定/转变天气、0–1/1–2/2–4h，规则由train/val冻结。
主指标：等权站点RMSE均值；辅助 MAE/MBE、逐lead RMSE。按目标日期整块、两站一起做配对bootstrap CI，保留重叠初始化相关性，另做多日block敏感性；种子间变异单报，不能把种子当独立天气样本。
四格交互以样本平均平方误差定义 I=MSE(A)-MSE(AC)-MSE(AN)+MSE(ACN)，负值表示联合降误差超过相加预期；只有区间与方向支持才称协同。

## 训练及门禁
先审计湖南 SimVP/COT/CPP/IFS availability，旧湖北 SimVP 权重不得直接进入湖南主表。已有合规权重则复用，不因本方案重训骨干。IFS release_time无法确认则暂停完整融合主张，仅先做A/AC，legacy结果独立标识。
复用第一篇 M0 时间划分：train 2024-04-02～2025-06-30；val 2025-07-01～2025-09-30；test 2025-10-01～2026-06-30；完整8+16支撑不跨split，最终样本计数以新CPP/IFS覆盖审计为准。
单种子 pilot 只看train/val；通过后核心四格各3种子(42/43/44)。AdamW lr=1e-3、weight_decay=1e-4，最多50 epochs，patience=8；共同候选lr {3e-4,1e-3}，每组同搜索预算。batch从64 profile后提高到安全利用一张卡，固定有效batch或记录变更；满卡后才扩卡。
不凭经验虚报耗时。GPU小时=一次共享外推缓存+必要COT训练+训练头单次实测耗时×正式次数；保存吞吐、显存峰值和ETA计算。
成功门：C1主对比ACN vs AN及AC vs A，在未触碰test的validation先看方向与稳定性；正式冻结后统一test出表。负结果写清范围，不能切换测试日期追求收益。

## 写作
正文顺序：数据与任务→固定外推与COT表示→公平消融→结果→讨论→引言/摘要/结论。两篇共享数据合同与基线 provenance，但各自主张、结果和文字独立。中英文同步；服务器路径、哈希、修复记录只入工程文档。

## Round 1 修订：锚点与简化检查
Problem Anchor逐字保留。主机制仍为显式COT表示；不加注意力/门控/diffusion。根据审稿补充如下，以本节替代上文冲突的候选接口。
- 同监督对照为AL和ALN：共享同一个冻结R、同一CPP训练样本和损失，R倒数层全局池化后由固定种子的非训练正交投影得到4维隐变量；显式c也是4维。投影只用train统计做标准化，不新增可训练投影。AL对AC，ALN对ACN；AN-aux为旧称，不再使用。两组输入维度、站点头和搜索预算相同。
- IFS只保留区间通量除以同区间晴空通量的单一量，不同时输入同源ssrd与kt。每个init选择已发布且覆盖16个lead的最新周期；没有满足条件周期就标缺，绝不使用未来发布周期。相邻step差分后按15min窗口与原生区间的重叠时长求加权平均；不足覆盖标缺。记录为何某周期被拒绝。
- R固定候选为两级小U-Net，base_channels=16，输入13AGRI+三项几何共16通道，无不明metadata；输出log1p(COT)的softplus回归，masked Huber(delta=0.1)对log1p目标；训练50epoch、patience8、AdamW lr1e-3/wd1e-4，只以有效CPP训练/val选模。若复用模型合同不同，完整登记而非强改输入。
- CPP有效QC要求有权威产品schema解析质量位、填充值和物理范围；有效质量值须写入cot_contract.json（现在未知，禁止猜测具体bit）；统一SOZ<80度作为候选检索评估筛选，必须与官方有效角度取更严格者并在train/val冻结。该QC只限制R监督和检索诊断，不能限制主GHI队列。未解析schema不放行R训练。
- C1只称受控实证假设。新颖性全文核验、权重/CPP/IFS数据审计仍为未完成前置任务，不以审稿评分冒充通过。
- N保留主表强参照；C-only和置换为附录可选，减少训练负担。A-wide/AN-wide及AL/ALN为条件性必须：若要声称物理表示优势必须完成。
## 定稿补充合同
- 同监督控制必须成对：AL对AC、ALN对ACN，分别无IFS/有IFS；AL是用四维隐变量替换显式c，A/AN不变。每一投影种子11/22/33均运行同预算的头，对比显式c时报告全部差异，不挑选有利投影。结论仅针对预注册隐特征对照，不声称胜过所有可能隐表示。
- IFS固定同一产品/stream/grid/version合同；候选周期按release_time降序、cycle init_time降序、product_id字典升序确定唯一选择。同周期重复文件若内容哈希不同则FAIL，不能任意选一个。只有覆盖全部目标区间的周期才可选。
- 几何通道为cosSOZ、cosRAA（无量纲）与day_mask（二值）；前两项使用train统计标准化，day_mask保持0/1。13 AGRI使用train期各波段统计，先恢复SimVP输出的物理单位再用R的norm，不能二次归一化。目标太阳几何由站点/像元坐标和target确定。
- C1主要是应用表示对照结论；检索监督、隐变量投影、汇聚算子都影响表示，若无法完全分离则明确称“显式COT统计管线相对预注册同监督表示”的效果，不做普遍物理因果断言。
