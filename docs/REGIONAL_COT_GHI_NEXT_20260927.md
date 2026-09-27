# 区域 COT 到单站 GHI：2026-09-27 执行交接

目标保持湖南四里、竺家两个单站的 +15…+240 min、15 min 区间 GHI/kt。先区分反演误差、未来云轨迹误差、COT 对 GHI 的实际价值，再决定是否训练 E_region。此文件为工程记录，不是论文结论；test 未读取。

## 1. 已完成：两版区域 R 的同像元验收

`scripts/audit_regional_r_pair_20260927.py` 用 PairDataset 的确切分片/记录顺序重建 3,488 个 val 时刻，逐帧核对两次保存预测的 CPP 参考值、有效掩膜，以及配对缓存的真实 CPP 行。没有调用未来预测图像，也没有把 R 输出当真值。对比 batch32/50epoch 上限与 batch512/680epoch 上限的已完成实验；两者优化过程不同，不能把差额单独归因于 batch。

同 212,757,698 个有效像元，区域 COT RMSE 从 3.725 降至 3.246。按 72 个 BJT 验证日做 2,000 次配对 bootstrap，区域 RMSE 差值（新−旧）的 95% 区间约为 [-0.532,-0.426]。这仅描述当前 CPP 参考拟合差异。

CPP 邻接有效像元绝对 COT 差 ≥3 的梯度 MAE 从 2.706 降至 2.460；该定义只是描述性边界诊断，不是物理云掩膜或边界位置真值。预测梯度绝对均值约 3.737/3.730，不能只凭平均梯度宣称更锐利。

薄云、中等云、COT10–30 和30–60的条件 RMSE 改善；近零（0≤COT<0.1）从1.242到1.253略退化；COT60–100从7.927到8.564退化，新模型此区间bias约−5.127。预测>100比例为0不是厚云压缩证据，因为缓存CPP范围本来限定0–100。保留这两项条件限制，采用新R作为下一轮**探索性冻结候选**，不再追加R轮次。

产物在 `results/regional_station_cot_20260927/r_pair_audit/`：summary.json、selected_cases.json、两站各4组 PNG/PDF（最差、改善、退化、典型）；各图显示中心前后15min的真实AGRI C03/C02/C01增强伪彩、CPP、两版R和误差。RGB以中心帧各通道2–98%范围对整图作线性拉伸，并在三时刻共享，未平滑。十字始终是同一站点像元；缺时刻明确留空。真实RGB仅作反演诊断，不进入可部署未来输入。

## 2. 已实现并提交：区域输入与GHI三组pilot

PBS `210854.tc6000`，`cot_region_ghi`，node22、1×A800 80GB、12h。2026-09-27 13:18核查仍为Q；尚未Python启动、profile、打包或训练。提交前账户另有1个运行作业占7张GPU；加入本作业后共2个未结束PBS、申请GPU共8张，符合任务/卡数上限。没有终止其他任务。

一次PBS串行完成：

1. 合成输入上的因果/梯度kernel见证：改变目标之后COT不改变输出；+240min输出仍对历史8帧有非零梯度。它只验证代码，不是预测精度证据。
2. 构建相同512 train/128 val序列的区域pilot，依据既有GHI cohort的BJT月份和manifest排名确定性抽取，不按CPP存在性或GHI大小筛选，保存原GHI row_ids，不修改正式全量cohort。
3. 保留64×64站点原分辨率AGRI/COT，以及完整256×256降采样后的32×32区域二维网格。区域降采样为每个8×8空间单元的均值，位置仍保留；它不是整图均值/p90。COT存储域为log1p。
4. 冻结修复版湖南 geometry24 SimVP和新区域R；可部署图像始终来自SimVP。真实未来AGRI仅在独立oracle COT分支读取，不混入任何组的图像特征。
5. 同参考覆盖上核对未来16时效的静止/历史C13相位相关搬运/SimVP→R，并单列oracle参考误差。搬运不循环回卷；记录出域覆盖。四种方法的诊断使用同一目标CPP有效掩膜与搬运域内支持交集，不能当作完整GHI cohort评价。
6. 同架构、相同seed/训练顺序/损失/验证键分别从头训练：零COT、预测COT、未来真实AGRI→同冻结R的COT。oracle是非部署诊断，仍含反演误差，不称物理真值。

H有区域/局地图像分支、区域/局地COT轨迹分支和站点/时效embedding。COT分支四层因果3D卷积，时间dilation=1,2,4,8，感受野31帧，覆盖8历史+16未来的可用前缀；末两层保留空间步长1。二维特征网格展平后用可学习读出，未对完整COT做全图平均读出。不同组仅COT输入来源不同。

SimVP推理batch按1/2/4/8/16/32单卡profile，GHI训练batch按16…2048上探，选择测试档位内显存峰值<75%的最大batch，同时记录吞吐。GHI max epoch至少对应600次优化更新；patience至少对应120次更新，三组共用。profile后恢复seed并重建模型，不把profile训练混入结果。银行复制到节点本地盘后训练。

脚本：`build_regional_cot_ghi_pilot_20260927.py`、`train_regional_cot_ghi_pilot_20260927.py`、`smoke_regional_cot_ghi_20260927.py`、`run_regional_cot_ghi_pilot_20260927_node22.pbs`。本地与实际服务器Python语法检查通过；新鲜gpt-5.6-sol xhigh代码审查发现两层时间卷积不足覆盖历史，已改四层并复审通过。审查为same-family/provisional，不替代真实kernel见证和数值验收。

## 3. 结果后的决策

保存逐样本预测后先核对三组row_ids/真值/站点/lead完全一致，重算两站等权RMSE、MAE、bias、16lead，并做BJT初始化日配对bootstrap。

- oracle可靠改善而预测COT不改善：先处理未来COT轨迹，按基线失败模式决定E_region。
- 两种COT均改善：扩大同键筛选和种子验证，再决定E预算。
- oracle也没有稳定改善：优先检查H对二维位置/时间轨迹的利用、太阳几何和AGRI信息重复，不盲目继续提升R。

本次是小规模单种子筛选；不得将其解读为正式全量GHI提升。CPP历史生产链/独立物理QA仍未闭合，IFS发布时间门禁未解除；不读取test，不自行恢复持续监控。

远端根目录：`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/regional_station_cot_20260925/`。数据bank为`regional_ghi_pilot_bank_20260927_v1`，训练输出`regional_ghi_pilot_seed42_20260927_v1`，PBS日志`regional_ghi_pilot_20260927_v1.pbs.log`。排队结束前不能宣称训练已启动。

## 2026-09-27 14:02 排队状态复核

Luna一次性状态核验及主线程SSH核对一致：PBS `210854.tc6000` 仍为Q态，申请node22单GPU，尚无`exec_gpus`分配字段。当前账户列表为210853运行、210854排队，共2项。PBS日志、kernel witness、pilot bank状态/profile、GHI训练状态均不存在；所以Python未启动，未进入profile、bank构建或训练。无运行错误日志。继续等待调度，不重复提交、不终止其他作业。

## 2026-09-27 14:08 排队原因与改派

作业210854最初指定node22；实时`qstat -f`显示无hold/comment错误。节点快照显示8张GPU均已分配：其它用户的210176占GPU0，运行中的210853占GPU1–7；node22另有维护备注`gv maint`。因此即使PBS节点概况显示`state=free`，指定A800仍没有可分配GPU，作业保持Q。

为使用node21空闲GPU而未重复提交，执行`qalter -l nodes=node21:ppn=8:gpus=1 210854.tc6000`。14:05:22 PBS prologue已启动，14:05:25分配node21 GPU2，实际为A100-SXM4-40GB。作业已运行，单卡资源和模型算法合同不变；脚本profile按实际卡显存选batch。14:07核验因果kernel见证通过；区域pilot bank处于train打包，21/512序列，test_used=false，无报错。SimVP profile在batch 1/2/4/8/16/32测得峰值约0.52/1.02/2.00/3.96/7.87/15.70GiB、吞吐约0.34/33.25/51.43/58.51/65.30/66.45序列/s；所测范围内选择batch32。初始脚本名含node22是提交时合同，当前实际分配由qalter确认为node21；后续输出须以qstat实际资源为准。

## 2026-09-27 14:56 区域 GHI 配对pilot完成

PBS 210854于14:56:28完成，PBS tracejob记录Exit_status=0，总运行51分06秒；输出状态为COMPLETE_MATCHED_REGIONAL_GHI_PILOT，test_used=false。因果kernel witness通过，512 train/128 val固定序列的bank构建完整。H训练batch profile在A100 40GB上测16…2048，选择batch2048，峰值13.66GiB；1024至2048吞吐仅约1.4%增加。每组预算上限100轮、最多600次更新（不是实际完成更新数），三组独立从头训练，输出三组逐样本预测。

站点等权RMSE（W m⁻²）：零COT 146.980，SimVP预测COT 146.410，未来真实AGRI经同一R得到的oracle COT 142.597。相对零COT，预测COT差值为−0.570，BJT日配对bootstrap 95%区间[−7.872,+6.051]；oracle差值−4.383，区间[−10.599,+1.278]。两个区间均包含0。单种子、按月份抽取的pilot说明当前结果尚不足以确认COT带来稳定GHI增益；oracle方向较有希望但仍不确定，且oracle不是可部署输入或独立CPP真值。该结果仅用于决策，不打开test、不据此扩大E_region训练、不添加IFS。

完整摘要、逐样本预测、逐站点/lead验证指标和batch profile已复制至本仓库results/regional_station_cot_20260927/ghi_pilot/；远端完整输出位于此前登记的regional_ghi_pilot_seed42_20260927_v1。当前PBS列表中210854已完成并清除，只剩原运行任务210853。

## 2026-09-27 目的导向复核：完成程度与下一步

重新读取远端训练日志和保存的future_cot_baselines.json：三组已完成，当前队列没有本区域COT任务；另有hn_shade_eval_te运行，不能当作本实验继续训练。

实际训练：zero完成29轮/174次更新，最佳第9轮/54次更新；forecast和oracle均完成25轮/150次更新，最佳第5轮/30次更新。epoch原始日志为0-based，此处改为1-based。minimum-updates=600当前仅提高max epochs，early stopping仍可提前退出，因此并未保障600次更新。保留原始结果，后续独立版本应明确最小更新与早停关系，记录训练损失、梯度及逐站验证曲线；不能把早期最佳和后续波动归因于batch本身。

验证pilot的CPP共同有效域区域COT RMSE，依次为persistence/transport/SimVP-R/oracle-R：+15min=4.324/4.358/6.924/3.356；+60min=7.754/7.855/8.309/3.294；+120min=9.690/9.886/9.164/3.183；+240min=11.957/12.221/10.892/3.216。SimVP-R在前5个时效差于持久性，在+90至+240min优于持久性；当前C13相位相关平移基线全部时效差于持久性。上述只代表pilot区域、CPP参考及transport共同有效域，不能推广为站点GHI收益或否定所有运动方法。

GHI forecast相对zero：四里RMSE降低7.087、竺家升高5.946 W/m²；oracle分别降低12.093、升高3.326。站点相消和验证波动均真实存在，原因仍待诊断。oracle是实际未来AGRI经同一R反演，不是真实CPP，也不是可部署输入。三组均独立训练，差异同时包含学习响应；不能把oracle减forecast直接当成纯上游误差。

建议顺序：先复用现有bank完成逐站/逐时效/真实云事件与训练收敛诊断；制定共同稳定训练设置并修正更新预算记录；再做公平扩大样本和种子验证。只在未来COT质量确为瓶颈、可靠COT在稳定头中有价值时推进E_region。R_region已有探索性改善，但极厚云和CPP来源仍有限制。IFS、独立test和论文主结论尚未放行。本轮仅核查和记录，未提交新训练。
