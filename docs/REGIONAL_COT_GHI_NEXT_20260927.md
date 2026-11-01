> Current status (2026-11-02): this is a retained historical planning/tracking document. Use [the repository index](../docs/REPOSITORY_INDEX.md) and [current model route](../docs/GHI_HEAD_ROUTE_20261102.md); old pending jobs and mandatory isolation steps below are not current instructions.

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

2026-09-27 四里厚云/非零运动核查完成：现有128条验证候选中124条有站点参考；63个CPP>=30记录中预测COT<10有9个，oracle为0；漏报集中3条序列。满足非零运动、有厚云且至少8有效时效仅1条。真实/预测RGB同通道共同拉伸显示预测空间结构平滑及厚云增强缺失。完整源数据、方法限制及下一步见results/regional_station_cot_20260927/sili_thick_motion/README.md。本轮不训练，不打开test。

## 2026-09-27 四里输运扩样本与厚度演变试验门禁

将固定BJT月份分层验证样本从128扩至512，选样未用CPP可用性、COT大小或GHI大小；冻结SimVP/R生成512条预测轨迹，test_used=false。PBS 210952完成，node22 GPU7 A800 80GB，推理用时约22.1分钟，峰值分配15.7GiB。纯输运评估由CPU完成，所有方法按lead使用相同CPP有效像元及所有平移方法的共同域支持；+240分钟共同域覆盖率约62%。

四里486/512条序列有CPP站点像元，55条至少有一个CPP COT≥30的时刻，共205个重叠序列时效记录。最新两帧相位位移取整为整数时79.3%显示零；保留亚像元后仅32.0%小于0.1像元/15min。七帧中位估计的近零率约32–34%。相位相关峰值均值0.00129，未校准为置信度，不能据此认定估计的位移真实。

在205个厚云站点记录（CPP≥30）中，预测COT<10的数目：SimVP→R 67，持久性56，32×32局地七帧输运48。输运相对SimVP将36条从<10移至≥10，同时17条反向恶化，净减少19条；对应12条挽回序列和8条新增漏报序列，但输入窗口重叠，不能当作独立事件。厚云时效站点RMSE：SimVP→R 28.33、32×32输运25.21；CPP峰值时刻平均绝对误差24.74和22.50 COT。全64×64图在+240分钟RMSE分别为11.54、12.18，站点像元RMSE为10.61、11.22，说明厚云条件的站点收益没有转化成总体图像误差改善。

云掩膜COT≥30的16时效平均IoU：32×32输运0.0317、持久性0.0286、SimVP→R 0.0238；有预测/参考厚云对象的平均质心差约23.31像元（输运）和25.56像元（SimVP）。绝对重叠仍低，近时效及+240分钟也非每项指标都更好；因此只把它视为弱的相对位置改善信号。CPP仍是卫星反演参考产品，不是独立云物理真值。

据此启动小规模“输运+厚度演变”门控试验：模型以历史R-COT、历史C13的32×32七帧亚像元位移、冻结SimVP→R的未来COT为输入，先平移历史COT，再学习厚度/演变残差；训练标签为CPP COT，推理时不读未来CPP或未来AGRI。512 train/512 val固定分层样本，最多40 epoch并按验证厚云误差早停；比较SimVP→R、持久性、纯输运和学习后输出，同时检查厚云漏报、COT RMSE/IoU、晴空误报。只作为验证pilot，不开test、不声称GHI改善。

脚本 `build_sili_cot_dynamics_bank_20260927.py`、`train_sili_transport_thickness_pilot_20260927.py`、`smoke_sili_transport_thickness_pilot_20260927.py`、`run_sili_cot_dynamics_pilot_20260927.pbs` 已执行。PBS `210963.tc6000` 于2026-09-27 22:46启动，node22 GPU7（NVIDIA A800-SXM4-80GB）；训练银行512/512 train与512/512 val均完成，模型9轮后按厚云RMSE早停，best epoch=1，日志写入完整标记并生成checkpoint。2026-09-28复核时作业已不在活动队列，scheduler历史接口不再提供Exit_status；完整脚本标记、`training_complete.json`、best权重与epoch曲线均齐全，判定正常结束。当前活动队列仅210935运行。test未使用。验证显示全场COT RMSE/MAE和晴空误报率改善，但厚云RMSE及COT≥30 IoU差于纯输运/持久性；不据此宣称外推或GHI提升。完整数值和限制见 `docs/SILI_TRANSPORT_THICKNESS_PILOT_RESULTS_20260928.md` 与 `results/regional_station_cot_20260927/sili_transport_thickness_pilot/`。

## 2026-09-28 四里输运+厚度验证诊断完成

修复并完成只读的64×64同掩膜验证审计（PBS 210984；此前两次尝试因审计脚本切片错误失败，未接受其产物）。最新验证中，24,355,048个CPP有效像元时效记录里，21,153,251个处于所有方法共用的输运有效支持域。学习残差在全场RMSE为9.056，低于step-0纯输运9.520；但CPP COT≥30条件RMSE为28.279，高于纯输运26.504，COT≥30 IoU也更低（0.1664 对0.2411）。站点上204个厚云时效同样退化：RMSE 28.82 对25.16，按初始化日配对bootstrap的95%差值区间为[+1.38,+5.55]。收益主要来自低COT参考像元；没有据此声称GHI改善或真实云物理精度提升。

已生成按统一支持域遮罩的代表性云增长/消散/持续厚云/近零案例图，以及16时效、厚度分层曲线：`results/regional_station_cot_20260927/sili_patch_case_audit_20260928_v3/`。图和指标中的CPP均为卫星反演参考产品。

## 2026-09-28 低COT额外惩罚单因素消融完成

PBS `211063.tc6000` 已以 `Exit_status=0` 完成。固定seed42，两臂各512条训练与512条验证、8轮/512次更新；唯一差别为是否使用CPP COT<5处的额外false-COT惩罚。去掉惩罚后，训练模型的最低厚云RMSE由控制臂28.279降至26.856，厚云漏报数由371,192降至328,427；但总体RMSE由9.056升至9.353，近零参考误报率由3.11%升至3.69%。两者仍未胜过step-0输运厚云RMSE 26.504，因此两个训练臂按预先约定均选择step 0。此为单seed pooled validation诊断，不含显著性区间；作业未保存未选epoch权重或逐像元预测，不能补算未选学习模型的站点/逐时效配对bootstrap。详见 `docs/SILI_LOW_COT_PENALTY_ABLATION_RESULTS_20260928.md`。CPP仍是卫星反演参考，test与GHI-head训练保持关闭。

## 2026-09-28 完整历史 COT 信息筛查与神经 pilot

为判断厚云误差是否主要来自当前残差头只读最近两帧 COT，在固定512 train/512 val bank上完成低成本线性条件信息探针。相同输入包括冻结 SimVP→R 未来COT、最后COT、末两帧差值、局地运动和lead；full8额外加入更早六帧。使用同一 CPP-valid ∩ transport-support掩膜，固定16×16局地网格和站点像元。

验证网格厚云 COT≥30 RMSE 为 recent2 28.991、full8 28.948；站点像元为29.788、29.656，差异很小。全COT RMSE差 −0.007，厚云IoU反而从0.1502到0.1490，站点厚云仅204条记录。该线性探针未给出历史轨迹具有明显增量的证据，但不能排除非线性神经表示能够利用轨迹。CPP仍只是卫星反演参考，样本窗口和像元不独立。

PBS 211080.tc6000（normal CPU）已完成，summary、逐样本预测与可视化归档于`results/regional_station_cot_20260927/sili_history_information_probe_20260928_v2/`；分析记录见`docs/SILI_HISTORY_INFORMATION_PROBE_RESULTS_20260928.md`。账户中另有完整历史神经网络 pilot PBS 211083.tc6000（node22、单A800）运行，不重复提交。该pilot的最终 step0/short-history/full-history 配对结果和初始化日bootstrap尚待完成；test与GHI头保持关闭。

## 2026-09-28 容量匹配全历史 vs 重复末帧 COT 对照完成

针对先前两帧/全历史pilot参数量不同、控制臂并非重复末帧的问题，另立同参数量15通道头和完全相同的零残差初始化，唯一改变为八槽输入“真实历史八帧”或“最近COT重复八次”；两臂都保留同一末两帧差值及SimVP→R、纯输运、运动和lead输入。最终PBS `211099.tc6000` 在node22单A800正常结束（日志有 `SILI_FULL_VS_REPEAT_LAST_COMPLETE`），batch512、每臂256次更新，最大PyTorch allocated 39.74 GiB；初始化数值见证两臂误差均为0。

同一CPP-valid∩transport-support域有21,153,251个64×64 patch pixel-leads，72个初始化日bootstrap。两臂均按预定厚云RMSE准则选择step-0纯输运：patch COT≥30 RMSE 26.504、站点25.163。最佳学习快照中，repeat-last对应patch/站点厚云RMSE 27.209/27.606，full-history为27.143/27.609；full−repeat配对95%区间分别为patch[−0.161,+0.029]、站点[−0.316,+0.293]。Full−transport站点差为+2.447，区间[+0.573,+3.977]。历史八帧没有稳定增益，学习修正降低低COT假云率却仍加重厚云RMSE，因此不训练GHI头、不打开test。完整结论、指标口径、限制与来源见 `docs/SILI_FULL_COT_HISTORY_PILOT_RESULTS_20260928.md`；原训练快照和逐时效曲线归档于 `results/regional_station_cot_20260927/sili_full_vs_repeat_last_history_20260928_v5/`。

## 2026-09-28 历史帧是否真正进入模型：只读反事实审计

PBS `211169.tc6000` 已以Exit_status=0完成同一full-history最佳学习checkpoint的512条验证序列只读前向。真实较早六帧与末帧平均相差4.515 COT，但把八个历史输入槽改为重复末帧后，公共域预测平均只变化0.326 COT，CPP厚云域变化0.982 COT；真实旧历史让厚云预测平均低0.590 COT，patch厚云RMSE为27.143，对照替换输入27.083，全域RMSE则9.603对9.669。因此旧帧确实被使用，只是当前学习头的小幅修正更偏向全域和近晴空，未改善厚云选模目标。训练损失继续下降时厚云验证误差上升，进一步支持目标取舍解释。完整统计、权重范数与限制见`docs/SILI_HISTORY_USE_COUNTERFACTUAL_20260928.md`。

同一审计还发现v5训练JSON的`simvp`诊断列错误地把首张未来图广播成16时效；corrected SimVP→R patch全域/厚云RMSE为9.665/27.700，站点为8.604/28.363。三个主候选的learned/transport比较不受影响，原JSON保留，训练原版脚本和修正后的侧车另名归档。CPP/AGRI标签对齐和输运支持域与训练损失掩膜的差异仍待单独核验；test/GHI训练继续关闭。
