# 湖南DyDiff推进记录

## 5.02%质疑后的独立审查：数值通过，范围WARN

用户质疑预报增益比真实图大，已启用experiment-audit fresh agent独立读代码/源NPZ，未调用原评分器，复现全部7对照及日期CI（差<4e-14），12头权重哈希和432标签/键匹配。来源、标签、指标、主预报因果输入未发现错误，但总体审查为WARN（same-family/provisional）：15/16窗口08:00BJT起报，1个10:45；400/432记录来自08:00，旧midday描述不准确，已加SELECTION_TIME_AUDIT说明并改图范围，保留原选样和预测。

同一组三帧头：完整真实图4226条为89.485→88.264（1.365%）；相同432条真实图99.574→96.155（3.433%）；预报图127.647→121.243（5.017%）。同432条“预报COT效应−真实图效应”差−2.985，日期区间[−6.603,+1.200]跨零，不能宣称预报比真实图更受益。+45—240分钟378条（三帧均预测）仍有4.896%点估计；432行无GHI零值或既有连续零值标记。

额外CPU216621：node7排队因满核，仅对自己的Q任务qalter到有16空核的node9；未加GPU、未动八卡训练。canary改变真实未来AGRI/QA后，两个主预报输入各198288值逐位不变，只有真实/Oracle分支改变。固定M10头：对齐COT121.243、仅未来槽置训练log1p(COT)均值152.723、全部槽置均值154.338、20次同站/时效打乱均值190.037。六组回放与原结果逐位相同；来源NPZ和20组donor也获独立复算。这说明固定头使用对齐COT，不能把破坏输入的损失当成COT增益或纯物理因果。

完整报告cot_ghi_diagnostic/EXPERIMENT_AUDIT.md和JSON、NUMERIC_AUDIT.json、COUNTERFACTUAL_AUDIT.json。5.02%仅保留为早间选定验证回放点估计，不提升为全年/独立test结论。未读取test载荷，未修改论文正文，训练继续。

## 最新：COT/GHI冻结头诊断完成（216596，CPU）

用户要求看COT和GHI收益，已完成16个原固定窗口的同键SimVP、修正几何R及M00/M10同容量三帧头回放，未新增训练或GPU；湖南216594继续。CPU216596 node7/16CPU、65.5秒完成，512 COT记录、432 GHI记录（四里192/竺家240），test读取0。来源及全部代码、逐样本输出和图见cot_ghi_diagnostic/。

独立保存预测FP64评分：DyDiff输入下无COT127.647→加预测COT121.243 W/m²，降低5.017%，两站六配对均改善；日期块区间Δ[−11.755,−0.447]，仅15日小样本验证回放，非全年/独立test或DyDiff适配重训。四里下降5.709%、竺家4.253%；15日期11好4差，去掉任一日期仍为负差。同期已训头重载最大差0.000245W/m²；新构建真实分支相对旧保存预测最大差0.064606W/m²，保留CPU/R批次漂移。

COT主站点像元对正确连接CPP（160记录/14日）的等站RMSE：DyDiff11.014、SimVP9.973、历史持平11.567；DyDiff没有整体优于SimVP，差区间跨0。GHI共同键上的COT方向一致，不能把GHI5%归因于COT普通RMSE更低。固定AC头换未来真实图→R的oracle为111.471，仅归因；历史COT持平为132.061。7月9日真实轨迹仍有厚云漏报和GHI突降未命中，保留负例。

下一步应在完整冻结湖南val检验是否保持，当前不修改论文或开启test。完整报告cot_ghi_diagnostic/RESULTS.md；未改变湖南正在运行的8卡训练。

## 当前：正式湖南八卡训练已启动（216594）

216594日志已出现HUNAN_FORMAL_8GPU_START，正式runs/seed42_150ep_hunan_v1/run_metadata.json已生成并回收FORMAL_RUN_METADATA_216594.json。PBS状态R、exec_gpus为node21 GPU0—7；world8/global80/per-GPU10，35320train/3140val，442步/轮，150epoch/66300计划更新，test_payloads_read=0。修复后的8卡合成断点恢复一致检查PASS；真实full-size pilot保存step2、准确加载step2并完成step3再次保存，之后新建正式随机初始化训练，未复用湖北权重。真实pilot单步约2.29秒，不以此代替整轮含加载/验证总耗时。完整记录见本目录代码、PBS三个attempt、归一化及跨区域结果。

## 最新：216590八卡适配启动

### 接续216591及跨区域诊断

### 216594检查点对象通信修复

216591真实pilot完成两次反向传播（约2.24秒/步、第二步reserved35.91GiB）后，在rank0收集Python RNG状态的NCCL gather_object报unhandled cuda error并退出1；尚未到正式启动标记，正式训练目录不存在，不能把该pilot写成正式训练。改为单独Gloo CPU group收集小型Python RNG对象，DDP梯度/验证张量通信仍保持NCCL，模型/优化器/样本设置不变。旧失败日志保留。

重新核对slfu全队列、node21空闲8卡后，提交216594.tc6000。只复测受修改影响的8卡断点保存/恢复和真实pilot，复用216591每卡10容量结果，不再跑输入检查或跨区域16窗口。正式global80、每轮442更新、150轮66300更新。

湖北模型→湖南16窗口诊断已完成PASS，原始metrics/selection、hourly_metrics.csv与两季真实云图对比已回收；结果见cross_hubei_on_hunan/RESULTS.md。1/2/3/4小时C02 RMSE为0.07117/0.09879/0.12862/0.14587，C12亮温为5.565/7.158/8.481/9.942 K，不是COT或GHI误差。所有13通道相对末帧持平的同支持MSE skill均为正；不能据此称全验证、优于SimVP或已有GHI收益。

216590实际退出1，原因仅为合成恢复测试的Adam状态逐位一致断言：8192元素中1个相差4.235164736271502e-22，model/EMA逐位一致已通过。原失败日志/产物保留。verify_resume仅对optimizer moments改FP32数值容差rtol1e-6/atol1e-20，模型/EMA仍要求逐位一致；复用已完成合成测试而不重跑数据检查。

重新查队列和物理GPU后提交216591.tc6000，八张node21 A100、40CPU。修正后的恢复验证PASS；真实湖南每卡10容量PASS：34.701GiB reserved、27.563GiB allocated，global80。随后8卡真实pilot与其续训通过后直接正式150epoch。

用户新增直接用湖北模型试湖南的授权已执行：冻结原第150轮EMA及湖北归一化，Hunan val按月选16窗口并排除湖北train日期；在216591容量测试期间用其他7卡并行推理，未另占第9张卡。部分窗口已完成。见HUBEI_ON_HUNAN_PLAN.md；完整指标以对应metrics.json为准。

用户明确取消重复全量源数据检查，适配湖南输入后node21测试通过即8卡开跑。已完成code_hunan_v1：复用冻结湖南train/val、原始QA和同train归一化，修正NOAA几何按UTC及湖南grid重算，原图UTC连接BJT质量位，按需本地缓存；不重新扫描全部帧，不重新拟合归一化，不读test载荷。训练日志loss修正为跨rank求和再平均，与实际全局有效像元梯度一致。

提交前slfu全队列为空，node21八张A10040GB均0MiB/0%。PBS216590.tc6000已分配node21 GPU0—7，状态R，40CPU，8卡不等/空rank有效像元梯度测试已PASS。后续顺序为8卡断点精确恢复、少量真实批次容量及8卡pilot自动续训，然后随机初始化湖南正式150epoch。以launch日志的HUNAN_FORMAL_8GPU_START和正式run_metadata/progress为正式训练启动证据；仅PBS为R不能代替正式训练开始。原CPU216589保留只读。例行监控交Luna，本轮不创建/恢复自动化。

2026-10-28：用户授权湖南独立底图。source_hubei六核心SHA与同门epoch150来源相同。学院GPU/队列实时核对完成；node21空闲8 A10040GB，node23空闲7，node22无可确认空闲。CPU216589正常队列初请求node21未运行，核对后仅对自身Q任务qalter到node7四CPU，C/exit0约30秒。全元数据和6帧头/网格PASS，但1验证帧IR全部不在有效范围，完整数值QC未完成。source adapter、train norm、本地缓存与2/4/8卡真实吞吐仍待做；未开正式训练、未读test。见EXPERIMENT_PLAN.md、READINESS_CPU.json和RESOURCE_SNAPSHOT.json。

## 2026-10-29 09:10 CST 实时训练状态收据（PBS 216594）

- 实时服务器时间 2026-10-29 09:10:49 CST。216594.tc6000 状态 R，自 2026-10-28 18:20:35 运行，已运行 14:49:03；PBS `exec_gpus` 为 node21 GPU0–7，共8张 A100 40GB。正式元数据与启动标记存在，训练合同为 world8、per-GPU batch10/global batch80、150轮、442 steps/epoch、66300计划更新；测试载荷读取数为0。
- 最新 progress.json：epoch45/batch362/global_step19810；日志已推进到 batch352/step19800，last.pt 约1.053GB，mtime 09:10:25（按每100步保存，应对应step19800附近）；best.pt mtime 07:47:33。最近完整轮为epoch44/global_step19448，轮耗时1051.35秒；epoch41–44普通轮平均1051.27秒（约17.5分钟）。epoch40执行完整验证耗时1614.76秒（约26.9分钟），相对普通轮多约563.5秒；完整3140样本验证RMSE/MAE为0.479235/0.260525，当前最佳RMSE为0.479235。epoch44固定64样本paired monitor RMSE raw为0.467226、MAE为0.258541；此口径不能与全量验证混同。
- 估算：epoch45尚余约80步，之后还剩105轮。按近期普通轮约1051秒、每5轮完整验证约额外563秒估算，剩余约34小时，吞吐稳定时预计2026-10-30 19:20 CST左右完成；数据/验证耗时变化会使该ETA浮动数小时。
- 作业内GPU CSV最新采样 09:10:38：8卡利用率99–100%，显存约31.6–39.5/40.5GiB；登录节点没有nvidia-smi，故使用PBS分配信息及作业内GPU采样确认资源使用。当前loss、grad_norm为有限值，日志未发现NaN/Inf、Traceback、OOM或运行错误；进度连续推进。没有读取test载荷。


## 2026-10-29 完整冻结验证配对评价已排程

用户批准扩至全部3140个val窗口，不限早间；同一修正R和M00/M10三帧头，无新训练或test。421窗口与湖北DyDiff训练日期重叠，同时固定报告2719非重叠子组。CPU216699完成原16窗口扩展脚本复现：DyDiff5.017036%、SimVP3.301113%。GPU216700已提交node21八卡/40CPU，afterok依赖当前216594训练与216699，H为依赖等待，不影响正在进行的训练、总并行不超过8GPU。自动小范围GPU复现通过后生成全val站点patch、八分片R/GHI推理、FP64全表与非重叠子组配对评分；未出全val结果，不能提前称5%保持。详细计划及脚本见full_validation_20261029/。


## 2026-10-29 取消GPU总并行8卡上限；全val改为node23七卡并行

用户明确取消8卡总上限，根AGENTS和子项目AGENTS已更新，保留PBS未结束≤3。实时node23 GPU0/1/2/4/5/6/7为0MiB/0%，GPU3有他人负载；只取消自己的未启动H任务216700，不损失任何训练/推理进度，替换提交216703.tc6000（node23/48CPU/7GPU、无依赖）。216594仍在node21八卡训练。full_validation_20261029/generate_full7_node23.pbs；合并/复现工具支持7分片，样本/随机种子/模型不变。新任务提交后为Q，实际启动由Luna核对。


## 2026-10-29 用户要求评价降为四卡

216703已在GPU复现阶段exit1（部分头重载差0.114—0.177W超0.1W），未进入全量生成，无预测进度损失，失败产物保留。关闭matmul/cuDNN TF32但不放宽容差，按用户要求新提交216704.tc6000 node23/24CPU/4GPU；gpu_smoke2与evaluation_shards2。216594八卡训练不变。本次评价优先3—4卡，不恢复总8卡上限。
