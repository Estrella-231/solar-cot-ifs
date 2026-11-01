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

四卡216704实时R，exec_gpus=node23 GPU0/1/2/4；GPU smoke2四片均PASS，共432行，最大差0.00048828125W（原0.1W阈值未改）。关闭TF32后恢复数值一致；全量生成命令已进入初始化，尚未见首窗口完成，不能提前称已有完整val进度。GPU_SMOKE_PASS.json已回收。


## 2026-10-29 完整冻结验证评价完成，5%未保持

216704于17:35:23正常exit0，3140窗全部完成；GHI同键63480行/51目标日期。DyDiff128.326→127.221（0.861%，日期CI[-2.671,+0.472]跨0）；SimVP127.421→125.340（1.633%，CI[-3.714,-0.561]）。预定无湖北训练日期重叠2719窗/54280行：DyDiff0.996%、CI跨0；SimVP2.137%、CI不跨0。DyDiff四里三seed均改善、竺家三seed均退化；SimVP两站三seed均改善。本地已回收逐样本输出并独立FP64复算六组RMSE及5000次日期CI，差<1e-10。只属于冻结头val回放，不是独立test，也不是正在训练的湖南DyDiff。详见full_validation_20261029/RESULTS.md与LOCAL_FP64_RECHECK.json。

## 2026-10-30 01:44 CST 实时监控（PBS 216594）

- 服务器 `tc6000` 实时于2026-10-30 01:44:56 CST核验。`qstat -u slfu`仅有216594.tc6000一项，状态R；运行31:24:21，PBS分配node21 GPU0–7共8张A100。正式训练尚未完成，未读取测试载荷。
- 正式progress为epoch96/batch270/global_step42260；95轮完整，计划150轮/66300步，约63.7%更新已完成。last.pt约1.053GB、mtime 01:42:41（按每100步保存，约step42200）；best.pt约1.053GB、mtime 01:35:02，对应epoch95全验证后更新。
- 最新完整3140样本验证为epoch95：normalized RMSE=0.442142、MAE=0.233725、matched persistence RMSE=0.791838、MSE skill=0.689434；test_payloads_read=0。最佳完整验证RMSE为0.442142。最新固定64样本paired monitor仅作独立小样本监控：raw RMSE/MAE=0.435940/0.231667，EMA=0.431141/0.228134，不与完整验证混用。
- 最近普通轮91–94耗时1050.10/1053.43/1051.72/1049.22秒，均值约1051.12秒；完整验证轮95耗时1616.59秒，每5轮额外全验证约565秒。按当前轮余172步、后续54轮（含11次全验证）估计，剩余约17小时36分，粗略预计2026-10-30 19:20 CST完成，吞吐变化可能造成数小时偏差。
- 作业内GPU CSV 01:44:48采样8卡利用率100%，显存31.6–39.5/40.5GiB。loss/grad_norm均为有限值，错误/NaN/OOM扫描无命中；progress与step日志持续推进。当前没有训练完成标记。

## 2026-10-30 17:58 CST 实时训练监控（PBS 216594）

- 服务器快照时间17:58:27 CST，客户端时钟18:17:33 CST，两者约差19分钟；ETA给出剩余时长并注明服务器时间推算，不用时钟差计算吞吐。216594.tc6000状态R，node21八张A100；progress为epoch146/batch360/global_step64450，计划66300步，已完成145轮，test_payloads_read=0，未发现完成标记。
- 最近完整3140样本验证为epoch145：normalized RMSE/MAE=0.4330496/0.2273480，matched model RMSE=0.4320450，persistence RMSE=0.7918377，MSE skill=0.7022954；当前最佳完整验证RMSE=0.4330496。
- 最近普通轮141–144耗时1052.09/1050.69/1052.49/1050.29秒，均值1051.39秒；epoch145完整验证轮1614.63秒，额外约563秒。按当前轮剩余82步、其后4轮及一次完整验证估算，剩余约4980秒（1小时23分），按服务器时钟约19:21 CST完成。
- 末尾5000行未发现Traceback、OOM、CUDA error、NaN、Inf或error。GPU CSV最新17:57:57.839 CST：8卡均100%，显存31557–39537/40536 MiB。last.pt/best.pt只stat未加载。未读取test载荷，未改动作业。
- 机器可读/文本收据：`monitor_20261030_latest/receipt.json`、`receipt.txt`。

## 2026-10-30 19:47 CST 湖南DyDiff训练完成快照（216594）

- 实时qstat已无216594；tracejob历史确认2026-10-30 19:22:45结束，exit status 0，PBS墙钟49:02:13。`training_complete.json`为PASS/150轮/66300步，progress为complete，test_payloads_read=0。
- 最终epoch150全3140序列验证：normalized RMSE/MAE=0.4332109082/0.2276047394，matched persistence RMSE=0.7918376841，MSE skill=0.7020667215。最佳全验证轮次epoch145，RMSE/MAE=0.4330495719/0.2273480072，skill=0.7022953972。
- last.pt和best.pt均约1.053GB；只读取文件stat。训练日志异常关键词未见Traceback/OOM/CUDA error/NaN/Inf。未加载checkpoint、未读test、未启动任务。
- 本次收据独立保存在`monitor_20261030_1947_final/receipt.json`及`receipt.txt`。

### 完成后回收的全量验证序列

按项目负责人追加要求，从训练run目录回收了全部150个`validation_epoch_*.json`轻量文件至`validation_records/`；其中31个`forecast_evaluated=true`的记录另复制到`forecast_evaluated/`。附`validation_trend.csv/json`紧凑趋势表，字段保留normalized RMSE/MAE、matched_model_rmse、matched_persistence_rmse、代码记录的MSE skill和0–1h/1–2h/2–4h RMSE。比较模型与persistence时应使用同支持`matched_model_rmse`，不能用全normalized RMSE替代。全部记录的test_payloads_read为0。未加载checkpoint。

## 2026-10-30 湖南独立第150轮COT/GHI完整验证已启动（216841）

用户已授权接入同一湖南R和配对GHI模型。新任务216841.tc6000已实际在node21 GPU0–3四张A100/24CPU运行，提交前后slfu未结束任务分别0/1。固定湖南last.pt内第150轮EMA，不选择第145轮best；使用湖南训练归一化/QA和训练验证采样seed，完整3140val窗口，两站×seed42/43/44，既有3×3三帧GHI模型不重训。16窗衔接测试完成432条GHI，主生成20:00服务器快照已122/3140、四卡100%，暂估40–60分钟完成生成及评价；未读test。结果目录hunan_full_validation_20261030/，详见其中EXPERIMENT_PLAN.md、EXPERIMENT_TRACKER.md和LAUNCH_216841.json。完整结果待生成与自动配对评分完成后报告。

## 2026-10-30 20:34 CST 湖南epoch150 DyDiff完整冻结val预测COT→GHI评价（216841）

- PBS历史`exit_status=0`，walltime 38:02；`forecast_attempt1/completion.json`与`evaluation_attempt1/FULL_EVALUATION_COMPLETE.json`均PASS。epoch150 EMA在3140冻结val窗生成；common-key核验63480条GHI记录，51个目标日期，明确与此前完整验证使用相同标签/clear-sky/keys及12个GHI头，test_payloads_read=0。范围是固定R和已训GHI头回放，不是独立test，也未对预测COT重训GHI头。
- DyDiff预测输入（M00无COT→M10加预测COT），站点/三种子等权RMSE：四里137.222→133.815 W/m²，下降2.482%，三个seed的RMSE差(after-before)为−3.794/−3.316/−3.110；竺家112.695→111.761，下降0.829%，seed差−0.688/−1.074/−1.042。两站等权124.958→122.788，下降1.737%；日期配对95% CI（加COT−无COT）[−3.685,−0.735]，5000次按目标日期配对重采样。
- 对照SimVP无COT→加预测COT：Sili132.082→129.227（−2.161%），三seed差−3.071/−2.862/−2.631；Zhujia122.760→121.453（−1.065%），seed差−1.763/−0.802/−1.357；两站等权127.421→125.340（−1.633%），日期CI[−3.714,−0.561]。DyDiff相对SimVP无COT和加COT的RMSE差CI均跨零（分别[−6.806,+1.937]、[−6.794,+1.743]），不能据此宣称DyDiff显著胜过SimVP。
- 全预测槽+45至+240min：DyDiff无COT→加预测COT 128.555→126.308（−1.748%，日期95% CI差[−3.776,−0.808]）；从lead指标CSV按站点/seed先聚合，再等权得到对应SimVP无COT→加COT 130.880→128.771 W/m²（−1.612%）。
- 本地回收收据/小型数据（不含大逐样本NPZ）位于`hunan_full_validation_20261030/recovered/`：score JSON/CSV、completion及COMMON_KEYS、forecast completion/provenance、RESULTS.md。oracle未来COT替换仅诊断，不作为预报主结果。

## 2026-10-30 湖南DyDiff预测COT完整评价完成（216841）

完整3140窗/63480共同GHI记录/51日期，固定12个GHI权重。无COT124.958→加预测COT122.788 W/m²，下降1.737%；四里2.482%、竺家0.829%，六组均改善。目标日期配对RMSE差95%区间[-3.685,-0.735]；+45至240全预测槽子组下降1.748%。已回收保存预测并由独立NumPy实现复算，数值差<3.1e-14，未读取test。详情hunan_full_validation_20261030/EXPERIMENT_TRACKER.md及recovered/evaluation_attempt1/RESULTS.md。
