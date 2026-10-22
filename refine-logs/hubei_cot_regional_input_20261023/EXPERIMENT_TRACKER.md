# 区域反演输入恢复进展

2026-10-23，本轮为 G1/G2 数据准备，尚未训练 R_region，不代表预报 COT 或 GHI 已提升。

|工作|当前证据|状态|
|---|---|---|
|修正 AGRI 精确连接|2828 train、296 val，共3124帧；152/16个日期|完成|
|AGRI 输入核对|全部3124帧哈希、形状、完整太阳几何；最大几何差0，2帧有红外零像元，原样保留待训练质量规则确定|完成|
|区域 CPP 提取|FD-201 PID2318421 cmdline 实时确认；最近进度1491成功、109首次失败，共3124帧|运行中；失败不是缺云或晴空|
|旧参考数值对照|首100个稳定文件，共25600个共同有效像元，最大差9.54e-7|部分完成，不能替代全量核对|
|I/O 定位|两个train源的直接读与本地暂存裁块一致；耗时受共享盘缓存/等待影响，不宣称普遍加速|完成|
|失败恢复|PID2347947 finalize_source.py 活进程；等待原批次终态，只重读失败帧并做全量旧参考核对|等待原批次|
|独立数据包验算|PID2359683 run_ready_audit.py 活进程；只在finalizer终态后运行独立输出哈希、键、mask与train-only log-COT统计|已部署等待|
|区域R训练|需先取得全量来源/参考一致性及独立统计结果，再冻结架构、预算和训练协议|未启动|
|G2区域打包接口|build_regional_pack.py已完成；两帧合成检查覆盖train-only统计、热红外无效mask、物理log1p目标与val变换|代码通过，未运行正式打包|

## 实际路径与进程

- 服务器：FD-201，复旦服务器，hostname 4090；此阶段4个CPU提取进程，不占GPU。
- 原始输出：`/tmp/hubei_cot_regional_input_20261023_attempt1/`，不覆盖。
- 重读与整合输出：`/tmp/hubei_cot_regional_input_20261023_ready/`。
- 代码与持久日志：`/home/Data_Pool/zjnu/hubei_cot_regional_input_code_20261023/`。
- 原始源：`/home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP/`；共享 NFS，已有 I/O 等待证据。额外GPU不能加速这一阶段。
- `audit_ready_statistics.py`严格只用train有效像元拟合log1p(COT)矩；输出是数据准备统计，不自动采纳为模型归一化，不使用val拟合，不读GHI标签或test/buffer。
- 本轮保留负结果，不因新时段站点收益修改样本，也不重跑已关闭的27参数残差筛选。

## 下一项

原提取、失败重读和独立验算全量通过后，转运一次小体积区域CPP数据到学院既有修正AGRI缓存旁，构建只读G2配对数据。若失败则先检查具体错误，不因观测超时重启原任务。长GPU训练按工作区规则先实测2/4卡，维持全局batch和更新预算；本阶段未提交PBS。

### 后续实时核查

同一提取PID2318421继续推进，最新Luna实查1846成功、154首次失败，共3124；finalizer及独立验算等待器均活着，未产出最终REPORT。不把首次失败计为最终缺失，不重启正在推进的进程。G2打包代码尚未连到自动训练，源数据完整通过后才正式运行。

### G2自动衔接已启动

原提取进度2608成功、192首次失败，总处理2800/3124；父进程及两项后处理实时确认存活。当前Windows执行会话95827运行`continue_g2.py`，根据远端真实进程和报告等待，不以锁文件认定存活；探测超时继续观察同一任务，不重启原提取。

源报告及独立统计同时PASS后，程序只做一次带`tar -h`的区域CPP流式转运（必须解引用ready内保留原输出的软链接），目的为学院实验根的`source_ready_attempt1`。两端退出码均0后，独立重验所有目标载荷哈希，再在node22已有修正AGRI旁构建`/dev/shm/hubei_regional_r_pack_20261023_attempt1`。输入已存在且本地空闲94,232,305,664字节；正式打包前再次要求至少20GB。失败保留目录，无自动重试转运。

本衔接不提交PBS、不占GPU、不启动训练、不触碰旧资产。正式数据包完成状态只以`G2_CONTINUATION.json`及远端pack报告确认，等待文件不算完成。转运、独立核验和打包日志写本目录`g2_*.log`；日志与临时等待文件不作为论文结果。

### 首轮源提取终态与失败归因

PID2318421已结束，终态`FAILED_RETAIN_PARTIAL`，不是训练失败。首轮完整报告`CPP_ATTEMPT1_REPORT.json`及简表`CPP_ATTEMPT1_SUMMARY.json`已保存：实际成功2926/3124（2646 train、280 val），失败198；最后PROGRESS只在第3100条写出，因此其中2902成功不是最终成功数。

198个错误均为NetCDF/HDF读取错误：196条RuntimeError、2条OSError(-101)。时间/网格检查未发现新增错误，成功帧坐标差0。此处不能据此宣布文件永久损坏或CPP来源缺失；finalizer PID2347947仍活，已见19条`PASS_LOCAL_STAGED_RETRY`，暂未见重读错误。完整源hash仅对重读顺序复制的文件计算；原始成功项保留裁块/轴hash，不混称全源文件hash。

独立统计等待器和G2转运继续会话95827仍在等待。只有最终198失败全部修复并且全量参考核对通过后才会转运。未因部分成功缩小样本或提前启动区域R。

### G3训练实现与CPU检查

为避免数据准备完成后再空等实现，已补`regional_model.py`、`train_regional_r.py`及`R_TRAINING_PROTOCOL.md`。仍是新随机初始化的原base16两层下采样UNet（119233参数），训练256×256整幅区域，非旧R权重扩大视场。固定全局batch128、最多50轮、AdamW及cosine更新计划、内部选模patience8；profile与正式训练目录分离，profile不接续为正式权重。

不读载荷的日期分组已冻结：R_fit1999帧/106日期、R_select314帧/19日期、邻日缓冲515帧/27日期，外层val296帧/16日期。内部选模不使用外层val；外层train输入统计沿G2（包含内部选模输入，不包含外层val），不把内部选模叫完全隔离OOF或独立测试。

CPU/Gloo两进程检查完成：有效像元计数分别1与3，掩膜外含NaN，DDP缩放后梯度与单进程4像元全局均值差0；见DDP_LOSS_WITNESS.json。新旧结构仅用相同随机权重严格映射，64×64随机输入前向差0；屏蔽NaN的反向全部梯度有限，见MODEL_CPU_WITNESS.json。不读取旧checkpoint；这两项不证明CUDA/SyncBN或训练已通过，正式GPU运行仍须profile及实际检查。

最新直接读取重读进度115/198成功恢复，finalizer与独立audit等待器均存活，未见重读失败。源数据依赖尚未全量PASS，训练未提交。

### GPU短测与计分程序已准备

新增2/4卡同全局batch梯度核对、FP32及AMP完整断点二步重放、同预算吞吐比较。模板`profile_regional_r.pbs`已在学院bash语法检查通过，但尚未qsub；提交程序要求实际G2报告、目的端独立审计及实时全队列/物理卡条件。CPU witness不能替代这些GPU结果。

新增独立物理COT计分`score_regional_r.py`，取原CPP物理值为参考，保存完整逐帧/空间范围/厚度分层误差和；合成数据已验证已知+2误差、有效晴空0与厚云计数、掩膜外高预测不会混入指标。这是代码检查，不是实验结果。

最近实际恢复175/198；后处理进程仍存活，无重读失败日志。自动转运与G2继续会话95827仍等待最终来源/参考及独立统计通过。正式训练和GPU短测均未启动。

### G1全量来源及独立验算完成

198/198首次失败均通过本地暂存重读恢复，finalizer终态PASS_REGIONAL_CPP_SOURCE_AND_REFERENCE：3124/3124帧，remaining_failures为空。2828 train/152日期/167222406有效像元；296 val/16日期/17443028有效像元。共同旧参考比较2637帧、673760有效像元，最大物理COT差3.8146973e-6，小于预定1e-4；完整报告保存CPP_READY_REPORT.json、CPP_REFERENCE_COMPARISON.json。

源端独立统计亦已完成PASS，G2_WAIT_STATE.json是程序实际读取两项PASS后的状态；源提取、finalizer及audit父进程均终态，不再重启。独立统计使用train有效像元，不读取GHI/test/buffer。转运会话95827活着，学院目的目录已实查397MB且继续增长；暂无完成收据，不能称完整转运或G2完成。fresh FD-201 SSH曾短暂超时，但既有转运连接仍推进，不因此重启。

目的tar仅有源/目的机器时钟不同的mtime未来警告，不用于物理时间对齐，也不影响以UTC/BJT键和载荷hash核对。需等两端真实退出码及目的端hash审计后再判转运完成。

源端独立矩统计曾默认启用大量BLAS线程；未修改或停止已运行分析。后续目的端audit_ready_statistics.py在导入NumPy前固定一CPU线程并记录代码hash，部署时目的端审核尚未启动，源端原审计代码未热改。本优化仅减少小向量统计的线程开销，不改变样本、掩膜或训练设置。

### G2转运、初版归一化问题及修复

转运完成，两端exit0，475580779压缩字节、381.253秒。目的端独立audit PASS，attempt1 CPU打包完成；未提交GPU。检查实际上下文发现day_mask由仅白天train像元标准化（mean1/std1e-6），导致R_fit50个有监督帧、R_select10个、val4个出现−1e6，超FP16范围。此处是本轮新包问题，不是历史模型或前述新时段分站差异的原因。

已完成attempt2重建，二值day_mask保留0/1；其余15通道逐帧完全一致，rows/target/mask/support文件hash不变。全量上下文audit PASS：最大绝对输入31.98265，三组均无超FP16值。attempt1原样保留，所有GPU入口仅接受attempt2。详见NORMALIZATION_AUDIT_attempt1.json、G2_ATTEMPT2_COMPLETE.json和attempt2 NORMALIZATION_AUDIT。

实时slfu全队列为空；node22仅GPU0空闲，不申请4卡；node23物理GPU3—7空闲，调度器仍有其他账户作业，绝不终止。已启动学院本地盘node22→node23单次数据包复制（会话89791），复制后逐文件hash核验，再实时重新查队列/物理卡提交4卡profile。源/目的/tmpfs均为本地盘，不再读复旦NAS；约14GB数据包只复制一次。正式训练尚未启动。

### G3正式训练已提交：215752

node23本地包逐文件哈希PASS；14,128,332,800字节传输完成，不再读取原NAS。215751四卡profile已C/exit0：2/4卡全局损失相对差4.17e-8、梯度最大差4.95e-6；FP32/AMP断点二步重放通过。2卡197.614帧/秒、4卡570.398帧/秒，训练短测约2.886倍，非全流程加速。

按预定吞吐规则选4卡，submit_training.py实时检查全账户空队列、node23物理GPU3—7空闲后提交215752.tc6000。请求node23四A100-SXM4-40GB、8CPU、2小时上限；随机初始化区域R，固定global batch128、最多750更新、内部日期选模，不延续profile参数。PBS先核对实际分配卡，再训练，完成后同一作业独立FP64计分原CPP。运行目录r_region_seed42_attempt1、日志train_attempt1.log、分数r_region_seed42_attempt1_scores。

这不是GHI训练或预报增益；后续仍需G4共同样本下的未来COT对照，通过后才进入G5配对GHI。test/buffer未打开。提交记录、代码哈希、profile终态和资源记录见TRAIN_SUBMISSION_attempt1.json。

### G4时间键准备（不读载荷）

prepare_g4_keys.py连接本轮完整修正AGRI清单和已恢复区域CPP清单，要求8帧历史AGRI均存在且与目标CPP同split；未来CPP只在对应评分时效要求，不要求24帧CPP连续。得到train 27224个init/target配对、4529个起报/152目标日期；val 2936配对、503起报/16目标日期。val +60/+120/+180/+240分别216/188/160/132条候选，随时效不同，不能假装是完全相同样本。

此清单仅是元数据候选；没有读取标签或test/buffer，尚未核验运动来源覆盖和共同逐像元mask，不是有效评分样本数或预测精度。G4各方法须在同一init/target和像元交集评价，先冻结R再生成；不得根据单方法误差删行。G4_CANDIDATE_KEYS.csv、G4_KEY_REPORT.json保存输入/输出哈希。215752已实时推进至epoch20/global_step315，尚未完成选模或评分。

### G3完成与冻结

215752已C/exit0，总291.026s；最终epoch39/global_step600，内部R_select最佳epoch31。285有效月度val帧/16日期，实际CPP物理COT对保存预测FP64计分完成：全域RMSE8.490809，三站像元10.046469/9.934704/9.405174；非forecast/GHI结果。逐帧误差和第二实现聚合150指标最大差8.88e-15。站点厚云仍偏薄、中等云偏厚，完整分层不隐藏。R_RESULT_REPORT.md给出意义、局限、checkpoint/代码来源；best.pt哈希ed7be2ea2c987557d3753b3eb7d8d0f9ffb4513b63e4afedfcbb54be132600e2。下一项G4共同键未来COT比较，不自动把反演分数当预报收益。

### G4冻结三路线推理已提交：215753

615张val历史原始区域图及aux从node22本地复制到node23本地，3,709,603,840 tar字节、84.989秒，两端exit0；1230文件hash全匹配。首次CPU准备因目的端未上传corrected_agri_manifest.csv在写任何选择/复制前退出；上传既有清单后完成，非源数据缺失。没有重建AGRI、读NAS或读test/buffer。

固定比较协议G4_FIXED_COMPARISON.md：同一新区域R、既有修正S、历史C13相位运动；val503起报/2936时效对，评价CPP参考共同像元的持平/输运/SimVP→R，另保留来源覆盖。运动符号及回溯CPU检查通过，已按PBS实时队列空、物理卡空闲提交215753.tc6000（node23，16CPU，4 A100）。执行g4_attempt1；history/forecast分片，实际G4成绩以保存预测及scores/REPORT.json为准。没有训练/GHI收益结论。失效来源不填晴空，各时效支持不混淆，test/buffer仍关闭。

### G4三基线完成，均未超越持平

215753已C/exit0，总约228.5s；4历史+4未来分片、评分完整结束。503起报、16目标日期，站点像元参考7672/共同7669（99.9609%）。等站点等时效物理COT RMSE：持平10.043288、输运12.070293、SimVP→R18.157870。日期块95%CI分别相对持平[+0.554996,+3.108864]、[+2.532028,+12.418143]，三站动态路线均退化。近乎完整上游覆盖没有自动转换为精度改善。

79,272个像元/时效行的唯一键、因果target关系及独立计分核验通过，第二RMSE最大差3.55e-15，5000次目标日期块区间。G4_RESULT_REPORT.md说明分层/不同lead样本量和限制。下一项只准备上级G4历史演变候选的train来源及预定方法，不基于此val调运动或删站；COT门未过，不新增GHI训练，不声称目标完成。

### G4区域轨迹候选已进入实测

后续独立目录trajectory_attempt1：已冻结20通道显式逐历史输运/full与repeat_last等容量对照、三种子、相同物理COT损失/global batch128/最多2000更新。仅train内106拟合/19选模日期；19420fit/2840select配对、4598历史帧已转运并校验，冻结R四卡缓存太阳几何差0。CPU数据包首轮因glob误入日志失败，复用有效缓存写全新attempt2修复完成，未重跑GPU。

模型CPU回溯与独立SciPy、step0和删除旧帧对照均通过。215755跨卡比较略超门槛，发现比较对象是分别更新13步后不同权重；完整失败保留，不放宽门槛。215756现正以共同非零梯度权重做同一步2/4卡重验并重测AMP/断点恢复。正式六拟合尚未启动。当前最新状态统一见trajectory_attempt1/EXPERIMENT_TRACKER.md，不将源码已实现或profile当训练完成。外层val/test/GHI未被新候选读取，GHI收益仍待验证。

215756同权重重验已C/exit0通过原门槛，选择4卡（AMP比2卡快18.26%）；215757正式区域轨迹六个配对拟合已提交，固定global batch128/每组最多2000更新。最新收据与进度见trajectory_attempt1；不能把早先profile失败状态当作仍未推进。COT/GHI科学增益仍未得到新证据。

## 区域轨迹候选完成：尚不进入GHI

trajectory_attempt1六组训练已完成（215757 C/exit0、node23四A100，18:53）。内部冻结repeat_last；215758六组预测完成，CPU参考空字段解析失败后，仅修复缺测读取并原支持复算，不重训。三seed未来COT RMSE11.3120，比SimVP→R18.1579好，但高于持平10.0433，三站及厚云均未通过预定继续条件。独立原预测FP64复算最大差3.55e−15。该候选按原协议封存，不自动开GHI，不将COT结果称作GHI收益；总目标仍未达成。

已继续在训练内部select19日期进行不训练的运动归因：中等云固定输运显著高估，厚云却有改善，需区分运动与学习厚度修正，不能凭外层验证一概否定历史运动。完整证据、模型/代码与实际预测见trajectory_attempt1/RESULT_REPORT.md。test/buffer持续关闭。

## 接续内部归因及GHI输入源核对

215759四A100冻结重放完成、exit0；内部学习修正总体可改善，但厚云已被下调，外层输运偏薄时进一步下调造成退化。证据见trajectory_attempt1/INTERNAL_ATTRIBUTION_REPORT.md。之后仅用历史可见误差的可靠性规则CPU内部筛选未通过；未打开外层验证，不继续扫规则。

新区域R也不能假定优于当前GHI来源：4778共同键下现有旧R预测COT RMSE12.6851，新区域R17.8443，日期区间跨0，三站点均不利新R。实际同S预测AGRI相差很小、几何相同，完整输入/来源/样本边界见ghi_source_followup_attempt1/RESULT_REPORT.md。保留旧R与原负结果；未启新GHI训练，目标未达成，后续回到GHI端明确新单因素及同容量控制，而非继续在旧外层验证换COT损失。

## GHI端固定辐射编码筛选完成，未通过

新增单因素候选hubei_cmf_ghi_increment_20261023：文献固定CMF衰减编码与原log-COT同27参数、同基线及C日期。215760在拟合前因太阳域断言停止；CPU定位旧未来COT有限值mask遗漏cosSOZ有效条件，将超80度的零占位也标为有效。独立attempt2在拟合前冻结修复协议，增加同掩膜log-COT对照，保留旧H/F数值重放与原全部样本。

215761在浙师大学院node23三A100完成，C/exit0、43s；九份旧预测重放差0，保存预测独立FP64复算差1.42e-14。仅训练内部C_select：基线103.9012、旧未来log-COT111.9013、同掩膜log-COT111.9029、CMF编码112.2883、C02对照112.9689 W/m²。CMF−基线+8.3871，日期CI[+2.7579,+16.3823]，三站三seed均不利；封存候选，不开外层验证/新时段/test，不调参数。掩膜修复只改变0.0016，不能解释线性修正大幅退化；旧神经可用性问题另查，不能据此宣称已修好。下一步核查C_fit与C_select残差条件分布及已拟合修正符号/幅度，避免无依据扩模型。总目标仍未完成。完整证据见同级hubei_cmf_ghi_increment_20261023/RESULT_REPORT.md。

215762只读固定系数重放已C/exit0，三A100约30s，九个C_select重放差0，无新拟合。拟合日期基线偏差−30.102，修正+29.122，RMSE119.294→112.765；选模基线偏差+14.555，仍修正+22.952，RMSE103.901→112.288。COT10–30条件下偏差也由−57.682变+21.081，修正仍约+50；厚云预测组拟合内已被错误上调。OOF来源8个月跨模型，但同来源月配对4个月子集仍退化，不能把原因唯一归天气或fold。已保存完整日期/分层/恒等式分解及可视化，见hubei_cmf_ghi_increment_20261023/attribution_attempt1/REPORT.md。下一项承接已有神经G2/G3的C日期方向/可用性记录，不重复“添加基线标量”旧方法，不据C_select扫参数；目标未达成。

215763神经G2/G3固定best/last C角色重放已C/exit0，三A100约38s，best原评分最大差3.44e−7。G3末轮拟合117.244→106.965，选模102.889→111.353，太阳gate只改约0.00026，支持日期修正失效而非mask主导；原best全部epoch1，未事后改选。报告hubei_neural_cot_diagnosis_20261023/REPORT.md。

据此新单因素日期内残差目标pilot（hubei_date_centered_cot_20261023），raw/center×无/有futureCOT，共同修复epoch0与太阳available、其余输入/初始化/预算配对。215764三A100、C/exit0、36s，36小头各60updates。仅C_select：G1 102.888750；raw无/有102.871998/102.873256；center无/有102.477216/102.504780。center有−G1=−0.383969，CI[−1.705503,+1.063972]；center有−无=+0.027564，CI[−0.428081,+0.510943]，继续条件失败。全部预测独立FP64复算差1.42e−14，封存目标变换，不开official val/新时段/test，不调中心化强度。部分修正失效缓解不等于预测COT有效，目标仍未完成。

215765日期覆盖候选完成：hubei_cot_calibration_coverage_20261023。只按元数据将未进入G1 F_fit/F_select的原train日期纳入修正拟合，12天扩为50/49/50天；四组small/large×无/有futureCOT，同一G1、同归一化、同9922参数、同60完整batch更新/92160槽，step0有效。三A100、C/exit0、46s。仅C_select：G1 102.888750，small无/有102.871729/102.873089，large无/有102.570535/102.595038；large有−G1=−0.293711，CI[−1.483414,+0.820059]，large有−无=+0.024503，CI[−0.464021,+0.456084]，三seed均未胜过同容量无futureCOT。预设条件失败、候选封存；独立点分数差1.42e−14，原C前向最大差1.14e−4已通过事先数值门槛。未读official val/新时段/outerbuffer/test，不再在该C_select调小头，转回未来COT来源及条件信息的新证据核查；不重复已有离线真实图替换。目标未达成。
