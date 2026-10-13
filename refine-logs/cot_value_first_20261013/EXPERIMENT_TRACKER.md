# COT价值优先任务表

日期：2026-10-13。当前会话统一执行，取消窗口分工和强制scope receipt。已实时确认HPC上的既有GHI标签、旧head pack、历史COT bank、区域R及新月度SimVP资产；新月度CPP监督R也已重训并独立FP64复核。T03复用已生产标签并按新日期合同选取覆盖。当前工作为新R对齐真实AGRI train/val COT bank，再用同一样本、同一3×3输入和GHI训练合同比较R-A/R-AC/R-AP；旧split或旧R结果只作历史参照，不重复旧合同实验。定时任务当前暂停。

| ID | 阶段 | 执行者 | 前置任务 | 工作 | 验收产物 | 资源 | 状态 |
|---|---|---|---|---|---|---|---|
|T01|B0|当前会话|—|既有211902/212019及R/S/H来源审计|asset_inventory.json|CPU|PARTIAL（旧R、历史COT bank、月度SimVP及CPP资产已按生产记录远端核验；211902精确作业映射仍未确认）|
|T02|B0|当前会话|—|实时队列、可用GPU/磁盘/路径核验|resource_snapshot.json|CPU/GPU|PASS_FOR_ZJNU_SINGLE_GPU（zjnu-hpc node22 8×A800-80GB物理空闲；PBS已接受单卡任务213623；FD-107/201本轮未参与）|
|T03|B0|当前会话|—|既有站点工作簿按新日期清单抽取标签，连接Ineichen晴空值并核对唯一target/掩膜|data_contract.json|CPU|PASS（train/val标签及clear-sky已落盘；最终主掩膜train 50,458、val 4,479）|
|T04|B0|当前会话|T01,T03|复用已训练R及CPP资产，核对新日期归属和覆盖|retrieval_contract.json|CPU/GPU|PASS_FOR_CPP_SUPPORTED_TRAIN_VAL（新月度R seed42训练及FP64验证完成；pack/转移审计通过；13个训练时刻CPP NetCDF/HDF读取失败，且CPP仅支持部分候选）|
|T05|B0|当前会话|T02,T03|节点本地盘缓存、checksum与读取吞吐|cache_manifest.json|CPU/IO|PASS_SINGLE_LOAD_IN_MEMORY（R训练已从已审核pack一次加载至内存并保持训练期间常驻；不逐epoch重读共享文件）|
|T06|B0|当前会话|T04,T05|按唯一train/val目标时刻复用冻结R生成观测AGRI的站点16×16 log1p(COT) bank|bank_manifest.json|GPU|PASS（55,733/55,733帧；train 51,236/val 4,497；source errors=0；R checkpoint SHA绑定通过；test未打开）|
|T07|B0|当前会话|T01,T03|3×3 A/AC/AP架构、参数及真实数据可用性witness|head_contract.json|CPU/GPU|PASS；54,937条共同train/val pack，独立键/形状/有限值/哈希审计通过，test未读取。|
|T08|B0|当前会话|T05,T06,T07|单卡吞吐、batch与30epoch预算冻结|profile.json|GPU|PASS；node22单张A800 profile选择batch=2000，至少12 updates/epoch，峰值分配0.189 GiB；已记录小头算力轻的限制。|
|T09|B1|当前会话|T08|四里R-A/AC/AP seed42|3 runs|GPU|PASS；PBS 213829，30 epoch；输出及独立FP64核验见 ghi_heads_new_r_seed42_v1。|
|T10|B1|当前会话|T08|竺家R-A/AC/AP seed42|3 runs|GPU|PASS；PBS 213829，30 epoch；输出及独立FP64核验见 ghi_heads_new_r_seed42_v1。|
|T11|B1|当前会话|T09,T10|六臂重载与逐样本FP64验收|pilot_audit.json|CPU/GPU|PASS_PROVISIONAL；保存预测FP64独立复算与checkpoint CPU reload均完成；CPU/A800最大逐点差0.452 W m-2，RMSE差≤0.0046 W m-2，作为跨设备浮点差记录。单种子未作主结论。|
|T12|B1|当前会话|T11|两站R-A/AC/AP seed43|6 runs|GPU|PASS；PBS 213832/213833，逐样本FP64复算完成；AC−A=-5.684、AP−A=-0.104 W m-2站点等权RMSE。|
|T13|B1|当前会话|T11|两站R-A/AC/AP seed44|6 runs|GPU|PASS；PBS 213838 Sili、213839 Zhujia均exit_status=0；各A/AC/AP 30 epoch；新test未读取。|
|T14|B2|当前会话|T12,T13|18臂主终点、配对CI与分层|experiments/monthly_R_station16_20261013_v1/results/GHI_HEAD_THREE_SEED_RESULTS_20261013.md|CPU|PARTIAL_PASS；FP64、2000次BJT日期/连续2日bootstrap通过；AC−A=-3.740、AC−AP=-3.582 W m⁻²，两站三seed方向一致，季节描述分层完成且各季station-equal均改善；天气分层仍待CPP/CLP公共支持。|
|T15|B2|当前会话|T11|固定头COT扰动与输入依赖诊断|experiments/monthly_R_station16_20261013_v1/results/COT_INPUT_DEPENDENCE_VALIDATION_20261013.md|CPU|PASS_VALIDATION_DIAGNOSTIC；物理零/中心保留外围置零/保值邻域置换/同站同月错配均已完成；置换敏感度很小，错配与置零会明显退化；结果仅为冻结头输入依赖诊断，不作因果证明。|
|T16|B2|当前会话|T03,T06,T14|预先选16例并输出真实图像/GHI案例|event_manifest.json, figures/|CPU|PLANNED|
|T17|B2|当前会话|T14,T15,T16|决定应用验证/错误修复/收缩结论|phase_decision.json|CPU|PLANNED|
|T18|B2条件|当前会话|T14|R-AL同监督隐特征两站三seed|6 runs + audit|GPU|PLANNED|
|T19|B3|当前会话|T01,T04,T05|S冻结来源、train/val预测与几何验收|forecast_contract.json|CPU/GPU|PLANNED|
|T20|B3|当前会话|T19|预测COT与历史COT train/val bank|forecast_cot_manifest.json|GPU|PLANNED|
|T21|B3|当前会话|T20|16lead COT站点/局地/区域共同支持评价|cot_lead_metrics.json|CPU/GPU|PLANNED|
|T22|B3|当前会话|T21|文献可比性、覆盖、厚云与到达诊断|cot_reference_comparison.md|CPU|PLANNED|
|T23|B4|当前会话|T17,T20|F-A/H/F/HF结构、因果梯度与预算见证|forecast_head_contract.json|GPU|PLANNED|
|T24|B4|当前会话|T23|四里F四臂seed42|4 runs|GPU|PLANNED|
|T25|B4|当前会话|T23|竺家F四臂seed42|4 runs|GPU|PLANNED|
|T26|B4|当前会话|T24,T25|八臂配对键/预测/重载验收|forecast_pilot_audit.json|CPU/GPU|PLANNED|
|T27|B4|当前会话|T26|两站F四臂seed43/44|16 runs|GPU|PLANNED|
|T28|B4条件|当前会话|T27,T14|C1成立而C2弱时F-O六臂及替换诊断|6 runs + offline audit|GPU|PLANNED|
|T29|B4|当前会话|T27,T21|预报增量、COT误差传播与公共cohort|forecast_results.json|CPU|PLANNED|
|T30|验收|当前会话|T14,T17,T22,T29|冻结模型、方法、阈值、主比较及test许可判定|analysis_freeze.json|CPU|PLANNED|
|T31|验收|当前会话|T30|新test一次正式评价、复算与结果保全|test_receipt.json|CPU/GPU|PLANNED|
|T32|交付|当前会话|T31|中英稿、科学图表、工程交接与结论范围|paper + handoff|CPU|PLANNED|

条件项T18/T28由阶段判定决定，未启动不算失败；若启动，T30必须纳入其审计结果。T17关闭预报扩展时T23–T29标NOT_APPLICABLE并记录理由，不无限等待，也不伪造预报结果；T30/T31只针对已有证据支持且符合门禁的同期反演范围。R重建如必要由T04拆出独立子任务/预算，不悄悄算作已完成。

复旦承接第4个及以上并发GPU任务；PBS账户≤3未结束作业。当前会话直接准备、提交和评价，不再需要交接、请求单或签署收据。
运行时记录host、job/pid、GPU、命令/config、数据及模型来源、时间、输出及结果。缺实际数据才记阻塞，缺scope receipt或窗口回复不算。旧windows目录保留已有产物。执行规则见EXPERIMENT_PLAN.md第10节。

## 2026-10-13 11:57 BJT：使用学院服务器空闲 A800

- `zjnu-hpc` (`tc6000`) 可访问；`qstat -u slfu` 提交前为空。GPU2 (`node22`) 备注 `gv maint`，但本机 `nvidia-smi` 实际列出8张 NVIDIA A800-SXM4-80GB，显存各约2 MiB、利用率0%；PBS调度接受了 node22 的单卡请求。
- 已提交并启动 `213623.tc6000`，状态 `R`，node22，PBS申请 `ppn=8:gpus=1`，请求8小时。启动日志确认Gold预留成功，实际只暴露分配的GPU 0；作业启动时该卡约2 MiB显存/0%利用率。
- 作业任务：用官方旧合同冻结R `cot_repaired_seed42_v1`（SHA-256 `0cbdb932f49945259d5f1704316edaf6481be7cfdb54692236f829988925799c`）和月度train-only SimVP归一化，对冻结新清单的唯一train/val真实AGRI目标帧生成四里/竺家站点16×16 `log1p(COT)` bank。候选表SHA-256 `b85c2841e33cc92eda848acb183f926d49564b6c131d6418730e6b53978e2807`；模型实现SHA与checkpoint绑定均通过。此为旧R对新月度真实输入的下游复用，不是新月度R训练或COT精度结论；无test行/载荷读取。
- 代码已通过本地与服务器Python语法编译，首个train帧及对应AGRI/几何文件存在。输出目录 `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/cot_value_first_20261013/results/real_agri_cot_bank_rseed42_monthly_v1/`。本记录只说明作业已运行，完整bank与覆盖状态待作业收据。
- 2026-10-13 12:00 BJT进度核验：作业仍为`R`，已处理1,408/55,733个唯一train/val目标帧，缺失输入0。node22 GPU0约571 MiB/80 GiB、利用率0%；其余7卡仍各约2 MiB/0%。当前任务受小文件读取吞吐限制，暂不并开更多卡以免增加共享存储竞争。

## 2026-10-13 12:35 BJT：开始新月度 R 重建

- 按用户要求，独立训练新的 `monthly_station16_COTUNet_seed42_v1`，不再把旧 `cot_repaired_seed42_v1` 的下游推理银行误称为新split R。新模型从随机初始化开始，采用新月度 train/val 日期键、AGRI 13+几何3输入、站点16×16 patch、train-only归一化及有限安全的 `log1p(COT)` masked Huber。
- 数据来源门禁通过：candidate manifest SHA-256 `b85c2841e33cc92eda848acb183f926d49564b6c131d6418730e6b53978e2807`，CPP inventory SHA-256 `0a303ec6965be360689f3891e96b5e4085ad3e6550b2db6504a8b2ac35565dd2`；输入文件只允许train/val。PBS node22不能访问CPP源，完整打包改由FD-107读取原始CPP与月度train/val AGRI。
- 更新后的pack脚本通过本地语法检查及FD-107真实源smoke：一个train时刻、一个val时刻，共4个站点样本；AGRI/CPP最大网格坐标误差0.0°；有效零COT保留；读取错误0；test载荷读取0。完整pack已在FD-107单CPU进程启动，PID `179560`，输出 `.../monthly_R_station16_20261013_v1/data/full_trainval_pack_v1`，日志 `logs/full_trainval_pack_v1.log`。该步骤I/O为主，不占GPU。
- 新训练模型拓扑已对照历史 `COTUNet16` 固定为两级下采样及 base width 32；训练脚本语法通过，已同步到PBS项目脚本目录且SHA-256核验一致。PBS全账户最新队列只有 `213623.tc6000`（旧R诊断银行，仍在运行）；本轮尚未qsub。
- node22两次1-epoch训练smoke均完成；第二次已修复只读memmap warning，成功保存/重载checkpoint与验证预测。独立CPU FP64审计通过，保存指标与独立复算一致，且确认预测样本、mask和pack标签逐条对齐。smoke仅含2个train、2个val站点样本且val均为厚云，误差不作科学解释。
- 12:39 BJT更新：FD-107 full pack 已访问1,000/17,600个有CPP支持的时刻，写出1,051个监督站点patch，读取约18.5 GB；进程继续运行，尚无完整数据门禁结果。
- 下一门禁：完整pack完成后核验train/val样本数、源错误/缺失分类、数组shape/hash、0值样本和网格映射；将pack转至PBS后再核验拷贝hash。随后实时复查全账户PBS队列与node22资源，提交单卡新R训练并做checkpoint重载及独立FP64验证复算。test保持关闭。

## 2026-10-13 13:02 BJT：完整数据包仍在NFS读取

- FD-107 PID `179560` 仍运行，日志最后一次flush为1,000/17,600 CPP支持时刻、1,051个监督站点patch；`/proc/179560/io`累计读取约26.8 GB（12:39时约18.5 GB）。当前等待点为`wait_on_page_bit_common`，表示正在等待读取页完成；尚无完整数据包验收结果，不能提交正式训练。
- 最新PBS账户队列仅1个未结束作业：旧R的`213623.tc6000`（R）；已完成smoke `213643.tc6000`为C，不计入并发上限。node22 GPU0为约571 MiB/80 GiB，其余7卡约2 MiB，利用率均0%。训练仍等待数据包完整审计和传输hash门禁。
- 13:04 BJT更新：pack日志已推进至1,500/17,600个CPP支持时刻、1,571个监督patch；FD-107进程读取约29.8 GB，仍运行。正式验证审计新增按站点、COT强度分层的FP64误差统计；更新后本地/PBS脚本SHA-256一致（`557c42dc66297f19fa420d616479ed2457d4e7488d21651e536cdbc0c4ab45c9`）。
- 13:10 BJT Luna只读复核：pack推进至2,500/17,600（2,613个监督patch），累计read_bytes约45.0 GB；NFS I/O仍持续。PBS仍仅旧R诊断任务`213623`未结束，新R训练尚未提交。
- 13:17 BJT：FD-107只读SSH命令开始明显延迟，短命令及日志查询均无法在正常窗口返回；上次可确认状态仍为2,500/17,600且I/O继续增长。PBS节点和`qstat`通道正常，仅`213623`运行。未重启pack进程。
- 13:23 BJT Luna复核：FD-107 SSH恢复，pack继续到4,000/17,600支持时刻、3,996监督patch，read_bytes约75.1 GB；进程仍处于NFS I/O等待/读取状态。全量包及训练尚未完成，PBS仍仅`213623`运行。
- 13:27 BJT Luna复核：pack推进至4,500/17,600（4,339监督patch），read_bytes约84.2 GB，PID存活；PBS仍仅`213623`运行。
- 13:30 BJT：FD-107本轮只读检查超时，暂未拿到新flush进度；最近已知pack PID仍存活且I/O增长。PBS qstat正常，仅`213623`未结束。
- 13:39 BJT Luna复核：FD-107 pack推进至6,000/17,600（5,472个监督patch），累计read_bytes约107.1 GB，PID存活。PBS仍只有旧R诊断作业`213623`未结束。
- 13:43 BJT Luna复核：新一轮监控确认pack到6,500/17,600（5,916个监督patch），read_bytes约114.2 GB；FD-107 PID存活，PBS仍仅一个旧诊断作业`213623`未结束。
- 13:46 BJT Luna复核：pack推进至7,000/17,600（6,468个监督patch），read_bytes约124.9 GB；PID仍运行，PBS仍仅`213623`未结束。
- 13:50 BJT Luna复核：pack推进至7,500/17,600（6,984个监督patch），read_bytes约133.8 GB，日志无错误；PBS仍仅`213623`未结束。
- 13:54 BJT Luna复核：pack推进至8,000/17,600（7,475个监督patch），read_bytes约144.1 GB，日志正常；PBS仍仅`213623`未结束。
- 14:07 BJT：FD-107连续两次只读状态查询无输出超时，暂不能取得新进展。最后确认仍为8,000/17,600、PID存活且读盘；PBS正常，仅旧R诊断作业`213623`运行。
- 14:09 BJT：FD-107 SSH shell仍有延迟，但SFTP成功读取到pack日志；进度已到10,000/17,600支持时刻、9,669监督patch。无完成标记；未重启任务。PBS仍仅旧R诊断作业运行。
- 14:18 BJT Luna经SFTP复核：日志到11,000/17,600（10,701监督patch），没有失败/完成提示；PBS仍仅旧R诊断作业`213623`处于R。继续等待完整pack状态与审计。
- 14:24 BJT：SFTP日志查询恢复，pack已到12,000/17,600（11,707监督patch），日志mtime为14:24，无错误/完成提示。PBS旧诊断作业`213623`状态C，新R训练仍等待数据门禁。
- 14:28 BJT：SFTP日志推进到12,500/17,600（12,189监督patch）；`pack_status.json`尚未出现。PBS当前唯一未结束作业为213666 `hunan_detail_MH`（其他实验），账户并发数1；新R仍未提交。

- 2026-10-13 14:27 BJT：实时复核 `213623.tc6000` 已从队列退出；远端 `bank_manifest.json` 状态为 `COMPLETE`，55,733/55,733帧齐全（train 51,236、val 4,497），源错误0、errors为空、test载荷未打开；日志含 `MONTHLY_REAL_AGRI_COT_BANK_COMPLETE`。当前PBS队列仅见另一作业`213666.tc6000`，不属于本bank任务。

- 2026-10-13 15:54 BJT：新月度R的train/val pack通过transfer后审核（16,812监督patch，train 15,414、val 1,398；CPP网格误差0.0°；13个train时刻NetCDF/HDF读取错误，单独作为来源错误统计）。PBS `213712.tc6000`完成seed42新R训练及FP64验证：best epoch 20/28（zero-based），CPP支持白天像元RMSE 9.074、MAE 4.646；厚云(COT>=30) RMSE 14.129；train-only归一化、test载荷0。新R COT bank PBS `213716.tc6000`已在node22 GPU2运行，目标唯一train/val 55,733帧。R训练包单次加载入内存；R训练profile选batch 1,284、13 updates/epoch。

## 2026-10-13 16:15 BJT：新R bank与GHI对照准备

- Luna只读核对PBS `213716.tc6000`仍在node22 GPU2运行；账户另有`213666.tc6000`，共2个未结束任务。新R bank日志进度6,400/55,733（约11.5%），source errors=0，test_payload_opened=false。速度受共享盘逐时次小文件读取限制；没有再启动同路径读盘任务。
- 完成3×3 GHI训练数据连接器 `experiments/monthly_R_station16_20261013_v1/prepare_ghi_head_pack.py`：按`(station,target_time_utc)`连接train/val GHI标签、目标时刻AGRI、真实太阳几何和新R bank的3×3 COT；从16×16 COT裁`[7:10,7:10]`，不池化；仅保留主标签掩膜，缺失项不按行序错连。AGRI及log1p(COT)统计量从train样本计算。
- 完成配对训练器`train_ghi_heads.py`与单卡PBS入口`run_ghi_heads_seed42.pbs`。训练固定同一3×3样本、逐epoch样本顺序、30轮预算与GHI物理误差损失；A/AC共用初始化主干与读出，AC/AP约等参数；PBS入口先在单卡做三臂profile，再依吞吐与每站至少12次更新/epoch选择batch，顺序训练两站六臂seed42。Python源码本地编译通过并已同步；远端`swc`环境Python 3.10编译通过。profile/训练作业等bank完整、连接pack完成后再提交。

## 2026-10-13 识别诊断：新月度 R 云/厚云固定阈值验证

- 对已完成并经FP64独立复算的新月度R验证预测做云/厚云识别评价；有效CPP验证像元356,169，1,398个站点样本，test载荷读取0。参考COT标签与预测均按物理单位比较，预测`log1p(COT)`仅逆变换一次。
- 固定阈值：COT≥0.1为云COT代理，COT≥30为厚云。合并结果：云POD/precision/CSI=`0.957/0.871/0.838`，厚云=`0.677/0.663/0.504`；厚云漏报16,425、虚警17,526。按站点结果见`COT_CLOUD_DETECTION_VALIDATION_20261013.md`和JSON原始计数。
- 解释范围：CPP支持的白天验证像元；CPP是检索参考而非独立真值。该云标签由COT阈值定义，不等于独立CLP云掩膜验证；仅seed42，不能推出多seed稳定性或GHI增益。未读取test，也未提交新训练。
- 复算脚本：`evaluate_cot_cloud_detection.py`（SHA-256 `FD062D1CCD75915B76AF0F2E1D1784DC9F3D8113E31233468E81188330DAEB84`）；服务器结果在`experiments/cot_value_first_20261013/results/cot_cloud_detection_monthly_R_seed42_validation.json`。

## 2026-10-13 CPP CLP 同样本云识别及文献口径对照

- 从704个唯一验证时刻的CPP NetCDF逐站裁取CLP；1,398/1,398验证样本均与COT有效掩膜完整重合，共356,169像元，覆盖率100%，经纬网格最大误差0°。候选清单SHA与grid_static SHA与pack审计一致，test载荷读取0。
- 固定预测为物理COT≥0.1，参考为同像元CLP∈{1,2}（有云）/CLP=0（晴空）。合并云POD 0.925、晴空POD 0.740、云FAR 0.123、HR 0.863、KSS 0.665；四里/竺家分项及全部混淆矩阵见`cot_cloud_vs_cpp_clp_monthly_R_seed42.json`。
- 公开FY-4B CLM质检（2024-12全圆盘、与MYD35匹配，92,528,101点）报告云POD/晴空POD/云FAR/HR/KSS=`0.922/0.816/0.045/0.901/0.737`。本项目的云POD相近，但参考产品、范围、COT阈值定义及日间站点样本不同，故仅作指标量级上下文，不能声称达到或优于业务CLM。
- COT RMSE对照也仅作背景：本项目9.074（CPP验证参考）；FY-4A全天MODIS基准研究报告12.79；FY-4B夜间CALIPSO基准报告不同太阳角条件1.26/1.99。卫星、参考、条件和分布不同，不能直接排序。未找到同阈值COT≥30公开CSI/FAR数字；厚云CSI=0.504保留为项目内部指标。
- 结论与严格限制见`COT_CLOUD_LITERATURE_COMPARISON_20261013.md`。CPP CLP与COT同属CPP产品链，非独立真值；真正业务产品头对头评估还需相同时次FY-4B CLM。此次没有读取test，也没有提交训练。
- CLP提取脚本`extract_validation_clp.py` SHA-256 `22BFA129173AF104130765DDB0F2F2C4C26B6C3ADBCE9B16FE7C57CEAF0B216C`；评分脚本`score_cot_cloud_vs_cpp_clp.py` SHA-256 `6F798B6ACCA161416AA9F693C217CF68B93797597E196792564D543FF9BBF3C3`。原始CLP数组及提取审计保存在`clp_reference_monthly_R_seed42.npz`和`.audit.json`。
- 17:00 BJT实时检查：`213716.tc6000`仍运行，账户未结束作业2个；bank到23,424/55,733，source errors=0。为减少下一阶段共享盘流量，pack生成器按唯一AGRI帧分组，同一时刻两站patch只打开一次AGRI/geometry源文件；同时记录3×3 AGRI有效像元比例、数组hash。修订版hash已重新同步并通过服务器Python编译和PBS shell语法检查。
- 17:05 BJT：PBS `213716.tc6000`继续运行，进度25,472/55,733，source errors=0，PBS账户仍有2个未结束作业。pack job继续等待，不与bank竞争同源I/O。
- 19:48 BJT：新R bank完成并核验：55,733帧完整（train 51,236/val 4,497），source errors=0，`test_payload_opened=false`；COT bank输出绑定R checkpoint SHA `342596…f001`。PBS账户实时查询只有`213666.tc6000`未完成，于是提交CPU数据连接作业`213781.tc6000`，开始生成同键GHI/AGRI/几何/COT 3×3 pack；账户当前未结束任务数2。待pack审计后再提交单卡profile/seed42训练。
- 2026-10-13 20:14 BJT：PBS 213781 live check: 8-core CPU pack is R, elapsed about 7m34s, log reached 6,001/54,937; pack manifest is not yet present. Account has 2 unfinished jobs (213666 and 213781). Do not overlap another shared-data reader; after completion audit counts/hashes/support then submit single-GPU GHI-head profile/training if queue remains below the 3-job cap.
