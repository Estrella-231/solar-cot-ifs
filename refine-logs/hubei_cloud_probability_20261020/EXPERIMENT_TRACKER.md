# 湖北云状态四组配对实验进度

更新：2026-10-20。范围仅湖北train/validation；test/buffer数值载荷保持关闭。

| 项目 | 状态 | 证据/当前处理 |
|---|---|---|
| 湖北既有判云器盘点 | 完成：未找到可复用湖北checkpoint或`p_cloud` bank | 已查湖北实验目录、代码与学院服务器可见Hubei资产；不复用湖南权重 |
| 太阳几何与站点索引 | PASS | 9个跨train/val原始AGRI样本公式最大差0；BJT−8h独立太阳位置SOZ差中位1.12°、最大2.39°；三站局部中心坐标沿用冻结映射 |
| CPP时间连接 | 原路径FAIL；修正UTC键已确定 | 输入BJT、CPP NetCDF UTC。旧路径有3,479/3,479同数字文件，但标签错8小时；本轮不使用旧R监督、权重或COT bank |
| 正确CPP/CLP支持 | PASS | FD-201 attempt2完整抽取3,479/3,479 train/val键，失败0；独立本地审计逐键验证BJT−8h UTC、manifest顺序、split、形状、掩膜、物理值范围及SHA。坐标误差0；test/buffer载荷读取0。sidecar SHA=`b27c1f9599591745c8bad85293c86b54ec3780a6b443760290542e1dca8982f0`，审计`cpp_attempt2_local/independent_audit.json` |
| GHI/AGRI共同支持 | 覆盖连接25,901行；输入质量过滤后四臂共同保留25,899行 | 3,450行因辅助源覆盖缺失；另2行因站点3×3内13个AGRI通道全零而从所有组一致剔除。无天气或结果筛选 |
| 湖北独立R训练 | 完成：正确UTC新R | 正确UTC CPP监督，随机初始化；COT银行覆盖22,061个train/val时刻。验证COT RMSE=13.67仅代表对CPP检索参考的一致性 |
| 湖北独立判云训练 | 完成：湖北随机初始化分类器及p_cloud银行 | 日期分组5折OOF + full-train，覆盖22,061个train/val时刻。CLP参考诊断POD=0.872、FAR=0.141、CSI=0.762，非独立真值精度 |
| 四组GHI配对 | attempt5共同包审计PASS；seed42/43/44训练完成，保存预测独立复算PASS | A/AC/AM/ACM共用3×3单帧输入、初始化、行键、标签和训练预算；该实验是同期GHI重建，不是未来预报 |

## 2026-10-20 全量正确UTC CPP抽取尝试

- FD-201 CPU attempt1：`/home/Data_Pool/zjnu/solar_cot_hubei_20261020/cpp_trainval_correctutc_attempt1/`，日志`/home/Data_Pool/zjnu/solar_cot_hubei_20261020/logs/cpp_extract_attempt1.log`。3,479个train/val键全部处理，3,424成功、55失败；失败类型为53个`RuntimeError('NetCDF: HDF error')`与2个同类`OSError`。由于脚本要求全键成功才发布压缩侧车，attempt1仅保留失败JSON，没有生成部分训练数据。
- attempt2：使用`extract_hubei_cpp_trainval_correctutc_retry_v2.py`，每键最多3次短退避重读、4个CPU worker；PID 2238278于2026-10-20完成，3,479/3,479成功、失败0。sidecar在`/home/Data_Pool/zjnu/solar_cot_hubei_20261020/cpp_trainval_correctutc_attempt2/`，日志`/home/Data_Pool/zjnu/solar_cot_hubei_20261020/logs/cpp_extract_attempt2.log`。这不是PBS/GPU任务。
- 独立本地审计`cpp_attempt2_local/independent_audit.json` PASS：train 3,178行/807,936有效CLP像元，val 301行/77,056有效像元；类别数train晴空305,854、云类1 347,688、云类2 154,394，val分别24,903、40,395、11,758。行序、UTC、mask、SHA一致，坐标误差0；test/buffer读取0。
- attempt1失败产物保留作读取故障诊断，不参与训练。重试版、建包、判云器、全AGRI推理、四臂建包和训练脚本均已通过本地`py_compile`。

## 队列与资源

历史队列行是先前快照，当前状态见下方“2026-10-20 当前执行审计”；每次PBS提交前后仍需实时查队列。

## 下一步

1. AGRI转运已完成；PBS名额释放后，qsub前后均查`slfu`全队列。先由单卡作业依次训练湖北R和湖北判云器，生成正确UTC COT银行与日期OOF/full-train `p_cloud`银行。
2. 银行完成后CPU建包并独立审计共同支持：源GHI 29,351行中按覆盖保留25,901行；逐键核验COT/p_cloud、原AGRI/geometry/GHI/clear-sky值、行序及split。
3. pack审计通过后再次查PBS全队列，再跑A/AC/AM/ACM四臂、三站、三个seed；逐样本预测FP64复算总体/逐站指标、日期块配对区间及CPP支持内COT/CLP分层。CPP参考覆盖以2026-03-02为上限；总体GHI成绩覆盖全部共同AGRI来源支持。

## 2026-10-20 当前执行审计

- FD-201可读冻结清单中的AGRI样本；ZJNU不挂载`/home/Data_Pool_3`。train/val清单对应的22,061帧现已转运到ZJNU，远端归档和staged manifest审计PASS。
- 共同支持预审计：源GHI 29,351行、AGRI源键22,061个，交集25,901行；缺源3,450行、1,481个时刻键，均为train/val来源缺失。三个站点train/val均有充足训练支持；无test/buffer数据进入。
- `build_hubei_fourarm_ghi_pack.py`现按两种银行键做共同支持连接，并保持原GHI行序；合成测试通过。`audit_hubei_fourarm_ghi_pack.py`将独立比较源/目标数组及辅助场裁块。
- 最新PBS队列实时查询显示3个运行任务：215161.tc6000、215162.tc6000、215163.tc6000；无新任务提交。node22显示节点free及8 GPU，但具体卡需以作业分配为准。node22/A800属于ZJNU学院服务器。

## 2026-10-20 15:34 CST 执行更新

- 最新实时 `qstat -u slfu`：`215161.tc6000`、`215162.tc6000`、`215163.tc6000` 均为运行态R；当前已达3个未结束任务上限，因此本轮没有qsub，判云/R训练均未启动。
- AGRI转运已完成。归档含22,061个train/val文件（train 20,206、val 1,855），归档SHA-256=`d16a026e007f112c5ad82cb8bdf24272867329f795f89d771e46dae05f6c46e2`；staged manifest SHA-256=`8b1f95c2a2cf1d66cabd11539b975fe3dccd486146bf3bec17fb18e537f72183`。
- ZJNU远端 `staged_agri_audit_attempt2.json` 独立审计PASS；本地归档副本为 `agri_stage_attempt1/staged_agri_audit_attempt2.json`，审计文件SHA-256=`525f6e71c521b5526e7a7f9da2708ea8c50c6a699ace8bbd35b3e2f57b36334`。抽查训练与验证端点，patch为20通道16×16，几何字段齐全且有限；test/buffer载荷读取/转运均为0。
- 本地 `py_compile` 覆盖code_attempt1全部Python脚本；四组pack builder合成测试1/1通过。ZJNU远端 `sha256sum -c code_attempt1/SHA256SUMS` 全部OK，银行训练PBS脚本 `bash -n` 通过，SHA-256=`a416bbeda685aa52ccfce13ef46644d54f847694e02bac71eddbf3852c178059`。四臂seed训练PBS脚本已同步远端并通过`bash -n`，SHA-256=`4b42291c098de0ae1da51a9956c92f30d73595ede53bad20b2e403bf041cd256`；该脚本仍需等待银行和共同pack审计完成后才可提交。
- 提交顺序保持：待名额释放后运行单卡R+分类器/银行作业；CPU建包与审计通过后，另行核队列再提交四臂GHI训练。node22/A800是浙师大学院服务器；目标卡以PBS实际分配结果为准。

## 2026-10-20 18:35 CST 执行更新

- 提交前 `qstat -u slfu` 实时为空；`215205.tc6000` 于18:35:25提交并开始运行。提交后即时全队列仅有该作业，状态`R`，未超过3个未结束任务上限。
- PBS实际分配node22的8个CPU线程及`node22-gpu/1`；在计算节点核实GPU 1为NVIDIA A800-SXM4-80GB，显存80 GiB。该卡属于浙师大学院服务器。
- 日志：`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cloud_probability_20261020/logs/banks_attempt1.log`。初次核验日志显示作业处于启动和输入校验阶段；当时GPU利用率为0%，因此仅确认PBS作业已运行，尚未确认模型训练已进入R或分类器阶段。
- 该作业只处理湖北train/val，test/buffer载荷保持关闭。四臂GHI训练尚未提交，须等银行生成、共同pack建立及独立审计通过后再单独核队列提交。

## 2026-10-20 18:54 CST 失败诊断与重跑状态

- Attempt 1（`215205.tc6000`）在分类器训练前失败：旧代码把CPP/CLP侧车和R pack逐行比较，误把不同文件的行序差异当作时刻错配。独立逐键复核确认两边均为3,479个唯一键，键集合相同、每个BJT键对应的train/val划分相同；仅2,663行的顺序不同。判云训练、R训练和银行推理都未开始；attempt1日志及空输出目录保留。
- Attempt 2（`215216.tc6000`）在CPU测试启动阶段失败：PBS用`python -m unittest`加载绝对文件路径，模块导入失败，退出码1；没有进入模型训练，也没有生成模型产物。已改为直接运行测试文件。
- 修正版key-join使用唯一`time_bjt`键重排CLP/掩膜至R pack行序，并逐键核验split；键缺失、重复或split冲突仍会立即报错。服务器乱序/重复键/划分冲突单元测试3/3通过；实际3,479行训练验证包审计通过，报告`sidecar_rows_reordered=2663`。代码哈希清单、py_compile及PBS bash语法检查通过。
- Attempt 3（`215217.tc6000`）于18:54:07提交。提交前`qstat -u slfu`无未结束作业；提交后该作业状态`R`，全队列当前仅此1个未结束作业。实际分配node22 GPU 1（NVIDIA A800-SXM4-80GB，80 GiB），浙师大学院服务器。日志：`.../experiments/hubei_cloud_probability_20261020/logs/banks_attempt3.log`。最新已见单元测试通过；当前还未确认GPU训练epoch开始，后续以该日志为准。
- test/buffer数值载荷未读取。四臂GHI头尚未提交；需等R、判云/p_cloud与COT银行完成，再CPU建立并独立审计共同pack。

## 2026-10-20 18:56 CST 银行作业阶段更新

- `215217.tc6000`仍为运行态，当前全队列只有该作业；已完成正确AGRI侧车审计、3项key-join测试、真实3,479行CLP键对齐审计、五折日期留出判云训练及full-train模型拟合。
- 判云验证诊断（CPP/CLP有效像元，阈值0.5）：n=66,630，POD=0.8717、FAR=0.1414、CSI=0.7623、Brier=0.1486。它衡量与CPP检索参考CLP的一致性，不是独立真值精度或GHI增益证据。
- 当前正在对22,061个AGRI train/val键推理连续`p_cloud`；日志最近进度为4,096/22,061。该阶段完成后脚本才会继续训练正确UTC湖北R并生成COT bank。test/buffer数值载荷仍未读取。

## 2026-10-20 19:37 CST：输入质量复核与GHI训练前就绪状态

- `215217.tc6000`已完成正确UTC湖北R、判云器及22,061帧train/val的COT与`p_cloud`银行；CPP/CLP比较只作为检索参考一致性诊断。
- 四臂覆盖连接初始共有25,901行。站点COT尾部审计发现王寨与米庄`202511221400`训练输入COT分别约89,882和99,490。原始16×16 AGRI帧逐通道审计确认13个光谱通道均为精确0，虽白天掩膜有效；因此这不是可用云物理输入。分类器对该异常帧输出`p_cloud=1`，R也产生极端COT。
- 为避免将无光谱帧当作云信号，attempt5构建时按固定输入质量规则从**全部四组共同样本**剔除这两个站点行；原始GHI pack和旧attempt3/attempt4包均保留。质量规则不使用云况、GHI误差或验证成绩，未剔除任何验证行。
- Attempt5独立审计PASS：原GHI 29,351行；AGRI/COT/p_cloud共同覆盖25,901行，另3,450行仅因源覆盖缺失；质量筛选后25,899行（train 23,704、val 2,195）。源AGRI、几何、GHI和晴空值在保留行逐元素相同；COT、判云值按站点/time/split及3×3裁块重核通过。test/buffer载荷读取为0。
- Attempt5 COT尾部审计：训练、验证站点3×3均无COT>1,000；三站训练最大约240、209、243。GHI头仍按既定train-only `log1p`标准化，不对物理值做任意裁剪。
- GHI训练代码快照`code_attempt4`的远端SHA256SUMS全部通过；四臂PBS脚本`run_hubei_fourarm_ghi_seeds_attempt2.pbs`通过`bash -n`，指向attempt5 pack、独立审计和新seed输出目录。PBS提交尚未执行；需在提交前再次实时查队列。

## 2026-10-20 19:55 CST：四臂GHI训练完成及验证预测独立复算

- PBS作业`215227.tc6000`已完成。日志末尾包含`HUBEI_FOURARM_GHI_ALL_SEEDS_COMPLETE test_payloads_read=0 buffer_payloads_read=0`；seed42/43/44的36个头均完成30轮并保存checkpoint和验证预测。实际设备为node22的`node22-gpu/1`，NVIDIA A800-SXM4-80GB（80 GiB），属于浙师大学院服务器。远端完整日志：`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cloud_probability_20261020/logs/fourarm_seeds_attempt2.log`。
- 独立复算代码为`score_hubei_fourarm_predictions.py`，输出在`results_fourarm_attempt2/`。36个验证预测和36个checkpoint哈希均与训练元数据一致；所有checkpoint重载复现预测的最大差为0；三站×三seed×四臂的验证行键与attempt5 pack逐键相同。GHI真值与晴空值相对pack的最大差分别为4.88e−5、3.05e−5 W/m²。全流程只读train/val，test/buffer读取为0。
- 全部2,195个共同验证站点-时次上，先逐站、逐seed算RMSE，再对三站和三seed等权平均：A=77.285 W/m²；AC（仅加预测COT）=69.142，较A下降10.54%（−8.143 W/m²）；AM（仅加连续判云概率）=74.053，下降4.18%（−3.232 W/m²）；ACM（两者都加）=68.815，下降10.96%（−8.470 W/m²）。20,000次配对两日日期块bootstrap（10块）中，AC−A改善区间为4.40%至18.91%，AM−A为2.40%至6.71%，ACM−A为5.38%至18.66%。ACM相对AC点估计只多改善0.47%，区间−1.23%至2.62%，没有显示总体上在COT之上再加判云概率有增益。
- 逐站三seed均值RMSE：Niushou A 76.851→AC 63.978（改善16.7%）；Wangzhai 70.539→63.137（10.5%）；Mizhuang 84.466→80.312（4.9%）。AC相对A在9个站点×seed组合中9/9改善；AM是8/9，ACM是9/9。
- CPP有效站点中心COT分层仅用于诊断，CPP是检索参考而非独立真值。厚云参考层[30,100]共173个站点-时次、4个两日日期块：AC RMSE从57.139降到34.527（改善39.6%；bootstrap区间28.0%至50.1%）；但ACM比AC高3.095 W/m²（改善率−8.97%，区间−14.32%至−3.22%）。分层结果只由4个日期块支撑，属于探索性观察。
- 薄云参考层[0.1,5)仅71个站点-时次、3个两日块：AC相对A点估计改善8.89%，但区间−3.22%至10.11%，不能认定稳定提升。分站均值中COT单独加入三站都未退化（Niushou +23.1%、Wangzhai +1.7%、Mizhuang +3.1%）；连续判云概率的分站结果并不一致，例如Mizhuang薄云层AM约退化6.6%、ACM约退化1.5%。
- CPP站点中心COT有效分层共723/2,195个验证站点-时次，受CPP可用时段及像元有效mask限制：2025-08/09有效覆盖分别为227/244、214/238行，2025-10至2026-03每月仅42–53行，2026-04/05为0。月度逐项覆盖表在`results_fourarm_attempt2/cpp_reference_coverage_by_month.csv`。总体四臂成绩仍覆盖全部2,195个共同验证行，不依赖CPP覆盖筛选。各厚度层仅有3–6个两日块，CI应谨慎解释。
- 该对照是**单帧3×3站点邻域的同期GHI重建**，不是未来GHI预报。每个头按本验证集选择checkpoint，所以日期块区间条件于已经选出的checkpoint；验证结果不是独立test，不能作为最终泛化结论。test/buffer仍关闭。
- 19:55 CST `qstat -u slfu`显示本四臂作业已退出；队列另有`215229.tc6000`运行态，本实验未对其进行任何操作。
