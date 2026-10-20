# 湖北三站判云条件输入重复验证

2026-10-20：用户明确要求在湖北三站复核湖南薄云退化是否区域特异。本授权覆盖湖南方案中“湖南C1通过后才开展湖北”的条件；湖南负结果全部保留。

## 问题与控制
比较牛首、王寨、米庄的同期GHI：AGRI+几何、加COT、加判云得分、加两者，四组每站seed42/43/44，共36个GHI头。仅判云/COT通道开关变化，公共容量、初始化、行顺序、训练预算和总体选模准则一致。判云作为连续独立通道，不乘COT、不按验证表现调阈值。湖南结果不并入训练，不迁移湖南权重。

复用湖北冻结日期划分与已完成COT/GHI资产；沿用湖南三帧3×3方法前先核验湖北现有包的时间维度，不能把单帧包直接称为同结构重复。需要三帧时从现有train/val按精确时间重组，四臂共同支持一次冻结。test/buffer载荷不读取。

## 已核对资产
学院服务器zjnu-hpc实时可达，hostname=tc6000。本轮qstat -u slfu无作业输出，未提交任务；后续提交前重新核查。
- R训练包：`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_r_runtime_gate_20261015_v1/r_pack_trainval`，3178 train、301 val。**2026-10-20追加的时间溯源审计发现其CPP路径按BJT原字符串匹配UTC文件，旧target/mask及其R checkpoint/COT bank不用于本次新对照**；目录仅作为AGRI/几何输入与旧结果来源保留。见同目录 `AUDIT_GEOMETRY_CPP_ALIGNMENT_20261020.md`。
- COT bank：同父目录cot_bank_trainval。
- GHI包：/public/home/slfu/ttzhou/swc/irradiance_forecast/hubei_al_same_time_20261017/pack，agri_raw/cot/geometry/ghi_wm2/clear_sky_wm2/rows/pack_report存在。
- CPP原始来源索引保存在R包pack_status.json，示例源根为/home/Data_Pool_3/chenyi/Auxiliary_data/HuBei_AGRI_Preprocessed/cpp_16_fixed_v2。
- 当前核查的湖北目录与可见服务器实验树未找到湖北专属判云分类器或p_cloud bank；这是有界搜索结果，不表示全服务器不存在。正确BJT−8h原始CPP NetCDF具有3,479/3,479个旧R train/val键覆盖，可提供CLP监督；不得用COT>0替代CLP类别。

## 执行顺序
1. 追溯湖北CLP字段、质量mask、时区与原生网格，清点train/val与三站覆盖；核查现有判云产物是否另存。
2. 若无独立湖北判云器，复用已批准分类结构在湖北train训练；先核验正确太阳几何，不继承湖南旧几何误差。按训练日期分组生成OOF得分，完整train模型仅推理val。所有归一化train-only。保留训练和验证判云质量、薄云/厚云召回；不依据验证结果反复调阈值。
3. 当前湖北GHI包为单帧，故本轮按同一单帧3×3包运行四组，不称为三帧预报。AGRI来源清单有22,061个训练/验证时刻。与原GHI包按station/time/split求共同支持后，保留25,901行，排除3,450行（1,481个时刻键），排除原因仅为原AGRI侧车来源缺失；所有四臂共享该子集。复核保留行的AGRI、几何、GHI、晴空值、行序和split逐项相同。复核两个输入开关与共同初始化，单卡实测batch/吞吐后运行36头。PBS每次提交前后查全队列，最多3个未结束任务。
4. 保存逐样本预测独立FP64复算：三站分别、等站等seed；两日BJT日期块配对区间。CPP站点中心厚度沿用[0,.1)、[.1,5)、[5,30)、[30,100]，缺参考不剔除总体GHI样本。补参考覆盖/各月/每层日期数；不得以湖北好结果替换湖南负结果。
5. 薄云逐站/种子/日期误差与真实案例，比较判云错误、COT幅值、GHI修正方向。两省方向不同只能提示区域/数据依赖，不能据此归咎仪器，也不能把三个相近湖北站当成三个独立气候区。

## 初始状态（2026-10-20早期记录；以下审计更新覆盖）
资产核对与方案已建立；历史太阳几何已通过复核。该时点旧R/CPP时间连接仍待修正，湖北R监督与判云器训练均未启动。后续正确UTC重建、训练数据转运和当前PBS状态见本文后续更新。不得重复已经完成的湖北A/AC历史实验作为新结果。

## 2026-10-20 时间与模型来源审计修订（优先于上文旧资产假设）

### 太阳几何与时区

- 湖北AGRI/站点样本键是BJT；CPP原始NetCDF路径时间是UTC。正确连接为 `UTC = BJT - 8 h`。旧R manifest把BJT数字串直接拼入CPP路径，虽然3,479/3,479个同数字文件存在，却错配8小时；这批旧R监督值、R checkpoint、COT bank不进入本次四臂实验，也不覆盖删除。
- 9个跨train/val时段原始AGRI样本的 `cosSOZ=cos(SOZ)`、`cosRAA=cos(SOA−SAA)`、`day_mask=(raw day_mask>0.5)` 均逐像元精确一致（最大绝对误差均为0）。按BJT−8h计算的独立名义太阳位置SOZ误差中位数1.12°、最大2.39°；误将BJT当UTC则中位数63.90°、最大90.69°。因此本实验使用经核验的AGRI几何通道；不重算或替换原AGRI角度通道。
- 站点在16×16 patch中的中心像元继续采用冻结映射：牛首(8,6)、王寨(9,8)、米庄(7,10)。CPP空间映射依据CPP NetCDF LAT/LON与AGRI `grid_static` 经纬网格逐像元最近邻并要求坐标误差小于1e−4°；已有同一正确UTC样本与fixed-v2裁片逐像元一致，最大差0。

### 分类器与监督源

- 湖北专属判云权重/`p_cloud` bank在已搜索的湖北项目输出中未发现。新建湖北随机初始化判云器，结构仅复用已审核的分类网络形式（16输入通道：13 AGRI + `cosSOZ` + `cosRAA` + `day_mask`；输出16×16 sigmoid云概率），不加载湖南权重。
- 监督来自正确UTC CPP NetCDF中的CLP：0=晴空，1/2=水/冰云；仅有效类别、白天且几何有效像元参加损失。CLP缺测/其他类别保持无效，不从COT、GHI或误差反推类别。训练归一化只用湖北train。训练部分按日期分组生成OOF云概率；train中没有CLP标签的键由full-train模型推理，val也只由full-train模型推理。验证云分类指标仅在CPP/CLP有效的时间支持上报告，不外推为缺参考月份的分类准确率。

### COT重建与四臂边界

- 先按正确UTC原始CPP COT重建湖北R监督包并从随机初始化独立训练湖北R；新COT bank覆盖22,061个现有train/val AGRI源键，推理时不读取CPP/GHI标签。原始CPP目前只覆盖到2026-03-02；晚于该时刻的GHI行仍保留在共同支持中，但R/分类器在其上属于模型外推应用，且没有同期CPP/CLP参考评分。
- 四组实际比较为：13通道AGRI加`cosSOZ`/`cosRAA`，再分别加入预测COT、连续判云概率或两者。判云分类器以`day_mask`参与标签有效像元筛选；GHI四臂的两个辅助通道只对应COT和判云概率。全组固定同一个3×3单帧湖北GHI数据包、station/time键、标签、训练顺序、初始化、更新数和头部容量；每站三个seed。当前包是单帧同期GHI反演，不称三帧预报，也不以真实未来CPP作为输入。
- 当前有效COT/GHI pack存在错时旧输入，不可直接用作AC/联合组。新pack只能由正确时间训练出的湖北R和湖北判云器产生；保留共同支持行上的AGRI、几何、GHI标签、晴空值、行键和split必须与源pack逐项相等。CPP监督覆盖与AGRI源覆盖分别报告；不因CPP参考缺失剔除GHI行，不填0。
- 作业顺序：CPU生成/审计正确UTC CPP标签包 → 一张卡训练R与日期OOF/full-train判云器并推理train/val bank → CPU逐键pack对照和代码/数据见证 → 单卡profile → 36次配对GHI头。每次PBS提交前后实时查 `qstat -u slfu`，最多3个未结束任务；使用节点必须以PBS分配/实际GPU信息为准，并注明node22 A800属于浙师大学院服务器。

## 2026-10-20 18:35 CST 执行状态

湖北正确UTC R与独立判云器/银行作业已提交为 `215205.tc6000`。提交前队列为空，提交后队列仅含该作业且状态为运行态`R`。PBS分配node22、8个CPU线程及GPU 1；计算节点实时查询确认GPU 1为NVIDIA A800-SXM4-80GB（80 GiB），属于浙师大学院服务器。启动日志位于远端`logs/banks_attempt1.log`。初次检查确认作业已进入启动和输入校验阶段，但GPU利用率仍为0%，因此目前只报告作业启动，待后续日志确认训练实际进入R或分类器阶段。作业仅处理train/val，test/buffer数据载荷仍关闭；四臂GHI作业未提交。

### 18:54 CST 失败修复与第三次提交

Attempt 1（`215205.tc6000`）的失败来自逐行对齐检查过严：R pack和CPP/CLP侧车的3,479个BJT键集合及逐键split完全相同，但文件排序不同，2,663行顺序不一致。新代码按唯一BJT键对齐CLP与mask，仍严格拒绝缺键、重复键及逐键split不一致。Attempt 2（`215216.tc6000`）只因PBS测试命令错误调用`unittest`绝对文件路径而退出；修正为直接执行测试文件。乱序、重复键、split冲突测试3/3通过，真实训练验证键连接审计PASS，代码哈希、Python编译和PBS bash语法检查通过。

第三次作业`215217.tc6000`于18:54:07提交，当前PBS状态`R`，实际分配浙师大学院服务器node22 GPU 1（NVIDIA A800-SXM4-80GB，80 GiB）。日志为远端`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cloud_probability_20261020/logs/banks_attempt3.log`。已见对齐单元测试通过，尚待日志确认真实GPU训练启动；R训练安排在判云器和概率银行之后。test/buffer数值载荷保持关闭。

### 18:56 CST 银行阶段进度

作业已完成train日期分组5折OOF判云训练和全train判云模型拟合。仅作为与CPP/CLP检索参考的一致性诊断，验证有效像元n=66,630，POD 0.8717、FAR 0.1414、CSI 0.7623、Brier 0.1486，不作独立真值或GHI收益结论。当前正在对22,061个AGRI train/val键生成`p_cloud` bank，日志进度4,096/22,061；其后再训练正确UTC R和COT bank。test/buffer载荷仍关闭。

## 2026-10-20 执行更新（历史记录；以末尾最新状态为准）

- 湖北判云器资产检索结论仍为：项目范围内未找到湖北专属权重或`p_cloud` bank；不复用湖南模型。正确UTC CLP侧车和R pack已独立审计通过，分类器与R训练代码均已准备，但尚未提交GPU训练。
- 原始AGRI路径位于FD-201；`zjnu-hpc`不挂载该目录。FD-201可读所有源清单路径；当时正在将仅含22,061个train/val文件的归档转运到学院服务器，以便在同一训练服务器上生成完整预测银行。test/buffer从清单和归档中排除。
- 当前GHI原包29,351行与AGRI源清单相交为25,901行；缺源3,450行、1,481时刻。已将pack builder改成覆盖支持内连接，并新增独立逐键审计；本地合成测试通过。
- 实时PBS检查曾见`215160.tc6000`状态C、`215161.tc6000`、`215162.tc6000`、`215163.tc6000`状态R。按未结束任务计数为3，不能提交。`node22`标识为ZJNU服务器节点，报告显示8 GPU；在作业分配前不假定其空闲。不得把FD-201的GPU计入或冒充ZJNU资源。

## 2026-10-20 15:34 CST 最新执行状态

- 湖北判云分类器在已查湖北项目目录、代码及ZJNU可见Hubei资产中未找到；新分类器将以湖北数据随机初始化训练，太阳几何已通过独立核对。
- train/val AGRI归档现已完成转运。22,061帧（train 20,206、val 1,855）远端抽查审计通过；归档SHA-256=`d16a026e007f112c5ad82cb8bdf24272867329f795f89d771e46dae05f6c46e2`，staged manifest SHA-256=`8b1f95c2a2cf1d66cabd11539b975fe3dccd486146bf3bec17fb18e537f72183`。远端独立报告`staged_agri_audit_attempt2.json`已复制到`agri_stage_attempt1/`；其状态为`PASS_HUBEI_AGRI_STAGED_TRAINVAL_AUDIT_TEST_CLOSED`，test/buffer载荷读取和转运均为0。
- 本地所有Python脚本`py_compile`通过；四组pack builder合成测试1/1通过。ZJNU端训练代码SHA清单全部通过，银行作业脚本语法检查通过。
- 最新实时队列中`215161.tc6000`、`215162.tc6000`、`215163.tc6000`均为运行态R，已达到每账户最多3个未结束任务。本次未提交任何新PBS作业，R和判云器尚未训练。PBS银行作业已准备好：名额释放后先单卡重训正确UTC湖北R与独立湖北判云器、生成两个train/val银行，再CPU建包及独立审计；审计通过后单独核队列提交四臂GHI配对训练。银行PBS脚本SHA-256=`a416bbeda685aa52ccfce13ef46644d54f847694e02bac71eddbf3852c178059`；四臂seed PBS脚本已同步并通过语法检查，SHA-256=`4b42291c098de0ae1da51a9956c92f30d73595ede53bad20b2e403bf041cd256`。`node22`上的A800属于浙师大学院服务器，实际GPU须以PBS分配结果确认。

## 2026-10-20 输入有效性执行补充

在统一COT与判云银行完成后，站点3×3输入尾部审计发现王寨、米庄在BJT 2025-11-22 14:00对应同一训练帧中，全部13个AGRI光谱通道在整个16×16 patch均为精确0，而day_mask为白天有效。该帧经R反演得到极端站点COT、分类器输出云概率1，不能作为有效卫星光谱观测输入。

因此四臂共同pack按纯输入有效性规则剔除该时刻两条已有站点记录，所有方法共用同一支持，原始包和旧pack均保留。此规则不查看云类型、CPP COT分层、GHI误差或模型表现，不剔除验证样本，不作COT数值裁剪。独立审计后pack为25,899行（train 23,704、validation 2,195）；另有3,450行仅因AGRI/COT/判云辅助源覆盖不足而不在比较共同支持中。之后四组GHI头均必须使用该审计包。
