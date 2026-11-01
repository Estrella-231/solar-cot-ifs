# 湖南按月连续留块划分与重新训练

## 当前合同

用户于2026-10-02批准：不再把2025-10至2026-06整段作为测试集；跨全期资料按月留连续2天验证、另外连续2天测试，每个留块前后各1天缓冲，余下日期训练。此版本回答全年未见日期块泛化，不宣称向后跨年度泛化。旧清单、权重、历史结果不覆盖、不作新测试成绩。

新split ID：`hunan_monthly_block_2d_buffer1d_seed42_v1`。北京时间范围2024-04-02至2026-06-30。日期从既有已审计manifest的唯一UTC帧路径并集构建，不读取云况、CPP、GHI或预测误差来选日期；不补造既有清单未包含的孤立帧。

固定seed42；每个月的独立种子取SHA-256(`42:YYYY-MM`)前16位。候选日期要求一天至少48个已有帧；两个留块起点至少相隔6天，缓冲可跨月。缺少20个合格日或没有分开的候选对的月份仅训练，单独报告，不动态缩减留块来追求指标。

26个完整月份各留2个连续日验证和2个连续日测试。2025-09实际只有上旬，列为稀疏训练用月份；2024-09仍有验证、测试。可用日分配：train588 / val52 / test52 / buffer104。序列分配：train35320 / val3140 / test3202；对应唯一帧51236 / 4497 / 4536。窗口都包含8历史+16未来，15分钟步长；24帧同集合，跨边界或缓冲窗口丢弃。帧集合两两不相交。

Manifest SHA-256：`f6703ed5e42f1ae00aac6538b108dc4d9361197ad57ad86a0f407767aa2e4cb2`。

清单、日期分配、逐月数量、完整审计：`experiments/monthly_block_20261002/manifests_v1/`。测试数量仅为元数据统计；测试载荷和测试成绩未读取。新测试日期部分被旧模型训练或诊断使用过，故旧模型不能进入新协议的独立测试比较。

## 第一阶段：新SimVP基线

保持原geometry24_v2架构与13 AGRI+3几何输入；完整256×256，8历史→16未来，两段8帧递归；未来实际AGRI不进入模型输入。先不新增doy特征、COT损失或其他架构。随机seed42初始化，从新train唯一帧重算均值/标准差；12轮AdamW，初始lr1e-4、weight_decay1e-4、余弦到1e-6，BF16，梯度裁剪1。归一化AGRI有效像元MSE使用当前单卡脚本的pooled valid pixel定义。

选模只用新val。每轮保留月份及DJF/MAM/JJA/SON的像元RMSE，以四季RMSE等权平均选best；这项评价约定预先冻结，不用test选权重。旧全像元加权val分数同时报告，不能把旧、新不同cohort分数直接当提升。

先单卡真实训练样本profile：batch4/8/12/16/20/24/32，最多85%显存；在计算吞吐距最佳不超过5%的候选中选较大batch。profile报告为驻留批次计算吞吐，不伪称端到端吞吐；训练日志另记录真实轮耗时。全部模型调用验证24几何顺序。

## 服务器执行

服务器：`zjnu-hpc`。隔离根目录：

`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hunan_monthly_block_20261002`

在node23本地盘缓存原始13通道、QA和几何，每帧仅从共享盘读取并写缓存一次；归一化进程锁避免重复写。缓存目录`/tmp/slfu_hunan_monthly_f6703ed5e42f1ae0`，保留25GiB磁盘余量。node22仅剩约46GiB，不在该节点扩大新缓存。CPU归一化PBS后，用afterok依赖启动一张A100-40GB的profile及训练。每次提交前实时查全账户队列，未结束任务最多3，不终止其他作业。

源码快照中triad_models.py的SHA-256 `9273447d357bad92bcf3cd976b65fb4c440187da456fb56d5f36ded349c681eb`与旧geometry24_v2完全相同；hunan_data.py、simvp.py也独立保存并校验。所有输出命名为新合同，不接续旧checkpoint。

## 2026-10-02 提交与核验记录

本地与远端独立重建的manifest SHA-256相同。独立审查逐一核对41662条完整24帧序列、帧集合和缓冲：PASS；验证样本DJF785 / MAM949 / JJA882 / SON524。远端目标Python `--help`、所有源码py_compile、两份PBS bash -n、deployment.sha256逐文件核验均通过。首轮CPU准备作业 `211814.tc6000`，依赖GPU作业 `211815.tc6000`（afterok211814）；提交后快照为CPU Q、GPU H，未宣称训练已开始。提交前账户无未结束作业，第二次提交前仅有211814，总任务数2。

英文LaTeX与中文正文已同步更新划分、缓冲、重新拟合与四季选模口径。内置编译器返回 `Unable to find standard directories for platform`，因此当前修改未确认PDF编译成功；保留源文件，不安装环境或声称已出PDF。

准备作业初始normal队列的node23请求持续Q；两队列`acl_host_enable=False`，不能用acl_hosts列表断言拒绝原因。只对自己的排队作业211814执行`qalter -l nodes=node23:ppn=8:gpus=1`和`qmove gpu`，改为GPU节点对应队列，保留原作业号及afterok依赖，未取消其他作业。准备阶段仍由CPU计算归一化；GPU队列资源请求暂保留1卡，实际CUDA计算要到后续profile。实际资源覆盖记录保存在服务器launch.json；原提交PBS及deployment校验文件保持原样，不篡改历史。

19:43:20 BJT准备作业211814已启动R，node23；19:44～19:45运行见证显示部署hash项全OK，归一化至少处理3000/51236唯一训练帧、六个CPU worker运行、缓存由0增长至约15GiB。不以包含lock的文件总数冒充帧数。无日志错误；GPU训练211815仍H，未产生profile或训练成绩。此为准备运行见证，不是模型训练完成或性能证据。

## 后续合同与停止条件

1. 先验收新train归一化、单卡有限loss/梯度、checkpoint重载和新val四季支持，再推进SimVP训练。
2. R_region、CPP监督清单、历史/预测COT bank及GHI头必须按同一新日期分配重建。缺CPP不得缩小GHI公平比较样本；先做逐月CPP与GHI标签覆盖审计，再训练新R/H。旧R/H目前不合格于新独立test。
3. 四季基线完成后，独立命名DOY sin/cos输入对照；本次不同时改变输入来混淆数据划分与模型收益。
4. Test只在S/R/H全部新合同、分析协议冻结后一次正式评价；当前只统计test日期/路径，不训练选模、不算test指标。
5. 缺任一季验证、跨集合帧重复、缓冲污染、输入未来观测、归一化引用旧统计、磁盘余量不足或profile无稳定batch，立即停在相应阶段并保留证据。
