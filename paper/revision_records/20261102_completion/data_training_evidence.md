# 表2、表3数据与训练来源补齐（2026-11-02）

本记录只汇总当前主线的既有train/validation记录及模型元数据。未读test载荷，未提交训练，未修改主稿。湖南GHI日期已通过只读SSH从实际训练pack行表统计；湖北使用已恢复本地pack。文中日期范围表示首末保留目标日期，不表示区间内连续无缺测。

## 1. 当前真实同期三帧GHI头：可直接填写表2

| 区域/站点 | train行数 | train有效日 | validation行数 | validation有效日 | train目标日期首末 | validation目标日期首末 |
|---|---:|---:|---:|---:|---|---|
| 湖北/牛首 | 6,776 | 214 | 663 | 20 | 2025-08-01—2026-05-26 | 2025-08-22—2026-05-04 |
| 湖北/王寨 | 6,660 | 212 | 607 | 19 | 同上 | 同上 |
| 湖北/米庄 | 6,599 | 213 | 635 | 20 | 同上 | 同上 |
| 湖北/合计与日期并集 | 20,035 | 214 | 1,905 | 20 | 同上 | 同上 |
| 湖南/四里 | 22,930 | 508 | 1,999 | 43 | 2024-04-02—2026-06-30 | 2024-04-19—2026-06-25 |
| 湖南/竺家 | 24,736 | 564 | 2,227 | 50 | 同上 | 同上 |
| 湖南/合计与日期并集 | 47,666 | 578 | 4,226 | 51 | 同上 | 同上 |

- 所有行是一条站点15分钟GHI目标和三个同期AGRI输入，不是原始一分钟记录数、卫星独立场次数或forecast窗口数。
- 湖南原始val为4,263行/52天；按既有质量规则剔除四里2024-09-05的37行后为4,226/51天。剔除依据是先前确认的观测异常日，不能描述为按模型效果筛选。train不受影响。源结果的minimum validation RMSE选模也使用此保留集。
- 湖北原始记录主要一分钟；先对同一右闭一分钟箱内记录求均值，再对15个完整分钟箱等权平均。湖南为5分钟区间末时刻记录，以target-10、target-5、target三个记录等权平均。AGRI步长均15分钟。
- 原始源记录覆盖可以更长；上表是当前实际用于GHI头的覆盖，不能把外层日期清单或DyDiff影像序列总数当作GHI有效样本数。

直接产物：
- 本目录hunan_ghi_trainval_keys.csv：仅split、station、target_time_bjt、excluded_station_day，适合绘图1日期rug。excluded_station_day=True的记录不属于当前保留验证集。
- 本目录hunan_ghi_trainval_rows_metadata.json：来自真实pack的日期集合/计数/pack_manifest，包含行键而不含GHI标签数值。
- 本目录hubei_ghi_trainval_summary.json：从当前本地train/val行表独立统计。
- 湖北来源：refine-logs/hubei_same_time_geometry_20261102/recovered/pack/rows.csv及COMPLETE.json。
- 湖南来源（远端）：/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/cot_interval_history_20261015_v1/data_attempt2/rows.csv和pack_manifest.json。
- 湖南结果核对：refine-logs/cloud_probability_input_20261020/results/four_arm_paired_v1_attempt1/setup.json、input_source_manifest.json、seed42/{sili,zhujia}/M00/metrics.json。实际权重与configs/hunan_active_ghi_heads.json一致。
- 原生采样/聚合实现：scripts/build_hubei_ghi15_trainval.py（window字段）、scripts/hubei_ghi15_contract.py；湖南定义见工作区IRRADIANCE_ERROR_PREVENTION.md及AGENTS.md。

## 2. 外层日期划分

两区都采用每个合格月连续2天validation、另外连续2天test，留出块前后1天buffer，余下为train；少于20个合格日或不能形成分离块的月份仅训练。湖北日候选的阈值是每天至少48个卫星目标时次，选择只用时间/路径元数据，不用GHI值或模型误差。湖北留块起点至少间隔6天，沿用seed42月规则并保留已冻结日期归属。全24帧预测窗口和三帧同期上下文不跨分区。

- 湖南协议：docs/MONTHLY_BLOCK_SPLIT_20261002.md；外层train/val可用日588/52，卫星序列35,320/3,140。GHI有效集更小，原因包括标签与上下文有效性。
- 湖北最终清单：refine-logs/hubei_cot_independent_20261014/frozen_split_20261015_v1/day_assignment.csv和FROZEN_REPORT.json。外层2025-08-01—2026-07-14，train250天、val22天；当前GHI头保留214/20天。
- 不使用EXPERIMENT_PLAN.md开头曾存在的“两个月暂定划分”描述，它已被同文件2026-10-15冻结说明覆盖。
- test仅保持预留。以上读取是既有日期元数据，不是test评价；当前文稿不能加入独立测试成绩。

## 3. R的真实设置及区域差异

| 项目 | 湖北R | 湖南R |
|---|---|---|
| 输入/输出 | 16×16；13 AGRI＋cosSOZ/cosRAA/day_mask；softplus输出log1p(COT) | 相同 |
| U-Net通道宽度 | 16/32/64 | 32/64/128 |
| 实际batch | 128 | 1,284 |
| 监督train/val patch | 2,738 / 264 | 15,414 / 1,398 |
| train/val有效像元 | 699,023 / 67,187 | 3,886,399 / 352,441 |
| 归一化拟合范围 | 3,178个train输入patch的所有位置、全部16通道 | 共同有效train CPP像元；前15通道标准化，day_mask保持0/1 |
| 优化/损失 | AdamW lr=1e-3，weight decay=1e-4；有效像元Huber，delta=0.1；梯度裁剪5 | 相同 |
| 预算/选模 | 最多50轮，patience8，按validation有效像元log-COT Huber选模；实际30轮结束 | 相同预算/选模；实际31轮结束（last_epoch=30是零基索引） |

湖北3,479个R候选pack记录中train/val分别3,178/301；其中440/37行没有有效CPP监督，因此实际参与损失为2,738/264；不得把3,178/301直接写成监督样本量。湖南CPP覆盖子集不是湖南全量GHI集。

来源：
- 本目录regional_R_metadata.json保存本次只读取得的实际R run_metadata/status/norm/pack_status和来源路径。
- 湖北：refine-logs/hubei_solar_geometry_repair_20261021/code_pipeline_attempt12/run_stages.py（显式batch128，覆盖训练脚本默认192），train_corrected_r.py（AdamW、损失、选模）；实际metadata路径写入regional_R_metadata.json。
- 湖南：experiments/r_solar_geometry_corrected_20261015/results/paired_v1/corrected/run_metadata.json、status.json和train_paired_r.py。当前下游绑定corrected R，不能复用10月13日旧R的batch或结束epoch。

## 4. CPP参考来源与有效掩膜

当前两区R使用已有归档FY-4B/AGRI云物理属性NetCDF参考。归档根为/home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP，文件命名FY4B_AGRI_UTC.nc；CPP经纬轴与AGRI网格逐像元匹配。COT是无量纲，范围0—100；CLP为0晴空、1水云、2冰云。当前数据不应自动写成国家卫星气象中心的CPD业务产品，也没有依据写官方质量位筛选。

- 共同基本条件：COT有限且0≤COT≤100，日间且太阳天顶角<80°；缺测保持缺测，合法零值保留。
- 湖南另要求13个AGRI通道有效、几何有限、CLP属于0/1/2；当前R沿用首次包的输入有效掩膜并与当前太阳几何有效区相交。
- 湖北COT sidecar的数值掩膜仅检查COT有限及0—100；CLP有效性单独保存，当前R构建未将CLP再AND入COT掩膜。两区不能笼统写成完全相同的CLP质量筛选。
- R监督/CPP分层仅在参考有效像元上做；主GHI评价不因CPP缺档排除样本；实际预报不读取未来CPP。
- 精确CPP生产批次/权重不能仅由邻近模型目录或用户后来提供的V2包推断；正文可用“归档卫星检索参考”，精确生产来源和版本局限放补充来源说明，不伪造DOI。

来源：experiments/monthly_R_station16_20261013_v1/scripts/pack_station16_cpp_agri.py；experiments/r_solar_geometry_corrected_20261015/build_paired_geometry_pack.py；refine-logs/hubei_cloud_probability_20261020/extract_hubei_cpp_trainval_correctutc.py；hubei_solar_geometry_repair_20261021/code_pipeline_attempt12/build_corrected_data.py；docs/CPP_QUALITY_PROVENANCE_20260905.md和CPP_PRODUCER_CODE_REVIEW_20260917.md。时间及数组采用现有归档，未重读任何原始test产品。

## 5. GHI头实际训练设置

两区均FourArmIntervalHead当前A/AC输入，3帧×17槽×3×3，42,529参数；真实同期图训练；冻结R；30epoch；AdamW lr1e-3/wd1e-4；余弦衰减到1e-5；梯度裁剪5；seed42/43/44；以各站validation GHI RMSE最小选模。物理GHI MSE除以1000²；softplus预测kt，不把kt封顶1。

- 湖北batch512；牛首每轮14次更新，王寨/米庄13次，30轮分别420/390/390次。每站分别用本站train拟合AGRI和log-COT统计，几何原物理值保留。
- 湖南batch1792；四里/竺家分别每轮13/14次，30轮390/420次。使用既有两站共享train AGRI统计；COT共享train标度固定为mean1.8354828989/std1.3974485910（由已批准的旧/当前R配对train输出联合拟合，来源细节留在补充/本记录），几何保留物理值。
- 同区同站A/AC使用同样初始化、行顺序、优化设置和选模；不能写“湖北和湖南batch/归一化完全相同”。

来源：湖北refine-logs/hubei_same_time_geometry_20261102/pipeline.py和recovered/training/*/COMPLETE.json；湖南refine-logs/cloud_probability_input_20261020/train_four_arm.py及results/four_arm_paired_v1_attempt1/setup.json、input_source_manifest.json、各站M00/M10 metrics.json。

## 6. DyDiff真实设置

两区独立训练同一结构：13AGRI、3几何，256×256，8历史→16未来；patch8、width384、12层、6heads、MLP ratio4，65,794,266参数；1,000扩散步，mixed-velocity MSE；150epoch；AdamW lr1e-4，最低1e-6，warmup2，wd1e-4，EMA0.999；25步采样，使用单个预测成员。选模记录为EMA固定种子单成员完整validation归一化RMSE。

| 项目 | 湖北 | 湖南 |
|---|---:|---:|
| 训练影像序列 | 34,854 | 35,320 |
| 验证影像序列 | 3,171 | 3,140 |
| 全局batch | 150 | 80 |
| 每轮更新 | 233 | 442 |
| 计划总更新 | 34,950 | 66,300 |

这些影像序列覆盖比实际GHI配对行更广，不可混为样本量。归一化仅用各区train；湖南继承本区冻结train的有效通道统计与QA位，湖北有效范围规则为反射率有限且正、日间≤2，红外50—400 K；不能泛写二者读取相同QA位。

来源：本目录hubei_dydiff_run_metadata.json为refine-logs/hubei_threeframe_dydiff_forecast_20261102/source_assets.tar内dydiff_run_metadata.json的只读复制；hunan_dydiff_run_metadata.json复制自refine-logs/hunan_dydiff_20261028/FORMAL_RUN_METADATA_216594.json。优化器源码为code_hunan_v1/train.py及对应湖北源代码。

## 7. 交付片段

- tables_data_training_en.tex：可替换主稿表2/表3并加入紧邻方法段落的英文片段，含当前范围，未混用test。
- tables_data_training_zh.md：同结构、数值和公式的中文对照。
- 本记录含工程路径及来源边界；正文片段不含服务器、哈希、任务号或历史修复流水账。
