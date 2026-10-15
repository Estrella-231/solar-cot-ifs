# 已有历史COT外推的GHI配对试验

用户2026-10-15授权：直接检验已有历史COT外推能否提高GHI预报，覆盖旧COT候选停止进入GHI的条件。本轮是旧湖南合同四里站探索性验证，不作为新月度独立test。

1. 冻结既有SimVP、区域R、C13历史运动和full_history v5 update96，不重训外推器。只读取既有train/val缓存；按新月度day_assignment元数据排除任何24帧窗口触及新test或buffer的样本，保留旧train/val角色；实际数量由审计记录。
2. 复用train512/val512候选COT bank，train图像来自regional_ghi_pilot_bank_20260927_v1，val图像来自regional_ghi_pilot_bank_val512_20260927_v1，按sequence索引逐项连接，不能按行位置拼接。
3. 固定3×3站点邻域。A=预测AGRI13通道+可提前计算目标太阳几何3通道；AC-T加8帧历史R-COT和16帧历史末帧输运COT；AC-H加相同历史及固定混合修正COT。后者还利用冻结SimVP→R预测，明确不是纯COT网络。原厚云准则选中输运，混合修正作为已冻结未选候选单列，不能事后择优当主模型。
4. 三组相同头结构/参数量/seed初始化/训练行顺序/优化器/1200次更新/验证选模频率100步。种子42/43/44共9次训练。未来时效之后的COT通道置零；位置保留，卷积后flatten，无全图均值/p90。输运不可达值通过显式mask置零，所有组共享相同mask支持；不填成晴空、不按CPP支持筛GHI。
5. 输入不含实测GHI；标签与晴空值来自已审计旧pack，以target为结束的三个5min记录均值，核对init/lead/target。输出非负kt乘区间clear；优化物理GHI MSE。COT标准化每臂仅train拟合。小头无Dropout。单卡先实测batch256至8192吞吐，选峰值95%内最小batch，留15%显存余量；资源足够时不盲目开多卡。
6. 保存逐行预测、标签、init/target、lead和键，checkpoint重载一致检查；FP64报告RMSE/MAE/bias及16lead，突出1/2/3/4h。2000次目标BJT日期块配对bootstrap，以三seed RMSE差均值为统计量；说明窗口重叠及验证选模条件。
7. 不保证正收益；分别报告T/H与A，不能因某臂退化删除。不得把旧合同、单站/小样本当全年两站新月度结果。若失败保留目录并新attempt修复；PBS账户最多3个未结束任务，提交前后实时查全队列。论文不改。
