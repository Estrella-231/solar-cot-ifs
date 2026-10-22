# G3区域R训练协议（首次拟合前固定）

目标：得到在湖北完整256×256范围监督训练的反演器，供上游历史云场输运和固定SimVP→R路线共用。此训练本身不证明预报或GHI增益，亦不把旧16×16权重扩大视场。

- 新随机初始化seed42；16输入，UNet base16，两次下采样，原有卷积/转置卷积及softplus非负log1p(COT)输出。多卡仅把BatchNorm转换为SyncBatchNorm；不使用任何湖南或旧湖北参数初始化，不增加分类分支或损失项。
- 数据及归一化沿G2固定协议。所有原外层train/val保留；训练用R_fit，选模只用R_select。按外层train逐月日期SHA256顺序选floor(n/7)天（n≥5时至少1天），选中日期相邻±1天从拟合中移除；该规则不读取云量/GHI或模型误差。R_INTERNAL_ROLES.json已冻结1999拟合帧、314选模帧、515缓冲帧，外层val296帧；实际损失仅用其中有有效像元的帧。
- G2输入归一化包含外层train所有有监督像元的输入统计，包括内部选模日期的输入，不含外层val。内部选模不是完全隔离的独立测试，不声称其为OOF；它的CPP标签不参与参数梯度。当前源只覆盖2025年8月至2026年2月的既定可配对帧，不宣称全年四季完整监督。
- AdamW，初始lr1e-3，weight_decay1e-4，全局batch128，最多50轮，每轮drop最后不足128的随机行，每轮seed=42+epoch洗牌；跨卡保持同一全局行顺序、batch、更新数及样本集合。每轮更新数=floor(有效R_fit帧/128)，训练完成记录实际global_step与每轮尾批数。
- 损失为有效像元等权Huber(log1p COT)，delta0.1。必须先where屏蔽无效位置再作损失；每个更新all-reduce有效像元计数，局部损失乘world/global_count以抵消DDP梯度平均。不得简单平均各卡均值。裁剪全局梯度范数5；fp16 AMP，但非有限未缩放梯度直接失败，不能静默跳过更新后声称更新数相同。
- 学习率按更新数cosine至50轮预算；内部R_select像元加权Huber最低保存best，连续8轮不改善停止。月度val不用于选epoch、结构或阈值，只在选模结束后保存一次逐帧预测；后续独立复算物理COT RMSE/MAE/bias、各站点1×1/3×3/16×16、全区域及厚度/日期分层。
- 新启动前实时检查slfu全队列与实际物理卡，最多3个未结束作业及8卡；同一PBS申请4卡，依次实测2/4卡同全局batch吞吐，按有效加速选择，profile后正式模型必须重新随机初始化。profile不是训练，不读取月度val标签来比较配置。
- last保存模型、AdamW、scheduler、AMP scaler、epoch、global_step、best状态、数据/模型/roles哈希。断点续训只接受完全相同绑定与global batch；多卡切换不从头重训。训练无dropout，逐轮全局行序由epoch确定，不依赖rank随机采样。
- 正式GPU运行还须通过实际forward/backward、SyncBN和2/4卡损失核验；CPU/Gloo检查只证明不等mask计数下的DDP损失缩放，并不证明CUDA训练已可用。
- 数据源/独立验算未PASS前不提交GPU训练。本协议及训练程序已准备，不等于已经启动；G4 COT路线通过后才做G5有无COT的GHI配对。
