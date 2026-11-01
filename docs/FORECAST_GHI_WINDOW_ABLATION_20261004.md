# 预测链路 GHI 空间范围消融

用户于2026-10-04要求重做尺寸消融。与211902真实目标时刻AGRI反演诊断分开保存；本实验使用旧湖南合同的冻结SimVP预测AGRI与区域R生产的历史/预测COT，不读取真实未来AGRI、oracle数组、CPP标签或test载荷。新月度划分R/H未闭环，因此仅作旧合同验证筛选，不进入新独立test或正式论文成绩。

## 固定合同

- 范围：1、3、5、16、32、64；保持原像元分辨率。两站中心为局地64画布[32,32]，对应区域Sili[163,148]和Zhujia[73,150]。奇数对称；偶数窗口偏移为[-size/2,size/2-1]。
- 数据：已有`regional_ghi_pilot_bank_20260927_v1`的512 train /128 val序列，所有原有有效白天GHI行相同，不按COT或CPP覆盖筛选。标签沿用已验收的15分钟区间GHI/晴空均值；lead为+15至+240分钟。
- 输入：目标时效的13个预测AGRI通道与cosSOZ/cosRAA/day_mask；8历史COT与16预测COT的截至当前lead前缀。历史COT来自历史真实AGRI，未来COT来自冻结预测AGRI。R始终在256区域生产后统一裁剪；不随GHI尺寸重新反演。1像元GHI头仍可能间接携带R的邻域信息。
- A/AC：A将全部COT值置零；AC保留时序COT。两个站点共用头并有站点embedding，所有组结构、参数量、初始化及样本顺序一致；不与211902独立分站模型作跨实验因果比较。
- 表示：24个按时间排列的COT输入通道，目标后的通道在卷积前清零。该实验是相同轨迹表示下的范围筛选，不声称检验运动输运或因果3D卷积优越性。
- 范围控制：先裁剪再零填充到64画布，在任何卷积前排除范围外输入。显式支持通道区别padding。固定两层空间卷积和展平二维网格读出；不使用全图均值、p90、额外区域分支。参数量与计算预算相同；不能把padding当真实晴空。
- 归一化：AGRI沿用冻结S训练统计；COT只由本次train完整64共同支持计算一个mean/std，所有尺寸同一归一化，禁止逐图标准化。
- 预算：seed42/43/44 ×6范围×A/AC，共36次训练。共同batch由最重64×64单卡profile确定，所有组固定1200更新，AdamW、LR余弦1e-3至1e-5、wd1e-4、clip5。均衡站点采样，损失为两站等权MSE((pred_kt×区间晴空GHI−观测GHI)/1000)。不早停；每100步验证保存最佳，同时保留最终分数及每100步的验证曲线和当步训练损失。

## 验收与解释

先验证范围外输入梯度与扰动影响严格为0、目标后COT梯度为0，再profile和训练。读取源数组一次进GPU，避免反复共享盘I/O。源数组/合同哈希、初值、checkpoint及预测哈希均保存；最佳权重严格重载，FP64从逐行GHI预测重算指标。报告两站与逐时效RMSE/MAE/bias及三seed均值/标准差。按初始化BJT日期2000次配对bootstrap，使用逐行跨seed平均平方误差，不等同三seed RMSE算术均值的区间；所有区间条件于验证选模。

可回答：固定这套预测输入和头部下，更大范围是否有用、COT增量是否依赖范围。不能回答独立全年泛化、最佳外推骨干或R是否绝对准确。源bank的CPP物理QA限制继承；结果不自动替换现有模型、不引入IFS、不打开test。

服务器独立目录：`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/forecast_ghi_window_20261004_v1`。

CPU见证命令：`/public/home/slfu/miniconda3/envs/swc/bin/python train_forecast_ghi_window_20261004.py --witness-only`。
训练通过PBS运行`run.pbs`，状态以`results/status.json`、`results/summary.json`与执行日志为准。
