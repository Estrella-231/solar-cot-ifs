# 湖北三站：正确UTC COT的真正预报配对实验

2026-10-20。用户授权重新进行真正未来GHI有/无COT配对。仅train/validation，test/buffer关闭。来源修订见 `../hubei_cloud_probability_20261020/AUDIT_GEOMETRY_CPP_ALIGNMENT_20261020.md`；旧预报分数不作为修正后成绩。

## 问题、输入及不变项

历史8帧AGRI经既有冻结SimVP外推16帧（+15至+240分钟），每帧预测AGRI与目标太阳几何经正确UTC监督的R得到预测COT，输出该目标结束的15分钟GHI均值。未来真实AGRI/CPP不作为推理输入。

复用原审计通过的五折SimVP折外预测和全train SimVP验证预测，不重训SimVP。严格保留旧forecast pack的70,171行、行序、AGRI、geometry、GHI与clear-sky数组：train 63,448、validation 6,723；验证牛首2316、王寨2112、米庄2295，22个目标日期。若复用预测与旧pack逐像元不一致，停止核查，不静默换样本。

CPP监督使用正确规则 `CPP UTC = AGRI BJT - 8h`。训练日期沿原五折划分，各折R只用其他train日期，从随机初始化训练；复用原折内train拟合的AGRI归一化，覆盖目标/掩膜为正确UTC版本。验证使用已完成的 `hubei_cloud_probability_20261020/R_correctutc_attempt3`，不使用任何旧错时R。

## 配对训练

- FA：预测AGRI+相同目标太阳几何；FAC：加预测COT。
- 两组均采用原FAC的3×3局地头（24,489参数），FA的归一化COT槽固定零；相同种子下初始完整权重哈希必须一致。相较旧FA少参数版本，这是明确的容量匹配修订，因此不把新旧基线分数差归因于R修复。
- GHI seed42/43/44，共3站×2组×3seed=18头；各30epoch，batch1536，AdamW lr1e-3、wd1e-4，余弦至1e-5、梯度裁剪5，末尾不足batch照常更新。
- 使用原训练器的GHI尺度MSE：`mean(((pred_kt*clear_sky_mean - GHI)/1000)^2)`。各站train拟合AGRI及log1p(COT)归一化，geometry保持物理值。
- 每epoch用同一个validation评价并按最低RMSE选best，同时保留last、逐轮曲线和更新数。两组每epoch样本顺序哈希相同；checkpoint重载必须复现保存预测。
- R沿已完成修正训练设置：seed42、最多50epoch、patience8、Huber(log1p COT)、AdamW lr1e-3/wd1e-4；折内batch128保证每轮至少12次更新，五折统一。完整R保持冻结。

## 评价与报告

主指标为三站等权、三seed等权GHI RMSE，差值FAC−FA，负值表示加COT有利。保存逐样本station/init/target/lead、预测与真值，独立FP64复算RMSE/MAE/bias；完整报告逐站、逐seed、16个lead及+1/2/3/4h。

按相同BJT目标日期配对bootstrap 10,000次，主报告单日簇并给连续两日块敏感性，不把重叠预报或seed视为独立天气样本。验收不删无收益站点、不择优时效、不依据效果改样本。只有总体配对区间及逐站方向支持才称当前验证有增益；否则报告未证实。所有结果是反复使用的validation，非独立test。

## 资源与执行

原预测数组和折内AGRI归一化由FD-201一次性导出到学院PBS；不导出未来真实AGRI或旧CPP目标。PBS优先单张node22 A800，归档在作业内解压到本地/tmp再训练/推理，避免反复读取共享盘。提交前后检查slfu完整队列，未结束任务不超过3，不终止其他任务。例行监控交Luna；实际PBS号、来源哈希和状态记录于tracker。
