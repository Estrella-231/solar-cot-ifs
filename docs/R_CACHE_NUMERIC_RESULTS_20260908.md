# R cache数值后端诊断结果

PBS207626.tc6000，单张A100，walltime00:00:50，正常C、exit_status=0。未训练、未读取GHI/kt/test、未改权重或原cache。总体状态COMPLETE_WITH_NUMERICAL_DIFFERENCES，不是全部数值比较通过。

GPU复算六个原始完整分片共130序列、4160个patch、1064960个像元；CPU只复算先前固定的六序列，共192个patch、49152个像元。所有比较在GPU执行前固定max absolute error≤0.001，不使用旧GPU FP16的0.01容差。下表为log1p(COT)或相应四维统计坐标的绝对差，不是GHI误差。

|比较|最大绝对差|汇总RMSE|0.001门|
|---|---|---|---|
|归一化GPU/CPU/NumPy（相同输入）|0|0|PASS|
|GPU benchmark/TF32均开 vs原缓存（全像元）|0.001033306122|9.19607754e-05|FAIL|
|GPU均开 vs原缓存（四维统计）|0.0006914138794|6.355342137e-05|PASS|
|GPU benchmark/TF32均关 vs原缓存|0.008136749268|0.00052457566|FAIL|
|GPU均关 vsCPU（相同归一化输入）|4.053115845e-06|3.433813814e-07|PASS|
|CPU vs原缓存|0.008136749268|0.0005136322841|FAIL|
|原四维统计→head标准化pack映射|0|0|PASS|

归一化在三条路径上逐值完全相同。切换计算设置后，GPU与CPU差降到4.053e-6，表明这些固定样本的差异明显依赖计算后端。因为benchmark和cuDNN TF32同时变化，不能单独归因为TF32。原缓存重放仍有最大0.001033306的超限，不能说旧缓存逐像元完全重现，也不能以接近阈值为由放宽。

目前没有证据表明通道、归一化或pack映射发生错误；也没有证明该后端差异导致A/AC无收益。八组原始负结果及其逐预测重算保持有效保存，不因这份诊断覆盖原缓存或自动重训。是否需要额外精度影响实验由具体差异和独立审查决定，不进行后端设置搜索以获取更有利GHI指标。

独立审查见R_CACHE_NUMERIC_REVIEW_20260908.md。下一项候选为既有R pack与forecast cache的同target/同CPP掩膜有限配对诊断，先封存按时间排序的抽样规则和来源映射。R(real AGRI)只作为oracle诊断，不能作为可部署主模型或参与GHI选模。

证据文件：audits/r_cache_numeric_witness_20260908.json；脚本：scripts/verify_r_cache_numeric_backends.py；日志：服务器r_cache_numeric_witness_pbs.log。服务器根：`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907`。
