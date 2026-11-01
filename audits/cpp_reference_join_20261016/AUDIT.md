# CPP参考错配核查与纠正

已确认根因是CPP参考绑定错误，而不是通过改色标或删除异常点修饰模型成绩。
训练代码的sample_indices索引全SUPERVISED表；profile构建器先筛val再enumerate，误把验证局部编号当全表编号。
逐项重建错误映射后，2150个旧bank支持记录的参考与掩膜全部和错误映射一致。
例：竺家2024-12-24 06:00 UTC被赋予四里2024-04-18 23:00 UTC的CPP。

## 正确原始R验证
直接从原始validation_predictions.npz及其全表索引复算：356169像元，RMSE=9.074，MAE=4.646，Bias=+0.279，r=0.821。
四里RMSE=9.674、r=0.788；竺家RMSE=8.433、r=0.850。与训练报告一致。

## 已有271个预报窗的局部修复
仅重新按(station,target_time_utc)绑定正确CPP参考和有效掩膜，模型输出、权重、时刻完全保留。
参考有效记录从550400改为668672；它是参考错误纠正，不是模型改善。
|站点|输入模式|修复前错误RMSE|修复后RMSE|修复后r|
|---|---|---:|---:|---:|
|sili|SimVP forecast AGRI -> R|29.132|11.297|0.648|
|sili|observed target AGRI -> R|27.123|12.172|0.656|
|zhujia|SimVP forecast AGRI -> R|23.043|11.192|0.670|
|zhujia|observed target AGRI -> R|24.903|10.428|0.742|

这些窗口当初通过错误参考映射筛选，因此仍需按修复代码重新枚举全val覆盖后生成完整预报评价；没有自动提交GPU任务。
原始R完整验证与这271窗回放的日期、重复权重、输入装载/几何不同，不能混为同一支持。

## 已完成代码修正
profile_forecast_cot_val_100.py、profile_forecast_agri3x3_bank_20261014.py、profile_forecast_agri3x3_allval_20261014.py均改为先保留全SUPERVISED索引，再筛val，并核验保存的全表索引集合。
仅修改本地工作代码；历史服务器代码和历史结果保留，未声称已重新运行全部推理。
旧CPP相关指标和本次错误绘图已添加INVALID_CPP_JOIN.md；两个权威历史报告加撤回标记。
所有绘图不得再把旧bank中27～30的RMSE和负相关解释为模型本身的性能。
