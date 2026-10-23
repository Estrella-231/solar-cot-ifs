# 旧15分钟COT窗口平均报告的差值列标签更正

只更正解释，不改变旧模型、预测或审计JSON。原FUTURE_COT_WINDOW_AVERAGE_RESULT_20261020.md表头写Difference from ACREL，但A/AH/AC三行差值实际是ACRELWIN减该行方法；ACRELWIN一行则是ACRELWIN减ACREL。混用参照容易误读。以原保存审计JSON的明确字段为准：

|配对差（前者−后者）|RMSE差W/m²|目标日期块95%CI|
|---|---:|---|
|ACRELWIN_minus_AH|-1.228655|[-1.329123, 0.036284]|
|ACRELWIN_minus_A|-1.095680|[-1.344082, 0.164695]|
|ACRELWIN_minus_ACREL|0.026988|[-0.027775, 0.051510]|
|ACRELWIN_minus_AC|-0.234457|[-0.407748, 0.159495]|

所有区间跨0；原窗口平均候选未证实收益的判断不变。此处仅按已保存聚合JSON复核算术与标签，没有新做独立逐样本bootstrap。

源：refine-logs/cot_aware_simvp_20261017/results/future_ghi_oof_cot_window_average_validation_audit_v1/future_ghi_oof_cot_window_average_validation_audit_v1.json

SHA256：e65b2c3d853d38de165c2b376290e2a07ddf6642fc42e58ea007dd5e647f84c5
