# COT 反演器代码核查（2026-09-22）

范围：湖南第二篇的冻结 COT U-Net `R`、训练打包、验证重载、真实/预测 AGRI 推理及历史轨迹侧车。本页是工程审计，不改变论文结果。

## 已核对

- `scripts/pack_cot_training.py` 输入顺序为 13 个 AGRI 通道、`cosSOZ`、`cosRAA`、`day_mask`。COT 打包为 `target=COT/100`；`scripts/train_cot_repaired.py` 在训练前乘回 100，并对 `log1p(COT)` 做有效像元掩码 Huber。输出用 softplus 保证非负；物理 COT 经 `expm1` 恢复。没有发现这部分的通道、单位或反变换代码错误。
- 冻结 checkpoint、模型源码、训练集归一化文件的 SHA 与 checkpoint metadata 相符。只读脚本 `scripts/audit_cot_retriever_code_20260922.py` 在服务器对固定的 3 个验证行（6011、15246、18513）重载前向：`model.forward` 与展开的 `forward_features → head → softplus` 逐值相等；CPU FP32 与原 GPU FP16 验证缓存的最大 log1p 差为 0.002063，符合既有小于 0.01 的复核阈值。未读取 test。
- `scripts/cache_frozen_forecast.py` 对 SimVP 外推 AGRI 先逆归一化为物理 AGRI，再拼目标时刻的逐像元太阳几何，以 R 的训练集统计重新归一化；其 R 前向运行在 FP32。既有 872 对真实/外推输入诊断把两支都置于 CPU FP32，几何逐值相同，因此该诊断中由输入替换产生的误差差异不受下述 BF16 历史侧车问题影响。

## 确认的不一致与修正

历史轨迹侧车 `scripts/build_history_cot_trajectory_sidecar_v1.py` 把 R 的归一化和前向包在 CUDA BF16 autocast 内；未来轨迹缓存的 R 则使用 FP32。这使 `AC_H` 与 `AC_F` 所用的冻结 R 虽然权重相同，数值精度合同却不同。**这是推理实现不一致，尚未证明它是 GHI 结果差异的主要原因。**

新增 `scripts/build_history_cot_trajectory_sidecar_v2.py`：在 FP32 中完成 R 归一化与前向，显式禁用 autocast，检查归一化统计、输入和输出，并在新侧车报告中写入 builder 版本、脚本 SHA 和 R 精度。原 v1 脚本、既有侧车及已有 B1-T 指标保持原样，不能把它们标成 v2 的结果。使用 v2 时必须写**新输出目录**，重新构建侧车；若要检验对 `AC_H/AC_ST` 的影响，应以相同模型头、相同数据与 seed 重跑成对对照。尚未执行这一步，因此不能更新既有 GHI 结论。

## 尚未由代码核查解决

CPP 上游 NC 生产批次和生成 checkpoint 的关联仍未闭合；CPP 在本工程只能称为反演参考产品。模型在验证集选模，3 行重载与现有 validation 指标不能替代独立时间测试，也不能证明未来 COT 预测对 GHI 有提升。
