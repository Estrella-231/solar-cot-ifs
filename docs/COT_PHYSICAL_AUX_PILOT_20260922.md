# 湖南 COT 反演精度试验：物理 COT 辅助损失

状态：2026-09-22 预注册；验证集 pilot，不读取 test，不替换当前冻结 R。

## 触发证据

当前 R 的已保存验证预测在 6,536 帧、1,664,013 个有效像元上 COT RMSE 4.2465；参考 COT≥30 像元 RMSE 11.9531、偏差 −4.5513；站点周围 5×5 RMSE 3.9440。误差分解由 `scripts/diagnose_cot_retriever_bins_20260922.py` 计算，服务器原始结果在 `audits/cot_retriever_bins_20260922.json`。这是 CPP 产品参考误差，不是独立物理真值误差。

## 单因素配对

两个分支均从同一个湖南冻结 R checkpoint 开始，使用同一训练/验证样本、归一化、模型结构、seed42、数据顺序、AdamW、学习率 1e-4、batch1024、16 epochs、FP32 和验证 physical COT RMSE 选模；初始 epoch −1 也参与选模。

- `log_control`：原始有效像元 `log1p(COT)` Huber，delta 0.1。
- `physical_aux`：同一个 Huber 加 `0.005 ×` 物理 COT SmoothL1，beta 2.0。

两组的唯一设计差异是辅助损失。候选输出独立写入 `runs/cot_physical_aux_pilot_seed42_20260922`；不覆盖 `runs/cot_repaired_seed42_v1`。训练时不使用验证 CPP 反向传播；验证只做选模。保存逐 epoch 指标、best 权重和逐像元验证预测。

## 判定边界

先看候选相对于**同样继续训练的 control**：整体 COT RMSE、COT≥30 RMSE、站点 5×5 RMSE 和晴空 RMSE 都完整报告。只有整体与厚云 RMSE 均降低、站点附近和晴空没有实质恶化，才考虑作为新的 R 候选。若只是物理 COT RMSE 改善，不直接宣称 GHI 提升；后续需固定 S、同一 GHI 头和样本做配对验证。现有真实 AGRI→CPP 与预测 AGRI→CPP 的差距也需用新 R 分别重算，不能只看真实输入。

PBS 提交前实时检查本账户跨项目未结束作业少于 3；单卡先运行，不影响另一个论文既有作业。`scripts/run_cot_physical_aux_pilot_20260922.pbs` 是唯一提交入口。完整结果出现前不能把提交成功写成精度改善。
