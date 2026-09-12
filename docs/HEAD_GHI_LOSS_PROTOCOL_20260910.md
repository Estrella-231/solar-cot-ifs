# GHI一致损失的A/AC成对诊断协议

2026-09-10，用户“你推进吧”授权执行本轮。基于208011实际残差证据开展单因素方法诊断；不将现有kt损失称为实现bug，不预设COT有效。

## 固定假设与唯一改变

保持区间kt输出、15min结束时刻标签以及pred_GHI=pred_kt×区间晴空GHI。唯一训练目标改变为mean[w×((pred_GHI−observed_GHI)/1000 W/m²)²]，w沿用各cohort训练站点×lead权重。尺度1000事前固定，不调参，不宣称与旧目标的数值/梯度量级完全相同。新损失等价于旧kt残差乘(Gcs/1000)²；保持FP32损失、BF16网络与FP64评价。仍以两站等权GHI RMSE选模；损失是加权平方误差代理，并非该RMSE终点的完全等价优化。

## 冻结合同与预算

- original为主对照538053训练行，QC1为敏感性537605行；各自沿用旧归一化和训练权重；同99849验证行。无test，无新增删样本或kt截断。
- 冻结湖南S/R与cache。网络仍是原CNN/MLP，COT仍为原4统计量，A将相同COT槽置零。
- 每cohort A/AC×LR{0.0003,0.001}×seed42，共8个新训练；旧8个kt目标候选保留为基准。每候选随机初始化、同epoch行排列、AdamW/weight_decay=1e-4、batch2048、最多50epoch、patience8。各组同两档LR预算；不因为结果不好追加搜索。
- 新输出runs/head_ghi_{original,qc1}_seed42_20260910_v1；不覆盖旧模型。train_head_ghi_core.py是原train_head_pilot.py的独立目标函数分支，必须验证Head/fit/overfit的非目标行为一致。
- 固定CPU数学见证验证损失值、解析梯度、kt等价式、原Head同seed参数。PBS内单卡batch2048有限梯度/显存检查，峰值<总显存80%；8固定随机训练序列A/AC各800step且目标下降至少50%，通过后随机重置正式pilot。不重新搜索batch，避免改变优化合同。
- 服务器现有swc环境复用，无依赖安装/升级。1张A100，8CPU，单PBS串行8个模型，walltime30min。每次提交前实时查询账户全队列；最多3个未结束PBS与8张GPU，不终止其他任务。

## 验收与判定

保存全部候选checkpoint、epoch、validation逐行预测与源/脚本/协议SHA。独立NumPy核验预测键、真值、mask、站点与lead，并复算两站等权RMSE/MAE/偏差/分时效误差；完整保留original和QC1。

四个选择模型CPU FP32重载：每站lead index0/5/10/15第一验证行，固定abs误差<=0.02×max(1,abs(saved BF16 pred_kt))，失败不改阈值。

主报告同时列旧/新损失下A、AC、AC−A及差中差；同LR差值也全部报告。A/AC都改善但AC仍更差，只支持目标改善，不支持COT增益。仅QC1改善属于敏感性结果，不升格主结论。单seed验证选模不作正式显著性或test结论。任何正向pilot只触发后续独立评估计划，不自动扩种子、改表示或打开test。

## 运行命令与停止规则

服务器根/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907，Python/public/home/slfu/miniconda3/envs/swc/bin/python。

CPU预检（根目录执行）：
```bash
/public/home/slfu/miniconda3/envs/swc/bin/python scripts/train_head_ghi_paired.py --cpu-check
```
GPU训练仅经scripts/run_head_ghi_paired_20260910.pbs调用，不在登录节点训练。数学/源哈希/显存/过拟合/数值任一失败立即停止后续阶段，保留失败产物，不盲目重投。

本次为一次性训练及验收，不创建持续监控、不自动追加实验。独立来源缺口CPP生成者绑定与IFS发布时间/steps仍保留，不影响本次相同缓存的优化诊断。
