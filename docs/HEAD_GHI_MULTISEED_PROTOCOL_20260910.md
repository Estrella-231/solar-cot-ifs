# GHI损失下COT收益的配对种子稳定性验证

用户于2026-09-10明确要求多种子验证。本协议在查看seed43/44结果前固定，不改变seed42结果或原科学协议。

## 设计

种子固定42、43、44。42复用PBS208020的已验收候选；原始组固定LR0.0003，QC1固定LR0.001，同cohort的A/AC使用相同LR。选择依据是seed42两组各自选优恰好使用上述相同LR。seed43/44不再搜索LR，每种子两cohort×A/AC，共新增8个候选。

其余全部沿用HEAD_GHI_LOSS_PROTOCOL_20260910.md：冻结S/R、原始和QC1各自样本/归一化/站点×lead权重、4维COT、kt输出、固定1000 W/m²的GHI损失、BF16/FP32/FP64、AdamW weight_decay1e-4、batch2048、最多50epoch、patience8，按同99849验证行两站等权GHI RMSE选checkpoint，无test。original538053train，QC1 537605train，站点0四里、1竺家。不删除低Gcs或大kt样本。

同seed的A/AC初始参数张量逐值相同，唯一区别为A的COT槽置零；每epoch排列使用seed+epoch，A/AC严格配对。不同seed的初始化与排列改变。固定seed42的小样本门禁只用于数值健全性，不作为研究结果；正式fit重新按43/44随机初始化。训练按seed、cohort串行，新输出runs/head_ghi_multiseed_{original,qc1}_seed{43,44}_20260910_v1，原结果不覆盖。

## 验收与判定

CPU预检验证损失值/解析梯度/原Head结构，独立检查新fit中所有随机路径及seed元数据。PBS内显存/有限梯度与8序列过拟合门禁通过再fit。用独立NumPy从12个固定LR候选逐行输出核验共同键/真值与主RMSE，全部checkpoint SHA绑定、seed/配置绑定；原Head CPU FP32重载固定每站lead index0/5/10/15第一行，容差继续abs误差<=0.02×max(1,abs(saved pred_kt))，失败不改阈值。

每cohort报告每seed的A、AC、AC−A（负值有利COT），以及3seed的均值、样本标准差(ddof=1)、改善种子数；单独报告新增43/44的配对差值，区别于参与选配置的42。按站点和各lead同样完整保留差值。12候选均报告，不能挑最好seed或cohort。

3seed均值改善且新增43/44均改善，才称本配置出现初步种子一致性；若方向混合就报告不稳定。站点若一好一坏必须明确，不能称跨站一致收益。3seed不能支撑强显著性推断，验证checkpoint选择仍有乐观偏差；不把99849行当独立重复、不据此打开test或预写论文提升。不通过则不自动加seed或调参。

## 部署

复用zjnu-hpc现有swc环境，不安装或更改依赖。服务器根/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907。

CPU预检（在根目录）：
```bash
/public/home/slfu/miniconda3/envs/swc/bin/python scripts/train_head_ghi_multiseed.py --cpu-check
```

GPU经scripts/run_head_ghi_multiseed_20260910.pbs执行，1 A100/8CPU/30min，最多3个未结束PBS及8GPU账户合计限制。提交前实时核队列，已有任务资源无法确定则停止；不取消他人任务。作业串行执行训练与独立验收；失败停止后续阶段，不盲目重投。未恢复小时监控，无自动追加实验。
