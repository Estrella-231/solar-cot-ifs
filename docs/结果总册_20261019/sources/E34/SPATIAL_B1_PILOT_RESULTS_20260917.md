# 空间 COT B1 探索性 pilot 结果

状态：完成；这是用户授权的现有冻结 COT 探索，不是最终论文证据。CPP 历史生产链仍未闭环，test_used=false。

## 协议

PBS 作业 209422.tc6000 在 node21 的一张 NVIDIA A100-SXM4-40GB 上运行。三组使用同一原始 train/validation cohort、同一14通道头、同一 GHI 损失、seed42、batch2048、学习率0.0003、AdamW weight decay 1e-4，各训练10个epoch：

- A_sp：13 AGRI；COT空间通道和四统计量槽均置零。
- AC_stat_sp：四个既有 COT 统计量进入读出槽；空间通道置零。
- AC_map：完整标准化 log1p(COT) 16×16图进入第一层卷积；四统计量槽置零。

每组验证样本数为Sili 49,806、Zhujia 50,043，共99,849。主指标为两站全时效 GHI RMSE 的等权平均；checkpoint按该指标取最佳epoch。

## 原始结果

| 组 | 最佳epoch | Sili RMSE | Zhujia RMSE | 等权RMSE | 相对 A_sp |
|---|---:|---:|---:|---:|---:|
| A_sp | 7 | 164.0630 | 133.7177 | 148.8904 | 0 |
| AC_stat_sp | 7 | 164.4489 | 135.2100 | 149.8294 | +0.9391 W/m²（+0.631%） |
| AC_map | 7 | 165.0922 | 132.2524 | 148.6723 | delta=-0.2181 W/m²（-0.146%） |

相对四统计量，AC_map 低0.77%，但这是单seed、10epoch pilot 的验证差异，不能计算成稳定提升。空间图在Sili上高于A_sp 1.0292 W/m²，在Zhujia上低于A_sp 1.4656 W/m²，站点方向不完全一致。

三组均完成有限值检查、保存验证预测和checkpoint。GPU峰值显存约为A_sp 9.05 GiB、AC_stat_sp 8.77 GiB、AC_map 8.77 GiB；没有test输入。

## 决策

按预先设定的B1 pilot门槛，AC_map相对A_sp和AC_stat_sp都没有达到1%的实际意义阈值，因此不从该pilot直接扩展到三seed完整训练。该结果支持继续做输入/空间排列归因的研究价值，但没有证明空间COT优越性；也不能据此判断COT物理质量或CPP生产链正确性。

下一步若继续，应先保留本次结果，补充同一seed的参数/输入激活审计和空间置换诊断，再由用户决定是否为探索目的扩展seed43/44。IFS和test仍保持关闭。

## 服务器证据

结果目录：/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/runs/spatial_b1_exploratory_seed42_20260917_v1

完成状态：COMPLETE_EXPLORATORY_B1_VALIDATION_PILOT；脚本和输入 sidecar 哈希见该目录的 complete.json 与 execution_contract.json。
