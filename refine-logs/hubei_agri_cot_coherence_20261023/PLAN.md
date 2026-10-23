# 固定GHI映射下AGRI与COT输入域是否一致

## 问题

已完成的固定头COT替换保持预测AGRI不变，换真实图R-COT后米庄退化。响应正则只带来微小开发改善，候选封存。本轮不训练，也不扫正则系数，而是检查替换后AGRI/COT联合输入是否不一致。

## 完整因子对照

固定原large_no_future_COT与large_future_COT九对权重、历史8帧及train归一化。未来AGRI有预测/实际两种来源；未来COT为无值、预测AGRI经同折R、实际AGRI经同折R三种，形成2×3。每个AGRI来源分别计算同一冻结G1基底，再送对应固定修正头，base标量随该AGRI正确重算。另保存两个G1输出。

全部8266条原C_select记录保留，8,111条共同有效时所有分支使用同一availability，其余回对应AGRI来源G1。真实AGRI和真实图R-COT两种输入都属于不可部署离线诊断，不能写成预报成绩或上限。预测AGRI/预测COT为可因果得到的来源，但本次仅重放，不训练新可部署模型。

实际AGRI复用已完成matched_R_dynamics的2491帧原train raw16缓存，不重新生产或下载。逐target_time连接，按本站meta中心取3×3；实际13通道使用原future_agri_mean/std，不能改用历史归一化或另拟合真实图统计。目标太阳几何必须与原输入一致（max≤1e−6），历史和已知几何保持逐值相同。核对原缓存整体sha与帧sha；source_role及COT引用仍每条同OOF R。先复现原预测三个分支（≤1e−3 W/m²），再解释2×3分数。

## 事先声明的分析

逐站逐seed先算RMSE/MAE/bias，再等权；日期簇5000次配对CI，四完整小时段和共同来源覆盖。主要诊断是：同一实际AGRI下，实际图COT−预测COT的误差变化，是否与同一预测AGRI下的变化不同。并报告两种AGRI各自有COT−无COT，不把AGRI更换的收益归给COT。

计算逐记录的MSE交互项：在实际AGRI下换COT的ΔSE，减去预测AGRI下换COT的ΔSE，保存全部日期/站点/seed。相同R-COT不是CPP独立真值；实际AGRI仍是瞬时快照，GHI为右端15分钟均值。条件反事实离开训练分布，结果只能定位联合映射，不识别唯一物理因果。

若两种AGRI下换参考COT都不利，不能继续把退化仅解释成AGRI/COT源不一致；若只在预测AGRI下不利，则联合输入域是候选原因，仍需另外的训练内配对方法验证。没有自动启动混合域训练的结论。不读取official val、新时段、outerbuffer/test，不改论文。GPU三站并行短前向，队列≤3/GPU≤8，原失败候选不恢复。
