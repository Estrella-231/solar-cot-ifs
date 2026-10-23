# 冻结GHI响应引导的局地COT校正

2026-10-23，承接条件信息诊断；不是重新调已封存的F_select校正器。新假设：在相同物理COT损失之外，约束COT误差在冻结GHI映射中的响应，可改善因果预报。模型不改S/R/G1；仅large C_fit日期三折OOF、排除验证日期±1日，禁止据C_select、新时段或test选择超参数。

## 预先固定比较

1. physical：308→16→16→9 SiLU，5369参数，末层零初始化；输出relu(log1p原预测COT+delta)，九像元物理COT MSE除100²。
2. task：同一初始化、输入、采样、预算，在上述损失外增加冻结教师GHI差的MSE除1000²，系数1。不扫描系数。
3. direct_AGRI：仅135个预测AGRI/太阳几何特征、lead、G1的kt共137维，137→36→11→1 SiLU，5387参数（多0.34%），末层零初始化，直接回归(GHI−G1)/1000。全部层参与前向、训练，无闲置参数。它是近似容量匹配的直接辐照度修正对照，架构差异如实披露，不冒充严格同构单因素。不能因为仅比弱physical好就宣称COT有增益。

三组global batch128、600更新、AdamW lr1e−3/wd1e−4、cosine末1e−5、clip5、固定last；三站×三G1来源×三外折×三臂=81拟合。physical/task成对样本次序和参数完全相同；direct使用相同记录顺序。每站每外折的标准化只拟合其训练日期。seed42/43/44对应G1来源和初始化，不是三个独立数据集。

## 教师与下游输出

使用同外折observed_R_OFFLINE ridge教师，它未见该折评分日期；拟合域教师为训练内拟合，不冒称训练内OOF。参考COT来自相同source_role的R(actual_AGRI)，是反演代理，不是独立真值；实际未来仅作训练监督。

未来logCOT九维响应s=1000*(beta_target/std_target + history_valid*beta_diff/std_diff)。当前feature的279:288为原始logCOT，288:297为相对最后历史的log差。教师完整线性输出在两种COT下之差应等于sum(s*delta_log)，须独立有限差分检查；不经过教师输出非负截断。

physical/task预报GHI=max(0,G1+sum(s*(logCOT_corrected−logCOT_original)))；direct为max(0,G1+1000*direct_output)。零初始化三组都还原G1。原预测COT不可用行三组均回原G1，不用实际COT质量挑评分行，不以输出>100清零。共同availability与原实验相同，direct不读取COT数值，但共享该评价/回退条件，需明示。

## 成功与停止

task对physical和G1的等站点RMSE差及5000次日期块95%CI上界均<0，三站无>1%退化，至少2/3 G1来源方向一致；报告direct对照，若task未优于direct则不唯一归因COT表示。COT代理RMSE相对physical恶化≤1%，厚云FAR较原预测增加≤2pp，高GHI≥800绝对bias较G1增加≤5W/m²。未满足则保留负结果，不扫系数、幅度、更新数；通过也须冻结新日期确认才可能认定目标达成。

## 执行顺序

先CPU double检查日期/梯度公式/零初始化/缺效回退/非零梯度/断点恢复；再PBS三卡按站点并行，记录实测耗时。只有短任务用此方式；若预计数小时，按根规则实测2/4卡吞吐。每次提交前后查slfu全队列、最多3作业/8GPU，不终止他人。现有环境复用swc，不安装或改变软件。
