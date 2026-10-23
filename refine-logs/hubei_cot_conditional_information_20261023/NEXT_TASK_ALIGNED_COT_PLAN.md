# 下一候选：用冻结GHI响应监督未来COT校正

## 新单因素

本轮日期OOF证明：实际R-COT可被学到的映射利用，但预测COT无法保留该收益。下一项不改S/R或扩GHI小头，而比较同一局地COT校正器的物理COT损失与物理COT＋冻结GHI响应损失。区别于此前F域逐像元MSE校正：新增监督明确针对COT误差在GHI中的作用，且只在large C_fit内部OOF，不到C_select调参。

## 训练/验证隔离

沿用本目录三折DATE_FOLDS及±1日排除。每个外折使用已保存observed_R_OFFLINE_fold模型作为冻结GHI响应教师：该教师只在该外折拟合日期训练，未见外折评分日期。校正器也只用同一拟合日期；不得为其训练行采用其它可能见过外折评分标签的教师。教师在校正器拟合域是训练内拟合，必须如实披露，不冒称训练内教师OOF，但这不构成外折标签泄漏。

输入仅当前预测AGRI/历史COT/mask/几何/lead及冻结G1标量，仍308列、3×3，无实际未来图推理输入。输入统计只来自该外折拟合日期，可复用同折forecast_COT探针的均值/std；不能混用实际COT教师的输入统计。预测侧availability固定，不用参考质量或校正输出大小选择评分行。

## 模型和损失

MLP308→16→16→9、SiLU，末层零初始化，5369参数；输出9个非负log1p(COT)：relu(原预测logCOT＋delta)。物理值expm1恢复；不全图平均/p90，不以>100自动清零。非有限数值应停止并记录，不能用裁剪隐藏。初始COT及最终初始GHI须逐行重放原值。

两组从相同初始化、相同样本顺序；本候选明确采用global batch128、600更新；每站每原G1来源每外折成对训练，seed42/43/44只对应其G1与初始化。AdamW lr1e−3/wd1e−4，cosine末值1e−5，clip5，末步固定，不在外折挑step或早停。27对/54个拟合，GPU按站点并行短实测，若总预计数小时按根规则2/4卡吞吐选择，提交前后PBS≤3/GPU≤8。

基础损失L_C=mean[(COT_out−R(actual_AGRI))²/100²]，参考为同条source_role的同R代理，不当CPP独立真值。新增L_H=mean[(sum_i s_i*(logCot_out_i−logCot_reference_i))/1000]²，系数固定1.0。

s_i是该外折冻结实际COT教师对第i个logCot的线性GHI导数：1000×(beta_target_i/std_target_i＋history_valid_i×beta_difference_i/std_difference_i)。只对未来Cot和相对原最后历史Cot的差值同步变化，AGRI/context/base不变。它等价于未截断线性教师在两个COT输入间的GHI差；须用独立有限差分及完整教师前向验证该等价。L_H只用训练参考，不在推理读取它；教师参数不更新。

原pixel损失组不加L_H；task组L=L_C+L_H。除此不改结构、样本、预算或归一化，不扫描系数。

## 下游预先固定的主评价

GHI主输出为G1＋sum_i s_i*(logCot_out_i−logCot_forecast_i)，再非负截断。它利用冻结教师的COT创新响应，而非换成教师绝对GHI基线（后者当前比G1差）。初始delta=0时必须严格等于G1。原G1、物理COT损失、加任务损失三组在外折全部行配对；缺效均返回G1。

主要task−physical及task−G1的RMSE点差与5000次日期CI上界均<0、三站无>1%退化、至少2/3基底方向一致，才继续。同期COT代理RMSE相对physical不得恶化>1%，厚云FAR相对原预测不得增加>2pp，高GHI≥800绝对bias相对G1不得增加>5W/m²。保护未过即封存，不扫loss系数/更新数/幅度。始终保留全部原行，报告MAE/bias/四小时/薄厚云，不能只按GHI倒推任意COT数值。

通过也只是候选：校正网络增加了容量，尚需直接AGRI同容量GHI对照及冻结后确认，不能据此唯一归因显式COT或宣布用户目标完成。该后续对照方案需在看到候选结果前明确，不用参数数相同但大部分闲置的伪容量控制。

## 启动前必需检查

同折教师未见外折标签的日期列表逐一审查；目标时间和source_role连接；logCot/差值/有效mask梯度等价；零初始化GHI回放；无效行回退；原始COT输入不可用不填晴空；损失有限、梯度非零、两步恢复optimizer/scheduler/sampler一致。只准同折训练统计。未完成这些CPU/小批工程检查前不启动54个正式拟合。

本计划尚未实现/训练。禁止触碰C_select、official val、新时段、outerbuffer/test以选择这个候选；不改论文，不改变已封存结果。

启动状态：暂不启动。直接AGRI同容量GHI对照的输入、输出映射与同预算细节尚未冻结，当前仅为待完善候选；本轮交付仍是新时段退化归因及条件信息诊断。
