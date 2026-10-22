# 固定GHI基底，单独增加修正训练的日期覆盖

## 依据和新增因素

旧神经头只在12个C_fit日期拟合；其基底G1于这些日期低估，C_select却高估，修正跨日期失败。去除日期整体残差后102.889→102.505点差区间跨0，且不输入未来COT的同容量对照102.477更好，候选已封存。本轮不重复改变编码或残差目标，检验修正训练日期覆盖不足这一新因素。

元数据核查发现原prepared_attempt1内部角色purged中仍有训练日期，未用于G1的F_fit/F_select。按日期元数据全部纳入而非按误差挑选：牛首/王寨/米庄C_fit由12日2691/2590/2641行扩大到50/49/50日11113/10760/10965行。全部额外行原split=train，初始化及目标同一天；新增拟合日期与F_fit、F_select和C_select目标日期严格互斥。这里是旧GHI两阶段划分中未使用的训练日，不是月度外层buffer或test；不得读取后两者。完整日期和键元数据哈希先冻结于METADATA_AUDIT.json。

## 四组公平对照

同一个已冻结G1，small/large拟合日期×无/有未来预测COT四组。small拟合原12日，large拟合全部上述可用日期；选择日期仍为原12个C_select，已使用的开发数据，不叫独立验证。未来COT仍冻结SimVP→旧R，无真实未来观测输入。所有特征、原仅train拟合的归一化固定相同，包括原C_fit拟合的未来COT尺度，不重新在large适配。3×3 ordered field；不加均值/p90。

307特征加同一G1标量，9922参数ConditionalHead，无未来COT组将未来值/diff18槽置0；两组共用太阳有效mask槽和available，缺效严格回G1。恢复cosSOZ>cos80仅作为本四组共同条件，非本轮被检验因素。

全局batch1536，固定60个optimizer更新，总92160个采样槽；以固定seed的完整随机排列循环取样，small/large均不丢尾且不按天气重采样。同范围有/无COT行序严格相同，四组共享初始化；AdamW lr1e−3/wd1e−4，cosine按60updates至1e−5，clip5，原GHI MSE/1e6。此更新式调度和固定完整batch对四组共同应用，不把旧epoch式结果直接拼成数据量单因素效果。

三站seed42/43/44，全部训练完60updates。C_select只对原观测选best，含epoch0精确零修正、平分保留较早。模型best/last、optimizer/scheduler/updates和全部选择日志保存。固定G1重新前向扩展日期；原C日期对旧保存预测预先规定物理最大差≤0.001W/m²，若超出先查数值/来源，不继续训练。

## 停止条件

large+futureCOT须同时优于G1、large-no-futureCOT、small+futureCOT：每项等站/seed RMSE负点差且5000次目标日期簇95%区间上界<0；至少2/3seed改善，每站相对G1退化≤1%，高GHI≥800平均绝对bias增加≤5W/m²。未过则封存覆盖候选，不继续改日期数量/权重/预算/阈值，不打开official val、新时段或test。若通过，先冻结并核对一次外层开发验证的完整来源，仍不叫目标完成。

同一C_select多次使用须明示，不据其结果宣称稳定独立收益。例行监控Luna；预计分钟级三个空闲A100并行站点，最多3个未结束PBS/8GPU，提交前后全队列核验。无上游重训或论文修改。
