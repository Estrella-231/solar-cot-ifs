# 下一项：当前固定GHI修正头对同折参考COT的响应

尚未执行；只读诊断，无新训练。先核对最新代码/输入/权重，不能从历史记忆直接加载名称相似的旧头。

## 为什么不是重复旧替换

旧fixed_head_cot_substitution_attempt12采用FAC seed42，在R validation bank可连接的约18%官方验证行上比较forecast/init持平/实际未来图R。其离线3.94%点估计不是当前两阶段GHI条件修正头的证据。新matched_R_dynamics已生成原C日期每条source_role对应的同fold真实init/target R参考，覆盖C_select8266候选、8111共同有效中心记录。新的大日期GHI修正头及此同折完整开发范围尚未做替换，本次只弥补此来源/覆盖/模型差异，不重复用旧分数说明当前头有效。

## 固定比较

1. 优先使用hubei_cot_calibration_coverage_20261023的large_future_COT最佳头、相同冻结G1及配对large_no_future_COT，三站×三seed。以实际保存来源和权重命名为准，原小日期G3仅作必要来源回溯，不自动增加模型。
2. 每个有future_COT头固定参数、预测AGRI、太阳几何、8历史输入、基线输出及原GHI归一化；仅替换目标3×3未来COT值及其与最后历史COT差槽。比较原forecast COT、同fold R初始化持平、同fold R实际目标图参考。后两者不混用full-R历史bank，且实际未来参考明确不可部署。
3. 逐行station/init/target/lead连接，检查参考源R及太阳有效性。所有来源用同一joint mask/available；缺效精确回同G1。保存全部原行及覆盖标记，原预测/同mask重放与离线源分列，不能因为替换缺失或误差好坏删行。首先复现原保存预测，物理GHI数值容差1e−3 W/m²，失败只修实现，不放宽数据规则。
4. 分别算相对原forecast输入、G1基底和无future_COT同容量头的变化，报告全样本和共同可替换行、逐站/时效/日期、修正幅度和方向，以及原forecast/目标R-COT误差。GHI为右端15分钟均值，瞬时R参考不能当严格辐射闭合。
5. 三seed分别算分数再等权，按目标日期配对区间；对保存预测独立FP64复算。替换实际未来R若无改善，说明“提高COT就必然提高当前头”不成立；若改善，只提示信息/映射利用空间，不称部署增益或严格性能上限，因为参考输入可能分布外。

只用已处理的C_fit/C_select开发资产，优先C_select固定头响应；不读取两个失败COT校正器的C_select输出，不恢复已封存候选，不打开official val、新时段、outerbuffer/test。短前向按站点/seed并行GPU，先实时核查PBS≤3/GPU≤8；无训练，不额外扫参考混合比例、阈值、幅度或选模。
