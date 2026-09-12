# Research Findings

## 2026-09-12：固定 AC checkpoint 的 COT 反事实前向

original-cohort AC seed42/43/44 在同一99849条湖南validation行上完成原COT、标准化全零、训练集晴天中位COT及四维逐项屏蔽前向；三个原COT前向与历史保存预测逐值完全一致。四里全样本中屏蔽COT均值和p90的RMSE分别平均改善1.568和1.539 W/m²，且三个seed方向一致；屏蔽空间标准差仅改善0.243，屏蔽中心像元则恶化2.690且三个seed均恶化。说明固定AC模型内，四里整体向下响应主要沿均值和p90出现，中心像元总体提供有益抵消。

四里预测晴天3046行中，训练集晴天中位替换三个seed均轻微改善（平均−0.221 W/m²）；均值/p90单项屏蔽跨seed不稳定，中心像元屏蔽使RMSE平均恶化19.259 W/m²并使预测再下移29.028 W/m²。全零替换跨seed强烈不稳，不能作为晴天默认值。独立NumPy从逐行保存预测重算168行/672标量通过，最大差1.14e-13。该实验是AC内部输入干预，AC(c=0)不等于独立A checkpoint，不能唯一解释全部A→AC差异。详见docs/AC_COT_COUNTERFACTUAL_RESULTS_20260912.md。

## 2026-09-08 后续：COT负收益原因分解

8候选/全部epoch/同99849验证预测重读与SHA核验通过。当前kt MSE在GHI残差上隐含1/Gcs²权重，与GHI验证终点不同；original AC在Gcs<300改善、≥600 RMSE+7.559，QC1分组模式不同。低Gcs<20仅约5.7%训练行，却占零预测参考kt损失95.69%/81.25%；这是参考敏感性，不是实际训练残差占比。两分支主要净退化在竺家；QC1相同LR0.001的AC稍好，不能称所有配置/条件均变差。四维COT压缩与既有R外推域退化为另外两条线索，未证明唯一原因。

四个已选best的全train/val只前向任务207702在PBS prologue因Gold异常失败，脚本未启动，没有模型或数据修改。待明确调度恢复后同协议重试，禁止绕过调度器；全部结果与限制见 docs/COT_FAILURE_CAUSES_20260908.md。模型/loss对照尚未冻结或执行，不预设COT必然有效。

## 2026-09-08：用户要求的区域与源像元复查

冻结两站×train/validation首中末12行后，8个唯一源CPP NC→修正sidecar→R target/mask逐值通过，2735有效参考像元；full AGRI→Combined全20通道及R实际16通道逐值一致。独立复算通过，最近站点像元均为[8,8]。原HDF固定4片的DN/LUT还原与full/Combined一致，另一程序只读取两张LUT后使用已存DN重建亦通过。本轮未发现这12行下游裁剪、转置/翻转、站点互换或一像元偏移错误。

实际CPP生产writer、批次、checkpoint、输出地理注册/拼接与真实输入时刻仍未绑定。正确的NC坐标轴和下游逐值相等不能排除生成时已错配；具体缺口见 docs/CPP_PRODUCER_GEOMETRY_REVIEW_20260908.md。已询问生成脚本/日志路径，小时任务改为处理新生产线索，不重复已完成检查。未改模型/数据、不重训；旧GHI负收益仍不能自动归因于空间或据此判断COT普遍无效。完整结果见 docs/COT_SPATIAL_REAUDIT_RESULTS_20260908.md。

## 2026-09-08：固定SimVP外推下首轮COT无增益

original与QC1两个训练分支各运行A/AC×两档LR，seed42，全部8次完成并保留。相同99849条validation预测重算后，original的AC比A RMSE高3.1781%，QC1高0.8958%。独立CPU模型抽样重载通过。完整结果与所有候选见docs/HEAD_PAIRED_PILOT_RESULTS_20260908.md。

本轮不支持当前四维显式COT统计拼接带来整体GHI收益；不能据此宣称COT普遍无效，也不能检验IFS互补。疑似平直段QC1没有逆转结果方向，不能选择QC1掩盖original结果。尚未证明负收益来源是损失、COT反演、外推域或模型容量。

执行原计划止损规则：检查通道/归一化、反演饱和、forecast输入域；只允许明确实现错误的新命名修复，不为追求正结果进行大规模结构/损失/样本搜索。P203归因扩展和正式test保持未放行。细化语义审查与后续动作见docs/HEAD_PAIRED_RESULT_TO_CLAIM_20260908.md和docs/HOURLY_NEXT_ACTIONS_20260908.md。

## COT数值诊断边界

全量小型COT特征和六固定序列映射诊断未发现通道、重复归一化或汇聚错误；R CPU与已存GPU FP32复算仍有最大log1p差0.008137，初始0.001检查未过。旧0.01规则针对GPU FP16，不能作为本次放行阈值。整体诊断保留warning，下一项为同原分片batch形状的GPU后端/TF32数值见证，不训练、不改变旧cache，不预先把差异认定为负收益原因。

## 207626数值见证

单A10050秒完成。归一化完全一致，benchmark/TF32均关闭GPUvsCPU最大log1p差4.053e-6；两者均开启重放旧cache仍有一分片0.001033306超过既定0.001，保留失败，不能称全部通过。后端组合可解释主要变化，但没有隔离TF32单因素；无具体特征链实现错误、无AC负收益机制证明。原数据/模型/8组结果均保留；可继续既有R pack与forecast的有限同target/同CPP参考掩膜诊断，不扩展正式模型搜索。

## 固定目标 COT 参考误差诊断完成

预先固定 64 validation 目标，58 匹配目标形成 872 配对；152 缺 lead 全保留。相同 R/CPU精度/几何/参考mask，仅替换真实与 SimVP AGRI，pair-pooled COT RMSE 4.293596→10.131402，全部 32 station×lead 组外推误差更高。独立审计 635 项保存数组指标精确重现，未发现配对、特征链或指标实现错误。详见 docs/COT_PAIR_DIAGNOSTIC_RESULTS_20260908.md。

这提供了固定样本内输入替换的 CPP-reference 检索退化证据，尚不能归因为 GHI 负收益机制或证明 COT 普遍无效；重复 target、不同 lead 的覆盖、R validation 选模和 CPP source teacher 未绑定均限制解释。旧cache数值warning仍开放。保留 8 次负结果，不重训 S/R、不调整 cohort 或 loss、不打开 test；按 docs/COT_PAIR_RESULT_TO_CLAIM_20260908.md 的独立语义判定执行下一步。

独立 Result-to-Claim 复审：claim_supported=no，same-family/provisional；confidence=high 仅针对本配置不放行判定。P202 当前配置正向扩展关闭，IFS 单独保持等待具体新来源。此判定不证明 COT 普遍无效。
