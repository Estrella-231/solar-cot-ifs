# 既有神经修正头的固定模型诊断

承接hubei_cmf_ghi_increment_20261023/attribution_attempt1：其27参数修正拟合内改善、C_select退化，不能推论已训练神经条件头同样失败。承接docs/HIGH_GHI_UNDERESTIMATION.md，不重复提出已具备的基线标量/AGRI上下文作为新方法。

只读现有hubei_forecast_cot_increment_20261022固定G1和G2/G3的best及last，加载既有C_fit/C_select张量；G1来自F训练，G2不读未来COT值，G3读预测COT。保留原模型、掩膜、选模和结果。重放选模RMSE应吻合原COMPLETE.json；不重新选择epoch，G1/step0只是诊断参照，不能回写训练结果。无新增训练或上游反演。

另做固定头干预：恢复原R要求cosSOZ>cos80，按原整3×3均有效规则生成available；任一像元缺效时精确回到同一G1。只改输出门控，不调整数值、权重或阈值，不删评价行。它未经过重新训练，只是可用性遗漏的影响诊断，不能叫修复后模型成绩或论文主表。

保存C角色、站点/目标日期/时效/OOF来源、各臂逐样本GHI、未来COT太阳有效性。分别计算RMSE/MAE/bias、相对G1修正平均值/幅度/方向、ΔMSE=2ed+d²、全体与太阳缺效子群；best/last分列，以实际更新计数核查训练过程。仅三站三seed等权；C_select已参与选模，非独立test，不新增日期筛选。GPU仅短前向，三张空闲A100站点并行；每次提交前后实时全队列≤3/总GPU≤8，Luna监控。

若只是较大训练预算导致选模退化，保留该负结果，不重新扫epoch。若可用性缺陷影响显著，再冻结独立修复训练方案；不能把其它分支的缺陷直接当作已定位的神经模型错误。official val、新时段和test/buffer数值不读取。
