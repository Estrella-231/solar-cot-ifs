# 当前大日期固定GHI头的同折COT来源替换

执行../hubei_directional_cot_correction_20261023/NEXT_FIXED_HEAD_RESPONSE_PLAN.md，先原预测重放再离线替换，不训练任何模型。旧attempt12 seed42/约18%官方val范围不替代当前三seed、大日期条件头和完整C_select开发范围。

原C_select所有8266行、三站三seed全部保留。读取matched_R_dynamics的同fold参考，逐station/init/target/lead/source_role连接；比较原forecast、同R初始化持平、同R实际未来图参考。joint mask同时要求预测及两个参考的太阳/数值有效，三种替换共用；缺效回同G1。另列原掩膜的G1/有无未来COT原预测，避免把掩膜变更误记成换源收益。

模型为calibration_coverage大日期large_future_COT与large_no_future_COT各best，以及原G1；权重、预测AGRI/几何、8帧历史、归一化、base标量均固定。替换目标9个COT值、与最后历史COT的9个差值槽及其共同valid槽；差值仍相对原输入的最后历史COT，不能偷换历史数据。参照初始化COT采用该条OOF R，非full-R历史bank。线性归一化使用原meta的C_fit统计。

原已存大日期G1/无/有未来COT必须复现，逐样本物理GHI最大差≤1e−3 W/m²；同掩膜分支不可用时逐位等于G1。保存全部键、真值、clear、COT物理值、joint mask、覆盖和逐样本预测。实际未来AGRI→R是不可部署的离线干预，既不是forecast成绩，也不是严格上限；不会进入任何训练。

评分逐站逐seed先算后等权，全部原行与共同可替换行分别报告，日期簇配对CI、1—4h、偏差、改变量及COT代理误差均保存；独立FP64复算。只用已处理C_select开发资产，不读失败校正器C_select输出或official val/新时段/outerbuffer/test。不扫替换比例/阈值/幅度或重新选模，不自动恢复封存候选。

短前向按三站三张空闲卡并行，提交前后查全slfu队列≤3、GPU≤8；沿用swc，不安装环境。Luna监控作业；本轮无GHI训练、无论文修改、无部署增益结论。
