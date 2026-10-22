# 下一阶段：冻结S/R后的预测域COT校正

尚未启动训练。不是再调当前C_select融合头；依据为同折R参照下可见动态信息、厚云漏报及两段开发日期均偏薄。这个假设不保证GHI收益。

## 数据与阶段隔离

1. 复用原prepared_attempt1的F_fit/F_select日期元数据，为这些日期补齐同fold真实历史/目标R参考。使用与当前相同raw/aux SHA、BJT/UTC、station patch和OOF R权重规则，复用已完成C域参考，不重建已有输出。仅原split=train，仍不读取official val/new-period/outer buffer/test。
2. COT校正器只用F_fit拟合、F_select选模；不使用C_fit/C_select的真实未来R、CPP或GHI做其优化/选择。原S、R和GHI基底G1均冻结。这样后续GHI在C拟合时，使用的是校正器未见日期的预测COT，避免将校正训练内“接近真值”的COT喂给GHI。
3. F阶段目标是同OOF R(真实target AGRI)；这是训练代理，不是独立COT真值。读取可用CPP时另列诊断，不把R代理改善升级成CPP改善。推理只能读预测未来AGRI/原预测COT、初始化及过去COT、已知几何/lead。

## 首个固定候选及控制

预测域残差校正：保持原16×16R输入场，34通道=预测AGRI/三几何16、原预测log-COT1、8帧因果历史log-COT8、历史有效mask8、lead/240一个常数平面。无全图均值/p90输入，归一化只用F_fit。34→16→16两层3×3 SiLU卷积，最后1×1输出log-COT残差、末层零初始化。输出expm1(relu(log1p(raw forecast COT)+delta))，太阳缺效位置保持缺效；数值step0需复现原预测COT。

目标仅物理COT均方误差/10000，对同R目标与预测共同有效像元；不增加GHI损失或天气权重。新校正器独立命名，不叫原R更新或新的SimVP。原forecast COT与同R初始化持平为必要对照，不能只优于更弱的新区域R。

先实测小批量前向/反向、断点恢复和吞吐。固定global batch128、最多600updates、AdamW lr1e−3/wd1e−4、cosine、每50updates评分；step0候选保留，三个seed42/43/44同来源；预算或架构若需变动，拟合前记录而不按F_select误差临时扩展。长任务按多卡规则实测，短小模型可按seed并行多卡，不浪费已有进度。

## 前置停止条件

F_select站点中心等站/时效COT RMSE必须同时优于原预测和持平，目标日期簇95%CI上界<0；三站不得退化，至少2/3seed同向；厚云R参考≥30 POD不得下降>1pp、FAR不得增加>2pp，增厚>3召回不得下降>1pp。全16×16和3×3误差、bias、峰值及全部日期也保存，防止只通过站点指标掩盖伪造厚云。

若失败，封存此候选，不查看C_select校正精度或再扫损失/厚云阈值。若通过，先冻结来源、权重及train-only归一化，再生成C_fit预测COT与C_select预测COT，真实参考不可进入GHI输入。

## 之后的GHI配对

保持冻结G1、既定9922参数修正头及相同全部C拟合日期、样本、行序、预算和step0选择，三组：不输入future COT、原future COT、校正future COT；唯一可变输入来源，掩膜共用且缺效回同G1。不混用旧不同预算头。只有校正COT胜过同容量无COT及原COT、跨日期区间和种子/站点保护条件通过，才考虑外层开发评价。最终仍需预报GHI证据，COT-only分数不完成总体目标。
