# 预测域COT校正的数据准备

执行上阶段 `../hubei_matched_R_dynamics_20261023/NEXT_COT_DOMAIN_PLAN.md`，保持其候选结构、预算和继续条件，不按旧C_select结果调参。

本次先生成原F_fit/F_select日期的同折R真实图参考；校正器只用F_fit拟合和F_select选模。S/R/G1冻结；C_fit/C_select不能参与校正器拟合或选择，official val、新时段、outerbuffer、test载荷均不读取。真实未来AGRI及其R输出仅作监督，不能作为部署输入。此步骤没有GHI提升结论。

prepare.py先仅用rows/roles/stage清单核对日期隔离、时间关系、来源和可用帧，再在学院node22 CPU低优先级提取原缓存。与上阶段相同的原始帧直接复用并核对缓存及patch哈希；新增帧核对原data/aux哈希。输出不覆盖已完成或部分产物；失败须定位并新命名attempt。

后续冻结R推理复用已通过五折重放的infer_roles.py；分配前实时检查全账户PBS≤3、GPU≤8及物理占用。预计只是短推理，按fold分配3卡，非多卡训练。训练尚未启动，前向/反向、恢复和吞吐须先实测。环境复用.aris/compute/pbs-swc-weather.md中的swc及既有PBS cuDNN recipe，不安装依赖。
