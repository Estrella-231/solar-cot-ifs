# 分站COT策略：四里固定，竺家残差修正

用户于2026-09-27批准：四里先使用当前预测COT模型；竺家按独立残差策略推进；先上传仓库。当前是工程候选选择，不是多种子或独立test验收后的业务发布。

## 当前模型和服务器位置

服务器SSH别名zjnu-hpc，工程根为/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907。

实验根为上述目录下experiments/regional_station_cot_20260925。四里指定regional_ghi_pilot_seed42_20260927_v1/forecast_best.pt，输入使用regional_ghi_pilot_bank_20260927_v1合同，站点id=0。竺家无COT基线为同一run的zero_best.pt，站点id=1。冻结湖南geometry24 SimVP及r256_full_exploratory_seed42_batch512_node22反演器。远端checkpoint不上传Git；启用执行前记录SHA-256及配置，保持资产只读。

四里当前验证RMSE159.667 W/m²；对应无COT166.753。仅为本次512训练序列/128验证序列、seed42 pilot。三组已完成，PBS210854退出码0；新竺家残差模型尚未实现、尚未提交训练，当前不得声称其提升。

## 固定四里

按站点路由：四里调用上述forecast模型；竺家调用独立候选。竺家训练不得更新四里任何参数或S/R。保存四里固定逐样本输出，后续验收模型权重哈希不变且相同输入的输出一致。当前仓库记录该选择；推理路由尚待实现。

## 竺家独立候选

冻结zero模型作为基线，新增竺家专用COT融合分支，输入完整局地64×64和区域32×32 COT场，8历史+16未来，保留目标时刻之前的因果轨迹；不恢复全图均值/p90特征。

输出为pred_kt=base_kt+delta_kt，pred_GHI=pred_kt×区间clear_sky_GHI。修正分支同时接收AGRI、太阳几何及COT轨迹；末层零初始化，初始预测与冻结基线一致。修正允许正负，约束幅度但不预设厚云必然减值。约束形式和强度须在执行协议中固定，不能使用验证集严重案例逐条调参。输出非负处理及kt大于1的云增强情形须明确记录，不能强行把kt截断到1。

配对组：冻结基线、同预算AGRI残差容量对照、AGRI+COT残差。旧共享融合模型作为历史参照，不能以不同收敛预算冒充新公平对照。先在现有pilot bank验证稳定训练，再扩大样本；不同时搜索多个结构或约束。验证按竺家独立选模，记录训练损失、梯度、实际更新次数及各时效指标。最低更新数必须真正约束早停，不能仅提高max epochs。

验收：样本/真值/白天掩膜一致，测试集关闭；报告总RMSE、MAE、bias、+15～45min、严重事件及日期bootstrap。未稳定优于冻结基线时，竺家仍使用基线。零输入干预为诊断，不作为物理因果证明。四里输出一致性为硬性工程要求；竺家收益仍待实验。

## 来源与结论边界

诊断见docs/ZHUJIA_COT_DIAGNOSIS_20260927.md及保存的summary.json/input_probe.json。最差12例局地分支驱动不代表所有样本；共享训练干扰、过拟合及云位置误判仍是待验证解释。CPP来源限制、IFS因果门禁继续有效。此次上传不修改论文结论，也不表示新训练已启动。

## 2026-09-27 残差pilot启动

PBS 210898因Python导入时cuDNN符号冲突退出，Exit_status=127，训练循环未启动、只留下空输出目录。PBS 210900沿用成功pilot的LD_PRELOAD顺序后通过导入，但数据worker在6.6GB共享盘bank读取停滞、GPU无训练利用，已在无任何checkpoint时主动qdel。PBS脚本已增加将bank复制到节点本地盘、仅复制所需小型标签数组后训练；重提第三次前先核对代码和队列。失败与处理记录保留，空目录仅在确认无文件后由PBS脚本rmdir。

第三次提交PBS 210903于19:06:02分配node21 GPU0。bank复制到本地/tmp完成；19:07首个zero臂状态为24/600 updates、epoch2，验证RMSE 208.968，GPU显存约6.7GB、利用率74%。这确认训练循环已启动；单一早期验证点不作模型效果结论。任务继续运行，下一阶段还包括fusion和residual各600 updates。

提交时账户另有PBS 210853一个运行任务，合计2任务/8卡。现用样本为既有regional GHI pilot bank的512条train/128条val序列，仅竺家监督行；同一600 optimizer updates、batch512、seed42、lr3e-4。无COT和direct-COT头从同一seed独立训练，残差以无COT验证选定checkpoint为冻结基线，使用direct-COT头初始化时空COT编码器；delta零初始化、tanh限制为±0.15 kt，delta平方惩罚0.05。每臂严格跑满600次更新，固定预算内按竺家验证RMSE选checkpoint，并记录训练损失和实际step。四里checkpoint不参与训练且其哈希写入完成结果。开始训练前零delta等于基线的kernel witness写入输出。

此为小规模单seed探索，不代表正式GHI提升。完成后先核验witness、600 updates、共同样本键、+15/+30/+45min和日期bootstrap；不以单次pilot决定部署或打开test。
