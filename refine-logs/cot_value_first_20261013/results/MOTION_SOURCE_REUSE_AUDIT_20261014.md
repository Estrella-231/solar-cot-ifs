# Motion+Source 旧 COT bank 对当前月度合同的复用审计

日期：2026-10-14 BJT。范围：服务器现存验证产物的只读来源核对；不加载新 test，不提交 PBS，不读取预测载荷数值。

## 核对方式与发现

- 通过 `zjnu-hpc` 只读连接成功。实时 `qstat -u slfu` 显示一个未结束作业 `213850.tc6000`（R）；本次没有提交或改动作业。
- 旧 Motion+Source COT bank 存在于 `HunanBackboneTriad_20260921/geometry24_v2/evaluation/four_backbones_cpp_cot_val_20260923_geometry24_v2/motion_source/forecast_cot_log1p.npy`，大小139,526,272 bytes，mtime为2026-09-26 14:44 CST。旧评估目录另有Motion+Source AGRI/COT数组及公共CPP评估契约。
- 旧目录 `contract.json` 记录 `split=val`、4,258条验证标签行、89,765个站点—目标—时效配对、旧 `geometry24_v2` 的8→16预测合同。Motion+Source权重是旧 `motion_source_geometry24_formal_e12_v1/best.pt`，SHA-256 `02e5f6cb96cd401d5f32f718435a2b6e2d5adc52cad832d2fc215f1375f0701b`；记录其随机初始化、12 epoch和旧训练manifest `9b6f5ff452a377a202db3d7dc305cebd74230cb084b6accf21df745b4b47dcf8`。当前新月度SimVP的split manifest为 `f6703ed5e42f1ae00aac6538b108dc4d9361197ad57ad86a0f407767aa2e4cb2`，并非同一训练合同。
- 旧冻结R记录指向 `cot_repaired_seed42_v1`（PBS 207372，checkpoint SHA-256 `0cbdb932f49945259d5f1704316edaf6481be7cfdb54692236f829988925799c`），且 `source_NC_generation_checkpoint_binding="UNVERIFIED"`。这不是当前 `monthly_R_station16_20261013_v1` 的R。当前B3 bank绑定新月度SimVP/R及当前split，故不能把旧Motion+Source阵列直接拼入当前S→R/持续性共同支持表，也不能用它补齐T21排行榜。

## 决定

T21保留为“当前合同下不含输运基线”的有限诊断，不把旧目录同名模型当成可比方法。若未来需要完整Motion+Source比较，必须由可核验的运动模型来源在当前冻结月度日期、输入几何及新R下生成独立验证bank，再按共同键/掩膜复算；这属于额外模型重建，不由本次只读审计自动放行。现有T17/B3仍不支持B4 GHI头训练。所有结论仍为validation-only，test载荷未读取。

## 可复核入口

- 旧产物目录：`/public/home/slfu/ttzhou/swc/irradiance_forecast/HunanBackboneTriad_20260921/geometry24_v2/evaluation/four_backbones_cpp_cot_val_20260923_geometry24_v2/`
- 新合同bank报告：`cot_forecast_eval_20261014_v3/COT_FORECAST_VALIDATION_RESULTS_20261014.md`
- T21 tracker状态：`../EXPERIMENT_TRACKER.md`
