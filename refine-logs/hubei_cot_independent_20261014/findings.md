# 湖北三站独立 COT 工程研究发现

## B2 同期 A/AC/AP validation 结果（2026-10-17）

**Verdict: partial；仅为 validation-cohort 证据，不是独立测试结论。**

预设湖北验证队列中，冻结反演器产生的局地 COT 表示加入紧凑 AGRI＋太阳几何头后，三站、三种子等权 RMSE 从 74.361 降至 71.428 W/m²（相对 A 降 2.932 W/m²，3.94%）；容量匹配的 AGRI 对照 AP 为 73.894 W/m²，AC 低 2.465 W/m²（3.34%）。点差在九个站点×种子单元方向一致。共享单日bootstrap的95%区间分别为[−5.070,−0.939]和[−4.599,−0.452] W/m²。

本结果可用于描述“当前湖北 validation cohort 下的增量证据”，不可称为独立泛化已确认、未来0–4小时预报提升，或COT物理语义因果贡献。A/AC/AP共用站点内时间键、真值和掩膜，且训练归一化仅用train；但反演器R与GHI头都用同一validation日期进行选模，存在选择偏差。AP只控制头部参数量，不匹配COT反演器的CPP监督、上游容量和感受野。AL对照、连续两日块敏感性及三站共同目标时刻敏感性现已补齐，具体结果见下文与`al_20261017/THREE_STATION_COMMON_TIME_SENSITIVITY_20261017.md`；独立test仍未完成。45°–90°太阳高度组中AC−A为+1.308 W/m²，不能声称各条件均有收益。设备计量/导出字段单位元数据仍未闭环。

旧bootstrap文件记录22个单日重采样单位；AL结果复算已补充连续两日block bootstrap及可复现脚本/收据。B2专用`EXPERIMENT_AUDIT.json`仍未完成，故完整性审计状态为unavailable；review为同模型家族，acceptance为provisional。

建议后续表述须纳入AL限制：“在湖北三站预设验证队列中，AC低于A及容量匹配AP；但R隐藏特征AL取得相近RMSE，且AC−AL日期块区间跨0。因此当前结果支持局地R/COT相关表示在本验证合同下有增量信息，不足以把收益归因于显式COT值，更不能推断未来预报收益；独立时间测试仍待完成。”

Evidence: `.aris/traces/result-to-claim/2026-10-17_run01/`、`.aris/evidence_precheck_hubei_b2_20261017.json`、`PROGRESS_20261016.md`及远端`logs/validation_date_block_bootstrap.json`。test/buffer载荷保持关闭。

## 三站共同目标时刻敏感性（2026-10-17；validation only）

从冻结共同pack和已保存的A/AC/AP/AL逐样本预测中，进一步取三站都存在的同一目标时刻交集：486个目标时刻，分布于21个预设validation日期；同一交集上每站均为486条、三种子齐全。用相同预测键与真值、站点等权/种子等权重算RMSE：A 76.067、AC 73.216、AP 75.577、AL 72.785 W/m²。按冻结连续两日validation日期对做共享配对bootstrap（2,000次），AC−A为−2.851 W/m²（95% CI [−5.392,−0.724]），AC−AP为−2.361（[−4.908,−0.229]）；AC−AL为+0.431（[−0.283,1.142]），区间跨0。此敏感性与完整站点支持上的方向一致：AC低于A/AP，但仍不能说明显式COT数值优于R隐藏特征。

可复现脚本、JSON与Markdown位于`al_20261017/audit_three_station_common_time_sensitivity.py`及同目录`THREE_STATION_COMMON_TIME_SENSITIVITY_20261017.{json,md}`。分析只读train/validation pack与保存预测，报告确认test/buffer值读取数为0。它仍是用于选模的validation证据，不改变独立test未完成、验证选模偏差及设备单位元数据待补的限制。

## B2 AL 表征控制（2026-10-17；仍为validation）

AL定义为A输入加冻结R dec1的局地16通道隐藏特征，参数量与AP/AC相差均小于0.3%。从共同pack中的相同站点、目标时刻、真值和掩膜独立FP64复算，三站、三种子等权RMSE为AL 71.089、AC 71.428、AP 73.894、A 74.361 W/m²。AL−A为−3.271 W/m²（共享连续两日块95% CI [−5.614,−0.889]）；AL−AP为−2.804（[−5.107,−0.457]）；AC−AL为+0.339（[−0.244,0.990]）。单日与两日块bootstrap均2,000次，seed=20261013；完整复算见`al_20261017/AL_RESULT_AUDIT_20261017.md`及同目录JSON/脚本。

解释：在当前验证队列上，R的隐藏表征本身可提供与AC相近的增量；AC相对AL的区间跨0，因而“显式COT值优于R隐藏表征”尚未成立。该控制不否认R表征与云物理有关，也不能将validation差异解释为因果或泛化收益；validation用于checkpoint选择，独立test仍关闭。设备计量/导出单位元数据仍未闭环。

## 预报输入三臂 GHI 结果（2026-10-18；validation only）

FA（仅预测AGRI）、FAC（预测AGRI+COT）和FAP（预测AGRI加AGRI派生辅助分支）已完成三个种子、三个站点的匹配预报输入验证。三臂使用同一预报包（SHA-256 `9421c0d5e19094454e5b94b98de797ac90248d3e5b56fdef6dacda339df359dd`）、同一trainer（SHA-256 `fd15bf6f939cb81d09bd05e6fc889f88e53e8a9e40366d417fa195848caf5999`）、相同的22个目标日期和键、实测GHI及共同支持；每臂30轮、batch 1536，test/buffer载荷读取计数均为0。FAP以预测AGRI主干特征的`32→4`投影替代COT分支，参数量24,613，与FAC相差0.51%。

站点等权、种子等权RMSE为FA 77.150、FAC 78.155、FAP 78.980 W/m²。按目标日期聚类的10,000次配对bootstrap中，FAC−FA为+1.005 W/m²（95% CI [−0.722,+3.780]），FAP−FA为+1.830（[−0.517,+4.778]），FAP−FAC为+0.825（[−2.657,+4.402]）；均跨0。FAP仅有单日目标日期区间，本结果不引用FAC的连续两日区间。可表述为“当前验证中未能分辨预测COT或新增AGRI派生分支的GHI增益”；不能称等效、显著变差或独立测试结果。验证集选模使区间属于条件性validation推断，独立test保持关闭。

复核收据：`audit_receipts/forecast_fap_three_seed_20261018/forecast_fap_three_seed_validation_recheck.json`；独立review建议限于主结果与讨论/结论，不挤入摘要，并保留validation/test边界。远端FAP单卡profile记录参数量与预算，不作为精度证据。

## 预报时COT增益主张复核（2026-10-18；validation only）

**Claim verdict: no。** 论文目前不能声称“预测COT能提升可部署的湖北 +15～+240 min GHI 预报”。最新归一化重载的同期诊断验证中，AC−A RMSE为−3.340 W/m²（95%目标日期簇区间[−5.800,−1.118]），AC−AP为−2.846 W/m²（[−5.147,−0.717]）；但这是同刻AGRI/COT输入的验证关联，不是未来预测证据，且A/AC/AP完整性审计为WARN。真正的forecast输入验证中，FAC−FA为+1.005 W/m²（FAC点估计差1.30%更差；95%区间[−0.722,+3.780]），FAP−FA为+1.830 W/m²（95%区间[−0.517,+4.778]），区间均跨0；预测COT的增益未得到证明。该独立claim review为同模型家族、provisional，置信度低；测试链审计为FAIL，test/buffer继续关闭。

因此将同期AC结果限定为“验证期诊断性关联”，forecast贡献暂记为未证实，不把二者混为一谈。下一步先补回16个精确AGRI train/val缺档，或在新的预注册合同中明确排除并重建清单；随后对FA/FAC/FAP同样本、同预算从随机初始化重训并重评。若forecast验证仍不稳定，不为此主张打开test；只有补齐测试来源与整链释放门禁后，才考虑一次性盲测。设备校准元数据仍未确认，RMSE单位按项目W/m²操作口径理解。

Evidence: `.aris/traces/result-to-claim/2026-10-18_run01/`、`.aris/evidence_precheck_hubei_cot_20261018.json`、`audit_receipts/same_time_three_seed_regen_20261018/independent_three_seed_validation_audit.json`、`audit_receipts/forecast_ghi_three_seed_normreload_20261017/three_seed_validation_summary.json`和`audit_receipts/forecast_fap_three_seed_20261018/forecast_fap_three_seed_validation_recheck.json`。不改论文正文、不读取test/buffer载荷。
