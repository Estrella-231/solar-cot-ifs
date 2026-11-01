# 湖北三站同期 GHI 对照：三种子验证结果

日期：2026-10-17。结果类型：validation 选模期结果；不是独立 test，也不是未来预报评价。

## 对照与复核口径

三站（牛首、王寨、米庄）分别训练 A（同期 AGRI+太阳几何）、AC（A+COT）和 AP（A+容量对照），种子42/43/44。三种子使用同一重建训练包、冻结日期表、训练器和 train-only 归一化合同。共用验证目标数为牛首856、王寨784、米庄816（每个 seed 和 arm 均一致）。指标由保存的逐样本预测重新计算，检查了站点/时间键、真值、晴空值完全配对及 checkpoint 重载；test/buffer载荷读取数为0。

## 结果

三站等权、每站三种子等权 RMSE：

| 站点 | A 无COT | AC 加COT | AP 容量对照 |
|---|---:|---:|---:|
| 牛首 | 76.65 | 75.00 | 76.14 |
| 王寨 | 66.07 | 64.27 | 65.27 |
| 米庄 | 80.28 | 73.70 | 80.10 |
| 三站等权均值 | 74.33 | 70.99 | 73.84 |

单位：W/m²，按原工程操作量纲合同；站点设备校准/供数方字段元数据仍未独立取得。

主配对差值按每种子先对三站等权、再对三种子等权汇总；负值表示右侧方法误差更小：

| 比较 | RMSE差 | 配对日期bootstrap 95%区间 | 连续两日块敏感性95%区间 |
|---|---:|---:|---:|
| AC − A | −3.34 | [−5.80, −1.12] | [−6.14, −0.73] |
| AP − A | −0.49 | [−0.80, −0.26] | [−0.85, −0.25] |
| AC − AP | −2.85 | [−5.15, −0.72] | [−5.50, −0.35] |

主 bootstrap 以22个 validation 目标日期为共同抽样簇，10,000次；连续两日块为敏感性分析。三种子各自 AC−A 的三站等权 RMSE 差为−3.50、−4.75、−1.77 W/m²，方向一致。

## 可支持的解释与限制

在这组湖北 validation 日期和当前输入合同下，加入同期反演 COT 的 AC 比 A 和 AP 的 RMSE 更低；容量对照 AP 相对 A 也有小幅改善。因此结果初步支持“COT条件与同期GHI估计误差降低相关，且不完全由本对照中的额外容量解释”。这不是独立test确认，也不能扩展为跨区域迁移或所有天气条件下有效。

AP 只控制网络容量，不控制 R 的 CPP 监督、COT 输入的空间/感受野差异等因素，故不能单凭该比较将收益归因于 COT 的唯一物理语义。更重要的是，独立的 forecast-GHI 归一化重载验证中 FA/FAC 总 RMSE 为77.150/78.155 W/m²，FAC−FA 为+1.005 W/m²，日期块区间跨0；当前结果不支持预测COT改善未来预报。同期反演结果不能替代预报结论。测试集继续关闭，直至来源、冻结范围及评价实现全部闭合并通过独立 test-freeze 复审。

## 审计产物

- 历史汇总回执（已由下方2026-10-18版本取代）：`audit_receipts/same_time_three_seed_normreload_20261017/independent_three_seed_validation_audit.json`
- 当前独立复算与 bootstrap 回执：`audit_receipts/same_time_three_seed_regen_20261018/independent_three_seed_validation_audit.json`
- 训练输出与逐站预测：`audit_receipts/same_time_normreload_runs_20261017/`
- 审计脚本：`tooling_20261017/audit_same_time_three_seeds.py`
- 统一 pack SHA-256：`e8c5b2dc587e8a938bba43fc564be43aec777788aca223c054341f75dd09a515`
- 统一 split SHA-256：`6ef8cbc7d2bb7d1b1d3defdfcba473cb171e06db0115c8206f7c289d8bad19c6`
- 训练器 SHA-256：`fd15bf6f939cb81d09bd05e6fc889f88e53e8a9e40366d417fa195848caf5999`
- 当前分析脚本 SHA-256：`719eeb731ac3a628c594423a9708031ec8b604019b05b105f4b24448fb5537fd`
- 当前汇总回执 SHA-256：`df0374c16ace142b5acb11b5d0f9b92b9138483ac1d6638bd06e3b1a6383ea6f`
- 同刻 checkpoint 冷启动/归一化重载回执 SHA-256：`eb72f580dabc3c523f7651cb1f70b936a296dc516046ebc35ce3ac42d1f65cde`
- 冷启动执行来源补充回执：`audit_receipts/coldstart_normreload_20261017_v3/COLDSTART_EXECUTION_PROVENANCE_20261018.json`，SHA-256：`f004411d081464d3896f795c1e12fd1f6a1a4c7e6287f07f96d072d450fce3d6`。它记录PBS作业214705、远端审核器当前SHA和运行前mtime、作业日志SHA；由于原运行回执未在执行时写入脚本SHA，不能将事后哈希称作执行时字节级证明。
- 2026-10-18 fresh read-only review：数值与27个bundle/checkpoint哈希复核通过；此Markdown旧版及paper claim audit曾指向旧汇总路径/旧脚本哈希，现已更新。该复核为同家族临时保证，不替代独立test-freeze审计。
- seed42/43/44 PBS：`214668.tc6000`、`214669.tc6000`、`214670.tc6000`，三者成功结束。
