## 最新进展（2026-10-21，本轮实况）

- 全量几何审计 PBS `215288.tc6000` 已完成，exit_status=0（node22 CPU-only）。覆盖 4,086 个 train 序列、407 个 val 序列，共 10,473 张唯一 AGRI 帧（train 9,560、val 913）和 3,479 条 R 样本（train 3,178、val 301）；train/val 日期键隔离通过，test/buffer 载荷读取均为 0。
- 审计报告列出需要从精确 UTC 原始 HDF 恢复 255 帧、1,344,223 个白天像元；检测到 28 帧旧 aux 几何不一致，因此旧数据不能直接用于修正后训练。
- 独立 NOAA 官方 JavaScript 几何参考核验已通过：train/val 共 62,802 个点，天顶角最大绝对差 `1.75e-13°`，方位角最大圆周差 `2.72e-10°`，大气折射关闭后比较几何角。报告为 `NOAA_ALL_TRAIN_VAL_REFERENCE_AUDIT.json`。
- 当前执行器 attempt3 已进入 `ALL_TRAIN_VAL_OFFICIAL_NOAA_REFERENCE_PASS`，会继续传输恢复请求至 FD-201、恢复白天反射率、重建 train/val 数据并按队列规则启动后续重训。最近 SSH KEX 前断连，故恢复、DATA_READY 与模型重训尚未完成；不要把公式/几何通过误报成重训已启动。
- 已有恢复需求归档与审计产物先保存在服务器原实验目录；下一阶段成功后再核验 DATA_READY、manifest/key/labels/AGRI/geometry 与 GPU 训练队列。PBS 仍最多3个未结束作业；215284 不是本任务，不得触碰。
# 湖北太阳几何修复与配对重训状态

## 当前状态（2026-10-21）

- NOAA/Meeus 太阳几何核心及预处理接入已完成；官方 JavaScript 交叉核验 864 个合成公式案例，最大角度差低于 1e-12°。本地回归脚本再次通过（864 cases）。这只证明公式实现，不等于实际训练/验证数据已审计。
- 全量实际数据审计 job **215288.tc6000**（node22，4 CPU、无 GPU）最后一次可连接时为运行态。需产出全量 geometry report 并通过独立 NOAA 对照后，才可继续恢复反射率和构建训练数据。
- PBS 215287 是部署路径错误导致的快速失败；215286 是 R pack 路径拼错并已在 10240/10473 帧处失败；215285 暴露旧 aux 与 raw 几何临近日落不一致。失败日志和证据均保留，未覆盖旧数据/权重。attempt2 本地执行器随后只在查询报告文件时遭遇 SSH key-exchange 断连而退出，未做远端写入、未 qsub。attempt3 已接续等待同一个 215288，不重复提交审计。
- `prepare_data_attempt2.pbs` 已修正为 `restored_attempt2`，日志改为 `prepare_data_attempt2.log`；已同步到服务器，远端/本地 SHA-256 均为 `185f9f908c1dbb9a03275f4ebf5db104834ebea98c7d711259ac656bed27314a`。
- SSH 连接存在间歇性 KEX 前断连；本地执行器 attempt3 已启动（tool session 94875），状态为 `RESUMING_WAIT_FOR_EXISTING_AUDIT_NO_RESUBMIT`；它只对明确发生在SSH认证前的KEX断连重试，避免命令执行结果未知时自动重放。审计成功后继续独立 NOAA 复算、恢复和数据准备。执行器只在审计和恢复依赖满足后推进，qsub 前后核对 slfu 全队列，绝不盲目重复提交。
- 最后一次成功读取的全队列快照：215284（其他 direct COT 作业）与 215288（本审计）运行中；215287 已完成。215288日志之后进到6400/10473帧（监控快照）。FD-201 CPU恢复节点只读探测通过，swc Python存在，根卷剩余211 GB。此快照在 SSH 恢复前会过期，任何新 qsub 仍必须实时查询。

## 尚未完成

1. 实际 train/val 全帧及 R 样本太阳几何核验、时间/日期隔离报告、官方 NOAA 全实际点复算。
2. FD-201 CPU 恢复旧掩膜误置零的白天反射率，并由修正版数据准备脚本生成修复后的 train/val。
3. 重建/重训受影响 S 与 R 模型，再重跑湖北三站有无预测 COT 的相同样本配对 GHI 实验。

**截至当前：正式全量审计未完成；数据恢复未完成；任何修正后的 GPU 训练和配对 GHI 重训均未开始，暂无本轮成绩。test/buffer payload reads = 0。**

