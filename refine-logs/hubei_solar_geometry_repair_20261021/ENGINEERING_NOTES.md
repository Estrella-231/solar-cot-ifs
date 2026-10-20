# 执行记录与复现说明

## 太阳几何修复

旧均时差公式漏乘 `180/pi`，使太阳天顶角/方位角、`cosSOZ`、`cosRAA` 与昼夜掩膜不可靠。新核心位于 `scripts/solar_geometry_noaa.py`，被冻结到本实验 `code/`。约定输入UTC、东经为正、几何天顶角不加折射，方位角自北顺时针。与 NOAA 官方 JavaScript 实现比对 864 个公式案例，误差小于 1e-12°。这是公式单元测试，不能替代实际训练/验证数据全量审核。

## 实际数据审计及历史失败

全部 10,473 个 SimVP 输入帧和 3,479 个 R 样本的实际几何审计由 PBS 215288.tc6000 执行（node22，4 CPU、无 GPU）。此前试次保留：215285 暴露旧aux/raw临近日落不一致；215286 因 R pack 文件路径 attempt3 不存在，在 10,240/10,473 帧处失败；215287 因远端尚无其引用的 code_attempt4 目录而快速失败。以上都未进入本轮训练，也未覆盖旧产物。

正确的 R 包路径是 `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cloud_probability_20261020/R_pack_correctutc_attempt2`。当前审计 attempt5 对旧 aux/raw 差异作记录而非断言，重新按独立 NOAA 核验后的几何计算；是否恢复 C01–C06 根据原始HDF保存的 raw day channel 判定，不做插值或合成。

## 续跑机制和资源

本地 `code/advance_pipeline_attempt3.py` 接续现有审计；通过后依次执行 NOAA 复算、FD-201 CPU 可见反射率恢复、node22 train/val 数据重建及经队列控制器的 S/R/GHI 配对训练。SSH 恢复后已手动同步并验证修正版 `prepare_data_attempt2.pbs`。attempt2 本地执行器在报告轮询时遭遇SSH认证前断连并退出；错误保存在 `PIPELINE_STATE_ATTEMPT2.json`，确认没有远端写入或qsub。attempt3 执行器（session 94875）安全接续同一审计任务，且仅在SSH KEX认证前失败时无限退避重试，不对未知qsub结果重放。任何提交前实时检查 slfu 账户全队列，最多 3 个未结束作业；其他作业不取消。

PBS 215284 是其他 direct COT 工作，不归属本实验，不能取消。SSH 曾间歇性在 key exchange 阶段被关闭；截至最近一次监控核验，215288 仍运行，报告进度到 6,400/10,473 帧。FD-201 CPU恢复节点连通，环境可用且根卷剩余211 GB。尚未提交数据准备或模型训练任务。test/buffer 数据未读取。本轮训练和 GHI 结果必须等实际审计、数据恢复、manifest 和训练产物完整后再报告。


Attempt2 execution receipt: only the read-only report-existence probe failed before authentication; no remote write or qsub occurred. Attempt3 state is isolated in PIPELINE_STATE_ATTEMPT3.json and resumes the same job 215288.
