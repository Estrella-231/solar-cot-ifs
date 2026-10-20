# 湖北修正COT未来GHI预报进度

2026-10-20。授权范围：真正+15～+240分钟预报，有/无COT，同样本和训练设置。

| 阶段 | 状态 | 证据 |
|---|---|---|
| 固定旧预报输入与新R来源 | COMPLETE | 旧forecast输入pack 70,171行；新R为正确UTC attempt3；不复用错时COT |
| 五折S预测/折内R输入导出 | COMPLETE | FD-201 export_attempt1，五折共4086训练序列、407验证序列；未导出真实未来AGRI或旧CPP监督 |
| 转运与正确UTC五折R包 | EXPORT_TRANSFER_PASS; FOLD_BUILD_IN_JOB | 归档1,422,069,760字节，学院实测SHA与FD-201导出一致；正确UTC折包在本地盘作业内构建 |
| 五折正确UTC R训练及预测COT | ATTEMPT2_SUBMITTED | 首次在训练前统计JSON阶段失败；类型修复后新PBS215245接续，模型/数据/预算未改 |
| 同支持18个GHI头 | PREPARING | 3站×2组×3seed，24,489相同参数、共同初始权重与顺序、30epoch、batch1536 |
| 独立FP64与日期块配对评价 | NOT_STARTED | 等实际保存预测，保留全部站点和时效 |

工程根（学院）：`/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_forecast_correctutc_20261020/`。

## 提交记录

- PBS `215241.tc6000`，单卡node22 GPU1，已由Q进入R；日志`logs/run_attempt1.log`。提交前slfu全队列无作业，提交后只有此一项，未超过3个未结束任务。
- 导出归档SHA `d36a46f106d614bd823428fc4ae8e8e7ab159b644c562a5d25320b59db17935e`，远端sha256sum复核一致；模型定义SHA `8af2d0e101b492bf2096223fe788b7f3a8a9b5ece2544db45903b95397620e05` 与已完成新R训练来源一致。
- 代码、本地`py_compile`、PBS语法检查已准备；登录节点CPU检查出现共享存储等待，因此作业内将代码、输入、模型和训练输出先放本地`/tmp/hb_fc_corr_<PBSID>`，每阶段再持久化。CPU/GPU数值检查在正式训练前执行，失败即停止。
- 例行监控已委托Luna，首次只见PBS prologue与资源分配，尚未宣称R/GHI开始训练；以日志实际阶段为准。
- 后续已确认`PAIRED_FA_FAC_CPU_WITNESS_PASS`：24,489参数、完整初始权重相同、零COT输出相同、FA不读取COT、FAC对COT梯度非零；实际卡为A800-SXM4-80GB。共享环境首次读取较慢，初始化期间GPU尚未进入训练，不据此报告训练吞吐。

## 数据包统计序列化修复与重提

- 215241以exit_status=1退出，发生在`build_correctutc_fold_packs.py`统计监督行数后的JSON写入，`numpy.int64`不可序列化；未进入R或GHI训练。前置archive本地stage、CPU配对检查和A800前反向检查均已通过。
- 唯一逻辑修复为统计量显式转Python `int`，mask判断转`bool`；不改目标、掩膜、样本选择或训练设置。旧源码在提交`833a561`及远端`code_attempt1`保留，失败日志在`ops/run_attempt1_failed.log`，旧临时目录未覆盖。
- 新尝试为`code_attempt2`、`run_attempt2.pbs`与`runtime_attempt2/`。PBS **215245.tc6000**已提交；提交前只有215241为C，未结束计数0，提交后只有215245为Q，未超过3项。
- 新结果预期在`runtime_attempt2/scored_attempt1/`；其中attempt1为内部子阶段名，完整路径与失败尝试完全独立。Luna已切换监控新作业。
