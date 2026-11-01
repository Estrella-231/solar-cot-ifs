# 2026-11-02 用户确定：真实图训练GHI头

当前主路线使用真实同期AGRI三帧训练的既有GHI头，冻结后接预测AGRI及预测COT进行预报。不再用预测图训练或微调GHI头；不得把这类变化隐含在“全train重训”中。此决定覆盖此前“两条训练路径均可默认推进”的表述。train/validation/test外层日期清单与其他数据规则不变。

Details: docs/GHI_HEAD_ROUTE_20261102.md (repository root).

---

# 第二篇当前执行摘要

更新：2026-10-30。旧“READY_FOR_ASSET_AUDIT、尚无训练、P200全部通过后才启动、先单卡”的立项状态不再适用；原文保存在`../docs/execution_simplification_20261030/runtime_fix_before/refine-logs/PIPELINE_SUMMARY.md`。

研究问题仍是：COT表示能否改善站点GHI，以及可用IFS是否提供额外收益。湖北三站和湖南两站分别训练与评价。当前结果见[结果总表](../docs/experiment_results_20261030/SUMMARY.md)，任务进度以相应实验目录tracker及实际产物为准；本文件不声明实时PBS状态。

- 默认全部train与现有合规资产级联；不自动拆U/H、不强制U-only R、五折OOF或多层内部校准划分。
- 有效预测与地面GHI齐备即可直接配对评价；不以CPP稀疏、COT辅助指标或每站每时效全部改善作为继续开发的通行证。
- 当前主路线为真实图训练的冻结三帧头接预测图；预测域训练头仅保留历史结果，不再默认推进。validation允许开发和选模，不要求每轮提供新观测。
- 新接口针对检查，未变的检查复用；长任务按现行多卡规则加速，例行监控交Luna，未结束PBS≤3。

普通湖北预测头已取消默认OOF代码阻断，显式`--require-oof`保留可选复现实验。最新湖南SimVP冻结头预报验证为127.421→125.340 W/m²（1.633%）；这是该设置的正结果，旧负结果保留，不据此声称每站均显著或已完成独立test。

当前规则见[CURRENT_POLICY.md](../docs/execution_simplification_20261030/CURRENT_POLICY.md)，八项落实见[IMPLEMENTATION.md](../docs/execution_simplification_20261030/IMPLEMENTATION.md)。外层划分、时间/空间/标签和共同评价样本不变；test/buffer关闭，未来真实AGRI/CPP不进推理。本文不新增任务、不恢复定时、不改论文正文。

## 2026-11-02 本轮完成

真实三帧同期训练、冻结后接DyDiff的路线已在两省形成预报结果。湖北24451条、17日，101.017827→96.071320 W/m²（4.896668%，日期差区间[-9.983211,-0.154002]）；湖南66086条、51日，122.588303→120.501835（1.702013%，区间[-3.527266,-0.597792]）。均是验证，各省权重、归一化及标签独立，不能按绝对误差排名。湖北+45～240全预测子组4.725470%，区间略跨零。

湖北来源、18份头及逐样本独立复算见[hubei_threeframe_dydiff_forecast_20261102/RESULTS.md](hubei_threeframe_dydiff_forecast_20261102/RESULTS.md)。PBS217111正常完成，node21四张A100，约6分钟；不训练或微调预测图GHI头，不读取test。
