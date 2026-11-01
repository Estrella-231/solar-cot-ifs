# 空间 COT B0：输入侧车与 GPU smoke

状态：通过实现与资源验收；不等于空间 COT 的预报收益已验证。

## 完成的内容

服务器工程为 /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907。pack_head_spatial_v1.py 从冻结的 train/validation forecast cache 创建独立 sidecar：

- 输出：data/head_spatial_cot_trainval_20260917_v2/cot_map.npy，形状 [637902,1,16,16]。
- 键：base head pack 的 (split, cache_index, station_index, lead_index)。缓存内夜间等无标签元组被跳过；每一条已有标签必须且只会填充一次。
- 来源：1332个 shard 的receipt和SHA-256逐个核验；S/R保持冻结；test_used=false。
- COT：直接使用缓存已有的log1p(COT)，未重复取对数。仅在训练集全部137741568个像元拟合一个共享标量均值2.038626076027692和标准差1.527321132764502，保留像元位置和厚度幅度。

SpatialHead 将13通道AGRI与该1通道空间场在第一层卷积前拼接，结构为14→16→32卷积、GAP和原有读出。COT统计槽仍存在，供B1的统计量对照使用；空间主组将其置零。

## PBS 证据

| 作业 | 作用 | 状态 |
|---|---|---|
| 209380.tc6000 | 首次B0 | 预期外停止：把无标签缓存元组当作键不一致；无训练、无smoke |
| 209381.tc6000 | 修复后侧车构建 | 完成；生成v2 sidecar |
| 209384.tc6000 | v2独立GPU smoke | 完成，PBS exit_status=0 |

209384使用node21的一张NVIDIA A100-SXM4-40GB，2048样本完成前向、反向与一次AdamW step。峰值显存0.1828575134GiB，占39.5861206055GiB的0.00461923；参数量14,425；有限损失0.56082958。该损失只为数值/梯度见证，不是训练或验证指标。

## 没有完成的事项与下一关

本次没有启动B1 pilot、配对三seed、容量/同监督/排列对照、IFS或test。CPP候选V2代码与现有2025历史NC的写出合同仍不一致，见CPP_PRODUCER_CODE_REVIEW_20260917.md。因此B0只能证明空间COT输入从当前冻结缓存到新头的实现和资源合同，不能解除历史CPP生产链门禁或支持物理/性能结论。

待该门禁由历史生产代码/日志或覆盖时次的重算像元对比闭合后，按../refine-logs/EXPERIMENT_PLAN.md执行B1：A_sp、AC_stat_sp、AC_map的seed42短pilot，随后才决定是否启动42/43/44完整训练。
