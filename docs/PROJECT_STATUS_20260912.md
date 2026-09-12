# 工程与服务器状态（2026-09-12）

## 范围和结论边界

本篇固定湖南 SimVP 外推，检验 COT 表示及其未来可能与 IFS 的互补。所有已报告模型均只使用 train/validation；test 未读取、未选模、未评价。当前没有可写入论文的 COT 增益结论，也没有 COT+IFS 结果。

最强的实测结论是：当前 AC 的四维 `log1p(COT)` 直接拼接头没有通过多种子稳定性门。它不是“COT 无效”的结论，只否定当前的表示、损失和站点读出配置。

## 服务器工程位置

主服务器工程根：

```text
/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907
```

关键可复查产物：

| 内容 | 服务器位置 | 状态 |
|---|---|---|
| 冻结 SimVP/COT train-val cache | `data/frozen_forecast_trainval_20260907_v1` | 已构建；只用于训练/验证 |
| GHI 标签映射 | `data/forecast_label_join_20260907_v1` | 已建立；BJT、canonical lead 和目标窗口合同已审计 |
| 训练/验证 head pack | `data/head_pack_trainval_20260908_v1` | 637,902 行：train 538,053、validation 99,849 |
| original 三种子 A/AC | `runs/head_ghi_{original, multiseed_original_seed43, multiseed_original_seed44}_20260910_v1` | 已完成、验证集限定 |
| QC1 三种子 A/AC | `runs/head_ghi_{qc1, multiseed_qc1_seed43, multiseed_qc1_seed44}_20260910_v1` | 已完成、验证集限定 |
| 天气分类头 v2 | `runs/weather_coverage_head_seed42_20260912_v2` | 已完成，单 seed 诊断 |
| 固定 AC COT 反事实 | `runs/ac_cot_counterfactual_20260912_v1` | 已完成，只前向、未训练 |

历史 P200 建议根 `/home/Data_Pool_3/wangyc/irradiance_paper/solar_cot_ifs_20260905` 仅是早期部署建议，不能替代上述实际运行根。

## 已完成的服务器使用

| PBS | 资源 | 工作 | 可用结论 |
|---|---|---|---|
| 207615 | 1 A100 | 小头 profile 与 8 序列 A/AC overfit | 数值训练链正常，不是泛化证据 |
| 207616 | 1 A100 | original/QC1 初始 A/AC pilot | 初始四维 COT 拼接未显示总体收益 |
| 207626 | 1 A100 | R 后端数值见证 | GPU/CPU 主要差异受后端设置影响；旧 cache 仍有一分片数值 warning |
| 208020 | 1 A100 | GHI 目标函数 A/AC 对照 | 单 seed 初步正向，不可作为稳定结论 |
| 208030 | 1 A100 | 43/44 新种子与独立验收 | 多种子方向不一致，COT 稳定性门未过 |
| 208402 | 1 A100 | 天气分类头 v2 | 仅提供分层诊断，未改变 A/AC 主结果 |
| 208492 | 1 A100 | 固定 AC 的 COT 反事实前向 | 定位到当前 AC 的特征响应，不等同重新训练 A |

执行规则：每次 qsub 前查询账户全部未结束 PBS；运行、排队、挂起等保守合计最多 3 个。每个任务先尽量使用一张卡的吞吐与显存，随后才扩卡；不得取消其他项目任务腾配额。历史任务已完成不代表当前队列为空，后续提交必须重新查询。

本次仓库更新前，`zjnu-hpc` 上执行 `qstat -u slfu` 未返回任务行，说明该次查询时账户没有可见 PBS 任务。它只是 2026-09-12 的瞬时快照，不可替代后续每一次 qsub 前的实时检查。

## 当前结果与最大问题

`original` 三种子的 AC−A 两站等权 validation GHI RMSE 分别为 −1.010、−1.081、+1.091 W/m²，均值 −0.333 W/m²；`QC1` 分别为 −0.507、+2.703、+0.741 W/m²，均值 +0.979 W/m²。负数表示 AC 改善。两个训练口径都没有满足“新增 seed 方向一致”的预注册门。

最大、已被直接测量的问题是 **COT 到站点 GHI 的当前读出机制不稳**。四里预测晴天的 QC1 分支中，AC 在三个 seed 都进一步降低原本已偏低的 GHI，RMSE 增加 13.92、14.17、6.29 W/m²；退化贯穿全部 16 个 lead、COT 分位箱和大多数行，并非少数异常、晨昏小晴空辐照度、天气分类混淆或已发现的站点像元错误。

固定 AC 反事实进一步表明，在当前 checkpoint 内，COT 均值和 p90 主要驱动四里整体向下修正，中心像元则提供部分有益抵消。四个统计量同时置零的效果又因非线性抵消而不稳定。因此不能简单删除某一个特征后宣布修复；下一步需要独立冻结、参数匹配的空间表示或残差调制对照，而不是继续调当前四维拼接头。

## 未闭合门禁

1. **IFS 因果合同**：候选 HRES 文件有 `ssrd` 与长步长线索，但未取得每个样本的可靠 `release_time <= init_time`、完整同 cycle step、时间坐标和累计量转换证据。AN/ACN 不得运行。
2. **CPP 生产溯源**：下游 12 行空间链和修正 sidecar 已核对，但实际 Result/CPP writer、checkpoint、归一化、输入 AGRI 时间及源产品地理注册仍未绑定。不能将现有源 CPP 当作已独立认证的物理真值。
3. **表示机制**：四维全局统计压缩了 COT 场，无法表达站点上游、云边和空间演变；这是待检验的研究假设，不是已证明的唯一失败原因。

这些门禁决定下一步：先取得 IFS 和 CPP 的可复查来源合同；与此同时，若继续 COT 研究，另建一个单因素、参数匹配的空间 COT 表示/残差调制实验，保留当前 A/AC 全部负结果，继续不打开 test。

## 本地与远程仓库

本地工程：`F:/CODE_Classify/辐照度预报/solar-cot-ifs`。
远程仓库：`https://github.com/Estrella-231/solar-cot-ifs.git` 的 `main` 分支。本仓库仅包含代码、协议、审计摘要和轻量复现材料；原始缓存、checkpoint、逐样本预测、大型审计数组与PBS日志保留在服务器或本地审计目录，不上传。

## 证据入口

- `docs/HEAD_GHI_MULTISEED_RUN_20260910.md`：三种子 A/AC 主对照。
- `docs/WEATHER_HEAD_RESULTS_20260912.md`：预测天气分组结果。
- `docs/STATION_WEATHER_COT_DIAGNOSIS_20260912.md`：四里站点特异退化排除检查。
- `docs/AC_COT_COUNTERFACTUAL_RESULTS_20260912.md`：固定 AC 的四维输入干预。
- `docs/SOURCE_FOLLOWUP_20260910.md`：CPP/IFS 来源和因果性门禁。
