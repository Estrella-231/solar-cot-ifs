# 全月度验证窗固定 AC 头输入归因

## 设计与复核

在同一固定 seed42 AC 头、同一 15 min GHI 标签键上，将真实目标 AGRI/冻结 R-COT 与 SimVP 预测 AGRI/其经同一冻结 R 反演的预测 COT 作 2×2 离线替换。T37 在 PBS 213865 的一张 A800 上完成 66,590 条预测记录，覆盖 2,817 个验证序列、52 个 BJT 目标日期块。服务器和本地 `paired_predictions.csv` 的 SHA-256 均为 `462f3ab7b23f82ca0d581b40b5becb62e94454830eb653a61489684b03e20129`。本地独立 FP64 复算通过，输出 `fp64_independent_audit.json` 与逐 lead 表；test 未读取。

四组含义：TT=真实目标 AGRI+同期 R-COT；FT=预测 AGRI+同期 R-COT；TF=真实目标 AGRI+预测 AGRI→R-COT；FF=预测 AGRI+预测 COT。TT 使用未来真实 AGRI，是离线 oracle 参照。AC 头训练于真实 AGRI 和同期 R-COT；因此 FT/TF/FF 是固定头上的输入分布/误差传播诊断，尤其 FF 不能作为可部署 GHI 预测分数。

## 全验证窗结果

两站各自 RMSE 后等权：

| 输入组合 | GHI RMSE (W m⁻²) |
|---|---:|
| TT：真实 AGRI + 同期 R-COT | 118.14 |
| FT：预测 AGRI + 同期 R-COT | 126.46 |
| TF：真实 AGRI + 预测 COT | 125.42 |
| FF：预测 AGRI + 预测 COT | 139.39 |

| 配对替换（新−基准） | ΔRMSE (W m⁻²) | BJT 日期块 bootstrap 95% CI |
|---|---:|---:|
| AGRI：FT−TT（同期 COT） | +8.32 | [+3.01, +14.14] |
| COT：TF−TT（真实 AGRI） | +7.28 | [+4.52, +10.54] |
| COT：FF−FT（预测 AGRI） | +12.93 | [+9.73, +16.58] |
| AGRI：FF−TF（预测 COT） | +13.97 | [+7.84, +20.69] |
| 联合：FF−TT | +21.26 | [+14.22, +29.14] |

置信区间以 52 个目标 BJT 日期为块，5,000 次配对重采样；每次两站、四组使用相同日期权重。窗口和 15 min lead 重叠，日期块处理了同日聚类，但并不令所有窗口独立。站点单独结果、样本数和 lead 指标见 JSON/CSV。

## 解释

在这个冻结头上，预测 AGRI 和预测 COT 替换均使 GHI RMSE 增大；预测 COT 的误差惩罚在预测 AGRI 输入下更大（+12.93），而不是更小。该结果支持“外推误差可通过 R-COT 支路传到既有 GHI 头”的判断，不能单独证明 COT 物理属性没有价值：AC 头未接受预测输入训练，且 TT/TF/FT 包含不同真值时刻的 AGRI/COT，带有明显信息和分布优势。要判断模型能否利用预测 COT，仍需预测输入条件下公平训练并与无 COT 对照；这由既有 T17/B4 门槛决定，当前证据不足以越过该训练门。

## 复核文件

- `paired_predictions.csv`：逐样本四组输出及共同真值。
- `inference_receipt.json`：来源、checkpoint/代码哈希及输入定义。
- `audit_fp64_allval.py`：本地独立复算脚本。
- `fp64_independent_audit.json`：核心汇总及bootstrap区间。
- `lead_metrics_fp64.csv`：逐站逐lead RMSE。
