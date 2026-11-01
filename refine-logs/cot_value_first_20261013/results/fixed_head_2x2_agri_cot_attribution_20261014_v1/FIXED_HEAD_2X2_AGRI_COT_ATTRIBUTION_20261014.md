# 固定 GHI 头的 AGRI × COT 2×2 离线归因（验证集）

## 比较设计

对 271 个冻结 SimVP 验证窗的两站、16 个 lead，使用同一 seed42 AC GHI 头和完全相同的 7,005 个预测实例；这些实例来自预先建立的CPP支持验证bank窗口，不是完整自然分布验证集。归因过程没有再使用CPP像元掩膜筛行。2×2 因子只改变模型输入来源：

| 代号 | AGRI 输入 | COT 输入 | 含义 |
|---|---|---|---|
| TT | 目标时刻真实 AGRI | 同期真实 AGRI 经冻结 R 反演的 COT | 双 oracle 参考 |
| FT | SimVP 预测 AGRI | 同期真实 AGRI 经 R 得到的 COT | 只替换 AGRI，COT oracle |
| TF | 目标时刻真实 AGRI | SimVP 预测 AGRI 经同一冻结 R 得到的 COT | 只替换 COT，AGRI oracle |
| FF | SimVP 预测 AGRI | 同一预测 AGRI 经冻结 R 得到的预测 COT | 同时替换两种输入 |

这里的“真实 COT”是目标 AGRI 经冻结 R 产生的同期反演值，不是独立 COT 真值；“真实 AGRI”是离线归因用的未来信息。四组都复用同一个已训练 AC 头，没有训练或调参，CPP mask 不参与筛样。

## 结果

| 输入组 | 四里 RMSE | 竺家 RMSE | 两站 RMSE 等权均值 |
|---|---:|---:|---:|
| TT | 71.16 | 79.41 | 75.29 W m⁻² |
| FT | 83.59 | 90.45 | 87.02 W m⁻² |
| TF | 75.89 | 84.96 | 80.43 W m⁻² |
| FF | 99.37 | 94.96 | 97.16 W m⁻² |

日期块配对差值（正值代表 RMSE 变差）：

| 对照 | ΔRMSE | 95% BJT 日期块区间 |
|---|---:|---:|
| 只把 AGRI 换成 SimVP 预测（COT 保持同期 R-COT） | +11.73 | [+2.27, +17.54] |
| 只把 COT 换成预测 COT（AGRI 保持真实） | +5.14 | [−3.98, +11.91] |
| 在预测 AGRI 下再把 COT 换成预测 COT | +10.14 | [−3.18, +18.28] |
| AGRI 与 COT 都换成预测输入（FF−TT） | +21.88 | [+1.09, +34.55] |

区间来自 5,000 次配对 bootstrap，每次整块重采样北京时间目标日期。共同支持只有 **8 个 BJT 日期**；同一天的不同初始化和 lead 窗口重叠，因此 7,005 行不是 7,005 个独立样本。点估计显示，固定头对 AGRI 外推输入的敏感性大于在真实 AGRI 条件下的 COT 替换；联合预测输入的误差也明显更高。COT 影响在真实 AGRI 和预测 AGRI 两个条件下的区间都跨 0。差值来自非线性模型，不能把两项单因素 RMSE 差简单相加当作“贡献分解”。

## 图示与边界

![Frozen AC head 2x2 attribution](fixed_head_2x2_attribution.png)

正值表示误差增大。图中是固定头、8 个 BJT 日期的成块配对区间。最重要的限制是：这个 AC 头训练时看到真实目标 AGRI 和同期 R-COT；因此 FF 只是“把运行时预测输入喂给旧头”的压力测试，存在输入分布偏移。它不是经过预测 AGRI 训练的 GHI 头的成绩，也不能证明预测 COT 一定无法带来业务收益。

因此当前能说的是：在这组数据上，冻结头遇到预测 AGRI 后误差明显扩大；预测 COT 的单因素影响方向偏差，但日期块不确定性仍大。下一步若继续外推 GHI，应先让新头在预测 AGRI/COT 的因果训练输入下学习，并继续满足 T17 的 COT 精度、覆盖、厚云和到达时刻条件；当前没有启动这样的训练，也没有读取 test。

## 可复核文件

- `paired_predictions.csv`：每个实例的真值和四种输入下的固定头输出。
- `inference_receipt.json`：输入银行、head 与 pack 哈希、271 序列和 test 门禁记录。
- `fp64_independent_audit.json`：独立 FP64 重算和 BJT 日期块区间。
- `lead_metrics_fp64.csv`：逐站逐 lead 的四臂 RMSE。
- `audit_fp64.py`、`plot_2x2_attribution.py`：分析和图件脚本。

Figure design followed general scientific visualization guidance: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. arXiv:2609.00065, v2. https://doi.org/10.48550/arXiv.2609.00065.

