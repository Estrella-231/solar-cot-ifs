# 湖南 GHI 头中的历史 COT 有用信号

日期：2026-09-23

状态：探索性验证结果，不作为论文最终结论。

## 主要发现

seed42 因果轨迹头 pilot 中，加入**由观测历史 AGRI 反演的 COT**后，`AC_H` 相对无 COT 组 `A_ST` 的验证 GHI RMSE 在两个站点都下降：

| 站点 | `A_ST`（无 COT） | `AC_H`（仅历史 COT） | 差值（`AC_H - A_ST`） |
|---|---:|---:|---:|
| 四里（Sili） | 167.795 | 165.537 | −2.257 W m⁻² |
| 竺家（Zhujia） | 133.782 | 128.683 | −5.100 W m⁻² |

数值来自验证快照 `results/cot_real_agri_20260922/v5_fixed_head_complete_snapshot.json`。服务器源运行目录为 `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/runs/causal_cot_trajectory_solarresnet_v5_b1_seed42_iofix_20260921/`。两组使用相同的逐站 SolarResNet 风格因果轨迹头结构、seed、验证样本、GHI 标签、AGRI/太阳几何输入、优化器和损失；区别是 COT 轨迹输入是否启用。

## 历史 COT 与未来 COT 来源替换是不同问题

四组的比较含义如下：

| 组别 | 历史 COT | 未来预测 COT | 回答的问题 |
|---|---|---|---|
| `A_ST` | 零 | 零 | 无 COT 参考组 |
| `AC_H` | 历史观测 AGRI → 冻结 R | 零 | 历史 COT 是否有用 |
| `AC_F` | 零 | 预测 AGRI → 冻结 R | 未来预测 COT 是否有用 |
| `AC_ST` | 历史观测 AGRI → 冻结 R | 预测 AGRI → 冻结 R | 两段轨迹合用的结果 |

2026-09-23 的 AFNO 配对诊断属于另一类比较：固定已有 GHI 头权重，只替换**未来** COT bank，对比 SimVP-COT 与 AFNO-COT。两站合并的 RMSE 差值为 `AC_F` −0.115 W m⁻²、`AC_ST` −0.087 W m⁻²；两者的初始化日配对 bootstrap 95% 区间都包含 0。该实验没有比较历史 COT 与无 COT，因此不能否定 `AC_H` 的历史 COT 信号。结果位于 `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/runs/fixed_ghi_head_afno_forecast_cot_swap_20260923_v2/complete.json`。

历史 COT 是从已经观测到的 AGRI 反演得到；未来 COT 则经过 AGRI 外推和 COT 反演两步，包含外推误差和反演误差，并可能与 GHI 头已有的 AGRI 预报信息重叠。此次 AFNO 替换还将 AFNO-COT 输入到用 SimVP-COT 训练的 GHI 头中。这些都是可能解释，尚未由当前实验逐项验证。

## 证据边界

这是单 seed、最多 10 epoch 的探索性验证 pilot；各组按同一验证期选最佳 epoch，batch 为 1536，未使用测试集。历史 COT 差值值得后续按预定多 seed 方案复核，但不能称为独立测试集上的确定增益。

历史与未来 COT 同时加入后的 `AC_ST` 结果也有站点差异：相对 `A_ST`，四里约改善 2.431 W m⁻²，竺家约退化 3.640 W m⁻²。因此当前证据支持继续单独检验历史 COT；未来预测 COT 是否能带来额外收益仍未确定。

## 后续确认

保留 `A_ST / AC_H / AC_F / AC_ST` 配对设计，在预先确定的多个 seed 上重训并比较相同站点、日期和 lead 的预测；保持样本键、头部容量、优化预算、标签和掩码一致，报告逐站及合并后的配对不确定性。满足验证协议和 CPP 来源门禁后，再一次性评估独立测试集。
