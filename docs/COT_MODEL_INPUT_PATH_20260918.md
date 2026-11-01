# COT 输入线路：静态候选与因果时空主候选

> 2026-09-18 更新：下文 `AC_gate` 是已实现的**静态单时刻**探索候选，不能回答
> 云厚度历史和时空轨迹的问题。当前主候选已改为
> `causal_cot_trajectory_v4`，其严格因果合同和消融见
> `docs/COT_CAUSAL_TRAJECTORY_DIRECTION_20260918.md`。静态候选保留为对照，
> 不再以全 patch 汇集或单图 readout 作为主方法。

## 当前主候选线路

```text
8 帧历史 AGRI -> 冻结 R -> 8 帧历史 COT
8 帧历史 AGRI -> 冻结 SimVP -> 16 帧预测 AGRI -> 冻结 R -> 16 帧预测 COT
历史 COT + 截至目标 lead 的预测 COT 前缀
    -> 时间因果、空间卷积的轨迹编码器
    -> 站点 [8,8] 特征读出 -> kt -> GHI
```

网络不读取目标 lead 之后的预测 COT 图；不读取未来真实 AGRI/CPP/GHI；不做
全图平均、全 patch pooling 或 COT 四统计量 readout。

## 历史静态候选说明

本说明对应当前候选 `AC_gate`，实现见
`scripts/cot_gated_station_head_v3.py`。它是一个尚未完成 GPU B0/B1
验收的探索性候选，以下只描述代码线路，不报告性能结论。

## 1. 推理时的可部署数据链

```text
8 帧历史 AGRI
    -> 冻结 SimVP
    -> 各未来时效的预测 AGRI patch [13, 16, 16]
    -> 冻结 COT 反演器 R
    -> 预测 COT patch [1, 16, 16]
    -> COT-gated station head
    -> 站点 kt
    -> 该 15 min 窗口的 clear-sky GHI 回算为 GHI
```

这里的 COT 是由预测 AGRI 反演得到的**整张 16×16 图**。它不是未来真实
CPP，也不是站点 GHI 输入。COT 与 AGRI 的每个格点严格对齐；两站在各自
裁剪 patch 内的站点对应像元均为 `(8, 8)`。

## 2. COT 如何进入 GHI 头

该候选没有把 COT 简单拼接成第 14 个 AGRI 通道，而是保留两条分支：

1. AGRI 编码器：`13 -> 16 -> 32` 的两个 3×3 卷积，得到每个格点的
   32 维特征 `a(r,c)`。
2. COT 条件器：`1 -> 64` 的 1×1 卷积，同一格点生成 `gamma(r,c)` 与
   `beta(r,c)`；它不会让其他位置的 COT 去直接调制本站点位置的 AGRI 特征。
3. 像元融合：

   ```text
   f(r,c) = SiLU(a(r,c) * (1 + 0.1 * tanh(gamma(r,c))) + beta(r,c))
   ```

4. 站点读出：拼接 `(8,8)` 站点像元特征、站点周围 5×5 均值、整张 16×16
   均值，以及现有四个 COT 统计量、太阳几何、lead embedding 与 station
   embedding，再以 MLP 输出 kt。

   **2026-09-18 修订**：按用户决定，上述"现有四个 COT 统计量"槽位随全局
   平均类特征从计划移除，代码修改前以本条注记为准；其余读出结构不变。

因此，完整 COT 图影响每个位置的云特征；最终 GHI 头又显式保留了站点像元、
站点邻域与背景三个尺度。它没有引入 attention、多尺度分支、SimVP 改造或
diffusion。

## 3. 可视化资产

`figures/cot_pipeline_20260918/` 的三张图均为固定合成 16×16 场，目的只是
准确解释代码合同：

- `fig_cot_pipeline_overview_20260918`：完整可部署推理线路；
- `fig_cot_patch_alignment_20260918`：AGRI/COT 同格点与站点邻域；
- `fig_cot_pixel_gate_20260918`：同像元门控公式与三尺度读出。

它们不能作为卫星个例、COT 质量或 GHI 精度的证据。真实个例图需要在 Gold
恢复后，从冻结 cache 中按统一 init/target/lead 抽取，并同时保存其样本键、
预测 AGRI、COT 和 GHI 真值。
