# T18 R-AL 隐特征对照：验证结果

日期：2026-10-14（北京时间）。范围：仅限冻结的按月划分验证；使用同样的 4,479 条 `(station,target_time_utc)` 记录，同样的 GHI 真值与掩码，A/AC/AP/AL 使用同样的三个种子。AL 消费冻结的新按月 R 的 `dec1` 特征块（32×3×3），并带一条与 AP 参数量匹配的 32→4 辅助分支。测试载荷仍保持关闭。

## 结果

站点等权 RMSE：先在每个站点内对三个种子取平均，再让两个站点等权。

| 头部 | 四里 RMSE | 竺家 RMSE | 站点等权 RMSE |
|---|---:|---:|---:|
| A | 135.156 | 112.805 | 123.980 |
| AC | 132.368 | 108.113 | 120.240 |
| AP | 135.073 | 112.571 | 123.822 |
| AL | 132.665 | 106.950 | 119.807 |

配对差为候选减对照，负值表示先命名的候选更好。

| 对照 | Δ RMSE (W/m²) | 95% 北京时间日期块区间 |
|---|---:|---:|
| AL − A | −4.173 | [−7.137, −1.390] |
| AL − AC | −0.433 | [−1.476, +0.677] |
| AL − AP | −4.015 | [−6.914, −1.265] |
| AC − A | −3.740 | [−6.074, −1.647] |

在该验证样本上，AL 相对 A 和 AP 降低了误差。AL−AC 尚无定论：区间跨零，因此这不能作为统计等效的证据，但它也没有显示 AC 相对容量匹配的 R 隐表示具有额外优势。分站点看，AL 在四里比 AC 差 0.297 W/m²，在竺家比 AC 好 1.163 W/m²，这提示不宜只看总体数值。

## 解读与边界

在现有验证对比中，COT 输入相对纯 AGRI 头部仍然有用。T18 表明，在相同 CPP 监督下训练的冻结 R 隐特征能提供相近的下游信号。因此当前证据还不能把显式 COT 的物理语义单独确定为 GHI 改善的唯一原因。隐特征既不是独立的云观测，也不构成可部署的替代方案主张；AC 与 AL 继承的是同一个冻结 R 及其监督。这只是验证集上的表示对照，既不是预报 GHI 评分，也不是测试集上的泛化结果。

日期块 bootstrap 使用 2,000 次配对重采样，在每个抽到的北京时间日期内保留全部方法与种子；站点日期支持分别重采样（四里 44 个日期、竺家 50 个日期），然后对站点均值取平均。某一站点缺失的日期不作插补。这是分站点独立重采样的敏感性结果。随后的一次配对方法核对，把本方法与预注册的共享 52 日期并集 bootstrap 都应用到同一批保存的预测上：共享日期的 AC−A 区间为 [−6.180, −1.597] W/m²，与 T14 主分析完全一致；分站点的 AC−A 区间为 [−6.074, −1.647] W/m²，与 T18 原始复算完全一致。点估计和 4,479 条配对记录完全相同。因此 AC−A 的主要结论应采用共享日期区间，T18 区间只保留作敏感性参考；差异来自重采样设计，而不是预测或指标本身的缺陷。参见 [bootstrap 方法核对](ci_method_reconciliation_20261014/T18_BOOTSTRAP_METHOD_RECONCILIATION_20261014.json)。

## 可复现性

- 6 个 AL 拟合在 PBS 作业 `213869.tc6000` 中完成，退出状态 0；30 个 epoch，batch 2,000，种子 42/43/44，单张 A800。训练输出：服务器 `results/r_al_formal_t18_20261014_v1/`。
- 独立审计：[fp64_independent_audit.json](../../experiments/monthly_R_station16_20261013/results/r_al_formal_t18_20261014_v1/fp64_audit_v3/fp64_independent_audit.json)
- 配对保存的输出与键：该审计目录下的 `paired_predictions_fp64.npz`、`paired_val_keys.csv`。
- 本地第二套实现从保存数组复算了所有点估计和 bootstrap 区间，最大绝对偏差为 0.0：[local_independent_recompute.json](../../experiments/monthly_R_station16_20261013/results/r_al_formal_t18_20261014_v1/fp64_audit_v3/local_independent_recompute.json)
- 独立复算代码：[verify_t18_r_al_saved_predictions.py](../../experiments/monthly_R_station16_20261013/verify_t18_r_al_saved_predictions.py)
- 所有预测产物都绑定到 GHI pack 清单 SHA-256 `27b1a10b172c5030194007bc1fdeb2ea6c929c86c304e78242255de9dee54bd4`；审计状态为 `INDEPENDENT_FP64_T18_AL_AUDIT_PASS`，本地复算状态为 `PASS_LOCAL_INDEPENDENT_FP64_RECOMPUTE`。
