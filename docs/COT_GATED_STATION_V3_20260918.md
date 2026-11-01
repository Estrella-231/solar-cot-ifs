# COT-gated station-aware v3 — implementation record

## Contract

`cot_gated_station_v3` replaces raw 14-channel concatenation with pointwise physical conditioning. The frozen normalized `log1p(COT)` at pixel `(r,c)` is mapped by a learned 1×1 layer to `gamma(r,c), beta(r,c)`, then modulates the aligned AGRI feature at the same position:

`z(r,c) = SiLU(a(r,c) * (1 + 0.1*tanh(gamma(r,c))) + beta(r,c))`.

`a` is the two-layer AGRI encoder output. The head then aggregates the exact station `[8,8]` token, its 5×5 neighbourhood, and the full 16×16 patch. `A_gate` and `AC_gate` have the same parameters; `A_gate` supplies a zero COT map and `AC_gate` supplies the frozen predicted COT map. Neither group reads test data, true future CPP, or IFS.

**2026-09-18 修订（计划变更，代码未改）**：按用户决定，读出中原有的四个 COT 统计量拼接槽（区域均值、空间标准差、p90、中心像元）随全局平均类特征一并移除；B0 重跑前须修改 `scripts/cot_gated_station_head_v3.py` 删除该拼接并重做 CPU witness，三尺度读出与门控公式不变。

## Implementation witness and execution status

Remote CPU witness passed: output shape `(4,)`, finite forward/backward values, and nonzero `cot_gate` gradient L1 `0.03721`. This validates the implementation only; GPU B0 and the validation pilot have not run.

PBS B0 submission `209559.tc6000` was stopped by the scheduler before Python launch: `Gold is abnormal`, zero resource time. This is a scheduler accounting fault, not a model result. Do not claim a GPU acceptance, performance change, or COT effect until a later successful B0 and paired seed-42 run.
