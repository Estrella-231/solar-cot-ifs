# 未来预测 COT 的单因素收缩试验

日期：2026-10-19。状态：AC50原始runner面向in-sample pilot；独立OOF follow-up runner、按AC−AH预设门禁的fail-closed检查和来源标记已补齐，并通过服务器`swc`环境合成审计、Python编译、PBS语法与校验和核验。该runner仍未提交，仅在当前五折OOF A/AH/AC审计未通过预设AC−AH门后才可启动。此文件只规定工程筛选，不修改论文正文、不读取test/buffer。

## 问题

当前全量 `R(SimVP预测AGRI)` 可能携带不稳定的幅度误差。如果预测 COT 的信息方向有用、但幅度噪声过大，直接输入完整预测场可能抵消信号。本试验只检验一个处理：保留预测 COT 的空间形状与正负异常方向，把未来 COT 的训练集标准化偏差乘以 `0.5`，向训练集均值收缩；历史 COT 保持原样。

这不是先验认定噪声是原因。需要由相同 GHI 标签和日期块配对比较确认。

## 唯一变化与配对臂

- `A`：不输入 COT。
- `AH`：只用初始化时刻之前的历史 COT，未来位置置为标准化均值（零）。
- `AC`：历史 COT 加完整未来预测 COT。
- `AC50`：历史 COT 不变，未来标准化 COT 偏差仅乘 `0.5`。

若 `z` 是 train-only 标准化 COT，历史掩膜是 `m`，则：

```text
COT_AC50 = m * z + 0.5 * (1 - m) * z
```

在四臂内保持 GHI 头、AGRI/太阳几何、样本键、站点、目标值、归一化、初始化、种子、样本顺序、优化器、30 epoch 和 batch=1792 完全一致。当前 pilot 的 `A/AH/AC` 结果另行保留，新的隔离运行重训相同三臂以保证与 `AC50` 处于同一程序和预算中。

**主线后续必须使用OOF版数据：** 若五折OOF筛选的AC−AH门未通过，AC50须从 `future_ghi_oof_head_pack_v1` 读取训练期五折OOF预测COT、验证期固定全train SimVP预测COT，并与同一pack上的OOF A/AH/AC结果配对。不得把此前in-sample训练预测的pilot输入混进该比较。若OOF主门已通过，则不运行AC50，转入日期隔离的独立确认。对应的PBS入口为 `run_future_ghi_oof_cot_shrink.pbs`（含`gate_future_ghi_oof_cot_shrink.py`预注册门禁）；原 `run_future_ghi_cot_shrink.pbs` 仅保留给旧in-sample pilot，不得用于替代OOF follow-up。

## 判定

- 主看 `AC50−AH`：收缩后的未来预测 COT 是否带来增量；负值表示 RMSE 改善。
- 同看 `AC50−AC`：50% 收缩是否优于完整幅度；`AC50−A` 与 `AC50−AH` 不能互相替代。
- 使用两站共同配对键、各 seed 逐样本预测及 BJT 目标日期块 bootstrap；按两站、+15…+240 min 和预设时段分开报告。沿用 `FUTURE_GHI_GATE.md` 的早期时效及分站退化护栏，不用某一个时效或 COT 分层挑模。
- 只有未来 COT 主对比稳定改善且两站无不可接受退化，才将该输入处理送去日期交叉拟合/独立训练期预测输入确认。pilot 仍不是论文主结果。
- 若 `AC50` 不优于 `AH`，下一步不再盲目调收缩系数；先用实际 pilot 的分时效及 COT 分层结果定位错误，再按单一可解释因素提出下一个配对试验。

## 数据与资源门

OOF follow-up复用通过完整性门的 `future_ghi_oof_head_pack_v1` 和同一OOF主比较结果；输入仍仅来自观测历史AGRI/COT与冻结SimVP预测AGRI经冻结R得到的预测COT。真实未来AGRI、未来CPP、test和buffer均不读取。提交前由脚本复核OOF五折provenance、baseline输出与pack manifest SHA一致、主AC−AH门确实未通过及四臂新输出路径；操作者还必须实时查询全账户PBS队列并遵守最多三个未结束作业。不得覆盖当前OOF筛选结果。

实现：`train_future_ghi_heads_cot_shrink.py`、`audit_future_ghi_cot_shrink.py`。训练与审计manifest自动保留 `train_forecast_oof` 来源标记。合成见证：`test_future_ghi_cot_shrink.py`、`test_audit_future_ghi_cot_shrink.py`与`test_gate_future_ghi_oof_cot_shrink.py`。旧in-sample pilot入口：`run_future_ghi_cot_shrink.pbs`；当前主线OOF后续入口：`run_future_ghi_oof_cot_shrink.pbs`。两个入口均未提交；OOF入口仅在五折OOF主筛选完成且预设AC−AH门未通过时可提交。
