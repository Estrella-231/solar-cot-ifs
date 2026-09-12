# 固定 COT 配对结果独立复核（2026-09-08）

独立 NumPy 审计通过：`PASS_INDEPENDENT_SAVED_COT_RESULTS_AUDIT`。全部 635 个原报告浮点指标与独立重算完全相等（最大绝对差 0，预定容差 1e-10）。固定样本的 forecast AGRI 输入对应的 R 检索误差高于 real AGRI 输入；两站各 16/16 个 lead 的 physical COT MAE/RMSE 和 log1p MAE/RMSE 均为 forecast 更高。没有发现配对、重复引用、参考/mask 或指标计算错误。

此结论限于固定 validation、同 CPP 产品参考和原 mask 的描述性诊断。它不构成 CPP 独立物理真值证明，也不证明 GHI 或 A/AC 结果的唯一机制。

## 同配对口径结果

下表两支使用同样的 station/init/target、原参考和原 mask。每个可用 target/lead 的有效像元参与池化；跨 lead 的目标及参考会重复计权。bias 定义为预测 physical COT 减参考 physical COT。不得把下面的 forecast 汇总与后面的 unique-real 汇总直接比较。

| 范围 | 输入 | 配对数 | 有效像元贡献 | COT MAE | COT RMSE | COT bias | log1p MAE | log1p RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 全部 | real AGRI | 872 | 219310 | 2.121013 | 4.293596 | -0.319098 | 0.269468 | 0.463342 |
| 全部 | forecast AGRI | 872 | 219310 | 5.376300 | 10.131402 | 0.469172 | 0.596258 | 0.908917 |
| sili | real AGRI | 431 | 109984 | 2.506761 | 4.659453 | -0.557370 | 0.299193 | 0.483540 |
| sili | forecast AGRI | 431 | 109984 | 7.345446 | 12.608411 | 0.738110 | 0.733816 | 1.021742 |
| zhujia | real AGRI | 441 | 109326 | 1.732943 | 3.890978 | -0.079392 | 0.239563 | 0.442093 |
| zhujia | forecast AGRI | 441 | 109326 | 3.395303 | 6.780822 | 0.198615 | 0.457872 | 0.779098 |

保留 inventory 原样 64 个目标、872 个配对、152 个缺失 lead、6 个完全无 forecast 的目标；未以误差、覆盖或参考值重新筛选。两站可配对唯一目标分别为 sili 28、zhujia 30。

## 全部 32 个站点/lead 结果

所有 lead 均展示，未按误差选择。完整 JSON 另外保存每组的 MAE、bias、log1p MAE/RMSE、n_unique_target 和缺失数。下表只压缩展示 physical COT RMSE；正差表示该组 forecast RMSE 更高，不表示误差随 lead 单调变化。

| 站点 | lead/min | 配对数 | 缺失目标数 | 有效像元贡献 | real RMSE | forecast RMSE | forecast − real |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sili | 15 | 24 | 8 | 6122 | 4.809672 | 9.927290 | 5.117618 |
| sili | 30 | 25 | 7 | 6378 | 4.714252 | 10.837726 | 6.123474 |
| sili | 45 | 26 | 6 | 6634 | 4.709401 | 11.042843 | 6.333442 |
| sili | 60 | 26 | 6 | 6634 | 4.709401 | 11.558396 | 6.848996 |
| sili | 75 | 26 | 6 | 6634 | 4.709401 | 12.044020 | 7.334619 |
| sili | 90 | 26 | 6 | 6634 | 4.709401 | 12.265782 | 7.556382 |
| sili | 105 | 27 | 5 | 6890 | 4.626179 | 12.768477 | 8.142298 |
| sili | 120 | 28 | 4 | 7146 | 4.621486 | 13.452491 | 8.831006 |
| sili | 135 | 28 | 4 | 7146 | 4.621486 | 13.646322 | 9.024836 |
| sili | 150 | 28 | 4 | 7146 | 4.621486 | 13.423198 | 8.801712 |
| sili | 165 | 28 | 4 | 7146 | 4.621486 | 13.469465 | 8.847979 |
| sili | 180 | 28 | 4 | 7146 | 4.621486 | 13.039025 | 8.417539 |
| sili | 195 | 28 | 4 | 7146 | 4.621486 | 13.274287 | 8.652801 |
| sili | 210 | 28 | 4 | 7146 | 4.621486 | 12.686804 | 8.065318 |
| sili | 225 | 28 | 4 | 7146 | 4.621486 | 12.948081 | 8.326595 |
| sili | 240 | 27 | 5 | 6890 | 4.626179 | 13.854967 | 9.228788 |
| zhujia | 15 | 28 | 4 | 6913 | 3.818353 | 3.893498 | 0.075146 |
| zhujia | 30 | 27 | 5 | 6657 | 3.891075 | 4.174725 | 0.283650 |
| zhujia | 45 | 27 | 5 | 6657 | 3.891075 | 4.472253 | 0.581178 |
| zhujia | 60 | 27 | 5 | 6657 | 3.891075 | 5.008708 | 1.117633 |
| zhujia | 75 | 27 | 5 | 6657 | 3.891075 | 5.373414 | 1.482339 |
| zhujia | 90 | 28 | 4 | 6913 | 3.823136 | 5.861492 | 2.038355 |
| zhujia | 105 | 28 | 4 | 6913 | 3.823136 | 6.262577 | 2.439441 |
| zhujia | 120 | 28 | 4 | 6913 | 3.823136 | 6.489765 | 2.666629 |
| zhujia | 135 | 28 | 4 | 6913 | 3.823136 | 6.705837 | 2.882701 |
| zhujia | 150 | 28 | 4 | 6913 | 3.823136 | 6.298103 | 2.474967 |
| zhujia | 165 | 28 | 4 | 6913 | 3.823136 | 6.911704 | 3.088567 |
| zhujia | 180 | 28 | 4 | 6913 | 3.823136 | 7.720173 | 3.897037 |
| zhujia | 195 | 29 | 3 | 7169 | 4.010054 | 8.568694 | 4.558640 |
| zhujia | 210 | 28 | 4 | 6913 | 4.079147 | 8.908959 | 4.829813 |
| zhujia | 225 | 27 | 5 | 6912 | 4.079442 | 9.319119 | 5.239677 |
| zhujia | 240 | 25 | 7 | 6400 | 3.924567 | 8.936543 | 5.011976 |

对 sili 和 zhujia，COT MAE、COT RMSE、log1p MAE、log1p RMSE 各自的 16 lead 差值方向均为：forecast 更高 16、更低 0、相等 0。未将重复 target/lead 当成独立样本做显著性检验。

## 单独的唯一 real 目标摘要

只在本表每个 station/target 使用一次原 mask、参考和 real 预测，合计 58 个目标、14,571 个有效像元。这是不同权重的 real 描述，不能与 872-pair forecast 直接混比。

| 范围 | 唯一目标 | 有效像元 | COT MAE | COT RMSE | COT bias | log1p MAE | log1p RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 全部 | 58 | 14571 | 2.140536 | 4.287924 | -0.215687 | 0.271861 | 0.465500 |
| sili | 28 | 7146 | 2.482232 | 4.621486 | -0.556520 | 0.303351 | 0.493328 |
| zhujia | 30 | 7425 | 1.811680 | 3.940321 | 0.112340 | 0.241554 | 0.437049 |

## 审计范围和实证

- 检查脚本独立实现 NumPy float64 池化 sum/count 计算，不导入诊断 runner 或 torch，不加载模型或重跑预测。全部原报告的 32 station×lead、58 unique-real-target、2 unique-real-station 和 all-pairs/unique-all 指标均重算并逐字段比较；635 个浮点值最大差为 0。计数、键、字符串字段严格相等。
- 原 NPZ 全部 872 个 station/init/target/lead/R 行/sequence/shard 键逐项与冻结 inventory 完全一致，所有 `(station,init,target)` 唯一；每个 target 的 real 预测、参考和 mask 在不同 lead 间逐元素相同。
- 重新核验原 R pack 的完整 target.npy/mask.npy SHA，仅通过只读 memmap 提取 58 个固定 validation 行。严格按 `float32(target * float32(100))` 后转 float64 重建参考，全部 saved reference 和 mask 逐元素相等，原有效像元数也一致。selected target/mask 数值读取合计 74,240 字节。
- 重新核验 inventory、冻结协议、推理脚本、R 权重、R 模型源文件、两份坐标/几何门禁证据、R rows、norm、原 target/mask 和结果 NPZ，共 12 个源文件，哈希流共 37,888,874 字节。权重只作字节 SHA，没有反序列化或模型调用。
- 审计实际耗时 0.98 秒，CPU affinity `[0,1]`，OpenMP/OpenBLAS/MKL/NumExpr 均设为 2 线程，CUDA_VISIBLE_DEVICES 为空。无 GPU、训练、GHI 或 test 载荷读取；原 R pack 只有 train/validation 行。
- 本轮未重新读取 116 个原始/sidecar NPZ；它们的坐标与 SHA 绑定继承已核验哈希的前置坐标报告，并通过 selected R rows 的原路径/原 SHA 键校验。本次不升级为对 CPP source NC/生成 checkpoint 的新证明。
- 本审计验证已保存数组、来源收据和报告指标，未独立重跑 R 前向，因此不另行宣称完成新的 model-to-prediction 数值绑定见证。

## UTC 显示检查

原始 NPZ 的 872 个 target_time_utc 和 init_time_utc，以及原 report.metrics.unique_real_targets 的 58 个 target_time_utc，全部使用显式 +00:00，且与原 inventory 和实际时差合同一致。不存在原结果 UTC 字段写入 +08:00 或平移 8 小时的问题。

已在本地复现一个查看层转换：PowerShell `Get-Content -Raw | ConvertFrom-Json` 默认将 UTC ISO 字符串解析成 `System.DateTime`，在本机 +08 时区再用 `ConvertTo-Json` 会改以本地 offset 显示。例如 zhujia / R row 16815 原文件为 `2025-08-03T00:45:00+00:00`，经过该查看管线后显示为 `2025-08-03T08:45:00+08:00`。二者为同一瞬间，且原文件未被写回。原文件中的 +08 字符串位于明确命名的 target_time_bjt/init_time_bjt 字段。

查看原样字符串使用 Python json，或 PowerShell `ConvertFrom-Json -DateKind String`。若在新展示中同时提供本地时间，应显式分列 UTC 与 BJT；无需修原数据、NPZ 或报告。

## 产物和复核命令

服务器别名为 `zjnu-hpc`；工程绝对根目录为：

```text
/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907
```

成功诊断位于 `audits/cot_pair_diag_20260908_v2`；其中预测文件仍按冻结文件名 `cot_pair_predictions_20260908_v1.npz` 保存，不代表读取了失败尝试。原失败尝试目录保留；本审计不修改诊断结果或冻结协议。

| 产物（相对服务器工程根目录） | SHA-256 |
| --- | --- |
| `scripts/audit_cot_pair_results.py` | `1a19e0bbb05065f6924ef221b3589314f1f9c22d06929f3c7aa4b57efaebcf4c` |
| `audits/cot_pair_results_audit_20260908.json` | `d0693805cfa6bc5e95d90e13fb09a1c21d80ab4bb76be6b994c7e500c02f9d76` |
| `audits/cot_pair_diag_20260908_v2/report.json` | `8fa50c03f3077e29c59a77dfe93e1e85ba6b572971c9dccff83b0c12001cc928` |
| `audits/cot_pair_diag_20260908_v2/cot_pair_predictions_20260908_v1.npz` | `eaa99223119786658acea876f9ae482a6fc7834fc5f3ef685f027caf2250a420` |

在服务器复核时必须使用尚不存在的新 output 名称：

```bash
ROOT=/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907
env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= \
  /public/home/slfu/miniconda3/envs/swc/bin/python "$ROOT/scripts/audit_cot_pair_results.py" \
  --root "$ROOT" \
  --diagnostic-dir "$ROOT/audits/cot_pair_diag_20260908_v2" \
  --output "$ROOT/audits/cot_pair_results_audit_20260908_recheck.json"
```

未发现需要更换输入、扩网络、调 loss 或开展超参数搜索的实现错误。现有结论支持记录有限样本中 forecast 输入检索误差更高；R 使用该 validation 选模、CPP 产品参考身份边界、旧 GPU cache 数值 warning 等范围限制保持不变。
