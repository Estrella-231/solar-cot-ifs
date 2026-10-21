# Attempt12：全量验证窗 forecast-COT 提前诊断（CPU FP32 预览）

时间：2026-10-21 07:20 CST。该诊断使用已完成的 full-train SimVP 验证预测 bank 和 corrected-R full checkpoint；407 个验证窗、R 验证键264个、精确匹配的 `(init_time,target_time,lead)` 组合1,164个。它没有读取 train/test/buffer 标签或未来真实 AGRI。CPP 是卫星反演参考，不是独立真值；`offline observed future AGRI -> R` 只作不可部署的反演参照。

预览用单线程 CPU FP32 按正式 R 的16通道 train-only 标准化和日间有效掩膜生成 COT，再用正式 forecast-COT 评价器按目标时刻键和 CPP common-valid mask 打分。结果覆盖16个时效；所有时效合并后是297,604个有效16×16像元记录。该 CPU FP32 预览用于提前诊断；正式生产仍由 attempt12 的 GPU FP16 COT bank 与评价器给出，最终数值以正式产物为准。

| 验证链路/参照 | 全时效像元加权 COT RMSE（无量纲） | MAE（无量纲） | Bias（预测−CPP，无量纲） |
|---|---:|---:|---:|
| SimVP预测AGRI → corrected R | 10.564 | 6.443 | +0.901 |
| 历史COT保持不变（persistence） | 9.229 | 5.599 | +0.312 |
| 离线真实未来AGRI → corrected R（不可部署） | 8.684 | 5.039 | +1.042 |

站点正上方像元、跨全部时效的像元加权 RMSE：

| 站点 | forecast AGRI→R | 历史COT persistence | 离线真实未来AGRI→R |
|---|---:|---:|---:|
| 牛首 | 12.118 | 9.512 | 9.543 |
| 王寨 | 11.903 | 9.547 | 9.486 |
| 米庄 | 10.659 | 9.090 | 8.295 |

在这份复用验证集预览中，forecast AGRI→R 的全时效 RMSE 比历史持平高1.335（约14.5%），站点像元三个站都更差。逐时效结果并非单调：+60 min forecast略优于 persistence，但日期块配对95%区间跨零；+15/+30/+45 min 和 +240 min 的 all-patch COT RMSE 差分别为 +2.64 `[+0.16,+4.40]`、+2.86 `[+0.09,+4.90]`、+2.28 `[+0.01,+3.71]`、+2.82 `[+0.34,+5.79]`（COT无量纲）。其余时效应以CSV/JSON中的逐时效估计和区间为准。

这说明当前全量SimVP预测图经corrected-R反演后的COT，在验证预览中仍未超过“历史COT不变”；它提示云图外推误差或预测图到R的输入域偏移仍明显。该结果不能证明哪一个环节单独造成误差，也不是独立测试或GHI收益结论。下一步要等OOF训练bank完成，先取得正式GPU FP16结果，再看同一批样本上forecast AGRI→R、persistence和离线未来AGRI→R的差异，并完成有无预测COT的配对GHI评估。

可复核产物：`FORECAST_COT_PREVIEW_CPU_attempt12.json/.csv`、`COT_BANK_PREVIEW_CPU_attempt12.json`、`COT_BANK_PREVIEW_CPU_attempt12.npy`、`PREVIEW_CPU_ATTEMPT12.log`。预览bank SHA-256：`1d737b56d29528d130b88314536a8ec252e5ca7e564f4a8021400ee08ba399ba`。所有载荷读取计数为 test=0、buffer=0。
