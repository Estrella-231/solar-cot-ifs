# Corrected-UTC COT: paired future GHI validation

Historical 8 AGRI frames -> frozen SimVP future AGRI -> corrected-UTC R COT -> 15-minute GHI. No future observed AGRI/CPP input; no test/buffer payloads.

Same architecture (24489 parameters), same initial weights, sample orders and 30-epoch training budget; seed42/43/44. Reused validation selected best checkpoints; intervals are conditional validation evidence.

| Station | Without COT | With forecast COT | Difference (W/m2) | Relative RMSE reduction |
|---|---:|---:|---:|---:|
| Niushou | 93.294 | 94.556 | +1.261 | -1.35% |
| Wangzhai | 102.835 | 105.221 | +2.386 | -2.32% |
| Mizhuang | 111.692 | 113.106 | +1.414 | -1.27% |

Equal-station/equal-seed RMSE: 102.606987 -> 104.294007 W/m2; relative reduction -1.644%.

Paired target-date 95% difference interval: [-0.5428162909593982, 3.990968336074935]; consecutive-two-day interval: [-0.3397844275474505, 4.089400231894688].

Full lead results include all +15..+240 min, including exact +60/+120/+180/+240 points. Do not select only favorable stations/leads. This matched-capacity baseline differs from the historical smaller FA head; new-vs-old changes cannot be attributed solely to R repair.
