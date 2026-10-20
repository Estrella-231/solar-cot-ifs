# Corrected-UTC COT: paired future GHI validation

Historical 8 AGRI frames -> frozen SimVP future AGRI -> corrected-UTC R COT -> 15-minute GHI. No future observed AGRI/CPP input; no test/buffer payloads.

Same architecture (24489 parameters), same initial weights, sample orders and 30-epoch training budget; seed42/43/44. Reused validation selected best checkpoints; intervals are conditional validation evidence.

| Station | Without COT | With forecast COT | Difference (W/m2) | Relative RMSE reduction |
|---|---:|---:|---:|---:|
| Niushou | 78.276 | 78.858 | +0.582 | -0.74% |
| Wangzhai | 72.369 | 72.999 | +0.630 | -0.87% |
| Mizhuang | 82.824 | 83.541 | +0.717 | -0.87% |

Equal-station/equal-seed RMSE: 77.822996 -> 78.465749 W/m2; relative reduction -0.826%.

Paired target-date 95% difference interval: [-1.6375919841658082, 3.0398066574352662]; consecutive-two-day interval: [-2.129379715022571, 3.2073569759504728].

Full lead results include all +15..+240 min, including exact +60/+120/+180/+240 points. Do not select only favorable stations/leads. This matched-capacity baseline differs from the historical smaller FA head; new-vs-old changes cannot be attributed solely to R repair.
