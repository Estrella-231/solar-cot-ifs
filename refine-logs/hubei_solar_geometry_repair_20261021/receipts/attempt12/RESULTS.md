# Corrected-UTC COT: paired future GHI validation

Historical 8 AGRI frames -> frozen SimVP future AGRI -> corrected-UTC R COT -> 15-minute GHI. No future observed AGRI/CPP input; no test/buffer payloads.

Same architecture (24489 parameters), same initial weights, sample orders and 30-epoch training budget; seed42/43/44. Reused validation selected best checkpoints; intervals are conditional validation evidence.

| Station | Without COT | With forecast COT | Difference (W/m2) | Relative RMSE reduction |
|---|---:|---:|---:|---:|
| Niushou | 88.062 | 87.570 | -0.492 | 0.56% |
| Wangzhai | 96.219 | 96.867 | +0.648 | -0.67% |
| Mizhuang | 102.828 | 102.390 | -0.438 | 0.43% |

Equal-station/equal-seed RMSE: 95.702754 -> 95.608877 W/m2; relative reduction 0.098%.

Paired target-date 95% difference interval: [-2.2712070072629436, 2.2819593700671064]; consecutive-two-day interval: [-2.0068826212911137, 2.1512877005626034].

Full lead results include all +15..+240 min, including exact +60/+120/+180/+240 points. Do not select only favorable stations/leads. This matched-capacity baseline differs from the historical smaller FA head; new-vs-old changes cannot be attributed solely to R repair.
