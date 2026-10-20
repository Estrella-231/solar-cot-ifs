# 湖北三站云物理输入四臂GHI结果（保存预测独立复算）

范围：湖北牛首、王寨、米庄；共同验证集；单帧站点3×3同期GHI重建；三种子；总体支持不由CPP覆盖筛选。
CPP站点中心COT/CLP仅用于分层诊断，是卫星检索参考，不是独立真值。训练集和验证集以外载荷未读取。

## Overall validation metrics

| Arm | Equal-station × seed RMSE (W/m²) | Across-cell SD | MAE | Bias | RMSE gain vs A |
|---|---:|---:|---:|---:|---:|
| A | 77.285 | 6.409 | 56.616 | +24.799 | +0.00% |
| AC | 69.142 | 8.455 | 46.723 | +13.236 | +10.54% |
| ACM | 68.815 | 8.953 | 46.221 | +12.092 | +10.96% |
| AM | 74.053 | 8.214 | 52.017 | +18.584 | +4.18% |

RMSE汇总先逐站、逐seed计算RMSE，再对三站和三seed等权平均；括号中的SD是9个站点×seed单元的描述性标准差，不是抽样标准误。保存的行级预测为float64，指标由本脚本重新计算。

## Paired two-day date-block bootstrap

正的gain表示候选组RMSE低于参照组。区间基于10个连续两日验证日期块、20,000次配对重抽样。

| Stratum | Candidate − reference | RMSE delta (W/m²), 95% CI | Relative gain, 95% CI | Blocks |
|---|---|---:|---:|---:|
| all_validation | AC − A | -8.143 [-14.999, -3.361] | +10.54% [+4.40, +18.91] | 10 |
| all_validation | AM − A | -3.232 [-4.811, -1.903] | +4.18% [+2.40, +6.71] | 10 |
| all_validation | ACM − A | -8.470 [-14.443, -4.266] | +10.96% [+5.38, +18.66] | 10 |
| all_validation | ACM − AC | -0.328 [-1.654, +0.834] | +0.47% [-1.23, +2.62] | 10 |
| all_validation | ACM − AM | -5.238 [-10.457, -1.671] | +7.07% [+2.22, +13.99] | 10 |
| CPP_COT_[0,0.1) | AC − A | -3.559 [-8.246, +1.164] | +4.01% [-1.40, +14.12] | 6 |
| CPP_COT_[0,0.1) | AM − A | -1.414 [-5.967, -0.293] | +1.59% [+0.30, +9.37] | 6 |
| CPP_COT_[0,0.1) | ACM − A | -3.800 [-8.550, -0.637] | +4.28% [+0.78, +14.08] | 6 |
| CPP_COT_[0,0.1) | ACM − AC | -0.241 [-2.562, +0.833] | +0.28% [-1.10, +3.27] | 6 |
| CPP_COT_[0,0.1) | ACM − AM | -2.386 [-4.736, +2.273] | +2.73% [-2.76, +7.81] | 6 |
| CPP_COT_[0.1,5) | AC − A | -10.353 [-13.271, +2.201] | +8.89% [-3.22, +10.11] | 3 |
| CPP_COT_[0.1,5) | AM − A | -1.141 [-9.428, -0.710] | +0.98% [+0.54, +33.83] | 3 |
| CPP_COT_[0.1,5) | ACM − A | -6.116 [-7.700, -0.921] | +5.25% [+1.35, +10.48] | 3 |
| CPP_COT_[0.1,5) | ACM − AC | +4.237 [-3.122, +5.572] | -3.99% [-4.72, +6.59] | 3 |
| CPP_COT_[0.1,5) | ACM − AM | -4.975 [-6.990, +6.506] | +4.31% [-35.28, +5.36] | 3 |
| CPP_COT_[5,30) | AC − A | -2.305 [-10.554, +0.314] | +2.97% [-0.41, +18.29] | 5 |
| CPP_COT_[5,30) | AM − A | -2.783 [-11.834, -0.468] | +3.59% [+0.61, +20.51] | 5 |
| CPP_COT_[5,30) | ACM − A | -3.444 [-14.868, -0.624] | +4.44% [+0.81, +25.77] | 5 |
| CPP_COT_[5,30) | ACM − AC | -1.138 [-4.384, -0.647] | +1.51% [+0.80, +9.15] | 5 |
| CPP_COT_[5,30) | ACM − AM | -0.661 [-3.027, +0.143] | +0.88% [-0.19, +6.62] | 5 |
| CPP_COT_[30,100] | AC − A | -22.611 [-42.497, -11.283] | +39.57% [+28.04, +50.05] | 4 |
| CPP_COT_[30,100] | AM − A | -2.508 [-7.721, +0.247] | +4.39% [-0.61, +12.85] | 4 |
| CPP_COT_[30,100] | ACM − A | -19.516 [-35.981, -10.075] | +34.16% [+25.08, +43.08] | 4 |
| CPP_COT_[30,100] | ACM − AC | +3.095 [+0.935, +6.973] | -8.97% [-14.32, -3.22] | 4 |
| CPP_COT_[30,100] | ACM − AM | -17.008 [-29.313, -10.306] | +31.13% [+25.51, +37.03] | 4 |
| CPP_CLP_clear | AC − A | -3.665 [-8.750, +1.000] | +4.71% [-1.04, +14.17] | 5 |
| CPP_CLP_clear | AM − A | -1.529 [-5.684, -0.072] | +1.96% [+0.09, +9.40] | 5 |
| CPP_CLP_clear | ACM − A | -3.842 [-8.562, -0.615] | +4.93% [+0.72, +13.97] | 5 |
| CPP_CLP_clear | ACM − AC | -0.177 [-2.582, +1.146] | +0.24% [-1.96, +2.95] | 5 |
| CPP_CLP_clear | ACM − AM | -2.313 [-5.590, +2.205] | +3.03% [-2.84, +9.62] | 5 |
| CPP_CLP_water | AC − A | -7.126 [-24.605, -0.282] | +7.44% [+0.30, +35.17] | 6 |
| CPP_CLP_water | AM − A | -2.277 [-7.455, -0.241] | +2.38% [+0.29, +11.39] | 6 |
| CPP_CLP_water | ACM − A | -6.464 [-23.109, -0.590] | +6.75% [+0.81, +32.75] | 6 |
| CPP_CLP_water | ACM − AC | +0.662 [-0.664, +1.891] | -0.75% [-4.25, +1.06] | 6 |
| CPP_CLP_water | ACM − AM | -4.187 [-15.509, +0.016] | +4.48% [-0.02, +24.75] | 6 |
| CPP_CLP_ice | AC − A | -13.424 [-14.652, +5.368] | +32.67% [-30.71, +37.01] | 4 |
| CPP_CLP_ice | AM − A | -0.880 [-1.110, +4.138] | +2.14% [-23.07, +2.85] | 4 |
| CPP_CLP_ice | ACM − A | -12.449 [-14.149, +11.835] | +30.30% [-65.97, +35.83] | 4 |
| CPP_CLP_ice | ACM − AC | +0.974 [+0.379, +6.467] | -3.52% [-27.75, -0.99] | 4 |
| CPP_CLP_ice | ACM − AM | -11.570 [-13.051, +7.697] | +28.77% [-34.86, +33.95] | 4 |

## CPP COT/CLP strata

| Stratum | Arm | N unique station-time | Blocks | Equal-station × seed RMSE (W/m²) | MAE | Bias |
|---|---|---:|---:|---:|---:|---:|
| all_validation | A | 2195 | 10 | 77.285 | 56.616 | +24.799 |
| all_validation | AC | 2195 | 10 | 69.142 | 46.723 | +13.236 |
| all_validation | AM | 2195 | 10 | 74.053 | 52.017 | +18.584 |
| all_validation | ACM | 2195 | 10 | 68.815 | 46.221 | +12.092 |
| CPP_COT_[0,0.1) | A | 268 | 6 | 88.704 | 64.497 | +6.218 |
| CPP_COT_[0,0.1) | AC | 268 | 6 | 85.145 | 58.057 | +14.284 |
| CPP_COT_[0,0.1) | AM | 268 | 6 | 87.289 | 60.830 | +6.250 |
| CPP_COT_[0,0.1) | ACM | 268 | 6 | 84.904 | 57.445 | +11.978 |
| CPP_COT_[0.1,5) | A | 71 | 3 | 116.483 | 79.516 | +7.271 |
| CPP_COT_[0.1,5) | AC | 71 | 3 | 106.130 | 67.119 | -0.909 |
| CPP_COT_[0.1,5) | AM | 71 | 3 | 115.342 | 79.407 | +12.562 |
| CPP_COT_[0.1,5) | ACM | 71 | 3 | 110.367 | 73.597 | +3.502 |
| CPP_COT_[5,30) | A | 211 | 5 | 77.587 | 57.288 | +14.490 |
| CPP_COT_[5,30) | AC | 211 | 5 | 75.281 | 50.054 | -11.023 |
| CPP_COT_[5,30) | AM | 211 | 5 | 74.803 | 53.294 | +6.068 |
| CPP_COT_[5,30) | ACM | 211 | 5 | 74.143 | 49.718 | -9.457 |
| CPP_COT_[30,100] | A | 173 | 4 | 57.139 | 43.196 | +36.097 |
| CPP_COT_[30,100] | AC | 173 | 4 | 34.527 | 25.401 | +6.093 |
| CPP_COT_[30,100] | AM | 173 | 4 | 54.631 | 41.034 | +29.618 |
| CPP_COT_[30,100] | ACM | 173 | 4 | 37.623 | 27.297 | +9.067 |
| CPP_CLP_clear | A | 233 | 5 | 77.895 | 56.493 | +9.088 |
| CPP_CLP_clear | AC | 233 | 5 | 74.230 | 50.209 | +14.179 |
| CPP_CLP_clear | AM | 233 | 5 | 76.366 | 52.595 | +7.398 |
| CPP_CLP_clear | ACM | 233 | 5 | 74.054 | 49.534 | +11.776 |
| CPP_CLP_water | A | 371 | 6 | 95.795 | 69.181 | +16.858 |
| CPP_CLP_water | AC | 371 | 6 | 88.669 | 57.920 | -3.245 |
| CPP_CLP_water | AM | 371 | 6 | 93.518 | 65.939 | +12.217 |
| CPP_CLP_water | ACM | 371 | 6 | 89.331 | 59.457 | -0.976 |
| CPP_CLP_ice | A | 119 | 4 | 41.090 | 31.494 | +25.108 |
| CPP_CLP_ice | AC | 119 | 4 | 27.666 | 19.178 | +2.688 |
| CPP_CLP_ice | AM | 119 | 4 | 40.210 | 30.617 | +22.171 |
| CPP_CLP_ice | ACM | 119 | 4 | 28.641 | 20.132 | +4.757 |

## CPP reference coverage by validation month

Counts are unique station-time records. Months with no valid CPP center pixel remain in the table as zero-coverage months; they remain included in the all-validation GHI score.

| BJT month | GHI val station-time N | Unique target times | CPP COT center valid N | CPP CLP center valid N | Val days | Two-day blocks |
|---|---:|---:|---:|---:|---:|---:|
| 202508 | 244 | 104 | 227 | 227 | 2 | 1 |
| 202509 | 238 | 95 | 214 | 214 | 2 | 1 |
| 202510 | 215 | 91 | 44 | 44 | 2 | 1 |
| 202511 | 205 | 86 | 50 | 50 | 2 | 1 |
| 202512 | 193 | 80 | 49 | 49 | 2 | 1 |
| 202601 | 189 | 80 | 42 | 42 | 2 | 1 |
| 202602 | 179 | 90 | 44 | 44 | 2 | 1 |
| 202603 | 221 | 87 | 53 | 53 | 2 | 1 |
| 202604 | 254 | 103 | 0 | 0 | 2 | 1 |
| 202605 | 257 | 106 | 0 | 0 | 2 | 1 |


## Station-specific CPP COT strata

| CPP station-center COT stratum | Station | Arm | N/seed | RMSE (W/m²), mean over seeds | Seed SD |
|---|---|---|---:|---:|---:|
| CPP_COT_[0,0.1) | Niushou | A | 92 | 78.793 | 0.316 |
| CPP_COT_[0,0.1) | Wangzhai | A | 81 | 72.543 | 0.377 |
| CPP_COT_[0,0.1) | Mizhuang | A | 95 | 114.775 | 3.186 |
| CPP_COT_[0,0.1) | Niushou | AC | 92 | 64.100 | 0.825 |
| CPP_COT_[0,0.1) | Wangzhai | AC | 81 | 72.387 | 1.145 |
| CPP_COT_[0,0.1) | Mizhuang | AC | 95 | 118.948 | 2.373 |
| CPP_COT_[0,0.1) | Niushou | AM | 92 | 75.091 | 0.345 |
| CPP_COT_[0,0.1) | Wangzhai | AM | 81 | 68.209 | 0.921 |
| CPP_COT_[0,0.1) | Mizhuang | AM | 95 | 118.569 | 2.129 |
| CPP_COT_[0,0.1) | Niushou | ACM | 92 | 65.668 | 2.770 |
| CPP_COT_[0,0.1) | Wangzhai | ACM | 81 | 70.151 | 1.080 |
| CPP_COT_[0,0.1) | Mizhuang | ACM | 95 | 118.892 | 2.637 |
| CPP_COT_[0.1,5) | Niushou | A | 27 | 109.849 | 0.934 |
| CPP_COT_[0.1,5) | Wangzhai | A | 26 | 120.248 | 0.727 |
| CPP_COT_[0.1,5) | Mizhuang | A | 18 | 119.353 | 4.513 |
| CPP_COT_[0.1,5) | Niushou | AC | 27 | 84.518 | 0.066 |
| CPP_COT_[0.1,5) | Wangzhai | AC | 26 | 118.208 | 1.268 |
| CPP_COT_[0.1,5) | Mizhuang | AC | 18 | 115.665 | 2.923 |
| CPP_COT_[0.1,5) | Niushou | AM | 27 | 104.104 | 2.192 |
| CPP_COT_[0.1,5) | Wangzhai | AM | 26 | 114.636 | 1.351 |
| CPP_COT_[0.1,5) | Mizhuang | AM | 18 | 127.287 | 4.671 |
| CPP_COT_[0.1,5) | Niushou | ACM | 27 | 92.981 | 10.725 |
| CPP_COT_[0.1,5) | Wangzhai | ACM | 26 | 116.972 | 3.862 |
| CPP_COT_[0.1,5) | Mizhuang | ACM | 18 | 121.149 | 8.562 |
| CPP_COT_[5,30) | Niushou | A | 77 | 86.949 | 3.318 |
| CPP_COT_[5,30) | Wangzhai | A | 68 | 70.328 | 1.798 |
| CPP_COT_[5,30) | Mizhuang | A | 66 | 75.482 | 4.610 |
| CPP_COT_[5,30) | Niushou | AC | 77 | 81.043 | 4.560 |
| CPP_COT_[5,30) | Wangzhai | AC | 68 | 70.273 | 3.072 |
| CPP_COT_[5,30) | Mizhuang | AC | 66 | 74.528 | 5.286 |
| CPP_COT_[5,30) | Niushou | AM | 77 | 80.856 | 2.648 |
| CPP_COT_[5,30) | Wangzhai | AM | 68 | 68.890 | 4.426 |
| CPP_COT_[5,30) | Mizhuang | AM | 66 | 74.664 | 4.488 |
| CPP_COT_[5,30) | Niushou | ACM | 77 | 79.012 | 2.378 |
| CPP_COT_[5,30) | Wangzhai | ACM | 68 | 69.040 | 4.671 |
| CPP_COT_[5,30) | Mizhuang | ACM | 66 | 74.376 | 6.238 |
| CPP_COT_[30,100] | Niushou | A | 61 | 61.056 | 2.389 |
| CPP_COT_[30,100] | Wangzhai | A | 52 | 57.922 | 3.494 |
| CPP_COT_[30,100] | Mizhuang | A | 60 | 52.438 | 3.505 |
| CPP_COT_[30,100] | Niushou | AC | 61 | 32.436 | 0.445 |
| CPP_COT_[30,100] | Wangzhai | AC | 52 | 34.206 | 0.737 |
| CPP_COT_[30,100] | Mizhuang | AC | 60 | 36.939 | 2.587 |
| CPP_COT_[30,100] | Niushou | AM | 61 | 58.905 | 2.977 |
| CPP_COT_[30,100] | Wangzhai | AM | 52 | 54.576 | 4.393 |
| CPP_COT_[30,100] | Mizhuang | AM | 60 | 50.411 | 2.539 |
| CPP_COT_[30,100] | Niushou | ACM | 61 | 34.668 | 1.927 |
| CPP_COT_[30,100] | Wangzhai | ACM | 52 | 39.166 | 2.962 |
| CPP_COT_[30,100] | Mizhuang | ACM | 60 | 39.034 | 2.537 |
## Interpretation boundary

This experiment measures same-time GHI reconstruction from the input AGRI frame. It does not measure future GHI forecasting, and it does not establish that an auxiliary field is accurate in an independent sense. Per-COT strata are restricted to the CPP reference-valid station-center support; the all-validation result retains every common GHI validation row.

The saved checkpoint for each station/arm/seed was selected using this same validation set. The date-block interval is conditional on those selected checkpoints; it is not an independent-test interval and does not include model-selection uncertainty.

Test and buffer payloads read: 0.
