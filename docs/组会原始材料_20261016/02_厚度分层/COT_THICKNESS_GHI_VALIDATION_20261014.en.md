# GHI retrieval gain by COT thickness (saved validation predictions)

Date: 2026-10-14 BJT. Same-time observed AGRI; no forecast result; no training, model execution, or test payload.

Thresholds 0.1, 5 and 30 follow the pre-existing monthly-R reference-bin audit. Classification uses the station-centre pixel, not a full-patch mean or percentile. R bins are based on expm1 of the unstandardized saved log1p COT once; CPP bins use physical reference COT directly.

Each value averages station-wise RMSE equally across the two Hunan stations and the three matched seeds. Relative reduction is (RMSE_A - RMSE_AC) / RMSE_A. Shared BJT-date bootstrap (2,000 draws, seed 20261013, 52-date full-cohort universe) preserves pairing; the overall primary point estimate and CI are reproduced to <1e-10. Stratum intervals are descriptive pointwise intervals conditional on validation checkpoint selection.

## all_validation

| COT bin | Sili / Zhujia rows | BJT dates | A RMSE | AC RMSE | AP RMSE | AC−A [95% CI] W/m² | RMSE reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 2128 / 2351 | 52 | 123.980 | 120.240 | 123.822 | -3.740 [-6.180, -1.597] | +3.02% |

## CPP_reference_centre

| COT bin | Sili / Zhujia rows | BJT dates | A RMSE | AC RMSE | AP RMSE | AC−A [95% CI] W/m² | RMSE reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| supported | 610 / 674 | 46 | 138.369 | 135.540 | 138.183 | -2.828 [-6.130, -0.260] | +2.04% |
| nearzero_[0,0.1) | 143 / 321 | 31 | 159.165 | 156.812 | 158.964 | -2.353 [-6.238, +1.157] | +1.48% |
| thin_[0.1,5) | 71 / 110 | 29 | 165.869 | 163.045 | 165.816 | -2.823 [-6.814, +1.268] | +1.70% |
| moderate_[5,30) | 316 / 150 | 36 | 122.509 | 120.430 | 122.288 | -2.079 [-7.179, +3.030] | +1.70% |
| thick_[30,100] | 80 / 93 | 25 | 74.929 | 58.105 | 74.687 | -16.824 [-21.862, -12.359] | +22.45% |

## R_retrieved_centre

| COT bin | Sili / Zhujia rows | BJT dates | A RMSE | AC RMSE | AP RMSE | AC−A [95% CI] W/m² | RMSE reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| supported | 2128 / 2351 | 52 | 123.980 | 120.240 | 123.822 | -3.740 [-6.180, -1.597] | +3.02% |
| nearzero_[0,0.1) | 356 / 734 | 37 | 171.543 | 170.980 | 171.429 | -0.564 [-5.068, +2.755] | +0.33% |
| thin_[0.1,5) | 258 / 353 | 41 | 164.395 | 162.751 | 164.299 | -1.644 [-4.997, +1.437] | +1.00% |
| moderate_[5,30) | 1048 / 767 | 51 | 103.665 | 97.350 | 103.403 | -6.316 [-9.526, -3.033] | +6.09% |
| thick_[30,100] | 466 / 497 | 42 | 61.734 | 46.721 | 61.514 | -15.014 [-18.976, -10.937] | +24.32% |

## Main findings / 主要发现

总体同期GHI估计：A 123.980 → AC 120.240 W/m²，降低3.740 W/m²（3.02%）。

按CPP参考的站点中心COT分层，厚云层（30≤COT≤100）A 74.929 → AC 58.105 W/m²，降低16.824 W/m²（22.45%）；AC−A的日期块95%区间为[−21.862,−12.359] W/m²。173条样本覆盖25个BJT日期，四里80条/15日、竺家93条/17日。三个种子的两站等权差值均为负。

厚云收益有站点差异：四里三种子平均A/AC为69.870/64.661 W/m²；竺家为79.987/51.549 W/m²。厚云子集内的平均GHI高估同时减弱：四里bias由+14.460降到+1.648，竺家由+48.356降到+19.432 W/m²。这支持描述“厚云参考条件下COT增强尤其有帮助”，但不能把误差变化唯一归因为COT物理语义。

CPP近零、薄云、中等COT层分别约降低1.48%、1.70%、1.70%，但各自配对区间跨0。各层并非都已确定有收益，也未证明随厚度单调增加。

全体4,479行按R检索中心COT分组的补充分解，近零/薄/中等/厚层分别降低0.33%/1.00%/6.09%/24.32%；后两层区间低于0。该分组由模型检索值定义，会受到R误差与样本跨层影响，应和CPP参考层表分开呈现。

厚云22.45%是CPP支持子集内的GHI RMSE相对改善，不是总体改善、COT自身反演准确率或未来GHI预报改善。各层RMSE的基准不同，不能把分层百分比直接相加解释总体3.02%。

## Interpretation limits

- CPP strata describe a sparse keyed subset of the full GHI validation cohort. CPP is a co-produced retrieval reference, not independent truth. Missing reference pixels are excluded only from this diagnostic; the full 4,479-row primary cohort is unchanged.
- R strata classify the model input itself and cover the full cohort; they must be called retrieved-COT groups rather than true-cloud-thickness regimes. Retrieval error can move samples between groups.
- Near-zero COT is not identical to independently verified clear sky. A single centre pixel cannot represent the entire cloud field or sun-path cloud obstruction.
- The intervals are not adjusted for multiple strata. Different gains across bins do not establish a statistically significant interaction or a monotonic physical law.
- All saved predictions and labels retain their original station/time order. CPP sample_indices were keyed to the CLP archive, then joined to GHI by (station,target_time_utc); no row-position join was used across products.
- No hypothesis of forecast gain or extrapolator-only attribution is established by these same-time strata.

Outputs: thickness_metrics.json (all seeds, station support and errors, hashes, CIs), thickness_summary.csv, summarize_thickness.py.
