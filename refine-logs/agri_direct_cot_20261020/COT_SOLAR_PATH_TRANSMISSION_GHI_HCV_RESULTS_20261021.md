# Solar-path COT representation H-CV result — 2026-10-21

## Question and scope

This 90-head, five-fold, date-block H-only cross-validation asked whether replacing raw target-frame log-COT with a solar-path attenuation feature helps a fixed 3×3 GHI head. It tests GHI use of COT, not COT forecast accuracy, and is internal development evidence rather than independent validation or test evidence.

The three arms used the same AGRI and geometry inputs, keys, five date folds, three seeds, architecture, optimizer, 30-epoch budget, and batch size: no COT, raw target-frame COT, and solar-path attenuation. The predeclared primary contrast is attenuation minus raw COT; positive means worse. Uncertainty uses 112 frozen calendar two-day blocks and 5,000 paired bootstrap resamples.

## Results

| Input representation | All leads RMSE (W m⁻²) | +15–60 min RMSE (W m⁻²) |
|---|---:|---:|
| No COT | 150.054 | 140.838 |
| Raw target-frame COT | 150.934 | 141.257 |
| Solar-path attenuation | 151.892 | 142.607 |

The primary attenuation-minus-raw difference was **+0.958 W m⁻²** (95% CI −0.553 to +2.467) for all leads and **+1.350 W m⁻²** (95% CI −0.210 to +2.928) at +15–60 min. Both point estimates are worse and both intervals cross zero. The preset gain gate failed all five conditions: all-lead and short-lead confidence bounds, both-station non-degradation, at least two of three seeds improving, and leave-one-block stability. Station means also worsened: +0.755 W m⁻² at Sili and +1.160 W m⁻² at Zhujia; only one of three seed-level regional differences was negative.

## Interpretation

The tested solar-path transformation did not help this GHI head use the supplied COT. This does not show that COT forecast maps became less accurate: COT forecasts were held fixed, and only their encoding in the GHI head changed. The experiment gives no basis for claiming a GHI benefit or a general physical failure of COT.

All 90 checkpoints and OOF predictions are recorded in the run manifest. Test/buffer payloads and future observed AGRI/CPP inputs remained closed. The result figure is results/direct_cot_solar_path_hcv_v1_local4060/score/solar_path_hcv_summary.png; machine-readable paired scores are in results/direct_cot_solar_path_hcv_v1_local4060/score/solar_path_hcv_metrics.json.
