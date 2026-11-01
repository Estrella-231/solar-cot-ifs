# T16 event cases used in the manuscript

Source: frozen validation audit `t16_case_figure_audit.json`. Case keys were selected before GHI values/prediction errors were inspected. The audit reports the mean of the three seed-specific within-case RMSEs over the 16 canonical future targets (+15 to +240 min). Negative AC−A favours AGRI + COT. These four temporally adjacent validation sequences are descriptive cases, not independent samples and have no event-level confidence intervals.

| Station | Init (BJT) | Selection | D (COT / 15 min) | AGRI RMSE | AGRI + COT RMSE | Δ (AC−A) |
|---|---|---|---:|---:|---:|---:|
| sili | 2025-07-09 13:15:00 | reference_cloud_change | 7.278 | 213.851 | 218.408 | +4.557 |
| sili | 2025-07-08 13:15:00 | deterministic_hash | 3.887 | 151.268 | 150.397 | -0.871 |
| zhujia | 2025-07-09 14:30:00 | reference_cloud_change | 2.103 | 129.933 | 138.871 | +8.938 |
| zhujia | 2025-07-09 13:45:00 | deterministic_hash | 1.963 | 126.700 | 136.617 | +9.917 |

The COT-augmented head has lower descriptive RMSE in one case and higher RMSE in three. This does not change the paired cohort result or establish an event-level causal mechanism. All four examples are from JJA because the available CPP archive does not provide complete 17-frame sequences for the other seasons. The figure shows CPP reference COT and R-retrieved COT as separate fields; its GHI curves use observed AGRI and are not forecast results.

Figure source and rendering audit: `figures/t16_partial_cases_20261013/`; T16 case audit confirms `test_payload_opened=false`.
