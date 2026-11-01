# Attempt12 validation COT texture and temporal-dynamics audit

## Scope and interpretation
This is a validation-only diagnostic of attempt12 corrected SimVP→frozen R. All target comparisons use exact BJT timestamps and the same CPP-valid target mask. CPP is a satellite retrieval reference, not independent truth. The observed-future AGRI→R curve is an offline proxy, not a forecast input. The result does not use test or buffer payloads.

The spatial-texture ratios compare forecast-map spatial standard deviation and neighbor-gradient RMS against the offline observed-future AGRI→R proxy on the same target support. The temporal-change comparison is forecast COT minus R(history AGRI) versus R(observed future AGRI) minus the same R(history AGRI); persistence has zero increment. Thus these are attribution diagnostics for forecast smoothness/change, not standalone truth scores.

Matched init-target-lead pairs: 1164. Validation sequences: 407. R validation frames: 264. Test payloads read: 0. Buffer payloads read: 0.

## Exact-lead results at +1/+2/+3/+4 h

| Exact target lead | Forecast RMSE vs CPP | Persistence RMSE vs CPP | Forecast/offline spatial std | Forecast/offline gradient RMS | Forecast Δ RMSE vs offline Δ | Persistence Δ RMSE vs offline Δ | Forecast/offline Δ RMS | Through-origin Δ correlation |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| +60 min | 11.055 | 11.505 | 0.677 | 0.776 | 5.371 | 4.043 | 1.258 | 0.325 |
| +120 min | 12.100 | 11.562 | 0.595 | 0.680 | 6.492 | 6.095 | 1.037 | 0.454 |
| +180 min | 11.903 | 11.642 | 0.519 | 0.680 | 5.707 | 6.844 | 1.084 | 0.682 |
| +240 min | 14.259 | 11.434 | 0.530 | 0.750 | 7.375 | 7.136 | 1.398 | 0.675 |

## Reconciliation with the earlier 1/2/3/4-hour table
The established attempt12 headline values (10.255/8.946, 9.787/9.238, 10.194/9.617, 12.063/9.086) are **one-hour-wide lead bins**, not exact point leads at +60/+120/+180/+240. Re-aggregating the current hash-verified per-lead RMSEs over +15..+60, +75..+120, +135..+180 and +195..+240 reproduces those values; see `horizon_bin_rmse.csv`. The exact +60/+120/+180/+240 values in the table above answer a different question. So the records are consistent, but the phrase “1 h” is easy to misread unless the lead window is shown.

## Reading the result
Across exact +2/+3/+4 h leads, forecast COT is worse than the history-persistence map against CPP; at exact +1 h it is slightly better (11.055 vs 11.505). In the one-hour-wide bins, all four bins are worse than persistence, matching the existing attempt12 headline. Forecast spatial standard deviation is about 0.52–0.68 of the offline future-AGRI→R proxy at the 1–4 h checkpoints, and neighbor-gradient RMS is about 0.68–0.78, consistent with smoother spatial fields. But the forecast temporal-increment RMS is 1.04–1.40 of the offline proxy, so changes are not uniformly too weak. Through-origin increment correlation is only about 0.33 at +1 h and 0.45–0.68 at +2..+4 h: the pattern/timing of change is an issue alongside texture, not simply insufficient amplitude. These are retrieval-proxy diagnostics; they do not establish that a physical-constraint loss will improve COT or GHI. Attempt13’s matched auxiliary-loss comparison remains the controlled training test before choosing a subsequent single-factor motion/evolution experiment.

## Reproducibility
Run `python analyze_texture_dynamics_attempt12.py` from this folder. Source hashes and counts are embedded in `texture_dynamics_summary.json`; the downloaded input files are limited to validation bank/keys/CPP and validation R prediction/mask files.
