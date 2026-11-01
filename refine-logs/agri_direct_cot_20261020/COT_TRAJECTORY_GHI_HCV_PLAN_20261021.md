# Predicted-COT trajectory for station GHI: H-only date-block CV

Date: 2026-10-21. Status: complete; all 90 fixed-budget H-only heads trained and scored. The predeclared gain gate failed. Paper body remains unchanged.

## Motivation

The frozen direct-COT predictor has a modest, statistically unresolved improvement over SimVP→R against CPP at the station neighbourhood, yet the GHI head does not reliably improve. The lead-conditioned head only sees the target-lead COT map; it cannot directly represent whether the local cloud is thickening, thinning, or arriving. This experiment tests one new factor: whether a short trajectory of *predicted* COT maps helps the same station GHI mapping.

This is a development experiment inside outer-training H. It does not reopen outer validation V for tuning and does not produce independent-test evidence.

## Fixed candidate and controls

At each target lead L, construct four ordered 3×3 COT maps for L−45, L−30, L−15, and L. If an index is earlier than the forecast initialization, pad it with the latest historical COT map from the model's existing history-only output. The GHI head receives only predictions available from the forecast chain at initialization; no real future AGRI or CPP enters the model.

Train three same-architecture arms with the same AGRI 3×3 map, solar geometry, sinusoidal lead encoding, GHI target, date folds, and optimization budget:

1. **No COT:** the 4-map COT input tensor is the exact train-fold neutral value.
2. **Target-frame COT:** only the final 3×3 map is active; the previous three slots are neutral.
3. **COT trajectory:** all four ordered forecast/history maps are active.

The primary contrast is trajectory minus target-frame COT. Secondary contrasts are trajectory minus no COT and target-frame COT minus no COT. All models start from paired random initializations. Each fold fits every normalizer only on its training dates.

## Data split and training budget

- Use only the existing H training-partition row bank and H-only forecast output. Reuse the already frozen 5-fold calendar two-day assignment; remove target dates within ±1 BJT day of each held-out block from that fold's training set.
- Each H target key is scored once out of fold. Keep both stations separate for fitting and report station-equal regional scores.
- Three seeds (42/43/44), 30 epochs, batch 512, same optimizer/loss and lead encoding as the completed lead-conditioned H-CV. Do not early-stop or select by a held-out fold.
- Run the three arms for both stations and all five folds (90 heads). Require rowbank-to-sequence COT alignment checks to pass before training; incomplete GHI labels do not permit borrowing target rows from neighbouring initialization sequences.

## Scoring and decision rule

Primary metric: paired station-equal GHI RMSE difference, trajectory minus target-frame COT, across all 16 leads. Also report +15–60 min, station, seed, leadwise descriptive metrics, MAE/bias, and exact common-key counts. Bootstrap the fixed calendar two-day blocks 5,000 times. Include leave-one-block-out sensitivity and seed-direction consistency.

Call this H-internal candidate promising only if both all-lead and +15–60 min point estimates are negative, both 95% interval upper bounds are below zero, neither station degrades, at least two of three seed contrasts are negative, and removing any single block does not reverse the regional estimate. These criteria are frozen before scoring. Otherwise retain the result and stop this mechanism; do not tune individual leads, folds, dates, normalizers, or thresholds.

Any passing result would still require genuinely unused future ground GHI dates for confirmation. The accessible Hunan inventory currently only verifies source workbook coverage through 2026-07-01; satellite AGRI extends later, but that alone cannot create a new GHI evaluation set. V remains development data already examined by earlier experiments; test and buffer payloads stay closed.

## Completed result and stop decision

The 90 heads completed on one local RTX 4060 after the remote PBS account remained 3/3 occupied. Primary trajectory−target-frame ΔRMSE was +1.888 W m⁻² (95% two-day-block CI [−0.030,+4.455]) across all leads and +1.543 [−0.787,+4.738] at +15–60 min. All five predeclared conditions failed; all 16 leadwise descriptive deltas were positive. See `COT_TRAJECTORY_GHI_HCV_RESULTS_20261021.md` and `results/direct_cot_trajectory_hcv_v1_local4060/score/trajectory_hcv_summary.png`. Per the frozen stop rule, this trajectory representation is closed for this H-CV contract; do not tune on these folds or relabel V as untouched confirmation.
