# Future-COT reliability calibration follow-up (2026-10-20)

## Motivation from completed OOF screens

The three-seed full-forecast-COT screen did not establish a robust GHI gain: AC−AH was −1.014 W m⁻² with a 52-date paired 95% interval [−2.292, +0.227]. The registered global 50% attenuation screen also did not pass: AC50−AH was −1.067 W m⁻², interval [−2.968, +0.563]. Global attenuation changed the pooled point estimate very little and traded performance across station/COT/lead strata.

The prescribed descriptive follow-up found that effects are heterogeneous. For example, at Sili, full COT in near-zero predicted-COT cases improved RMSE over AH by about 3.1–3.4 W m⁻² across lead groups, while thin-COT cases at +15–120 min worsened it by about 0.8 W m⁻². At Zhujia, moderate-COT cases improved by about 1.6–4.4 W m⁻² with full COT, while near-zero-COT cases were slightly worse. These are descriptive strata with overlapping instances, not significance claims or selection gates.

A separate train-only OOF reliability audit exactly matched all 714,914 GHI training rows to observed-training-AGRI→frozen-R COT references. Center-pixel log1p-COT correlations ranged 0.793–0.959 at Sili and 0.856–0.976 at Zhujia; forecast-vs-reference regression slopes ranged 0.765–0.941 and 0.843–0.972. The forecast COT therefore carries substantial signal, while its calibration drifts with station and lead. Fold-excluded cross-fit affine calibration reduced mean log1p-COT RMSE from 0.723 to 0.675 at Sili and from 0.615 to 0.596 at Zhujia. These are training-only calibration diagnostics, not GHI outcomes.

## Single-factor GHI test

Test one train-derived transform of future forecast COT. For each station and future lead, fit a center-pixel affine map in log1p-COT space from date-heldout OOF forecasts to observed-training-AGRI→frozen-R reference COT:

```text
calibrated_log1p_COT = intercept[station, lead] + slope[station, lead] * forecast_log1p_COT
```

Clip slopes to [0, 1], apply the same map to all nine pixels of that future COT frame, then apply the existing train-only COT normalization. Leave historical COT, AGRI, geometry, GHI labels, masks, station sample sets, model, optimizer, epoch/batch budget, and selection rule unchanged. For each training row from OOF fold k, fit its transform on the other four folds; use all five training folds to calibrate validation inputs. No validation COT/AGRI/CPP target is used to fit coefficients.

Fit five paired arms: A (no COT), AH (history COT), AC (full forecast COT), AC50 (registered 50% global attenuation), and ACREL (cross-fit station/lead calibration). Use seeds 42/43/44, 30 epochs, batch 1792, and identical initializations and row orders across arms. Run seed 42 first, then seeds 43 and 44 concurrently on separate GPUs after the structural path passes. Aggregate only after exact arm/station/seed coverage and saved-prediction hashes pass.

The primary contrast is ACREL−AH, with date-block bootstrap over the existing 52 BJT target dates. Also report ACREL−A, ACREL−AC, ACREL−AC50, each station, every lead, and the existing +15–60-minute guard. This is validation screening only. Test, buffer, future CPP, and future observed AGRI as inference inputs remain closed. If the primary contrast or station/short-lead guard fails, do not tune the affine slopes against the same validation set; use the result to decide whether to stop COT input work or move to a separately justified data/label diagnosis.

## Provenance boundary

Observed future AGRI is read only as a training target reference to derive the reliability map through the frozen R model. It is not supplied to the GHI predictor. The calibration is bound to the five-fold OOF pack, the actual-R train pack, fold manifests, sequence-to-fold map, and saved array hashes. Independent test payloads and buffer payloads remain unopened.
