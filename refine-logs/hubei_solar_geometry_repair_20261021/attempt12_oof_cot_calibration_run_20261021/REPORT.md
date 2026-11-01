# Attempt12 OOF COT calibration: validation result

**Status:** complete, validation-only diagnostic against CPP retrieval reference. This is not independent COT truth or a test result.

## Question and method

The raw future-COT forecast sometimes has a systematic amplitude offset. A single global affine map was therefore fitted in `log1p(COT)` space on date-held-out training OOF forecasts, using only matching training CPP/mask rows. The fit used 149 BJT target dates and 4,382,924 valid pixel observations; every target date had equal total fitting weight. The frozen map was `z_cal = 0.86956 * z_forecast + 0.17836`, with physical output clipped to `[0,100]`.

The held-out OOF cross-fit reduced date-balanced log1p RMSE from 0.91540 to 0.89863 (−0.01678). Four of five date folds improved; the fifth was slightly worse. This is a training-only cross-validation diagnostic.

## Frozen validation comparison

The map was applied once to the full-train validation COT bank. Raw forecast, calibrated forecast and initialization-time COT persistence were compared on the exact same 1,164 `(init,target,lead)` keys, 16 target dates, and CPP-valid pixels. The main pooled score covers the full 16×16 patch; station-pixel and station-3×3 scores are also saved.

| Full-patch method | COT RMSE | Difference from raw forecast (95% date-block CI) | Difference from persistence (95% date-block CI) |
|---|---:|---:|---:|
| SimVP→R raw forecast | 10.567 | reference | +1.338 [−0.388, +2.941] |
| Train-only calibrated forecast | 10.755 | +0.188 [−3.881, +4.505] | +1.527 [−2.434, +5.465] |
| Historical COT persistence | 9.229 | −1.338 [−2.886, +0.449] | reference |

Positive difference means the left method has larger RMSE. The global calibration did not improve pooled physical-COT RMSE and did not beat persistence. The date-block intervals are broad and include zero.

At +240 min, the point estimates were raw 14.259, calibrated 11.586 and persistence 11.434. Calibration’s difference from raw was −2.673 (95% CI [−5.626,+1.598]); it did not establish a significant gain and remained 0.152 worse than persistence. The short-lead and +240-min responses differ, so a single global amplitude map is not an adequate trajectory correction.

For CPP near-zero pixels (`COT <= 0.5`), the share with forecast `COT > 1` was 11.03% raw and 11.26% calibrated. Thus this calibration slightly increased near-zero false alarms. Station-pixel changes were mixed: Niushou −0.059, Wangzhai −0.243, and Mizhuang +0.280 RMSE; all paired intervals were broad and crossed zero.

## Conclusion

The small cross-fit improvement in log1p space did not transfer to the physical validation RMSE used for the forecast-COT claim. This rules out this one global train-only amplitude calibration as the next improvement. It does **not** show that physical constraints are useless, nor does it justify tuning a more flexible correction against these same validation dates. The experiment used no new training and read no test/buffer payloads.

Historical evidence already includes direct future-COT-versus-persistence scoring, temporal-change loss, spatial-gradient loss, and frozen-R COT auxiliary supervision. Those losses are proxy/label constraints, not explicit cloud-water conservation or radiative-transfer closure. The Hubei attempt13 paired COT-auxiliary experiment is still the relevant active model test; wait for its matched result before defining a new physical evolution model.

## Reproduction

- Plan: `EXPERIMENT_PLAN.md`
- Reproducer: `score_oof_calibration.py`
- Full coefficient, fold cross-fit, source hashes and validation receipt: `attempt1_output/calibration_receipt.json`
- Lead/scope FP64 metrics and paired date-block intervals: `attempt1_output/validation_metrics.csv`
- The one-time local cache was deleted after scoring. Source arrays and original server artifacts were not modified.

The local scoring calculations and paired bootstrap use NumPy float64. Input copying was limited to the existing attempt12 train OOF arrays/reference and attempt12 validation forecast/reference/persistence arrays. The test and buffer payload read counts are zero.
