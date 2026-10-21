# Train-only station-center R-input shift

**Status: complete; descriptive training-input diagnostic only.** This checks whether the previously observed month-dependent difference between full-R and date-held-out R outputs is also present at the three Hubei station pixels used by the patch experiment.

## Data and calculation

The comparison uses the existing full-R train/validation bank and the five-fold date-held-out R predictions. Only the 20,206 training-frame keys were indexed from either COT prediction array. The validation rows in the metadata were used only to verify the frozen split/key inventory; their prediction values were not read. Test, buffer, CPP target/QC, AGRI, and GHI-label payloads were not read.

For each training frame and station center, the calculation keeps daytime records where both outputs are finite and in `[0,100]`. The center pixels are Niushou `(8,6)`, Wangzhai `(9,8)`, and Mizhuang `(7,10)` in the cached 16×16 patch. The common support contains 29,786 paired station-frame values from 220 training dates. Means and RMSE below are computed in physical COT units on that same paired support. Months are BJT calendar months. Frames overlap in time, so these are descriptive summaries; no independent-sample confidence interval is claimed.

## Result

Across the three centers, the full-R train inputs have mean COT `13.221`; date-held-out R train inputs have mean `11.225`. The paired held-out-minus-full mean is `−1.996`, and the paired input-disagreement RMSE is `6.735`.

| Station | Paired train values | Full-R mean COT | Date-held-out-R mean COT | Held-out minus full | Paired input-disagreement RMSE |
|---|---:|---:|---:|---:|---:|
| Mizhuang | 9,927 | 12.934 | 10.989 | −1.945 | 6.794 |
| Niushou | 9,932 | 13.306 | 11.220 | −2.086 | 6.756 |
| Wangzhai | 9,927 | 13.422 | 11.465 | −1.957 | 6.654 |
| **All three centers** | **29,786** | **13.221** | **11.225** | **−1.996** | **6.735** |

| BJT month | Paired values | Full-R mean | Date-held-out-R mean | Held-out minus full | Paired disagreement RMSE |
|---|---:|---:|---:|---:|---:|
| 202508 | 3,527 | 7.137 | 5.632 | −1.505 | 5.608 |
| 202509 | 3,212 | 20.663 | 16.658 | −4.004 | 8.754 |
| 202510 | 3,032 | 21.609 | 17.130 | −4.478 | 9.775 |
| 202511 | 2,633 | 5.156 | 4.495 | −0.661 | 4.444 |
| 202512 | 2,705 | 11.387 | 9.702 | −1.684 | 7.532 |
| 202601 | 2,802 | 13.502 | 11.570 | −1.932 | 6.049 |
| 202602 | 2,451 | 11.453 | 11.849 | +0.396 | 5.974 |
| 202603 | 3,311 | 15.972 | 14.443 | −1.530 | 6.569 |
| 202604 | 3,490 | 10.253 | 8.853 | −1.400 | 4.822 |
| 202605 | 2,623 | 14.407 | 11.734 | −2.672 | 5.688 |

The monthly summary reproduces the station-level CSV by pooling squared errors and means with each station-month's paired-record count as its weight. September and October show the largest downward input shift; February is a small reversal. All three station centers show a similar overall mean difference.

## Interpretation and boundary

This confirms that the train-only source-origin shift is present at the station pixels, not only in the full spatial field. It is consistent with a potential train/deployment feature mismatch if a future-COT model is trained on full-R in-sample histories but receives date-held-out/validation R histories at inference. The size and seasonal variation make that mismatch worth addressing in a later, properly sourced experiment.

It does **not** show which R output is physically more accurate, does not measure future-COT forecast skill, and does not establish that this mismatch caused the earlier validation result. It is not evidence of a GHI gain. The existing G1S forecast-COT gate remains failed and regional G1 remains blocked on complete full-resolution CPP input provenance; this audit does not unlock more tuning or training.

## Reproduction

The row-level monthly/station output is `train_only_station_center_shift.csv`; the key-checked extraction is `analyze_train_only_station_center_shift.py`. Reproduce on the source host with the existing full-R bank and fold OOF directory:

```text
python analyze_train_only_station_center_shift.py \
  --full-bank <cot_bank_correctutc_attempt3> \
  --oof-root <oof_history_inference_attempt3/results/oof_train> \
  --out train_only_station_center_shift.csv
```

The source paths used for this run were:

```text
Full-R bank:
/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cloud_probability_20261020/cot_bank_correctutc_attempt3/

Five-fold date-held-out R train outputs:
/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cot_station_patch_forecast_20261022/oof_history_inference_attempt3/results/oof_train/

Station-center metadata:
/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cot_station_patch_forecast_20261022/cache_attempt5/cache/cache_report.json
```

The audit uses memory-mapped arrays and indexes only training rows; it does not launch a GPU job. The output CSV SHA-256 is `d769601ad8c9271457adfd294b379f03339f16f9b94f36132b8c8d61d6c0a43a` (verified against the remote copy).
