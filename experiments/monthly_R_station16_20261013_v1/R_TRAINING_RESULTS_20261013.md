# New-split AGRI-to-COT R: training and validation record

## Status and scope

`monthly_station16_COTUNet_seed42_v1` completed on PBS job `213712.tc6000`
using the independent monthly split
`hunan_monthly_block_2d_buffer1d_seed42_v1`. It was initialized from random
weights and trained on 16×16 patches centered on Sili and Zhujia. The 16 input
channels are the frozen 13 AGRI channels plus `cosSOZ`, `cosRAA`, and
`day_mask`; the output is CPP COT. Validation measures agreement with the CPP
retrieval reference on supported daytime pixels, not error against independent
cloud-thickness truth.

The archived model used the first complete train/validation pack: 15,414 train
examples and 1,398 validation examples. The first pack pass recorded 13 train
CPP files with intermittent NetCDF/HDF read errors (26 station rows were
excluded rather than labeled clear sky). A read-only retry later read all 13
files successfully. Reconstructing those rows yielded 14 additional supervised
station examples; the other 12 station rows had no valid daytime pixels. The
recovered pack passed a second integrity audit with 15,428 train and 1,398
validation examples, no unresolved source errors, and `test_payloads_read=0`.
The existing checkpoint was **not** retrained on that recovered pack; its
provenance remains tied to the first pack. The recovered pack changes only 14
of 15,414 training examples (0.09%) and does not change validation support.
Keep this distinction when citing or regenerating downstream COT products.

## Validation results

The selected run stopped at epoch 28 (50-epoch maximum), used batch size 1,284
on one NVIDIA A800-SXM4-80GB, and completed with validation masked Huber loss
0.03485 on `log1p(COT)`. Independent FP64 scoring of saved validation
predictions produced:

| Support | Pixels | COT RMSE | COT MAE | Mean bias |
|---|---:|---:|---:|---:|
| All valid validation pixels | 356,169 | 9.074 | 4.646 | +0.279 |
| Near-zero reference COT `[0,0.1)` | 128,247 | 10.555 | 3.577 | +3.576 |
| Thin reference COT `[0.1,5)` | 50,154 | 1.658 | 1.187 | −0.383 |
| Moderate reference COT `[5,30)` | 126,844 | 6.111 | 4.375 | +0.801 |
| Thick reference COT `[30,100]` | 50,924 | 14.129 | 11.421 | −8.673 |

The dominant errors are conditional bias: predictions average about 3.58 COT
too high for near-zero pixels and 8.67 too low for the thick-cloud pixels.
Thin and moderate regimes are closer to the CPP reference. This makes the
model a valid new-split R baseline, but the aggregate RMSE alone hides the
clear-sky false-positive and thick-cloud underestimation problem. These COT
metrics do not establish an improvement in GHI forecasting.

Station breakdown:

| Station | Valid pixels | RMSE | MAE | Bias | Thick-bin bias |
|---|---:|---:|---:|---:|---:|
| Sili | 178,019 | 9.674 | 5.508 | +0.751 | −7.150 |
| Zhujia | 178,150 | 8.433 | 3.785 | −0.192 | −10.288 |

FP64 rescoring matched the run's pooled COT RMSE, MAE, and valid-pixel count
exactly. No test payload was opened. Do not generalize these metrics beyond
CPP-supported daytime pixels.

## Provenance

- Split manifest SHA-256: `b85c2841e33cc92eda848acb183f926d49564b6c131d6418730e6b53978e2807`
- CPP inventory SHA-256: `0a303ec6965be360689f3891e96b5e4085ad3e6550b2db6504a8b2ac35565dd2`
- First-pack status SHA-256 used by the checkpoint: `1b15f3e22a778f6b4d9c1546cc2d8381f14faae6c95b1350a937731efad2928b`
- Checkpoint SHA-256: `342596094d1a41aa8b141972a7b4adadece179b4b9f58170f4e7f409a894f001`
- Saved validation-prediction SHA-256: `08c9cc92df549f12412873e9b8e50f308f35f79451693fa10b229a3346c15785`
- New recovered train/validation pack audit: `PASS_MONTHLY_R_STATION16_PACK_AUDIT`; the compact audit and recovery receipts are archived in `results/data_recovery_pack_v3/`. This is not the pack used for this checkpoint.
- The 13-file recheck report is `results/data_recovery_pack_v3/source_error_recheck.json`; first-pass errors remain preserved in `source_errors_first_pass.csv` in the recovered-pack evidence folder.

## Next step

Continue the already-running matched GHI evaluation chain using the frozen
`seed42_v1` checkpoint and its existing COT bank. Keep its outputs tied to that
checkpoint. If the COT-bias pattern motivates a retrieval improvement, treat
it as a separately named model/training run and regenerate its downstream COT
bank; do not silently replace the checkpoint under the existing bank.
