# New-monthly-R matched GHI head results (three seeds)

Date: 2026-10-13 BJT  
Scope: validation only; no new-test payload was opened.

## Result

On the frozen new-monthly-split validation cohort, the matched AC head that receives the local COT field had a lower station-equal RMSE than both the no-COT A head and the capacity-control AP head in all three seeds. The three-seed average RMSE was 123.980 W m⁻² for A, 120.240 W m⁻² for AC, and 123.822 W m⁻² for AP. The mean paired differences were AC−A = −3.740 W m⁻² and AC−AP = −3.582 W m⁻². Negative values favor AC.

| Seed | A | AC | AP | AC−A | AC−AP |
|---:|---:|---:|---:|---:|---:|
| 42 | 122.933 | 119.765 | 122.721 | −3.168 | −2.956 |
| 43 | 125.217 | 119.533 | 125.113 | −5.684 | −5.580 |
| 44 | 123.791 | 121.422 | 123.631 | −2.369 | −2.209 |
| Three-seed mean | 123.980 | 120.240 | 123.822 | −3.740 | −3.582 |

Values are station-equal means of the two station RMSEs in W m⁻². In every seed, AC−A and AC−AP were negative at both Sili and Zhujia. AP−A was also slightly negative (three-seed mean −0.158 W m⁻²), so the capacity control is not literally identical to A; the substantially larger AC−AP difference is the more relevant control comparison.

Descriptive season stratification also favors AC in every season at the station-equal, three-seed level. The per-season sample counts are fixed across seeds; no per-season bootstrap or weather claim is made.

| BJT season | Sili n | Zhujia n | Mean AC−A across seeds (W m⁻²) | Seed-wise station-equal AC−A |
|---|---:|---:|---:|---|
| DJF | 340 | 501 | −4.428 | −3.008, −7.614, −2.661 |
| MAM | 712 | 787 | −5.271 | −5.685, −6.314, −3.813 |
| JJA | 726 | 731 | −2.271 | −1.685, −4.198, −0.931 |
| SON | 350 | 332 | −6.734 | −5.304, −9.641, −5.259 |

Weather-stratified results remain pending; do not infer cloud-regime effects from this seasonal grouping.

## Paired uncertainty

The independent audit recomputed RMSE in FP64 from the saved predictions and the common validation labels. It resampled 52 BJT target-date clusters 2,000 times, retaining seed, station, and arm pairing and recomputing each station's RMSE difference before averaging stations and seeds.

| Paired difference | Mean | BJT date-cluster 95% CI | Consecutive 2-day block 95% CI |
|---|---:|---:|---:|
| AC−A | −3.740 | [−6.180, −1.597] | [−6.174, −1.483] |
| AC−AP | −3.582 | [−5.977, −1.479] | [−5.973, −1.375] |
| AP−A | −0.158 | [−0.233, −0.092] | [−0.223, −0.095] |

The bootstrap means need not equal the point estimates because RMSE is recomputed after each resample. Both prescribed resampling schemes put the AC−A and AC−AP intervals below zero. These are validation-cohort intervals conditional on selecting checkpoints by validation RMSE; they do not establish independent-test generalization.

## Experimental contract and integrity checks

- The 54,937-row pack is bound to `hunan_monthly_block_2d_buffer1d_seed42_v1`, with 50,458 train and 4,479 validation rows. The validation support was 2,128 Sili and 2,351 Zhujia rows.
- All arms used the same train/validation rows, 30 epochs, batch 2,000, physical-GHI MSE objective, seed-specific paired initialization/order, and frozen new-monthly R COT bank. A receives the observed target-time AGRI 3×3 field and per-pixel `cosSOZ/cosRAA`; AC adds the co-registered 3×3 COT field; AP adds a similarly sized AGRI-derived auxiliary branch.
- AC and AP have 24,489 and 24,613 trainable parameters (0.51% difference); A has 22,177. The AP arm therefore provides a close extra-capacity control, not a complete control for the R model's larger receptive field or shared CPP supervision.
- All 18 checkpoints were reloaded on CPU and rerun against the frozen validation inputs. The maximum pointwise CPU-reload versus saved A800 prediction difference was 0.607 W m⁻²; the maximum change in station RMSE was 0.0078 W m⁻². The FP64 result table is computed from the saved A800 predictions, not from the CPU reload.
- Both seed-44 PBS jobs (`213838` Sili and `213839` Zhujia) completed with exit status 0. Each trained all three arms for 30 epochs; no OOM or training error was reported. The queue was queried live, and there were no unfinished jobs after completion.
- The pack, each run, and the audit record `test_payload_opened=false`. The independent test remains closed.

## Interpretation and limit

This passes the plan's validation-stage direction criterion for the matched real-AGRI retrieval experiment: AC improves over A in all three seeds, and the paired date-block intervals are below zero; AC also outperforms AP, making added head capacity alone an unlikely explanation for the observed difference. The result supports continuing to the preplanned COT-dependence and physical-case diagnostics.

It does not prove that COT's physical semantics alone caused the improvement. The COT is retrieved from AGRI by the frozen CPP-supervised R model, which has a wider native receptive field than the GHI head's 3×3 readout; AP does not control that pathway or its supervision. The result is for same-time observed AGRI and is not a forecast-chain gain. It is limited to two Hunan stations and validation windows with validation-selected checkpoints. No IFS or new test result is included.

## Reproducible artifacts

- `ghi_heads_three_seed_validation_audit.json`: independent FP64 rescoring, common-key checks, checkpoint/prediction hashes, and 2,000-resample paired CIs.
- `ghi_heads_three_seed_cpu_reload_audit.json`: CPU reload comparison for all 18 checkpoints.
- `../audit_ghi_heads_three_seed.py`: independent scoring and bootstrap implementation.
- `../reload_ghi_head_checkpoints.py`: CPU checkpoint-reload implementation.
- Per-seed model files and validation predictions are in `ghi_heads_new_r_seed42_v1/`, `ghi_heads_new_r_seed43_v1/`, and `ghi_heads_new_r_seed44_v1/`.

Next: finish validation-only weather stratification using the quality-screened CPP/CLP common subset, then freeze a 16-case physical-figure manifest before inspecting per-sample GHI errors. The fixed-head COT input-dependence diagnostics are recorded separately. Do not open test or begin the forecast-head stage until the phase decision is documented.
