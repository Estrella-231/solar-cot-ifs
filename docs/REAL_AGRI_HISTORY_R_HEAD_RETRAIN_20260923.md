# Real-AGRI COT retriever retraining comparison

Date: 2026-09-23

Status: exploratory validation-only retraining; no test split was used.

## Question

Does replacing the original real-AGRI-to-COT retriever with its continued/improved checkpoint change the value of historical COT for the same-from-scratch GHI head? The comparison uses the same Hunan frozen-SimVP historical AGRI frames, rows, labels, station inputs, forecast-COT bank, head, optimizer, seed, and training budget. Only the historical COT sidecar source changes.

The selected continued-R checkpoint was trained on real observed AGRI, but its supervision is the CPP reference COT product and its objective is pixelwise COT loss—not GHI error or independently measured physical COT. It is the `log_control` continuation of the old R; the separate `high_cloud_weighted` candidate did not beat its initial checkpoint and was not selected. On real AGRI validation, the selected continuation reduced full-grid COT RMSE from 4.2465 to 4.1688 and COT≥30 RMSE from 11.9531 to 11.2890, while clear-sky COT RMSE increased from 0.7711 to 0.8474; 2025-09 aggregate error also worsened. These COT results use the same validation period involved in model selection and do not by themselves predict GHI impact. See `docs/COT_REAL_AGRI_HIGHCLOUD_PILOT_20260922.md` for the retrieval audit.

## Historical COT sidecars

The FP32 sidecar builder reads 8 observed AGRI frames ending at each initialization time through the audited frozen-Hunan-S loader. It writes train and validation COT separately for the original and continued R checkpoints. Each route uses the same original-R train-only AGRI/geometry normalization. There are 27,675 training and 4,258 validation sequence slots; no test sequences are included.

| Route | R checkpoint SHA-256 | Train sidecar SHA-256 | Validation sidecar SHA-256 |
|---|---|---|---|
| original R | `0cbdb932f49945259d5f1704316edaf6481be7cfdb54692236f829988925799c` | `c13fc38b1c346078c8e5a689a9190fd084480b24050aada02f921414ab10e581` | `7f73f6c6a0e000e64afcaf6cfe4c947f6ad897d9a9ead65d0c4c148858100fb1` |
| continued R | `18d285c2b1349ca9890c48109b3a149520638674bae69293ff9480938fcfe069` | `cbb87a0725a76e520df45d605c7ffdea5a8640768254d1f3e5b95b296ce50c72` | `71397af9944b39eaa3948537e3e72d0e7b8b80fc44382b3d2b6441198116b230` |

## GHI-head protocol

Both routes use the same SolarResNet image trunk with causal dilated 3D COT-trajectory fusion and the same four groups: `A_ST`, `AC_H`, `AC_F`, and `AC_ST`. Each station/group head is initialized and trained independently with seed 42, AdamW (learning rate `3e-4`, weight decay `1e-4`), 10 epochs maximum, and the same validation-based best-epoch selection. The GPU profile selected batch 1536. Validation predictions are audited by pack row and recomputed in FP64; station RMSEs are equally weighted in the primary endpoint, with a paired bootstrap over shared initialization days. Results are exploratory and do not authorize a test-set claim.

## Normalization confound and correction

The first independent retraining pair fitted a different train-only COT mean/std for each retriever. This also changed the scaled future-COT values in `AC_F` and `AC_ST`, so those differences cannot be attributed only to historical R. That first pair is retained as a diagnostic artifact and excluded from causal interpretation.

The corrected retraining freezes the exact old-R train-only COT normalization for the new-R route. The old and corrected new-R contracts must match exactly for `mean`, `std`, and `count`; the independent audit rejects any mismatch. This makes the forecast-COT numeric input identical between the routes, while the historical COT maps remain the sole intended input difference.

## Results from the corrected paired audit

The corrected new-R route completed all 8 station/group heads. The independent audit recomputed GHI RMSE from saved validation predictions and matched pack rows. Its 2,000-resample bootstrap used shared initialization days across stations and leads (72 days). New-minus-old equal-station RMSE differences are:

| Group | Original R | Continued R | Difference (W m⁻²) | Paired 95% interval |
|---|---:|---:|---:|---:|
| `A_ST` | 150.789 | 150.789 | 0.000 | [0.000, 0.000] |
| `AC_H` | 147.643 | 148.955 | +1.313 | [−0.192, +2.650] |
| `AC_F` | 149.580 | 149.580 | 0.000 | [0.000, 0.000] |
| `AC_ST` | 152.572 | 151.067 | −1.505 | [−3.182, +0.148] |

`AC_H` is numerically worse with continued R at both stations (Sili +1.698; Zhujia +0.928 W m⁻²), but the pooled interval crosses zero. `AC_ST` is numerically better at both stations (Sili −1.411; Zhujia −1.600 W m⁻²), and its pooled interval also crosses zero. Exact `A_ST` and `AC_F` prediction equality (maximum absolute difference 0 at both stations) confirms that the shared training setup and future-COT input remain unchanged when historical R changes.

These results do not show a reliable GHI improvement from the continued R checkpoint. They also do not establish that it harms GHI, or that historical COT is generally unhelpful: the comparison is one seed, a 10-epoch exploratory head, and validation only. It supports no test-set or paper-level improvement claim.

### Result-to-claim review

A fresh same-family Codex reviewer judged the intended claim **partial**, with **medium confidence** and **provisional** acceptance. It found a consistent numerical improvement in `AC_ST` across both stations, but the paired interval crosses zero and `AC_ST` remains worse than `AC_F` by 1.487 W m⁻² and worse than `A_ST` by 0.279 W m⁻². The `AC_H` branch moves in the opposite direction: its historical-COT benefit relative to `A_ST` shrinks from −3.146 W m⁻² with original R to −1.833 W m⁻² with continued R. This does not support the intended claim that the continued R improves historical-COT value for GHI.

Checkpoint selection also warrants caution: all 8 original-R heads and 6 of 8 continued-R heads selected epoch 0 on validation. Extend only the `AC_H` and `AC_ST` original/new-R pairs to at least 3–5 seeds with shared normalization and a predeclared checkpoint-selection rule before considering an independent test-set confirmation. The evidence precheck helper and a separate `EXPERIMENT_AUDIT.json` were unavailable; the paired-row audit did pass.

## Current run and artifacts

- PBS job: `210398.tc6000` (completed with exit status 0 in about 52 minutes; one A100; seed 42; 10 epochs per station/group).
- Corrected new-R run: `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/runs/causal_cot_v5_real_agri_Rpair_seed42_commonnorm_20260923/continued_R/`.
- Existing old-R comparator: `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/runs/causal_cot_v5_real_agri_Rpair_seed42_20260923/original_R/`.
- Corrected paired audit: `results/cot_real_agri_history_rpair_20260923/independent_retrain_pair_audit_commonnorm.json`.
- New-R run summary, contract, and profile: `results/cot_real_agri_history_rpair_20260923/runs/continued_R_commonnorm/`.
- Result-to-claim trace (same-family, provisional): `.aris/traces/result-to-claim/2026-09-23_run01/`.

The first route-normalized pair remains in the result folder as `diagnostic_normalization_confounded_pair_audit.json` and `runs/continued_R_route_norm_diagnostic/`. It is not used in the corrected conclusion because its group-specific normalization changed forecast-COT scaling as well as historical COT.
