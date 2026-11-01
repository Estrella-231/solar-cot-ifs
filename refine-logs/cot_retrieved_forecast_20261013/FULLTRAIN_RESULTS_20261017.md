# Direct COT forecast screen — full-data seed 42

Date: 2026-10-17. This is a validation-only, single-seed screening result. CPP is a sparse retrieval reference, not independent physical truth; no test payload was opened.

## Question and comparison

The screen asks whether an 8-to-16 history-COT forecaster can provide a better COT trajectory than the existing fixed SimVP-forecast-AGRI → frozen-retriever chain. P-R is trained against retriever COT derived from future observed AGRI. P-Hybrid substitutes CPP COT where the training reference is valid and uses retriever COT elsewhere. Both arms use the same initialization, full eligible training population, sample ordering, geometry, architecture, optimizer schedule and 12-epoch budget; target values are the only experimental difference.

The date split is `hunan_monthly_block_2d_buffer1d_seed42_v1`. Each arm uses 28,806 eligible training sequences (57,342 station-sequence samples), batch 1,024, and 12 epochs. The paired PBS job `214510.tc6000` completed normally on one A800-80GB; its log ends with `PAIRED_COT_FULLTRAIN_COMPLETE`. The training contract records `test_payload_opened=false`.

## Scoring

The saved predictions were independently rescored with FP64 squared-error accumulators on the CPP-common pixel mask intersected with valid forecast-frame support. There are 3,140 validation sequences, 46 CPP-supported target dates, 5,224,460 common 16×16 patch pixels, and 183,564 station-centred 3×3 pixels. The support counts exactly match the earlier Fig. 7 forecast-COT audit. RMSE is first computed in each station-by-lead cell and then macro-averaged, matching the paper's forecast-COT metric. The direct models' checkpoint is selected using this same validation score; differences are therefore screening point estimates, without independent inference.

## Results

| Forecast source | Full 16×16 patch RMSE | Station-centred 3×3 RMSE |
|---|---:|---:|
| Existing SimVP-forecast-AGRI → frozen R | 10.953 | 10.664 |
| Direct COT, R-derived training targets (P-R; best epoch 3) | 12.512 | 12.275 |
| Direct COT, CPP/R hybrid training targets (P-Hybrid; best epoch 9) | 12.407 | 12.163 |

The CPP/R hybrid's point RMSE is 1.454 (13.3%) higher than the existing forecast chain on the full patch and 1.498 (14.1%) higher on the local 3×3. P-R is 1.560 (14.2%) and 1.610 (15.1%) higher, respectively. The hybrid arm is slightly better than P-R on the selected point score, but neither direct model reaches the fixed SimVP-to-R baseline. There is no paired date-block interval for this single-seed screen.

For P-Hybrid, full-patch station-by-lead macro RMSE by CPP reference thickness was 8.988 for COT <1 (2,119,268 pixels), 5.140 for 1–5 (575,625), 9.594 for 5–30 (1,831,804), and 25.668 for COT ≥30 (697,763). This shows that high reference thickness remains a difficult stratum for this candidate. It does not establish how much the direct model changes the fixed forecast baseline within each stratum; a same-key baseline-by-stratum comparison was not part of this screen.

## Claim boundary and next action

This screen does not demonstrate physical COT accuracy, a COT-induced GHI gain, or independent test performance. It does not include a GHI head trained on forecast inputs. The result is sufficient to stop unconditional multi-seed expansion of this direct-COT candidate: its selected validation point score is materially worse than the fixed SimVP-to-R forecast. The next paper-critical step remains the matched forecast-input GHI-head comparison if its data/training gate is satisfied; otherwise retain forecast COT as a bounded negative application result and do not claim downstream forecast benefit.

## Reproducible artifacts

- Training configuration and audit contract: `results/fulltrain_20261017_seed42_b1024_e12/training_contract.json`
- Epoch metrics and saved per-sample validation predictions: same result directory (`P_R_metrics.json`, `P_Hybrid_metrics.json`, `P_R_val_pred_epoch03.npy`, `P_Hybrid_val_pred_epoch09.npy`)
- Independent FP64 score and prediction hashes: `results/fp64_independent_validation_rescore_20261017.json`
- Recompute program: `recompute_fulltrain_validation_fp64_20261017.py`
- Earlier fixed SimVP-to-R reference: `paper/figures/fig7_forecast_cot_summary_20261017.json`

## Follow-up: fixed physical-amplitude loss

The one-factor follow-up is documented separately in `AMPLITUDE_LOSS_RUN_20261017.md`. It selected epoch 11, but FP64 same-support scores (12.473 full patch; 12.209 local 3×3) were slightly worse than the base P-Hybrid (12.407; 12.162) and remained worse than original SimVP→R (10.953; 10.664). A small ≥30-COT stratum change did not translate into overall gain. The predeclared gate failed, so the direct-COT objective-tuning branch is closed; no extra seeds, weight sweep, GHI head, or test access follows automatically.
