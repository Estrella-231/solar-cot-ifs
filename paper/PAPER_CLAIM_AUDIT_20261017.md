# Manuscript evidence audit — 2026-10-17

Scope: current English LaTeX and Chinese Markdown draft, validation-only evidence. No test payloads were opened and no new training was started. This is an internal evidence ledger, not manuscript text.

## Changes made in this pass

- Removed the COT-retriever hidden-feature control from the manuscript narrative and model table because the current paper evidence bundle did not provide a sufficiently traceable source for its score and paired interval. The capacity-matched auxiliary-feature control remains.
- Removed the unsupported qualitative statement that the retriever has “moderate” agreement with CPP. CPP is now described only as a retrieval reference, not independent physical truth.
- Updated the figure plan so Table 3 is described as the capacity control, matching the table that is actually in the manuscript.
- Added a validation-only Hunan solar-resource diagnostic to the English and Chinese manuscript, using saved three-seed GHI predictions and common date support. The paired daily-irradiation absolute-error result is explicitly not described as PV generation or independent test performance.
- Recompiled `main.tex` successfully after the synchronized manuscript update.

## 2026-10-17 event-case manuscript revision

- Added an error-blind, preselected Sili high-COT-change validation case to Fig. 6. The panel explicitly separates observed AGRI, CPP reference COT, frozen-R retrieved COT, solar geometry, and same-time GHI-head outputs. It is not a forecast figure.
- Added the four available summer-case RMSEs to Results §5.4: AGRI + COT reduced within-case RMSE only in one case (by 0.871 W m-2) and increased it in three (+4.557, +8.938, +9.917 W m-2). These are descriptive scores over consecutive targets, without event-level intervals; the text does not use them to infer causality or replace the cohort result.
- Added a reproducible summary script and output under `analyses/t16_event_cases_20261017/`. Source audit confirms that case selection did not use GHI values/errors and that test payloads were not opened.
- Updated the figure plan to reflect the observed evidence: one error-blind main-text example and all four cases in supplementary material. The seasonal CPP support remains limited to JJA.
- Recompiled the synchronized English LaTeX after this event-case revision with the local TeX Live toolchain; the updated 17-page PDF includes the Figure 6 image. Rendered page 12 was visually checked for layout and legibility. The built-in editor compiler reports diagnostics only and does not refresh the on-disk PDF, so it was not treated as PDF-generation evidence.

## Added solar-resource diagnostic

The reproducible analysis is `analyses/solar_resource_validation_20261017.py`; its machine-readable outputs and definitions are in `analyses/solar_resource_validation_20261017/REPORT.md` and `summary.json`. It reads the saved validation pack and A/AC/AP predictions only, and verifies reconstructed overall RMSE against the existing FP64 audit before producing the subgroup analysis.

- At station-centre solar altitude >=45 degrees, the common Hunan validation support contains 1,527 station-target rows across 35 dates. Station- and seed-equal RMSE is 154.111 W m-2 for AGRI and 148.645 W m-2 for AGRI+COT; paired date-block 95% CI for the difference is [-8.990, -2.152] W m-2.
- For complete daylight coverage at both Hunan stations, the paired support is 30 dates. Daily irradiation is calculated as 0.25 times the sum of 15-min interval-mean GHI, in kWh m-2. Station- and seed-equal daily absolute error is 0.5491 for AGRI and 0.5032 for AGRI+COT; paired date-block 95% CI for the difference is [-0.0861, -0.0067] kWh m-2. The corresponding daily bias shifts from -0.0164 to -0.0561 kWh m-2, so lower absolute error does not mean bias correction.
- This is a secondary validation diagnostic on the Hunan split. It does not establish PV yield, cloud-event causality, an independent test result, or a corresponding benefit in Hubei. The five-station main result remains reported region by region under the existing sampling contracts.

## Figure and claim status

| Item | Status | Evidence boundary / next action |
|---|---|---|
| Fig. 1 study regions and coverage | Present; built and visually checked | Reproducible station locator and first-to-last date-support timeline from audited inventory metadata; endpoints do not imply continuous daily coverage. No test payloads opened. |
| Fig. 2 method/workflow | Present | Describes the implemented synchronous input path and the fixed forecast application; not evidence of forecast GHI gain. |
| Fig. 3 retrieval quality against CPP | Present; source-key audit recorded | Uses source-key-matched validation arrays, common valid pixel masks, audited BJT mapping, and date-cluster bootstrap. CPP remains a retrieval reference, not independent truth. Hunan valid CPP support is 46 BJT dates (not 57 raw key-date components); Hubei is one shared patch covering three stations. Reproduction and source hashes are recorded in `analyses/cot_retrieval_quality_20261017/REPORT.md` and the figure summary JSON. |
| Fig. 4 synchronous GHI increment | Updated 2026-10-17 | Both Hunan and Hubei show paired date-block intervals. Hubei values use the source-closed `e8c5…` pack; the old `7990846…` AC value is superseded. Regions remain separate. |
| Fig. 5 Hunan CPP-COT strata | Present | Uses station-centre CPP COT, shared rows, and date-block uncertainty; does not use patch mean/p90. Descriptive validation subgroup, not independent truth. |
| Fig. 6 cloud-event cases | Present, bounded | One preselected Sili high-change case is in the main text; four-case numerical summary is reported. All examples are JJA; no event-level interval or causal interpretation. Full set should be retained in supplement. |
| Fig. 7 forecast-COT lead curves | Present | Validation-only, CPP-valid local 3×3 support, 3,140 windows / 46 CPP-supported dates; RMSE and CSI are distinct aggregations as stated in the caption. CPP is a retrieval reference. Matched forecast-input GHI heads are evaluated separately in text. |

## Claims deliberately limited

- Hunan and Hubei results are reported separately because their ground-label time-sampling contracts differ. Hubei's source-closed validation intervals do not constitute independent-test confirmation and cover three nearby stations.
- The stronger signed-residual comparison is reported with its OOF-training/full-baseline-validation distribution shift and its interval crossing zero; it does not establish an incremental COT benefit.
- Matched forecast-input GHI validation is reported: three-seed FA/FAC station-equal RMSE 77.150/78.155 W m⁻², FAC−FA +1.005, paired date-block 95% CI [−0.722,+3.780]. It is validation-only and does not show a COT gain. Only one of three seed-level station-equal comparisons favoured FAC. Forecast-COT scores do not establish physical COT accuracy; fixed-head replacement remains an input-sensitivity diagnostic. A cold-start normalization parity audit is still pending, so this draft is not freeze-ready.
- No independent test-set conclusion is made.

## Synchronization and compilation

English LaTeX and Chinese Markdown retain corresponding section structure, equations, experimental populations, formulas, numerical values, and conclusion scope. The paper remains a working draft: matched forecast-input GHI heads have been evaluated on validation only, and no independent test conclusion is available.

## Figure 1 and manuscript build update

- Replaced the Figure 1 placeholder in both manuscript versions with a reproducible five-station locator and a regional date-support timeline. Station coordinates are sourced from the station-construction scripts; date endpoints are from the audited actual-data inventory (2026-10-16). The timeline explicitly represents first-to-last dates, not continuous coverage; CPP support is separately labeled as sparse.
- The figure uses only station-coordinate and train/validation inventory metadata; test payloads were not opened. The manuscript names the exact overall date bounds and CPP-support endpoint in both languages.
- `latexmk -pdf -outdir=build -interaction=nonstopmode -halt-on-error main.tex` completed successfully on 2026-10-17. The 18-page PDF includes Figure 1 on page 3; rendered page 3 was visually checked. No undefined-reference or undefined-citation warning was found in the build log. Remaining overfull/underfull box warnings are minor typography issues and do not block the build.

The manuscript remains a working draft: matched forecast-input GHI heads have been evaluated on validation only, and no independent test conclusion is available.

## 2026-10-17 validation event-cohort update

- Added a second, separate event analysis for forecast COT in Results §5.6 and Discussion. The prespecified event rule is station-pixel CPP COT >= 30 for at least two consecutive valid 15-min targets; overlapping windows were collapsed by target time, and each event used the longest-lead validation forecast covering the entire event.
- The analysis found seven unique events (four Sili, three Zhujia) on only three BJT dates. Event-equal local-3x3 RMSE for SimVP-forecast-AGRI→R was 19.53/13.48 by station, versus 6.87/7.38 for the nondeployable observed-future-AGRI→R hindsight diagnostic; pooled thick CSI was 0.149/0.000 versus 0.814/0.617. Direct CPP/R-hybrid COT prediction scored 24.47/26.79. No event-level confidence interval is reported.
- The older full-validation bank's embedded CPP values were found to disagree with the exact current compact-pack CPP values by up to 68.20 COT despite a matching valid-mask count. Those embedded values were quarantined; the analysis reconstructed CPP targets and masks from current validation frame indices and applied current R frame-valid support. Forecast keys, R checkpoint, and prediction checksum were verified; test payload reads remained zero.
- Revised the Chinese Discussion to match the English event-cohort scope and updated `PAPER_PLAN.md`. The paper treats the events as a candidate mechanism for a later SimVP COT-auxiliary-loss test, not as causal or generalization evidence. No new figure or bar chart was added.
- This result does not establish full-validation COT improvement, physical COT truth, or GHI forecast gain. CPP remains a retrieval reference, and the episodes are concentrated on three dates.

## 2026-10-17 physical-amplitude loss follow-up

- The single-factor physical-amplitude loss run (`214511.tc6000`) completed 12 epochs; best epoch 11. The independent FP64 same-support score used the same 3,140 validation sequences, 46 CPP-supported target dates, all 16 leads, common CPP mask, and new-R frame-valid support as the direct-COT comparison. No test payload was opened.
- The manuscript now reports the added loss as a bounded negative screening result: full-patch/local-3×3 RMSE 12.473/12.209, compared with base P-Hybrid 12.407/12.162 and original SimVP→R 10.953/10.664. The ≥30 COT stratum changed by only −0.133/−0.358 versus P-Hybrid, without improving the overall score.
- This result is validation-only, one seed, and does not establish independent significance, physical COT truth, or forecast-GHI skill. It does not identify whether the remaining error arises from thickness, cloud location/timing, retrieval disagreement, or optimization.
- English LaTeX and Chinese Markdown were updated in parallel in the short-term forecast Results, Discussion, and Conclusion. The manuscript does not change the synchronous GHI primary result or introduce a figure/table for this loss screen.

## Full-data direct COT forecast screen and bilingual revision

- PBS `214510.tc6000` completed normally on one A800-80GB with 28,806 eligible train sequences, 12 epochs per arm, batch 1,024, and paired P-R / P-Hybrid target contracts. The job log has the `PAIRED_COT_FULLTRAIN_COMPLETE` sentinel.
- Independent FP64 rescoring used saved per-sample predictions, the CPP-common mask, and forecast-frame valid support. The common validation key support is 3,140 sequences, 46 CPP-supported target dates, 5,224,460 full-patch pixels, and 183,564 station-centred 3×3 pixels, matching the audited Fig. 7 counts. `test_payload_opened=false` is retained in the training contract and independent score JSON.
- Selected P-R (epoch 3) scored 12.512 on full patches and 12.275 on local 3×3; selected P-Hybrid (epoch 9) scored 12.407 and 12.163. The fixed SimVP-forecast-AGRI→R reference is 10.953 and 10.664 on those same supports. This is one-seed validation screening with checkpoint selection on the scoring cohort; report point estimates only and do not claim inferential superiority or physical COT truth.
- P-Hybrid full-patch CPP-thickness-stratified station–lead macro RMSE is 8.988 (<1), 5.140 (1–5), 9.594 (5–30), and 25.668 (≥30). These candidate-only strata show high error in the thick reference class; a per-stratum baseline contrast was not produced.
- English and Chinese abstracts, forecast results, discussion, and conclusion were synchronized. The new numbers are explicitly separated from GHI forecast skill; no matched forecast-input GHI head was trained or scored.
- Recompiled with `latexmk`; the 18-page PDF was produced and copied to `paper/main.pdf`. The conclusion and supplementary table/reference pages were rendered and checked after pagination adjustments. No undefined references/citations or LaTeX errors were found; minor box and locale warnings remain.

## 2026-10-17 forecast-event mechanism follow-up

- Reused the four preselected JJA validation events from the frozen event manifest, exactly matching station, initialization time, 16 target leads, compact CPP support, saved SimVP→R predictions, and direct-COT candidates. All methods were scored on the per-pixel intersection of CPP and R frame-valid masks; test remained closed.
- On Sili's largest-change event (2025-07-09 13:15 BJT), SimVP→R local 3×3 RMSE was 30.27 and thick CSI 0.0066 (10/1,504 reference-thick pixel-leads); direct P-Hybrid had no thick hits. The hindsight observed-future-AGRI→R diagnostic scored 5.37 local RMSE and CSI 0.582. On Zhujia's largest-change event (2025-07-09 14:30 BJT), observed-future-AGRI→R improved local RMSE 7.59→3.79 but achieved only 0.105 thick CSI; this indicates substantial residual retrieval/reference or case-specific discrepancy.
- The two deterministic-hash controls were mostly clear and provide no evidence about thick-cloud recovery. The four events are deliberately selected and are not an estimate of validation skill. The manuscript therefore reports only a descriptive two-cloud-event contrast and explicitly avoids station-wide causal claims.
- Updated `paper/main.tex` and `paper/main_zh.md` in parallel in Results §5.6 and Discussion. The conclusion and abstract remain unchanged because the new evidence is too narrow to change the study-level claim. Updated the forecast-COT tracker and standalone diagnostic report. No chart or training was added.
- The desktop LaTeX compiler reported success after the bilingual update; `git diff --check` found no whitespace errors in the edited manuscript and evidence-ledger files.

## 2026-10-18 AL provenance and manuscript correction

- Supersedes the earlier notes above that described the Hubei retriever-feature arm as untraceable or its cold-start normalization reload as pending. The AL source pack, frozen-R feature bank, split, checkpoints, normalization files, keyed predictions, and current A/AC/AP receipts were bound and independently rechecked. The cold-start audit recomputed train-only normalization from raw train rows and reran validation from raw train/validation inputs: normalization-statistic max difference 0; cross-device prediction max difference 0.299 W m$^{-2}$; maximum RMSE difference 0.0112 W m$^{-2}$; test/buffer payload reads 0. Receipt SHA-256: `a17e73415ec78a9781707c1ed5c8e5abe31b6efbdd22524839eea337d34a6b3b`.
- On complete three-station/three-seed validation support, RMSE (W m$^{-2}$) was A 74.333, AC 70.993, AP 73.839, AL 71.090. On the shared 486 station-target rows over 21 dates, AC/AL were 72.867/72.785; AC minus AL was +0.082 W m$^{-2}$, with consecutive two-day-block 95% interval [−1.490,+1.713]. The interval crosses zero. Thus the Hubei validation does not isolate an explicit-COT increment over the frozen-retriever feature control; this is validation evidence, not independent test evidence.
- Updated English and Chinese abstract, question framing, Results, Discussion, Conclusion, and paper plan to report this control and narrow the Hubei interpretation. `main.tex` and `main_zh.md` remain aligned on population, metric, interval, and claim boundary. LaTeX compilation and rendered-page check are pending in this update.
- FD-107, FD-172, and FD-201 were retried with password-only SSH after the user supplied credentials; each server closed TCP/22 before an authentication prompt. No credentials were persisted, no remote job was launched, and FD-248 was not accessed. ZJNU PBS remains the available compute route. Test/buffer remain closed pending refreshed full-provenance and test-freeze review.

## 2026-10-18 method and gate review

- Corrected the bilingual training-method text to distinguish Hunan synchronous batch 2,000, Hubei same-time A/AC/AP and AL batch 512, and Hubei forecast FA/FAC batch 1,536. The Hubei same-time primary and two-day sensitivity intervals use 10,000 resamples; AL common-support sensitivity uses 2,000; forecast-input GHI intervals use 10,000. The figure plan now describes Figure 7 as forecast-COT RMSE/CSI only and keeps the GHI result separate.
- Independent integrity/lineage reviews still conclude FAIL / NO-GO for test unseal. Remaining evidence gaps include exact result-analysis source-to-receipt hashes, a fail-closed frozen test evaluator, complete validation full-bank S/R date-exclusion provenance, and source-pack/checkpoint metadata enforcement. These do not reverse the current validation-only findings, but they block opening test/buffer.
- A fresh run of the existing same-time aggregation script against the legacy independent-recheck bundle failed its precondition `checkpoint_reload_parity == PASS`; no refreshed aggregate receipt was produced. This exposed that the script's input contract is stale relative to the normreload receipts. Published metrics were not altered on the basis of this failed run; the source/receipt freeze remains unresolved.
- The current `latexmk` rerun succeeded (20 pages); PDF pages 1 and 11 were rendered and visually checked after removing a duplicated abstract sentence. No undefined-reference/citation warnings were found. Two minor overfull-box warnings remain, unrelated to the new AL passage. The machine-readable claim audit has been updated to show AL and the successful build, but `paper_ready_for_submission` remains false and `fresh_experiment_audit` remains FAIL/NO-GO.
