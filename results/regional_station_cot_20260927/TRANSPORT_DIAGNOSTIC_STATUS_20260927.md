# Sili transport diagnosis status — 2026-09-27

## Purpose and scope

This is a validation-only diagnostic for whether improving cloud displacement can recover thick-cloud COT at the Sili station. The reference is the CPP COT retrieval, not an independent in-situ cloud measurement. No GHI model was trained and no test split was used.

## Cohort and paired scoring

The first expanded comparison uses the existing fixed, month-stratified 128-sequence validation cohort. CPP availability did not determine cohort selection. For each lead, every persistence, translation, and SimVP prediction is scored against CPP on the exact intersection of valid reference pixels and in-domain support for all translation variants. Coverage of this strict common spatial support falls from 94% at +15 min to 64% at +240 min. Station-pixel scores use the same sequence/lead eligibility across all methods.

Motion is estimated from historical C13 frames only. Compared estimates are the latest pair, medians over the latest four or seven pairs, and seven-pair medians in 64, 32, or 16 pixel windows centered on Sili. Each estimate is held at constant velocity to the future lead. This is a deliberately simple pure-transport baseline; it cannot create, dissipate, or thicken clouds.

## Preliminary 128-sequence findings

- Rounding each latest-pair displacement to integer pixels makes 78.9% appear exactly zero. Keeping subpixel shifts reduces the near-zero fraction (norm < 0.1 pixel per 15 min) to 31.3%. Thus rounded-zero counts are not evidence that clouds are stationary.
- On the shared 64×64 support, seven-pair global transport RMSE is 10.81 versus 10.92 for persistence at +240 min; at +120 min it is 8.62 versus 8.80. The improvement is small. SimVP is 10.77 at +240 min on the same spatial support.
- At the station pixel, seven-pair transport does not beat persistence consistently: +240 min RMSE is 8.30 versus 7.50, while SimVP is 9.39. Local-window transport also lacks a stable advantage across leads.
- Among 124 sequences with valid CPP station references, the cohort contains 63 thick-reference lead records (CPP COT ≥ 30) and 9 SimVP severe-miss records (CPP ≥ 30, SimVP COT < 10). Because the current 128-case sample contains only one qualifying nonzero-motion thick-cloud event, it is too sparse to conclude whether transport recovers thick peaks.
- The normalized phase-correlation peak is weak (mean 0.00131 in this cohort) and is not calibrated as a reliability score. It cannot certify that the estimated vector is physically correct; small fractional shifts may reflect rounding, weak texture, evolving cloud shape, or retrieval/image noise. Dense-flow or independent tracking validation remains a follow-up if phase transport survives the expanded thick-event gate.

These values are exploratory and adjacent leads within one sequence are dependent. They do not establish a GHI benefit or support stage 3 training.

## Expanded 512-sequence findings

The fixed validation bank has 512 selected sequences; 486 contain valid CPP at the Sili pixel, 55 contain at least one CPP-thick lead, yielding 205 thick station/time records. Adjacent windows overlap, so these are not 205 independent cloud events.

- Integer rounding again makes 79.3% of latest-pair vectors exactly zero. With subpixel motion retained, only 32.0% are below 0.1 pixel per 15 min. Median seven-pair vectors are under 0.1 for about 34.2% globally and 31.6% in the local 64×64 window. The “mostly zero” impression is chiefly a rounding artifact, although the phase-correlation peak remains weak and its vectors are not yet physically validated.
- On the 205 CPP-thick station/time records (CPP COT ≥ 30), the severe low-prediction count (method COT < 10) is 67 for SimVP→R, 56 for persistence, and 48 for 32×32 local seven-pair transport. The local transport recovers 36 SimVP-below-10 records and loses 17 SimVP-above-10 records, a net reduction of 19 thresholded misses. The count spans 12 and 8 overlapping sequence windows respectively, not independent events.
- Thick-lead station COT RMSE is 28.33 for SimVP→R and 25.21 for local32 transport; mean absolute error at the CPP peak is 24.74 and 22.50. Both remain large, so translation does not solve cloud-thickness evolution. For all 64×64 valid pixels at +240 min, local32 RMSE is 12.18 versus 11.54 for SimVP→R; station-pixel RMSE is 11.22 versus 10.61. Thus event-conditioned station gains do not imply a general spatial or downstream GHI gain.
- Example 2025-07-14 10:30 BJT: CPP COT grows from about 0.8 to 55.6 during the 4 h forecast, while history-only local transport stays below 1. This is cloud formation/intensification that pure transport cannot create. Example 2025-07-09 14:00 BJT: CPP peaks at 68.7 by +60 min and falls to about 5 by +240 min; local transport briefly raises the station COT, then transports the cloud away, while SimVP→R remains near 5–8 throughout.

These results meet the trigger for testing whether spatial object location improves: local32 reduces thick-event errors while leaving strong intensity/evolution errors. The paired cloud-mask audit is complete. Across the 16 leads, the unweighted mean thick-pixel IoU is 0.0317 for local32, 0.0319 for local64, 0.0286 for persistence, and 0.0238 for SimVP→R. Mean centroid distance when both masks exist is 23.31 pixels for local32 versus 25.56 for SimVP→R. Absolute IoU remains low, and +240 min differences are not uniformly favorable, so this is a modest conditional signal rather than robust transport skill.

The next small pilot uses historical R-COT sequence, local32 multi-frame C13 motion, and frozen SimVP→R forecast COT as inputs. Its output starts from the transported history field and learns a residual thickness/evolution correction. CPP COT serves only as a retrieval label for train/validation; it is not an input and is not independent physical truth. A 512-train/512-validation bank and 40-epoch early-stopped pilot are packaged as PBS `210963.tc6000`, currently queued for one GPU because the existing 7-GPU plus 1-GPU jobs fill the node allocation. No training has started yet. Test remains closed.

The 4-case station-trajectory figure is `sili_transport_case_tracks_val512_20260927.png` (and PDF); cloud-mask scores are in `sili_transport_variants_val512/metrics.json`, with a lead-wise figure `sili_transport_cloud_object_scores_val512_20260927.png`. All thresholds are descriptive diagnostics against CPP, not physical cloud labels.

## Current next gate

A fixed month-stratified 512-sequence validation bank was generated by frozen SimVP and frozen regional R inference. Its completion record confirms `test_used=false`; GPU inference took 1,328 seconds and reported peak allocation 15.7 GiB on the A800 80 GB. The paired transport, cloud-object, and dated-case audits are complete. Their results support only a small, gated transport-plus-thickness pilot: translation has a modest conditional thick-event/location signal, but does not improve all-pixel long-lead skill and cannot represent cloud growth or decay.

The pilot uses historical R-COT sequence, local32 multi-frame C13 motion, and frozen SimVP→R forecast COT. CPP is used only as a train/validation retrieval label. The training/validation bank builder and CPU smoke test are ready; PBS `210963.tc6000` is queued (`Q`) for one GPU and training has not started. The test split remains closed. The next gate is to verify the training data bank and run the pilot only when the job starts; after completion, compare against frozen SimVP→R, persistence, and transported history on identical test cases before deciding whether the learned evolution branch has value.

## Artifact locations

- Fairly matched 128-sequence summary: `sili_transport_variants/summary_v2.json`.
- Original 128-case figures and details: `sili_transport_variants/`.
- Expanded inference job: PBS `210952.tc6000`; bank is complete on the server, outside Git. Output root: `experiments/regional_station_cot_20260925/regional_ghi_pilot_bank_val512_20260927_v1`.
