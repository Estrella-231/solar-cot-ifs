# Future AGRI-to-COT forecast-domain diagnosis

## Why this is next

The nested train-only temporal-phase audit selected the current target time in all five outer folds. Moving forecast COT by one 15-minute step did not improve its frozen-R reference error, so time phase is not a supported correction. Simple adjacent-frame averaging and temporal-increment calibration also failed to produce a reliable GHI increment. The next question is whether the SimVP-predicted AGRI field loses or biases channel/spatial information that the frozen COT retriever needs.

## Fixed diagnostic

Use the held-out validation dates descriptively. For each exact forecast-pack `(station, init, target, lead, GHI-row)` key, compare the predicted AGRI 3×3 station neighborhood against the real AGRI frame at that target time. Real future AGRI is opened only as a validation reference; it is never an inference input or a fit/selection target. Compare per-channel normalized error, signed bias, predicted-to-observed spread, and their association with calibrated forecast-COT error against the same-time real-AGRI→frozen-R reference. Report station and fixed lead bands `+15–60`, `+75–150`, and `+165–240 min`.

The raw target AGRI patch is extracted from the 256×256 Hunan preprocessing frame using the audited `frame_keys.csv` station grid coordinates; the forecast patch comes from the already-built five-fold-OOF train / full-train validation bank. R output remains a retrieval reference, not independent cloud truth. This diagnostic does not fit parameters, choose thresholds, use GHI labels, or access CPP, test, or buffer payloads.

## Decision after diagnosis

If COT errors track specific AGRI channel bias/variance loss or forecast-field smoothing, design one train-only COT-aware SimVP loss ablation while retaining the original AGRI reconstruction loss. Train the five OOF folds under the fixed date split and budget, parallelizing independent folds/seeds across available GPUs subject to the account PBS limit. Rebuild forecast-COT OOF outputs, then compare GHI heads on identical keys. If AGRI forecast errors do not explain COT errors, test a fold-cross-fitted predicted-AGRI-domain calibration of frozen-R output before changing the SimVP backbone. In either branch, retain validation as final screening only and keep test/buffer closed.
