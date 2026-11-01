# Train-only lead-phase check for forecast COT

The three-seed target-window average screen did not improve on the existing calibrated future-COT input. A prior cross-fitted temporal-increment calibration also changed GHI very little. This next CPU pre-gate tests a distinct error mode: the predicted COT trajectory may have the right cloud signal but place it one 15-minute step early or late.

## Fixed diagnostic

Using only the five-fold OOF training forecast-COT bank and the exact training-row future-AGRI→the same frozen R reference from the oracle pack, compare three predetermined target-slot sources: current predicted COT at lead `L`, the preceding context COT at `L−15 min`, and the next forecast COT at `L+15 min`. The oracle COT is read only as a training diagnostic target; it is never an inference input. Apply the existing train-only reliability calibration to each forecast source frame at its own source lead. The `+240 min` endpoint is excluded from this pre-gate so all three candidates have identical keys. Do not read CPP or validation references to choose the shift.

Use nested date-fold selection: for each outer forecast fold, choose one global shift from `−15/0/+15 min` using the other four folds, then score the chosen shift on the untouched outer fold. Report center and full 3×3 log1p-COT RMSE, both stations, equal-station/equal-lead macro metrics, and target-BJT-date-block bootstrap intervals. A future GHI-head screen is unlocked only if the nested center-RMSE difference has a 95% interval wholly below zero, neither station worsens, and the same shift is selected in at least four of five outer folds. Otherwise stop this candidate and proceed to an AGRI→R forecast-domain error diagnosis.

If the pre-gate passes, train the same three-seed GHI head on the unchanged causal OOF pack, changing only the target future-COT frame by the single shift selected on all train folds. For `+240 min`, retain the unshifted target frame. Compare exact common-key A/AH/AC/ACREL/phase-shift predictions by date-block bootstrap. Validation is evaluation only; independent test and buffer remain closed.

### Implementation correction before numerical rerun

The first corrected-path run reached the data but stopped when a station GHI row at lead `L+15` was absent for some `L=225` samples. The COT source itself is available in the complete 16-frame OOF forecast bank; it must not be fetched by borrowing a neighboring GHI-pack row, because GHI label availability is unrelated to whether the forecast COT frame exists. The rerun reads the same-sequence, same-station bank frame directly at `L+15`, applies that source lead's already-frozen train-only calibration, and cross-checks against packed `L+15` rows wherever those rows exist. The scored targets and all candidate methods remain on the same original GHI keys. This is an implementation-only fix; it does not change the candidate shifts, fitting folds, metric, gate, or any held-out boundary.
