# T16 seasonal coverage audit

Date: 2026-10-14 (BJT)

## Finding

The four T16 cases being JJA-only does **not** mean the AGRI input series or the new monthly-R COT bank cover only summer. It means the T16 manifest was selected from the CPP reference pack and required 17 consecutive CPP-supervised timestamps at 15-minute cadence (initial time plus 16 leads through +240 min), with validation GHI key/main-mask support at every timestamp.

The selector does not require all 256 pixels to be valid: it intersects the 17 CPP masks and accepts a candidate if at least one common pixel remains. Therefore the primary seasonal bottleneck is temporal CPP availability, not a 256/256 spatial-support rule.

## Evidence from the project record

- New-split candidates contain 51,236 train and 4,497 validation unique frames. Raw FY-4B AGRI filename support exists for 50,924 train frames (99.39%) and 4,485 validation frames (99.73%). Thus AGRI source coverage is nearly complete, with 324 unmatched frames.
- The new monthly-R inference bank is marked `COMPLETE`: 55,733/55,733 train+validation frames, including both stations, with zero source errors. This bank is a learned R inversion of real AGRI, not an independent CPP reference.
- The CPP support inventory shows a different cadence pattern: most months from 2024-04 through 2025-05, and again from 2025-10 onward when present, have CPP files for about one quarter of 15-minute candidate times (approximately hourly sampling). June–September 2025 have dense 15-minute CPP support in the available split. Some 2026 months have no archived CPP at all.
- The October 13 T16 manifest consequently found 181 Sili and 183 Zhujia complete 17-frame validation sequences in JJA, and none in DJF/MAM/SON. This is consistent with the CPP support table and the selector code; it is not evidence of missing AGRI across those seasons.
- Project history records an independent CPP V2 15-minute production route beginning 2025-08-04 UTC. It writes to a separate output tree. The gap-fill plan identified missing historical CPP times whose AGRI HDF and ERA inputs existed, but only pilot/small-scope production was verified; the full all-month gap-fill was not completed. It must not be assumed that the V2 output already filled the historical archive.

## Correct interpretation and next use

Keep the four original CPP-selected cases labeled as a CPP-reference subset. For all-season illustrations, select a separate case set from the complete monthly-R bank applied to observed AGRI and label its COT as **R-retrieved COT**, not CPP truth. CPP can be overlaid where present as a separate reference layer. This restores seasonal coverage without changing or retroactively relabeling the preregistered T16 CPP cases.

If the goal is four-season 15-minute *CPP-reference* event evaluation, first complete and QC the CPP V2 historical gap-fill on the frozen train/validation frame list, then rerun CPP support inventory and create a new, separately versioned event manifest. Missing CPP must never be represented as clear sky or silently filled from another hour.

## Sources

- `windows/window_3/T16_EVENT_SELECTION_RULE.md`
- `experiments/monthly_R_station16_20261013_v1/select_t16_event_manifest.py`
- `results/t16_cpp_event_selection_v1/T16_EVENT_MANIFEST_20261013_v2.json`
- `results/trainval_data_20261013/cpp_support_v1/cpp_support_audit.json`
- `results/trainval_data_20261013/cpp_support_v1/raw_hdf_support_audit.json`
- `experiments/direct_cot_monthly_20261005_v1/m0_cpp_support/cpp_temporal_support_by_split_season_month.csv`
- `experiments/monthly_R_station16_20261013_v1/results/new_monthly_R_seed42_cot_bank_v1/bank_manifest.json`
- `docs/CPP_V2_15MIN_PRODUCTION_20260918.md`
- `docs/CPP_V2_GAPFILL_20260925.md`

No test payload was accessed for this audit.
