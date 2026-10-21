# Train-only monthly R-COT input-origin shift

This descriptive CPU audit groups the existing date-held-out R-vs-full-R comparison by BJT month. It reads only the frozen train-only `per_frame.csv` audit output and its scope summary. No AGRI, CPP target/QC, validation prediction, test, or buffer payload is read.

The values below use only common daytime pixels for which both R outputs are finite and in [0,100]. Means and disagreement RMSE are weighted by the number of such pixel pairs in each frame. Frames overlap in time and are not independent samples; no inferential interval is reported.

Aggregation check against the saved train-only analysis summary passed: 2,545,664 pixel pairs; full-R mean 12.631678; date-held-out R mean 11.256024; pooled map-disagreement RMSE 5.967576 COT.

| BJT month | Train dates | Frames | Pixel pairs | Full-R mean | Date-held-out R mean | OOF − full | Map-disagreement RMSE | Full-R >100 (%) | OOF-R >100 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 202508 | 23 | 1205 | 302,764 | 6.92 | 6.14 | -0.78 | 4.79 | 1.351 | 0.757 |
| 202509 | 22 | 1075 | 273,757 | 19.46 | 16.32 | -3.14 | 7.81 | 0.081 | 0.000 |
| 202510 | 23 | 1030 | 260,290 | 20.91 | 17.23 | -3.68 | 9.04 | 0.677 | 0.000 |
| 202511 | 22 | 880 | 224,456 | 5.03 | 4.77 | -0.26 | 3.41 | 0.141 | 0.106 |
| 202512 | 23 | 922 | 231,329 | 10.63 | 9.74 | -0.89 | 6.48 | 0.002 | 0.000 |
| 202601 | 23 | 941 | 239,049 | 12.90 | 11.58 | -1.32 | 5.29 | 0.000 | 0.000 |
| 202602 | 19 | 820 | 209,138 | 10.92 | 11.70 | +0.78 | 5.11 | 0.017 | 0.000 |
| 202603 | 24 | 1111 | 283,187 | 15.24 | 14.44 | -0.79 | 5.81 | 0.165 | 0.000 |
| 202604 | 23 | 1167 | 297,841 | 9.84 | 8.75 | -1.09 | 4.37 | 0.000 | 0.000 |
| 202605 | 18 | 878 | 223,853 | 13.81 | 11.66 | -2.15 | 5.19 | 0.000 | 0.000 |

## Interpretation boundary

The source gap varies by month: September–October show the largest negative OOF-minus-full mean differences, while February reverses sign in this train-only comparison. This is a seasonal feature-distribution warning for downstream models trained on full-R in-sample histories and applied to held-out R histories. It does not show which R output is physically more accurate, does not measure future-COT forecast skill, and does not prove that this shift caused the validation result.

The monthly validation cohort remains frozen for tuning. The next model experiment, if later unlocked by the source/plan gate, should train its COT forecaster with date-held-out R histories on train dates and preserve one fixed monthly-validation evaluation; do not fit monthly offsets or weights to these validation dates.

Source audit state: COMPLETE_OOF_HISTORY_INPUT_DISTRIBUTION_DIAGNOSTIC; train rows=20206; OOF rows=20206; validation rows inferred=0; test/buffer/CPP-target/QC/GHI-label reads=0.
CSV: `monthly_shift.csv`. Reproduce with `python analyze_train_only_monthly_shift.py` from this directory.
