# Hubei R-bank to CPP forecast-key overlap audit

State: `PASS_MANIFEST_ALIGNMENT_TEST_BUFFER_CLOSED`

This is a metadata-only join of the frozen R-derived 16x16 COT bank, the frozen train/validation AGRI index, and the existing corrected-UTC CPP reference manifest. No arrays, labels, test, or buffer were opened.

- R-bank rows: 22061; AGRI rows: 22061; exact key/split mismatches: 0
- CPP manifest: {'train_rows': 3178, 'train_full_support': 3156, 'train_zero_support': 22, 'val_rows': 301, 'val_full_support': 301}
- Train-history COT maps are frozen-R predictions on its own training AGRI frames (not OOF); validation maps are held-out R predictions.
- CPP targets are the existing station-local 16x16 retrieval reference, not independent cloud truth.

| Split | Lead | Scorable init/target pairs | Init dates | Target dates |
|---|---:|---:|---:|---:|
| train | +15 | 3020 | 153 | 153 |
| train | +30 | 3016 | 153 | 153 |
| train | +45 | 3019 | 153 | 153 |
| train | +60 | 3019 | 153 | 153 |
| train | +75 | 3017 | 153 | 153 |
| train | +90 | 3023 | 153 | 153 |
| train | +105 | 3023 | 153 | 153 |
| train | +120 | 3023 | 153 | 153 |
| train | +135 | 3021 | 153 | 153 |
| train | +150 | 3023 | 153 | 153 |
| train | +165 | 3019 | 153 | 153 |
| train | +180 | 3005 | 153 | 153 |
| train | +195 | 2976 | 153 | 153 |
| train | +210 | 2946 | 153 | 153 |
| train | +225 | 2913 | 153 | 153 |
| train | +240 | 2869 | 153 | 153 |
| val | +15 | 298 | 16 | 16 |
| val | +30 | 298 | 16 | 16 |
| val | +45 | 298 | 16 | 16 |
| val | +60 | 297 | 16 | 16 |
| val | +75 | 297 | 16 | 16 |
| val | +90 | 297 | 16 | 16 |
| val | +105 | 297 | 16 | 16 |
| val | +120 | 297 | 16 | 16 |
| val | +135 | 297 | 16 | 16 |
| val | +150 | 295 | 16 | 16 |
| val | +165 | 293 | 16 | 16 |
| val | +180 | 291 | 16 | 16 |
| val | +195 | 289 | 16 | 16 |
| val | +210 | 287 | 16 | 16 |
| val | +225 | 284 | 16 | 16 |
| val | +240 | 277 | 16 | 16 |

These counts establish only metadata coverage. They do not prove CPP file/QC support beyond the existing sidecar, forecast skill, or adequacy of date-independent validation. This audit does not authorize model training.
