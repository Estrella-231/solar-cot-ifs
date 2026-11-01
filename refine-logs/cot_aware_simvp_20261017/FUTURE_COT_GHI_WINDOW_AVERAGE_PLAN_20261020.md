# Target-window COT alignment screen

Date: 2026-10-20. The current GHI target is a 15-minute mean over `(T-15 min, T]`, while the forecast-COT feature in the target slot is a snapshot at `T`. The earlier temporal-increment calibration improved frozen-R reference RMSE slightly but did not change GHI, and a validation diagnostic found essentially zero linear association between per-instance center COT error and the paired GHI squared-error increment (Pearson r=0.011). This candidate tests temporal support alignment directly.

## Single-factor transform

Use the existing train-only ACREL reliability calibration. Replace only the target COT frame by a trapezoidal approximation to mean optical thickness over the label window:

```text
COT_window(T) = 0.5 * [COT_cal(T-15 min) + COT_cal(T)]
```

The average is computed in physical COT space, then converted to log1p and the original train-fitted normalization. The preceding frame, earlier COT history, AGRI, solar geometry, GHI labels, masks, samples, station/lead contracts, model, optimizer, learning rate, seeds and 30-epoch budget remain unchanged. At +15 min the preceding frame is the observed history endpoint; at later leads it is the previous forecast-COT frame. The run is named `ACRELWIN`.

## Evaluation and limits

Train seeds 42/43/44 concurrently on three separate A800s in one PBS job. Compare exact-key validation predictions against A/AH/AC/ACREL using 52 target-BJT date-block bootstrap intervals, both stations, every lead and the established lead bands. The primary future-COT contrast is `ACRELWIN−AH`; also report `ACRELWIN−A` and `ACRELWIN−ACREL`. This directly tests GHI compatibility with the existing 15-minute mean labels; it is not an instantaneous-COT accuracy claim. No validation COT reference is used to fit the transform or choose a coefficient. The actual-R/CPP inference inputs, test and buffer remain closed.
