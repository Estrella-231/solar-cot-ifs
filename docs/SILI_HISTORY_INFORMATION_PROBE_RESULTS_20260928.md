# Sili full-history COT information probe (2026-09-28)

## Question and setup

Does adding the six older COT maps provide useful validation information beyond the current head's recent two-frame COT representation, after conditioning on the same SimVP→R forecast COT, local motion, lead, final COT, and last-step COT change?

The probe fits two weighted linear ridge regressors on the fixed 512-sequence training bank and evaluates them on the fixed 512-sequence validation bank. Both use the same CPP-valid pixels intersected with the 64×64 transport source support. It scores a fixed 16×16 lattice over the local patch plus the exact station pixel. The target is log1p CPP-reference COT; weights follow the pilot's regime weighting. The `full8` model adds historical frames 1–6; the `recent2` model uses the last frame and last-minus-previous frame. This is a low-cost information probe, not a neural model comparison.

## Results

| Validation scope / metric | Recent two frames | All eight frames | Full-history minus recent |
|---|---:|---:|---:|
| Lattice all-COT RMSE | 8.942 | 8.936 | −0.007 |
| Lattice COT≥30 RMSE | 28.991 | 28.948 | −0.042 |
| Lattice COT≥30 IoU | 0.1502 | 0.1490 | −0.0012 |
| Lattice thick miss rate (prediction<10) | 33.33% | 32.98% | −0.35 pp |
| Lattice false-COT rate (CPP<5, prediction≥10) | 2.594% | 2.655% | +0.061 pp |
| Exact station pixel all-COT RMSE | 7.644 | 7.626 | −0.017 |
| Exact station pixel COT≥30 RMSE | 29.788 | 29.656 | −0.132 |
| Exact station thick records | 204 | 204 | same rows |

The lead-wise thick-RMSE curves almost overlap. On the exact station pixel, the 204 thick records are sparse and occur on few dates; their small differences are descriptive only. The probe does not estimate an uncertainty interval.

![Lead-wise and station-pixel thick-COT probe results](../results/regional_station_cot_20260927/sili_history_information_probe_20260928_v2/history_information_probe.png)

## Interpretation and decision

Older COT history adds almost no signal to this linear conditional probe. The result weakens the simple hypothesis that the current thick-cloud miss is caused only by discarding six older COT frames. It does not rule out a learned nonlinear or spatial-temporal representation, and it does not establish that future COT cannot improve GHI.

The matched fixed-seed neural pilot has completed as PBS `211083.tc6000` (`Exit_status=0`) on one node22 A800. It compares the current two-frame representation against all eight COT frames with the same train/validation bank, step-0 transport candidate, loss, update budget, initialization-function witness, and paired initialization-day bootstrap. Both arms selected step-0 transport. The best unselected full-history checkpoint improved patch COT≥30 RMSE relative to the best short-history checkpoint (26.938 versus 27.313; paired day-bootstrap interval for full minus short `[-0.594, -0.172]`), but remained above transport (26.504; interval `[-0.199, 1.096]`) and its exact station thick-COT RMSE was worse than transport (27.148 versus 25.163; interval `[+0.280, +3.420]`). See `docs/SILI_FULL_HISTORY_PILOT_RESULTS_20260928.md` for the complete regime metrics and post-selection caveat. The input stem grows from 8 to 14 channels in the full-history arm, so the total parameter count is slightly larger; interpret this as a representation comparison with a small capacity difference, not an exactly parameter-matched result.

The neural pilot did not pass the gate. Retain pure transport and stop adding historical COT channels to this pointwise residual head. The next useful work is to distinguish event displacement/timing from thickness evolution and CPP retrieval mismatch on validation cases; only then define another one-factor candidate. Test remains closed.

## Provenance

- Fixed bank: remote `experiments/regional_station_cot_20260925/sili_cot_dynamics_bank_train512_val512_20260927_v1/`; `complete.json` SHA-256 `7a7df500b1ac8e9606c66b0a64c35846907e0e2ac5294d2295076d9c36c41c0e`; test not used.
- Probe source SHA-256 `8364b350eff4adadc17b4797abcf2c8d7040a3cdfa5ae4ebafe2ccfc70835890`.
- PBS probe: `211080.tc6000` (`normal`, node12, 8 CPU cores, 16 GB); completed and emitted `SILI_HISTORY_INFORMATION_PROBE_COMPLETE`.
- Remote output: `experiments/regional_station_cot_20260925/sili_history_information_probe_20260928_v2/`.
- Local arrays, summary, and figure: `results/regional_station_cot_20260927/sili_history_information_probe_20260928_v2/`.
- First attempt `211078.tc6000` stopped before scoring on a NumPy axis-order error. It produced no metrics; the corrected v2 used a fresh output path. Do not use the failed attempt as evidence.
- CPP is a satellite retrieval reference rather than independent cloud truth; sequences overlap in time and sampled pixels are not independent observations.
