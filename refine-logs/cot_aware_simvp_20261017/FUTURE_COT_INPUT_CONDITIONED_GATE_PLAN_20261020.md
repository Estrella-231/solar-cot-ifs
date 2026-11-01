# Input-conditioned future-COT gate experiment (2026-10-20)

## Motivation

The validation-only forecast-domain audit found that AGRI forecast error and frozen-R future-COT retrieval error co-vary (overall Pearson r=0.306 at Sili and r=0.443 at Zhujia). Channel-wise association was strongest for C01/C02/C03 (r=0.404/0.428/0.393), while C04 had the largest standardized RMSE (0.706) but weak COT-error association (r=0.159). These are descriptive results from 66,590 validation rows and are not used to fit weights or thresholds. They motivate an input-conditioned trust factor rather than uniform COT shrinkage.

## Single-factor candidate

Start from the existing train-only, fold-cross-fitted ACREL future-COT calibration and the unchanged `IntervalHead`. Add a small gate that predicts one multiplier for each of the three AGRI/COT context frames from only inference-available features: normalized AGRI center/3x3 mean/3x3 spread, solar-geometry center/mean, calibrated COT center/mean/spread, and forecast lead. Apply it only to future COT anomalies; observed-history COT is unchanged. The multiplier is `1 + tanh(z)` in [0,2], with the final linear layer zero-initialized, so the model initially reproduces ACREL exactly and can learn either suppression or amplification. The loss remains the established 15-minute GHI/kt objective. No CPP, future observed AGRI, test, or buffer content is an inference input.

## Fixed training and comparisons

- Same five-fold OOF forecast input pack, train-only calibration, keys, labels, masks, station heads, optimizer, batch size 1,792, 30-epoch budget, and epoch selection rule as the existing ACREL run.
- Seeds 42/43/44 run independently and concurrently, one process per distinct A800 GPU in one PBS allocation. No DataParallel or DDP overhead.
- Compare the new `ACRELGATE` predictions against the already saved A, AH, AC, and ACREL predictions using exact `(station, init, target, lead, GHI-row)` keys and the existing target-BJT-date-block paired bootstrap.
- Primary application gate: ACRELGATE−AH must have a negative point estimate and a 95% interval wholly below zero; both station point differences and the +15–60-minute equal-station difference must be non-positive. To attribute improvement to the new conditional gate, ACRELGATE−ACREL must also be negative with its 95% interval wholly below zero. Otherwise retain as a screened candidate without claiming a gain.

## Boundaries

This is validation screening on five-fold-OOF training inputs and the fixed full-train validation forecast pack. It is not independent-test evidence. The validation-domain audit only motivated the hypothesis; no validation-derived channel weights, thresholds, or calibration parameters are used. Paper正文 is unchanged.
