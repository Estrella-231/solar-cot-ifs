# Sili transport-thickness loss ablation protocol (2026-09-28)

## Question and scope

Test one narrow hypothesis: does the pilot's additional physical-domain penalty on predictions above COT 10 at CPP-reference COT<5 pixels cause or contribute to the learned residual suppressing thick COT? This is a validation-led COT retrieval-reference experiment, not evidence of independent cloud truth or downstream GHI benefit. Keep the test split closed.

The observed symptom motivating this ablation is consistent across station and 64×64 patch validation: the learned residual improves frequent low-COT cases but lowers thick-cloud amplitude. It is not yet a causal result. Removing the penalty is therefore a single-factor diagnostic, not an assumed fix.

## Factor changed

Compare the existing loss against one variant with exactly one term removed:

- **Control:** masked log1p-COT SmoothL1; pixel weight 2 for CPP COT<5, weight 3 for CPP COT≥30, weight 1 otherwise; plus `0.015 * mean(max(pred_COT - 10, 0)^2 | CPP COT<5)` in physical COT units.
- **Ablation:** identical loss and weights, with only the additive `0.015` false-COT penalty omitted.

Do not change the network, inputs, residual initialization, transport, clipping, optimizer, learning rate, batch size, sample order, seed, maximum epochs, early stopping, or preprocessing. Keep the 512-sequence train and fixed 512-sequence validation cohorts and their CPP masks unchanged. Log actual optimizer updates and every loss component separately. The causal inference is limited to the effect of removing this one term under this training setup.

## Frozen candidates and paired evaluation

Evaluate all methods on the exact per-lead intersection of CPP-valid pixels and transport source support, with identical `(sequence, lead, pixel)` keys:

1. Frozen SimVP→R, persistence, and pure transported history.
2. **Zero-residual / step-0 candidate:** the model's zero-initialized output after its output clamp; verify numerically that this equals clipped transport on the shared support.
3. Every validation checkpoint from control and ablation. Select the candidate with the lowest predeclared 64×64 patch COT≥30 RMSE, with step 0 included in the candidate set. If no learned checkpoint beats step 0, retain step 0; do not select on all-COT RMSE after seeing results.

Report both 64×64 patch and station pixel (local `[32,32]`) metrics: all-valid RMSE/MAE/bias; CPP-reference COT bins 0–5, 5–10, 10–30, 30–60, 60–100; COT≥30 RMSE, COT<10 miss rate, and COT≥30 IoU; clear-reference false-COT rate (`prediction≥10 | CPP COT<5`); and all 16 leads from +15 to +240 min. Include valid counts, common-support coverage, residual sign/magnitude and the four fixed growth/decay/persistent-thick/clear case maps. Gray-mask unsupported pixels consistently in every map.

Use paired BJT-initialization-day bootstrap intervals (5,000 resamples, seed `20260928`) for ablation-minus-control and each candidate-minus-step-0 differences. The independent resampling unit is initialization day; overlapping sequence windows remain dependent within days. Include per-bin day counts and flag sparse 60–100 results rather than treating pixel count as independent sample size.

## Decision gate

The ablation supports advancing this loss change only if the selected learned candidate improves patch COT≥30 RMSE versus step 0 with a paired 95% day-bootstrap interval below zero, and the station COT≥30 result has no statistically supported deterioration. Also require that any change in the 0–5 bin and clear-reference false-COT rate is reported and does not show a statistically supported deterioration. If thick improvement is absent or comes with a supported low-COT/clear regression, keep step 0 and close this loss variant. A passing single-seed result only justifies a separately approved seed-stability experiment; it does not authorize a new GHI-head run, use of test, or paper claim.

## Provenance and execution gates

- CPP is a satellite retrieval reference, not independent in-situ cloud truth; disclose product QA/alignment limitations.
- Inputs remain historical R-COT, historical C13 motion, frozen causal SimVP→R COT trajectory, and lead. Future CPP and future AGRI are labels/reference only.
- Preserve the current pilot and all outputs. Use a new run/config/result directory; do not overwrite the best checkpoint or validation audit.
- Before any PBS submission, revalidate the bank/checkpoint/config hashes and masks, then query the complete live `slfu` PBS queue. Count running, queued, held, and suspended jobs toward the maximum of three. This protocol is not authorization to submit training; wait for explicit execution authorization after protocol review.
- Test remains closed.
