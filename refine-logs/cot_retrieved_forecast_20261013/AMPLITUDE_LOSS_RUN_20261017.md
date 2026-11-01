# Physical-amplitude COT loss screen

Date: 2026-10-17. Validation-only, one seed; CPP is a retrieval reference, not independent truth. Test payloads remain closed.

## Frozen comparison

The only changed factor was adding a fixed physical-amplitude term to the P-Hybrid objective:

```text
base: equal-lead masked MSE(log1p(COT)/log(101))
screen: base + 0.1 × equal-lead masked MSE((COT_prediction − COT_target)/100)
```

The architecture, hybrid targets, masks, monthly split, training population, seed 42 initialization, batch 1,024, 12-epoch budget, optimizer, learning-rate schedule, sample order, and checkpoint-selection rule followed the completed P-Hybrid screen. This was one screening seed and did not train a GHI head.

## Execution and scoring

PBS `214511.tc6000` completed normally (exit status 0) on one node22 A800-SXM4-80GB after 12 epochs. The predeclared station-by-lead validation score selected epoch 11. Independent FP64 scoring used the exact same 3,140 validation sequences, both stations, all 16 target leads (+15 to +240 min), compact CPP reference, and `CPP-common × new-R-frame-valid` mask as the baseline comparison. Sequence IDs and every target timestamp matched exactly. The prediction hash is `deecf646de44881d138a758ae4abdaf254a176b308ee295759dab46878a7994c`.

## Result

| Candidate | Full 16×16 RMSE | Local 3×3 RMSE |
|---|---:|---:|
| Original SimVP forecast-AGRI → frozen R | 10.953 | 10.664 |
| P-Hybrid, base log-space loss | 12.407 | 12.162 |
| P-Hybrid + fixed physical-amplitude term | 12.473 | 12.209 |

Relative to the base P-Hybrid, the added term worsened the overall score by 0.066 COT on the full patch and 0.046 on the local 3×3. Relative to the original SimVP→R, it remained worse by 1.520 and 1.545. On the CPP COT≥30 stratum, it improved RMSE only slightly versus P-Hybrid (−0.133 full patch; −0.358 local 3×3), while worsening COT<1 (+0.160/+0.197) and COT 5–30 (+0.182/+0.213). Thus, the amplitude penalty did not repair the overall deficit; the small thick-bin change does not pass the predeclared gate.

This does not prove that the residual error is purely amplitude-related: cloud placement, temporal phase, retrieval/reference disagreement, and optimization remain possible contributors. The current test supports only a one-seed validation screening statement.

## Decision and provenance

RC12 fails its overall-score criterion. Stop this direct-COT loss-tuning branch; do not continue with more weights/seeds, a GHI-head run, or test access under this result. The next paper work should characterize the existing forecast error by event timing/position versus thickness evolution, using the frozen event list and common support, before considering a different forecast formulation.

- Independent scorer: `analyze_fulltrain_vs_simvp_same_support_20261017.py`
- Machine-readable same-support scores: `results/simvp_vs_direct_cot_strata_20261017.json`
- Full-training baseline report: `FULLTRAIN_RESULTS_20261017.md`
- Test-payload state: `test_payload_opened=false`
