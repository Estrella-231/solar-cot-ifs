# Fresh reviewer response record — 2026-10-17

Reviewer: `/root/hubei_test_freeze_audit` (fresh same-family reviewer; provisional). Read-only; no test/buffer payloads decoded; no training or server jobs started.

Verdict: **FAIL / NO-GO — keep held-out test closed.** Existing validation can be recomputed, but result lineage, checkpoint freeze, and one-time test execution are not closed.

- A labels: PASS with WARN. Station-observation `ghi15_raw` is the GT, forecast chain creates input not labels; W/m² is still an operational legacy convention without independent instrument/export-unit verification.
- B metrics/support: validation PASS, test freeze FAIL. Paired keys/truth/clear-sky and raw RMSE/MAE are valid. Exact station train-only normalizers are not stored in checkpoints; fold-specific S training manifests do not prove OOF exclusion.
- C claims/lineage: FAIL. Current same-time pack is `e8c5…` with A/AC/AP 74.361233/71.429292/73.894113; manuscript AC 71.428 derives from old `7990846…`. Paper text incorrectly says the Hubei interval and forecast-GHI evaluation are unavailable.
- D output/script freeze: FAIL. Saved predictions recompute to FA 77.249908 and FAC 78.580996 W/m², difference +1.331088. The summary uses 10,000 resamples/two-day bootstrap and per-lead CSV from a different analysis path not frozen with the evaluator. Audit does not reload and infer from checkpoints.
- E scope: WARN. Three nearby stations, 22 validation dates, three seeds; use “separately trained Hubei validation cohort,” not language implying independent test or multiple climate regions.
- F classification: PASS. Both same-time and forecast-head results use real station-observation validation GT; neither is held-out test.
- Current test/buffer reads: 0 per current candidate receipts; older touched 2025-08-01 observations were excluded from the new split. Do not claim those file bytes were never accessed.

Required before test release: pin `e8c5` lineage; serialize/hash normalization; add reload parity; bind actual summary/CI/per-lead scripts; verify fold-specific OOF train manifests; freeze arms/seeds/checkpoints/support and the purpose of reporting a negative/uncertain result; implement fail-closed test pack/inference/evaluator and synthetic/validation-only dry-run; synchronize tracker, plan, claim ledger and bilingual manuscript; request independent read-only review.
