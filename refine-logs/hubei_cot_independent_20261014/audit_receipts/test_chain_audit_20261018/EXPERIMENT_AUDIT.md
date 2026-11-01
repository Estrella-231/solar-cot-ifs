# Hubei frozen-test chain review — 2026-10-18

**Verdict: FAIL / NO-GO for actual independent-test scoring.** Review was read-only; no real test/buffer values were inspected.

## Main findings

- Ground-truth provenance is not independent: the synthetic scorer currently reads truth from the same NPZ as predictions.
- There is no complete fail-closed builder → inference → scorer workflow or one-use ledger. The current manifest SHA strings are not verified against actual files by the scorer.
- The complete paper metric set from the frozen plan is not implemented (including per-lead, bias/R², common-support and clear-sky sensitivities, and block-bootstrap reporting); non-finite bootstrap groups are silently skipped.
- At review time the synthetic output was mislabeled as an independent real test. It is now corrected to `SYNTHETIC_CONTRACT_TEST_PASSED` / `synthetic_proxy`.
- The review rerun caught stale test fixtures passing digest strings where the evaluator requires a split `Path` (1/11 passed, 10 errored). Fixtures now write a temporary frozen split CSV with its actual SHA; fail-closed tests instrument `np.load` to ensure invalid release/split gates reject before opening bundles.

## Current verification

After the above limited software-only repair, `python -m unittest -v test_hubei_forecast_heads_frozen_test.py` passes **11/11**. This is only a synthetic software-contract witness; it is not scientific test evidence and does not change test/buffer sealing.

## Required before any real test access

Build and independently audit a test pack with separately sourced GHI truth; bind exact split, source files, test pack, model checkpoints, normalization, trainer and inference code by actual path and SHA; emit a complete inference receipt; enforce exact keys and all frozen metrics; implement an atomic one-use ledger; then obtain a fresh independent release review. Keep test/buffer closed until every condition is satisfied.

Full reviewer trace: [`.aris/traces/experiment-audit/2026-10-18_run01/`](../../../../../.aris/traces/experiment-audit/2026-10-18_run01/).
