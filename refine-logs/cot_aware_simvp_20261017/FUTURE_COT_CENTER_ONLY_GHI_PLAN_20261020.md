# Future-COT center-only GHI diagnostic

Date: 2026-10-20. This is a new, predeclared downstream ablation after the spatial-contrast calibration failed its training-fold COT gate.

## Question

Does the spatial variation in forecast future COT's 3×3 station patch help the GHI head, or does retaining only the calibrated center value improve the forecast? Existing diagnostics found strong center agreement but weak neighbor-pattern agreement. A separate train-only spatial-contrast scaling failed to reduce cross-fitted patch COT error, so this experiment directly tests downstream value instead of tuning another spatial coefficient.

## Single change

Starting from the same train-only ACREL station/lead calibration, for future COT frames only replace all eight 3×3 neighbors with that frame's already-calibrated center value. Keep center COT, every historical COT frame, AGRI, geometry, GHI labels, mask, exact rows, IntervalHead, normalization, initialization, optimizer, row order, batch size and 30-epoch budget unchanged. The candidate is named `ACRELCENTER`; the existing ACREL predictions are the paired reference.

## Compute and evaluation

Run seeds 42, 43 and 44 concurrently, one independent process per A800, inside one three-GPU PBS allocation. Require exact common station/init/target/lead/GHI-row keys and identical truth arrays. The primary paired comparisons are ACRELCENTER−AH and ACRELCENTER−ACREL; also report ACRELCENTER−A, both stations, each lead and +15–60, +75–120 and +135–240 minute bands. Use the frozen 52 target-BJT-date paired bootstrap and the existing screening gate. No validation-selected threshold or alternate spatial scale is permitted.

This is a validation diagnostic on five-fold OOF training forecasts and frozen train-only validation forecasts. It does not open test/buffer payloads or use future observed AGRI/CPP as inference input. A favorable result diagnoses forecast-COT spatial-detail utility; it is not by itself an independent-test paper claim.
