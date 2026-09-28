# Sili capacity-matched full-history COT control (2026-09-28)

## Purpose

The first full-history pilot compared an 8-channel recent-history head with a 14-channel all-history head. Its full-history arm therefore had 5.7% more parameters, and the control did not repeat the final map across the history slots. This follow-up isolates the historical values with exactly the same model and parameter count.

## Frozen comparison

- **Full history:** provide the eight actual historical COT maps.
- **Repeated-last control:** put eight copies of the final historical COT map in those same slots.
- Both arms also receive the same explicit last-minus-previous COT trend, transported COT, frozen SimVP→R forecast COT, forecast lead, and local motion.
- Both use one shared 15-channel Conv3d model (same weights/shapes/parameter count), identical zero-residual initialization mapped from the current head, optimizer, objective, seed 42, update cap, batch, sample order, fixed 512/512 bank, masks, and evaluation keys. A numerical witness requires each arm's initial output to equal the original short-history head within 1e-6.
- Loss is the penalty-off weighted log1p SmoothL1 used in the prior loss ablation. The extra CPP COT<5 false-COT penalty remains off in both arms.

## Selection and reporting

Keep step-0 pure transport as an explicit candidate in each arm. Select by minimum 64×64 patch CPP-reference COT≥30 RMSE. Save both the selected candidate and the best learned candidate even when step 0 wins. Retain every 16-update validation snapshot, per-day sufficient statistics, patch/station aggregate and 16-lead metrics. Compare both learned candidates and the selected candidates with paired initialization-day bootstrap (5,000 draws, seed 20260928), stating that checkpoint selection and intervals reuse validation and are descriptive/post-selection.

This can resolve whether older COT maps add useful information to this residual head. It cannot establish independent cloud truth or GHI benefit. Do not open test or train a GHI head unless the selected full-history model beats step-0 transport on thick COT and has no supported station/low-COT regression.

## Execution

Use one A800, profile through the full 512-sequence cohort batch (the largest possible batch without replacement), reserve at least 10 GiB VRAM, and check all `slfu` PBS jobs immediately before submission. Copy only the fixed bank to node-local scratch; keep outputs in a new versioned result directory.
