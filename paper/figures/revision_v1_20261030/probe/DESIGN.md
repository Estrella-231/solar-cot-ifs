# Conditional-information diagnosis: design card

## Purpose

Answer two distinct questions: does a future-COT probe beat the existing forecast baseline, and does replacing the COT source preserve the information usable by a fixed fitted mapping? This is a development-date out-of-fold diagnosis, not a main independent forecasting comparison. Place it in the explanatory diagnosis or supplementary material after the real paired forecast results.

## Original-paper lessons

Papachristopoulou et al. (2024), Fig.11 pairs absolute error with improvement relative to a baseline. We combine those in a compact table-and-interval panel rather than allocating two nearly redundant full panels. Their Fig.15 motivates a conditional/horizon diagnosis; our horizon panel keeps paired uncertainty and explicitly identifies the comparison. Chen et al. (2023), Table3 motivates retaining the strong baseline and the relevant controls together, so a small gain over a weak control cannot masquerade as a method improvement.

## Layout

(a) A shared row structure names all four models, reports precise RMSE and places paired differences with date-cluster intervals on one horizontal scale. (b) A 2×2 source-substitution matrix directly separates fitting source from inference source; the observed-future inference column is outlined and explicitly offline. (c) The four hourly intervals show how usable observed-image and forecast-image COT increments differ relative to the same context probe.

This retains the old four-panel content with less repetition and larger labels. The source matrix is not a theoretical physical response and its absolute colors do not imply probability. No smooth curves are fitted between horizon aggregates. Markers and line styles distinguish sources even without color.

## Locked data and uncertainty

All 32,838 station–initialization–target rows, 50 dates, three stations and three original frozen GHI model sources are unchanged. Scores are recalculated from saved daily SSE/counts. Each station×baseline-source RMSE is computed first and averaged equally. Paired 95% percentile intervals use 5,000 target-date-cluster bootstrap replicates with the original fixed seed. These dates belong to development data; the deterministic probes use date-block out-of-fold fitting, not an independent held-out test. Actual-image COT is a frozen retrieval proxy and requires unavailable future observations during inference.

## Intended size

Provisional manuscript or supplement figure, 177×135 mm. Export PDF/SVG and 300 dpi PNG, inspect the actual outputs and preserve a printed-size preview. No journal compliance is asserted.
