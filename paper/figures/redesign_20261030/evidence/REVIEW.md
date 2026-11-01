# Figure review — 2026-10-30

## Evidence

- Re-read actual 36 saved prediction files; no log-only or rounded-table reconstruction.
- Exact equality checked for station/init/target/lead keys, labels and clear-sky values across all four arms and three seeds.
- Verified paired initialization, sample-order, update-count and parameter-count receipts for the two controlled comparisons; verified frozen bases were unchanged.
- Independently reproduced all four aggregate RMSEs and all five existing overall bootstrap comparison intervals to absolute tolerance 1e-10 W m⁻².
- Additional station and lead intervals use the same target-date clusters, 5,000 draws and random seed as the original score; they are explicitly pointwise descriptive intervals.
- No independent-test, causal-mechanism or significance claim is added by plotting.

## Visual checks and revisions

1. Rendered 350-dpi PNG and opened it. The initial in-axis legend overlapped the error curves; moved the shared four-arm legend outside the plotting panels.
2. Rendered again. The initially added bottom numeric annotations crowded the x-axis labels after legend space was reserved; removed those annotations rather than shrinking fonts. Exact values remain in the paired CSV and caption text.
3. Exported the PDF through Poppler at 120 dpi and visually inspected the result (`forecast_ghi_print_preview.png`). All 16 leads, uncertainty boundaries, station names, unit exponents, panel labels and zero references remain visible. No axis label or legend overlaps data. Editable SVG text is retained; PDF uses embedded TrueType text.
4. Harmonized the capacity-control color to blue (`#4574A0`); gray remains no COT, teal historical COT, and orange forecast COT. Purple is reserved elsewhere in the figure set for actual-future COT used offline. Regenerated all exports and the source manifest; numerical results are unchanged.

## Improvements over the old forecast chart

- The outcome is directly GHI error, not a COT-only surrogate.
- Equal-capacity correction control isolates the future-COT increment from adding a correction head.
- Whole-cohort curves and all three stations are shown, including non-improvement.
- Weather-date uncertainty and seed variation are distinct visual encodings.
- Numeric scale makes the forecast-COT effect (~0.026 W m⁻²) distinguishable from historical-COT point change (~−1.925 W m⁻²), without displacing nearly coincident curves.
- Legends use input descriptions; no internal experiment IDs are printed.
- Figure provenance, source data summaries and a repeatable plotting script accompany the vector output.

No manuscript source, original figure, experiment result or raw data was modified.
