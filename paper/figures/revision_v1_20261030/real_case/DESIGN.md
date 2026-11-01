# Real cloud event: design card, 30 October 2026

## Scientific purpose and placement

Show how loss of a cloud-related signal in forecast imagery coincides with a different retrieved-COT trajectory and a missed GHI decrease under a fixed residual mapping. Place this figure in the forecast-information diagnosis, after the population-level paired forecast evaluation. This is an explanatory development-date out-of-fold event, not an independent forecast result or proof of a unique physical cause.

## Original-paper choices adopted

- Chen et al. (2023), Fig.2: display actual multispectral imagery so the reader can inspect what entered the method. Retain native pixels rather than inventing texture.
- Chen et al. (2023), Fig.6 and Fig.11: connect a meteorological situation to a time series and retain the failure of the model to reproduce observations. Do not use our retrieval as independent cloud truth or use a single event to estimate population benefit.
- Papachristopoulou et al. (2024), Fig.15: compare changing conditions against the response of the method. Here the optical-thickness proxy and GHI have aligned time axes; no theory curve or causal attribution is inferred.

## Changes from the first draft

The image block is grouped by source: observed AGRI next to its R-COT, then SimVP AGRI next to its R-COT. This makes the AGRI-to-retrieval relationship visible. Two source groups use distinct headings and explicit offline status. Below, station COT and GHI occupy aligned full-width time panels, avoiding the duplicated short time axes in the draft. The three image times are marked on both series. A small aligned 3×3 box shows the actual GHI input support inside each 16×16 retrieval image.

## Data and transformation lock

Case, station order, three displayed frames, all 16 trajectory times, masks and scores are unchanged. The selected initialization is 2025-08-06 12:00 BJT; the source CASE_SCOPE.json specifies the earliest eligible calendar date and closest-noon initialization without using errors. Niushou is the first station in the fixed station order. Images are C03/C02/C01, with the identical whole-image square-root display mapping of reflectance 0–1 for both sources. There is no per-image contrast adjustment, spatial interpolation or synthesized detail. R-COT uses a common linear 0–140 scale; outputs above 100 remain visible and are identified as retrieval extrapolation rather than independently verified thickness.

GHI is the target-time-ending 15-minute mean. Curves average the three existing frozen-baseline source variants for display only. Missing scored GHI remains missing. Purple future observations are offline inputs. The gray baseline already includes historical COT and must not be labelled “no COT.” Source and data hashes are recorded in PROVENANCE.json.

## Physical-size and revision checks

Provisional full-width manuscript figure: 177 × 190 mm, PDF/SVG plus 300 dpi PNG. Font sizes 8–9.5 pt. Check the exported PNG and PDF render for labels, shared limits, source grouping, axis alignment and missing intervals. Journal-specific requirements are not asserted.
