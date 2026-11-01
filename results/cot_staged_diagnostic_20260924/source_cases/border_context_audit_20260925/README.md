# Validation COT border and same-pixel recrop audit

Scope: Hunan Sili/Zhujia, 6,536 saved validation frames, real observed AGRI only. The staged retrieval weights are frozen. CPP COT is the retrieval reference in physical COT units, not an independent cloud measurement. Near-zero means valid CPP COT <0.1. No test data, new training, or GHI evaluation was used.

## 1. Is the 16×16 border associated with false cloud detections?

Across 648,648 valid CPP near-zero pixels, the outermost ring has 12,753/153,582 false-cloud classifications (8.30%). Pixels at least four cells from an edge have 7,308/164,722 (4.44%). The near-zero to predicted-COT≥10 rate is 0.160% on the outer ring and 0.059% in the inner region. The latter is rare; the four previously selected most severe errors are not a random sample of all false-cloud errors.

Immediate proximity to CPP cloud is a strong factor. For CPP near-zero pixels adjacent to a CPP COT≥0.1 pixel, false-cloud rates are 37.85% on the outer ring and 20.25% in the inner region. Without such adjacent CPP cloud, the respective rates are 3.37% and 0.96%. The border association remains when separating these two contexts. Direct standardization over 20 shared station × broad cosSOZ × adjacency strata gives outer 9.13% versus other pixels 5.02% for false-cloud classification, and outer 0.182% versus other pixels 0.088% for predicted COT≥10. These are descriptive, since reference cloud texture, geometry and source registration may still differ within a stratum.

![Validation border and cloud-context rates](border_context_rates.png)

Source: `border_context_audit.json` and `border_context_rates.csv`. The original saved-model validation row ids, masks and reference COT values are checked before aggregation. `scripts/audit_cot_patch_border_20260925.py` reproduces this analysis.

## 2. What happens if the same AGRI pixel is moved within a 16×16 crop?

The four selected CPP-zero pixels were re-extracted from their original observed 20-channel AGRI frames. For each variant, the exact same geographic/source pixel is sampled at a different local row in a 16×16 window. All 16 target-pixel model input values are bit-identical to the original packed input; the frozen retriever, classifier and regressors are identical. The window's surrounding pixels change when the window moves. CPU replay of the original crop agrees with saved GPU probability to at most 0.0000044 and saved final COT to at most 0.012.

| Case | Original row → shifted row | CPP reference COT | P(cloud), original → shifted | Final COT, original → shifted |
|---|---|---:|---:|---:|
| 1 | 0 → 8 | 0 | 0.9875 → 0.9844 | 36.93 → 30.17 |
| 2 | 0 → 8 | 0 | 0.9873 → 0.9829 | 36.58 → 27.22 |
| 3 | 8 → 0 | 0 | 0.9859 → 0.9894 | 35.27 → 46.06 |
| 4 | 0 → 8 | 0 | 0.9858 → 0.9764 | 32.40 → 18.55 |

In every recrop the classifier remains highly confident that these CPP-clear pixels are cloudy. Moving the three outer-row pixels toward the center reduces the predicted thickness by 6.8–13.8 COT units, while moving the interior pixel to the border increases it by 10.8. The model is sensitive to crop context/position for COT magnitude, but shifting the crop alone does **not** resolve this classification mismatch. Because the surrounding 16×16 scene changes, this experiment cannot isolate padding as the sole cause.

![Actual AGRI pseudo-RGB for the same-pixel recrops](same_pixel_recrop_rgb.png)

Red × marks the same source pixel in each row of the figure. Colors use one shared validation-wide stretch (R=C03, G=C02, B=C01). Detailed values, source pixel indices and checkpoint hashes are in `same_pixel_recrop_validation.json`. Reproduction: `scripts/audit_cot_recenter_same_pixel_20260925.py` on ZJNU CPU with existing frozen weights, then `scripts/plot_cot_same_pixel_recrop_20260925.py` locally.

## Interpretation and next check

The outer ring has a larger error rate even after coarse matching for station, solar angle and immediate CPP cloud adjacency. The four severe cases retain high cloud probability after recentering, so the evidence points to **both** edge/context sensitivity of the predicted thickness and unresolved pixel-level clear/cloud discrimination. It does not prove a padding bug or that CPP is physically correct at these pixels. The next targeted check is the original AGRI↔CPP spatial/scan registration around representative border and interior errors, plus correctly identified clear pixels in comparable cloud scenes.
