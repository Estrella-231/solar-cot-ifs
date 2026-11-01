# COT retrieval-quality figure: source and scoring record

Updated: 2026-10-17. This analysis supports manuscript Figure 3 and Results §5.1. It uses validation predictions only; no test payloads were opened and no training was started.

## Comparison and support

The comparison is frozen-R retrieved COT against the source-key-matched FY-4B CPP COT retrieval reference. The scores are agreement diagnostics, not independent physical validation. A pixel is scored only when both prediction and CPP reference are valid. Time support is assigned from the audited source-key mapping to BJT target dates; UTC date strings are not treated as BJT dates.

Hunan uses two station-centred 16×16 patches (Sili and Zhujia), with pixel-level common support pooled for RMSE/MAE/bias and station-equal RMSE reported separately. Its support is 356,169 pixels from 1,398 station-time patches across 46 BJT target dates. Scores: RMSE 9.074, MAE 4.646, bias +0.279 COT; station-equal RMSE 9.053.

Hubei uses one shared 16×16 regional patch covering Niushou, Wangzhai and Mizhuang, not three station-specific patches. Its support is 66,304 pixels from 259 validation frames across 16 BJT target dates. Scores: RMSE 13.357, MAE 9.465, bias −1.303 COT. Hunan and Hubei supports and CPP-supervision contracts differ; these scores are not a controlled regional ranking.

For the pre-existing CPP-reference strata, the mean retrieved-minus-reference bias in the ≥30 bin is −8.719 (95% BJT date-cluster interval [−11.533, −5.588]; 50,924 pixels, 34 dates) in Hunan and −14.109 ([−19.487, −9.022]; 15,869 pixels, 12 dates) in Hubei. These are descriptive, unadjusted subgroup results; Hubei has sparse date support.

## Reproduction

Run `python paper/figures/plot_fig3_retrieval_quality.py` from the repository root. The script writes PDF, PNG, SVG, summary JSON, and strata CSV beside itself. The JSON records source SHA-256 hashes, paired bootstrap settings (2,000 BJT-date resamples; seed 20261017), support sizes, and the no-test-payload boundary. The figure uses logarithmic pixel-count colour scaling and date-cluster intervals; it does not imply that CPP is ground truth.

## Date-support clarification

The older Hunan retrieval-validation summary reported 57 date labels from raw time-key date components. For Figure 3, source keys were joined to the audited `target_time_bjt` mapping and only dates with common valid CPP-reference pixels were counted; this yields 46 BJT target dates. Figure 3 and the manuscript use the latter count. This distinction is recorded to prevent UTC calendar dates from being presented as BJT dates.
