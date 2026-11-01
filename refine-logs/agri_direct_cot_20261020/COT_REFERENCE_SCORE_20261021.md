# Direct future-COT model: frozen CPP-reference diagnostic

**Scope:** outer-validation development data, not independent test evidence. CPP is a retrieval reference, not independent truth. Test/buffer numeric payload remained closed; future observed AGRI and CPP were not model inputs.

The completed U-only P-C and P-XC predictors were scored against the original supervised CPP validation reference, using the pre-existing causal H/V forecast outputs and the fixed 16×16, 3×3-neighbourhood, and centre-pixel masks. The scorer matched 20,500 forecast-pixel records across 1,362 unique station-target keys (16.9% of all forecast keys) and 23 of the 26 frozen monthly two-day blocks. The limited coverage reflects CPP-supervised reference availability, not model missingness.

At the station 3×3 scale, P-XC had COT RMSE 11.77 versus 12.82 for SimVP→R and 12.31 for P-C. Relative to SimVP→R, the paired point difference was −1.055 COT units (95% two-day-block interval [−2.032, +0.082]); thick-cloud CSI (COT≥30) differed by +0.017 ([−0.074, +0.113]). P-XC reduced COT-only thick-cloud underestimation compared with SimVP→R, but its absolute thick-cloud bias was 2.46 units higher ([−2.14, +7.18]). The intervals cross zero, so this is a promising but inconclusive COT-skill signal.

Against P-C, P-XC reduced station-neighbourhood RMSE by 0.547 units ([−2.020, +0.841]) and increased thick-cloud CSI by 0.052 ([−0.007, +0.105]); the absolute thick-cloud-bias difference was −5.45 ([−7.63, −2.96]). This suggests historical AGRI adds useful COT information beyond historical COT alone, but it does not establish statistically resolved improvement across the limited CPP-supported dates.

The downstream GHI comparisons remain negative/uncertain: the direct P-XC input did not pass the frozen GHI gate, and the lead-conditioned H-only CV also did not pass. Therefore better COT point metrics alone have not translated to a dependable GHI gain. The next experiment should test whether the GHI head needs the *trajectory* of predicted COT (recent thickening/thinning and cloud arrival) rather than only the COT map at the target lead.

![COT reference score and paired intervals](results/direct_cot_cpp_validation_score_v1_local4060/direct_cot_cpp_reference_summary.png)

Machine-readable results: `results/direct_cot_cpp_validation_score_v1_local4060/cot_cpp_reference_metrics.json`. The scoring script verified the immutable CPP-array hashes and the inference isolation flags.
