# Solar-path COT encoding for GHI: H-only date-block CV

Date: 2026-10-21. Status: fixed 90-head H-only CV running locally on one RTX 4060; synthetic scorer test and local causal input/model/feature smoke passed. This is the next single-factor candidate after the predeclared COT-trajectory encoding failed. It does not reopen or tune the failed trajectory experiment. Manuscript body remains unchanged.

## Hypothesis

The target is 15-minute interval-mean GHI, while predicted COT is a vertical cloud optical thickness at the target time. The GHI head currently receives log-scaled vertical COT. A physically motivated alternative is to encode the slant-path optical depth using known target-time solar geometry, then express the corresponding direct-beam attenuation proxy:

`A = 1 - exp(-tau / max(mu, cos(80 deg)))`, where `mu = cos(SOZ)`.

This is a feature transformation only. It is not a complete cloud radiative-transfer model and does not equate GHI with direct-beam attenuation; diffuse radiation and cloud scattering remain learned from AGRI, geometry, and labels. The test asks whether the transformed map is easier for the same GHI head to use.

## Fixed arms

Use the same forecast P-XC target-time 3×3 COT map and known target-time 3×3 geometry for every row. Three same-architecture arms:

1. No COT: exact neutral zero COT tensor.
2. Raw target COT: existing train-normalized `log1p(tau)` map.
3. Solar-path attenuation: `A` map above, standardized using training dates only.

Only the COT feature representation changes. AGRI channels, geometry channels, lead embedding, GHI/kt target, station rows, fold assignment, date gaps, architecture, loss, optimizer, learning-rate schedule, initialization seeds, batch size, epochs, and checkpoint selection stay the same as the completed H-CV. No COT trajectory maps are supplied.

## Split, training, and scoring

Use only the existing H training partition rowbank and causal forecast-COT outputs. Reuse the frozen five calendar two-day folds and remove training targets within ±1 BJT day of each held-out block. Each eligible H key is scored once out of fold. Fit all normalizers on the training dates for that fold; for the transformed map fit only sunlit (`cosSOZ > cos80`) training pixels and set off-sun COT features to neutral zero.

Train both stations × five folds × three seeds × three arms = 90 heads, each for 30 epochs at batch 512. Same seeds initialize paired arms identically. Do not early-stop or pick a checkpoint by a held-out fold.

Primary contrast: solar-path attenuation minus raw target COT. Report all leads (+15…+240 min) and the short gate (+15…+60 min), station, seed, leadwise descriptive metrics, MAE, bias, exact key counts, and paired 95% bootstrap over the 112 frozen calendar two-day blocks (5,000 replicates). Positive RMSE difference means the transformed arm is worse.

The candidate passes the internal H screening gate only if both all-lead and short-lead point estimates are negative with 95% upper bounds below zero, both stations are nonworse, at least two of three seed differences are negative, and leave-one-block-out estimates do not reverse the sign. Report contrasts against no COT regardless of the primary outcome. Do not adjust any threshold after seeing results.

## Evidence limits and follow-up

This is an exploratory internal cross-validation because H has supported previous model development. It cannot establish independent-test benefit. Previously examined outer V can only be a reused development check, not untouched confirmation. Never read test/buffer numeric payloads or use future observed AGRI/CPP/GHI as inputs. Any real-time chain confirmation requires unused ground-GHI dates with verified provenance; current accessible Hunan label inventory is verified only through 2026-07-01.
