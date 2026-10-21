# Hubei CPP sequence support audit (train/validation only)

State: `PASS_FULL_SUPPORT_SEQUENCE_AUDIT_TEST_BUFFER_CLOSED`

The audit used only the existing timestamp manifest; it opened no CPP payload, label, test, or buffer data.

| Split | Manifest frames | Full-support frames | Zero-support excluded | Dates | 24-frame windows, 15-min stride | 60-min stride | Non-overlap stride | Longest segment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 3178 | 3156 | 22 | 153 | 764 | 204 | 57 | 53 |
| Validation | 301 | 301 | 0 | 16 | 59 | 15 | 4 | 51 |

Only frames with all 256 COT and CLP pixels valid are used to form candidate sequences; zero-support frames are excluded and break continuity. A sample window consists of 8 history frames including initialization plus 16 future frames at 15-minute cadence. Overlapping windows are highly dependent; the independent uncertainty unit remains the target date. Validation counts therefore support only a feasibility screen, not a final skill claim.

The manifest SHA-256 is `6e307b98f27df523bd5d7d7200d412e58ba768991b9b4e265290f6b3fe025957`.

A passing result establishes only that station-local 16x16 CPP references have contiguous train/validation timestamps. It does not establish full-region CPP coverage, independent cloud truth, adequate independent validation sample size, or forecast skill.

## AGRI timestamp opportunity before the last known CPP product time

This second view counts allowed AGRI keys and candidate 24-frame sequences only. It does not assert that CPP exists or passes QC for every key.

| Split | AGRI frames ≤ cutoff | Candidate 24-frame windows | 60-min stride | Candidate init dates |
|---|---:|---:|---:|---:|
| Train | 14433 | 9673 | 2463 | 155 |
| Val | 1469 | 1022 | 258 | 16 |

Counts are AGRI key candidates only; raw CPP per-frame file/QC/time/grid coverage was not verified, so these are not confirmed CPP-supervised windows.

The train/validation-only AGRI manifest SHA-256 is `8b1f95c2a2cf1d66cabd11539b975fe3dccd486146bf3bec17fb18e537f72183`.
