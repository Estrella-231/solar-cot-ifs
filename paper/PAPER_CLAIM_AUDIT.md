# Paper Claim Audit

- Audit date: 2026-10-18
- Paper source: `paper/main.tex`
- Verdict: **FAIL**
- Acceptance status: `provisional`
- Review independence: `same-family`
- Fresh reviewer: `gpt-5.6-sol`, reasoning `ultra`
- Test result available: **no**
- Audited inputs: 91 files (the paper, Chinese mirror, 88 original JSON/CSV inputs, and one post-audit direct-COT result JSON)
- Paper SHA-256: `44ee5e2da1dc27b54f430021ce62d7ebe4559e777553d2a47f2c88e7b243c91`
- Chinese mirror SHA-256: `30565eb04fd34843276ae4297293845b449cdf1af9fb1af63bc7f0608e1b233c`
- Post-audit direct-COT evidence: `../refine-logs/cot_retrieved_forecast_20261013/results/simvp_vs_direct_cot_strata_20261017.json` (SHA-256 `6910862e2c2005a5c1b09a61df92b35d68a8a797dd49f7d9c0a24fd01a851abd`)
- Complete per-input SHA-256 manifest: `PAPER_CLAIM_AUDIT.json` → `audited_input_hashes`
- Trace: `.aris/traces/paper-claim-audit/2026-10-18_run01/`

## Verdict

The principal Hunan and Hubei synchronous validation results, the Hubei matched forecast-input result, the mapped retrieval results, and the mapped forecast-COT results reconcile with the mapped raw evidence. A post-audit check of the direct-COT result JSON resolved the 0.001 local-RMSE discrepancy: the source value is 12.162478360, which rounds to 12.162; both language versions now use 12.162. The paper-to-evidence gate remains **FAIL** because 25 claim groups lack eligible evidence, two are unsupported by the permitted sources, and three have ambiguous mappings. This is a targeted post-audit correction, not a fresh independent review of all claims.

No independent test result was inferred. The eligible artifacts consistently identify validation-only scope and zero test-payload reads.

## Status summary

| Status | Count |
|---|---:|
| `exact_match` | 10 |
| `rounding_ok` | 16 |
| `ambiguous_mapping` | 3 |
| `missing_evidence` | 25 |
| `number_mismatch` | 0 |
| `unsupported_claim` | 2 |
| `config_mismatch` | 0 |
| `aggregation_mismatch` | 0 |
| `scope_overclaim` | 0 |
| **Total** | **56** |

## Material findings

1. **Resolved after the fresh audit:** the direct-COT raw JSON gives local RMSE `12.162478360`, rounding to `12.162`. The English and Chinese paragraphs were corrected and the LaTeX source recompiles successfully.
2. No eligible raw evidence supports the signed-residual control, direct-COT screen, amplitude-weighting screen, seven thick-cloud episodes, adjacent-change diagnostic, or fixed-head replacement diagnostic.
3. The Hubei AC-versus-AL common-support result (`1458` rows, `21` dates, AC `72.867`, AL `72.785`, difference `+0.082`, CI `[-1.490,+1.713]`) is not present in the eligible files. The available AL JSON instead covers a different full support of `2456` rows and yields AL mean RMSE `71.089743`; it cannot verify or contradict the manuscript's common-support result.
4. The eligible Hubei forecast artifacts do not bind the upstream SimVP checkpoint, regional lineage, or five-fold OOF prediction bank. Therefore no Hunan/Hubei upstream-model provenance mismatch can be established, but the manuscript's Hubei-specific provenance statement remains unverified.
5. `fig1_cot_value_and_forecast_limits_data.json` contains an older eight-date forecast-COT block. The manuscript's current forecast-COT values map exactly to the newer Fig. 7 summary, so the older block was treated as stale/unmapped evidence, not a contradiction.

## Per-claim reconciliation

Repeated claims in the abstract, results, discussion, conclusion, captions, and appendix are grouped into one record.

| ID | Lines | Status | Claim and reconciliation |
|---|---|---|---|
| C01 | 31–33 | `unsupported_claim` | Prior-work scope claims require external literature, which was outside the permitted evidence roots. |
| C02 | 37, 40, 50–56, 230, 234 | `exact_match` | Sili/Zhujia are Hunan; Niushou/Wangzhai/Mizhuang are Hubei; regional scores are kept separate. |
| C03 | 37, 40, 172 | `missing_evidence` | Hubei artifacts are region-labelled, but eligible files do not establish both complete regional training lineages or rule out transfer end to end. |
| C04 | 42, 46 | `missing_evidence` | Exact AGRI/GHI/CPP date spans are absent from eligible files. |
| C05 | 50–56, 85 | `exact_match` | Hunan `4479`, `2128/2351`, `52` dates and Hubei `2456`, `856/784/816` reproduce exactly. |
| C06 | 85 | `missing_evidence` | Hunan training count `50,458` is absent. |
| C07 | 56, 63–80 | `ambiguous_mapping` | Only the Hubei 15-minute target is bound; incomplete-window wording is not consistently mapped. |
| C08 | 56 | `exact_match` | Hubei engineering-unit and calibration-metadata limitation is stated in run metadata. |
| C09 | 58, 93, 103, 115–139 | `ambiguous_mapping` | 16×16/3×3 support and 13 normalization entries are present; exact channels, geometry, and full architecture are not. |
| C10 | 60, 204–218, 274–289 | `exact_match` | CPP is identified as a sparse retrieval reference rather than independent truth, with common-valid support. |
| C11 | 83 | `missing_evidence` | Monthly split, buffers, sparse-month rule, and shared mapping lack an eligible manifest. |
| C12 | 85, 172–181 | `missing_evidence` | Sixteen +15…+240-minute leads are present; eight-frame history and SimVP implementation details are not. |
| C13 | 97–113 | `missing_evidence` | Retriever architecture and training hyperparameters lack eligible configuration evidence. |
| C14 | 115–139 | `missing_evidence` | Train-fitted normalization is supported; detailed branch dimensions and softplus output are not. |
| C15 | 144–157 | `rounding_ok` | A/AC/AP parameters are `22177/24489/24613`; AP differs from AC by `0.50635%`. |
| C16 | 159–169 | `exact_match` | Hubei seeds, 30 epochs, batches `512/1536`, and physical-GHI MSE match metadata. |
| C17 | 113, 167–169, 179 | `missing_evidence` | Hunan training settings, paired initialization/permutations, freezing, and 12 epochs are absent. |
| C18 | 172–181 | `missing_evidence` | AL binds a frozen Hubei retriever, but upstream SimVP identity and five-fold OOF/full-bank provenance are not bound. |
| C19 | 187–198 | `exact_match` | Station-within-seed RMSE, equal station/seed averaging, and non-ensemble endpoint match. |
| C20 | 198, 285 | `rounding_ok` | Forecast-input one-day and two-day intervals both use 10,000 resamples and match. |
| C21 | 198, 228, 249 | `missing_evidence` | Other asserted 2,000/10,000 bootstrap draw counts are not recorded in eligible artifacts. |
| C22 | 25, 226, 293, 302 | `rounding_ok` | Hunan `123.980→120.240`, gain `3.740` (`3.02%`), CI `[-6.180,-1.597]` matches. |
| C23 | 226 | `missing_evidence` | Eligible Hunan evidence has station means, not station×seed values needed for “every seed.” |
| C24 | 25, 228, 293, 302 | `rounding_ok` | Hubei A/AC/AP `74.333/70.993/73.839`, deltas, `4.49%`, and 8/9 paired wins reproduce. |
| C25 | 25, 228 | `rounding_ok` | Hubei AC−A CI `[-5.800,-1.118]` equals the stored A−AC interval after sign reversal. |
| C26 | 228 | `missing_evidence` | AC−AP and two-day same-time intervals are absent. |
| C27 | 26, 228, 293, 301 | `ambiguous_mapping` | Available full-support AL evidence cannot verify the manuscript's different 1,458-row common-support comparison. |
| C28 | 228 | `exact_match` | AL definition and AL/AC parameters `24549/24489` match metadata. |
| C29 | 218 | `rounding_ok` | Hunan pooled retrieval `n=356169`, RMSE `9.074`, MAE `4.646`, bias `+0.279`, `1398` patches, `46` dates match. |
| C30 | 218 | `missing_evidence` | Station-equal Hunan retrieval RMSE `9.053` is absent; only pooled `9.074092` is present. |
| C31 | 218 | `rounding_ok` | Hubei retrieval `n=66304`, RMSE `13.357`, MAE `9.465`, bias `−1.303`, `16` dates match. |
| C32 | 218 | `missing_evidence` | Hubei frame count `259` is absent. |
| C33 | 218 | `rounding_ok` | CPP≥30 retrieval biases, CIs, pixel counts, and date counts match for both regions. |
| C34 | 222 | `exact_match` | Retrieval-bias intervals use 2,000 BJT target-date resamples. |
| C35 | 236 | `rounding_ok` | High-sun A/AC/AP RMSE `154.111/148.645/153.854` and 1,527 unique examples match. |
| C36 | 236 | `missing_evidence` | Point difference is derivable, but union-date scope and CI `[-8.990,-2.152]` are not eligible. |
| C37 | 236 | `rounding_ok` | Thirty-date daily MAE and A/AC bias values match. |
| C38 | 236 | `missing_evidence` | Daily MAE point difference is reproducible; CI `[-0.0861,-0.0067]` is absent. |
| C39 | 238, 249, 312–322 | `rounding_ok` | All four CPP-stratum totals, dates, A/AC RMSEs, reductions, and intervals match. |
| C40 | 242, 319–322 | `missing_evidence` | Per-station stratum counts and thick-stratum station RMSE/bias are absent. |
| C41 | 244 | `missing_evidence` | Retrieved-COT grouped gains `0.33/1.00/6.09/24.32%` are absent. |
| C42 | 252, 258 | `rounding_ok` | All four event-case RMSE pairs and identifiers match the case CSV. |
| C43 | 254 | `unsupported_claim` | Case CSV does not establish trajectories, smoothing behavior, or the asserted archive-wide seasonal reason. |
| C44 | 261, 265–269, 293, 302 | `rounding_ok` | Hunan capacity control `123.822`, gain `3.582`, and CI match after orientation reversal. |
| C45 | 25, 202, 263, 293, 302 | `missing_evidence` | Signed-residual metrics, CI, parameters, training, and OOF details have no eligible artifact. |
| C46 | 25, 274, 295, 302 | `rounding_ok` | Original/persistence forecast-COT RMSE/CSI and all three fine-tune scores/deltas match Fig. 7 raw evidence. |
| C47 | 25, 274 | `missing_evidence` | Equal training budget for the three loss candidates is not bound. |
| C48 | 25, 275, 295, 302 | `missing_evidence` | Direct-COT design/results lack eligible raw evidence; common support counts alone are insufficient. |
| C49 | 275, 277 | `rounding_ok` | The direct-COT result JSON gives `12.162478360`; both language versions now round it to `12.162`. |
| C50 | 277, 295 | `missing_evidence` | Amplitude-weighted results lack eligible raw evidence. |
| C51 | 279, 295, 302 | `missing_evidence` | Seven-episode scores and “all three Zhujia misses” lack eligible raw evidence. |
| C52 | 281 | `missing_evidence` | Transition counts, change correlations/slopes, RMSE deltas/CIs, and CPP-change results are absent. |
| C53 | 283 | `missing_evidence` | Fixed-head replacement counts, RMSEs, deltas, and CIs are absent. |
| C54 | 25, 183, 285, 297, 302 | `rounding_ok` | Hubei FA/FAC scope, `77.150/78.155`, `+1.005`, `1.30%`, one favorable seed, and both CIs match. |
| C55 | 25–26, 85, 212, 228, 285, 297, 302 | `exact_match` | Mapped evidence is validation-only; test remains closed with zero test-payload reads. |
| C56 | 289 | `exact_match` | Fig. 7 local 3×3 support, station-lead macro RMSE, COT≥30 threshold, and pooled aggregate CSI convention match. |

## Evidence and scope limitations

- Read scope was limited to `main.tex` and eligible `.json`/`.csv` files in the nine user-listed locations.
- Files whose basename contained `AUDIT`, `REVIEW`, or `PROGRESS` were excluded.
- No NPZ, HDF, XLSX, checkpoint, test payload, buffer payload, radiation array, image, project plan, progress note, or prior audit report was opened.
- Figure-caption numerical claims were checked against raw tabular data, but visual fidelity of PNG/PDF/SVG figures was not inspected because those formats were outside the authorized read set.
- The complete 89-file path/hash manifest is in the JSON companion. The trace also records the exact paths read.
- The trace helper and tracing reference files were present. The collaboration backend exposed the reviewer canonical agent path but no skill-thread UUID, so no UUID was fabricated.

## Publication gate

`FAIL` is provisional because the reviewer belongs to the same model family as the executor. Before the manuscript can pass, the internal direct-COT value must be made consistent and every material `missing_evidence`, `ambiguous_mapping`, or `unsupported_claim` group must either receive a permitted raw artifact or be removed/qualified in the manuscript. Independent test claims remain closed.
