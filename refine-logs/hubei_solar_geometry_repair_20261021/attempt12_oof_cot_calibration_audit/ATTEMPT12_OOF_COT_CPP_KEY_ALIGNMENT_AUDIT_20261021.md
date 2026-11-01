# attempt12 OOF forecast-COT / CPP key alignment audit

**Snapshot:** 2026-10-21 16:21:59 CST. **Scope:** train and validation metadata only. No test/buffer paths were queried; no NPY numerical payload was loaded. NPY files were inspected through headers only.

## Finding

The five OOF forecast-COT banks can supply training pairs for a train-only calibration, but their CPP targets are **not** the `R_runs_attempt12/fold_i/val_*` arrays. Those per-fold run outputs are each model's validation set and contain no OOF training-frame target keys. The correct OOF training reference is the held-out date fold's rows inside `prepared_attempt12/R_packs/full/{rows.csv,target.npy,mask.npy}`: `rows.csv` identifies train dates, `heldout_fold` selects the matching fold, and the row's `index` binds it to the corresponding `target` and `mask` entry. The builder appends each metadata row in the same order as `target[ii]` and `mask[ii,None]`.

Using that keyed source, every fold's forecast sequence has the expected 16 leads at +15…+240 min, no duplicated `(init_time_bjt,target_time_bjt,lead)` tuples, no fold/date mismatch, and no duplicate CPP time key. Matching is by target time, never by row position. Of the available forecast frames, 17,616 train frames match a same-fold CPP key; 17,129 of those keys have a nonempty audited CPP/SZA80 mask, while 487 have zero valid CPP pixels and must be excluded. Many forecast frames lack a CPP reference row because the reference product is sparse; those are reported below and are not silently filled.

The full-train validation COT bank also aligns by target time to `R_runs_attempt12/full/val_keys.csv`: 1,828 forecast frames match 197 reference timestamps and all matched keys have nonempty masks. The existing forecast-vs-persistence comparison uses the narrower common history/reference support of 1,164 `(init,target,lead)` keys. Any calibration comparison against persistence must retain that exact common key set; validation must not be used to fit or choose calibration parameters.

There is no station field in the sequence-key CSVs: each COT prediction is a full 16×16 patch shared across the three station footprints. The station component is introduced by the fixed spatial crops (Niushou `(8,6)`, Wangzhai `(9,8)`, Mizhuang `(7,10)`). This key audit proves time/fold alignment for the full patch and its target mask row; it does not inspect station-pixel mask values.

## Fold and validation counts

`S sequences` counts OOF/full validation sequence rows; each contains 16 forecast frames. `Matched frames` are exact `(init,target,lead)` rows whose target time exists once in the relevant CPP key table. `Usable mask rows` are matched frames whose metadata reports `valid_cot_pixels > 0`. `Unmatched frames` have no CPP key row. `Unused CPP times` are reference keys not targeted by any forecast frame.

| Role | S sequences | Forecast frames | CPP key rows | Matched frames | Usable mask rows | Zero-mask matches | Unmatched frames | Unused CPP times | COT bank SHA-256 (from source manifest) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fold_0 | 823 | 13,168 | 631 | 3,211 | 3,134 | 77 | 9,957 | 231 | `5dd86edf76650e27329e580dcbd06bb4486181fab689dd16b41ba8a2886ce70a` |
| fold_1 | 797 | 12,752 | 631 | 3,399 | 3,332 | 67 | 9,353 | 211 | `f2e29b8e1e3cc02d4b0c07b8c53f6febd6a2476c8d2cbb77843dcb0524c6c852` |
| fold_2 | 850 | 13,600 | 644 | 4,073 | 3,919 | 154 | 9,527 | 180 | `af988aa936100d5c08d293c607354d64c358c7cd1fdce48aed202c4b4e1b8dd4` |
| fold_3 | 809 | 12,944 | 627 | 3,196 | 3,091 | 105 | 9,748 | 230 | `56ccbc96a12a5bcc9af4e42ea76aebde8a5bf1041d46cbbd010c8ba2470281de` |
| fold_4 | 807 | 12,912 | 645 | 3,737 | 3,653 | 84 | 9,175 | 197 | `c6b9fee222e38df86ce56d5cd3ef1009ea25b0e9a402da001b078eba3aa002b0` |
| Full validation | 407 | 6,512 | 264 | 1,828 | 1,828 | 0 | 4,684 | 67 | `3e4a04d575f09fe678fea87da9585db8b47005299e2703a0b93c2b7bf52e6393` |

Across OOF folds, the source contains 4,086 sequence rows and 65,376 forecast frames. The 17,129 usable-mask frame pairs are key-aligned calibration candidates, not 17,129 independent cases; date is the resampling/grouping unit. The five fold banks' target dates all map to their assigned held-out training fold (init/target fold mismatches: 0). Forecast key rows and CPP key rows have different order and multiplicity, so positional pairing is invalid.

## Files and hashes

Remote base:

```text
/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021
```

- OOF sequence indexes: `prepared_attempt12/indexes/fold_i_oof.csv`; rows and SHA-256 are recorded in `remote_key_audit.json`.
- OOF predicted AGRI key rows: `S_banks_attempt12/fold_i/val/keys.csv`; each has matching row count and SHA in the captured audit JSON. The key SHA values are the five `source_indices_and_keys.fold_i.key_sha256` entries in `OOF_TRAIN_BANK_REPORT_attempt12.json`.
- OOF forecast COT: `COT_banks_attempt12/fold_i.npy`, shape `(sequences,16,1,16,16)` from header-only inspection. The per-fold COT SHA values above are declared by `COT_banks_attempt12/source_manifest.json`, which binds each bank to its corrected-S checkpoint, corrected-R checkpoint, train-fitted R normalization, predicted-AGRI SHA and future-geometry SHA.
- Train CPP/masks: `prepared_attempt12/R_packs/full/rows.csv`, `target.npy`, and `mask.npy`; `rows.csv` SHA-256 `2322b0a8dd8d480a65e702e7b10def6c4ddea737e9183a5f44b28e13c1744511`; `SHA256.json` SHA-256 `6977ee87feddb964ca58db730124e378bfa5bfabac8136ca8aa056c25f677b2f`; `target.npy` payload SHA-256 `befbf167cee3cde086ca64464a0e5d5c9b897c6cae4230e5eb211c3c046c2563`; `mask.npy` payload SHA-256 `433eda0b165d2067441139c4793aedf9c4b9d23e944f1f29e978ea7bbd44ae18`. These payload hashes come from the pack's SHA manifest; this audit did not read or recompute array payload hashes.
- Full-train validation COT: `COT_banks_attempt12/validation.npy`, shape `(407,16,1,16,16)` by header. Declared COT SHA-256 is `3e4a04d575f09fe678fea87da9585db8b47005299e2703a0b93c2b7bf52e6393`.
- Validation CPP/mask keys: `R_runs_attempt12/full/val_keys.csv`, 264 rows, SHA-256 `87b7e95549074f53d99ad66735726f1b67cfcfbeca99164d15104c32ec7412f1`; reference and validity mask are `val_reference_cpp_cot.npy` and `val_valid_mask.npy`, each 264 rows. Existing metric receipt `forecast_COT_eval_attempt12/forecast_cot_metrics.json` records CPP-reference SHA `f401ab35f1ea5f0ad411efa07eb836a7676cabb440c43d83158030c33e84e67f`, mask SHA `ee2b8ea789de0f876110801648d8aba640e69545af7bddf2d2d0561483b2d7a3`, and 1,164 common forecast/history/reference keys.
- Fold R run outputs `R_runs_attempt12/fold_i/val_keys.csv` are not the OOF training labels: each fold file has 264 validation keys, with no target-time matches to that fold's OOF forecast frames. Use the full R pack's training rows filtered by held-out date fold, as above.
- Bank manifest SHA-256: `e0966f96c71f0ba8045b5ed159055e0d759ccbd216c438b223718dad3844ffb1`. OOF lineage receipt `OOF_TRAIN_BANK_REPORT_attempt12.json` SHA-256: `179693374e2fe85604d3d8cc5c88bc732368d257dbeece8c12c577271c10c67c`. It records 4,086 OOF sequences, 153,515 train rows, 15,275 validation rows, and `test_payloads_read=buffer_payloads_read=0`.

The complete per-role row/key/checkpoint hashes and NPY header dimensions are preserved in [`remote_key_audit.json`](remote_key_audit.json). The local read-only audit driver SHA-256 is `707c9ef3c9994e67d6bd7ef4477b26ce64fba75e94fbfaa57f00cd19b5352da3`.

## Calibration design (not fitted here)

Use one global two-parameter affine correction in log1p space, fit only on the 17,129 OOF forecast-frame rows with nonempty CPP masks:

```text
z_pred = log1p(clip(COT_forecast, 0, 100))
z_cpp  = log1p(100 * target.npy)  # target.npy stores CPP COT / 100
z_cal  = a * z_pred + b
COT_cal = clip(expm1(z_cal), 0, 100)
```

Minimize masked pixelwise squared error in `z_cal`; constrain `a >= 0`, and weight target dates equally (or use a predeclared date-balanced weighting) so dense dates do not dominate. Keep the identity mapping `(a=1,b=0)` as the uncalibrated forecast baseline. Do not select the transform, parameters, clipping policy, or weighting using validation. After freezing `(a,b)`, evaluate calibrated forecast, uncalibrated forecast, and historical-COT persistence on the exact 1,164-key common validation support with paired target-date bootstrap; CPP remains a retrieval reference, not independent truth. Validation does not enter fitting.

## Reproduction and live queue check

Read-only audit command:

```powershell
Get-Content -Raw -Encoding UTF8 'solar-cot-ifs/refine-logs/hubei_solar_geometry_repair_20261021/attempt12_oof_cot_calibration_audit/audit_attempt12_oof_cpp_key_alignment.py' |
  ssh zjnu-hpc '/public/home/slfu/miniconda3/envs/swc/bin/python -'
```

The script reads only the CSV/JSON metadata and NPY headers. At 2026-10-21 16:21:59 CST, `qstat -u slfu` showed 2 unfinished jobs: `215356.tc6000` R on node22, 8 CPU/1 GPU; dependent `215359.tc6000` H, 8 CPU/1 GPU. No PBS job was submitted, cancelled, or changed. Test/buffer payload reads remain 0.


