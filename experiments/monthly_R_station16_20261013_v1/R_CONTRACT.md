# R contract

- Split: `hunan_monthly_block_2d_buffer1d_seed42_v1`.
- Scope: train/validation only; 16x16 patches centered on Sili and Zhujia.
- Model: fresh `COTUNet` initialization, 16 inputs to nonnegative log1p(COT).
- Inputs: AGRI channels `[0,1,2,3,4,5,8,9,10,11,12,13,14]` and
  `cosSOZ`, `cosRAA`, `day_mask`, aligned to the matching 16x16 crop.
- Label: CPP COT in physical units, finite and within [0,100].
- Loss mask: valid label, CLP in `{0,1,2}`, all 13 AGRI channels valid,
  `day_mask > 0.5`, and `cosSOZ > cos(80 degrees)`. Zero COT is retained.
- Normalization: per-channel mean/std fit on train samples and the train
  supervision mask only; validation never contributes to normalization.
- Objective: masked Huber loss on `log1p(COT)`, delta 0.1, finite-safe
  masking. Validation selects the best checkpoint on the same support rule.
- Target coverage: CPP is available for only part of the monthly train/val
  candidates and has known monthly gaps. The model's proven domain is limited
  to its CPP-supported daytime pixels; it is not an all-weather or independent
  truth retrieval claim.
- Provenance required at completion: frozen candidate and split hashes,
  CPP source paths/availability, grid mapping tolerance, input/target counts,
  train-only normalization, model/source/checkpoint hashes, seed, epochs,
  GPU, PBS job ID, and explicit `test_payloads_read=0`.

No old checkpoint or old-split cache may initialize this model or enter its
new-split validation. The separate `cot_real_bank` job uses the historical
R checkpoint and remains a diagnostic product.
