# Train-only date-held-out R-COT input bank

**State: complete; data preparation only. No forecast model was trained or scored.** This package assembles the five existing date-held-out R output shards into one train-key-ordered bank so a future COT-forecast cache can use out-of-fold R histories on its training rows.

## What was assembled

- Exact train keys: **20,206** rows from 220 BJT dates, split across five held-out-date folds (`4,174 / 4,052 / 4,008 / 4,031 / 3,941`).
- COT shape: `(20,206, 1, 16, 16)`, float32; day-mask shape is identical.
- Join: `time_bjt` keys from each fold were matched exactly to the frozen train rows. A post-write check compared every COT map and day mask to its source fold array and passed for all 20,206 keys.
- Raw outputs were preserved without clipping. There are **2,564** COT values outside the R training range `[0,100]` (maximum `12,648.736`). Downstream cache construction must continue to mark such values invalid using the existing rule.

The only full-bank file read was its `rows.csv` split/key/order metadata. Its full-R COT prediction array was not opened. Validation prediction values, CPP labels or QC, GHI labels, test, and buffer payloads were not read. This bank contains no CPP targets and is not a forecast result or accuracy evaluation.

## Why this matters

The train-only audit found that full-R and date-held-out R histories differ at all three station centers. This bank makes it possible to form a separately named forecast-model cache with OOF R-COT histories for train frames, while leaving CPP targets and sample keys unchanged. The purpose would be to reduce the measured train/inference source mismatch. It does not guarantee better future-COT forecasts, establish physical COT accuracy, or pass the current COT skill gate.

The current experiment plan still requires source/evaluation gates before further model fitting. This data artifact does not authorize validation tuning or GHI-head training. A future cache must record the five fold checkpoint and row-manifest provenance, retain the original CPP target masks, and use a new output path; the existing cache and checkpoints remain unchanged.

## Files and reproduction

- `assembled_v1/cot_pred_train_oof.npz`: compressed COT and day-mask arrays, train time keys, and held-out fold IDs.
- `assembled_v1/rows_train.csv`: the exact train row order and fold assignment.
- `assembled_v1/MANIFEST.json`: source and output SHA-256 values, split counts, shapes, and read-scope receipt.
- `assemble_train_only_oof_cot_bank.py`: deterministic assembler; refuses to overwrite an existing output directory.

Reproduce from the repository root:

```text
python refine-logs/hubei_cot_transport_source_20261022/oof_history_inference_attempt3/train_only_oof_bank/assemble_train_only_oof_cot_bank.py \
  --bank-rows experiments/hubei_r_runtime_gate_20261015_v1/cot_bank_full_trainval_v1/rows.csv \
  --oof-root refine-logs/hubei_cot_transport_source_20261022/oof_history_inference_attempt3/evidence/remote_215541/oof_train \
  --out-dir <new-output-directory>
```

The assembler refuses to overwrite an existing output directory. The compressed bank SHA-256 is `62ff023ee299af4e613c84b371bba5873788a1e036a3dd8619a494dca245143a`.
