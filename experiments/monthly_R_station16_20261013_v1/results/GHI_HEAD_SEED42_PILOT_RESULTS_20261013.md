# New monthly R: seed-42 paired GHI-head pilot

Date: 2026-10-13 (BJT)  
PBS job: `213829.tc6000` (completed, exit status 0)  
Scope: train/validation only; the independent test payload remains closed.

## Contract and data audit

This pilot joins observed AGRI, physical solar geometry, COT from the newly trained monthly-split R checkpoint, and the frozen 15-minute GHI labels by `(station, target_time_utc)`. It uses the fixed `hunan_monthly_block_2d_buffer1d_seed42_v1` split and the frozen main mask. The head sees 13 AGRI channels, `cosSOZ` and `cosRAA`, plus a co-registered local 3x3 COT field in AC. A and AP are the paired no-COT and capacity-control arms. No COT spatial/global pooling is applied.

The pack contains 54,937 unique station-time rows: 50,458 train and 4,479 validation (Sili 26,283; Zhujia 28,654). Independent key, split, shape, finite-value, nonnegative-log1p-COT, and SHA-256 checks passed. The R checkpoint SHA-256 is `342596094d1a41aa8b141972a7b4adadece179b4b9f58170f4e7f409a894f001`; pack manifest SHA-256 is `27b1a10b172c5030194007bc1fdeb2ea6c929c86c304e78242255de9dee54bd4`. Both the GHI pack and R bank record `test_payload_opened=false`.

## Seed-42 validation pilot

Each station/arm used the same 30-epoch budget, seed, sample-order sequence, physical-GHI MSE objective, and selected batch size 2,000. Best checkpoints were selected by that station's validation GHI RMSE. The table reports independent FP64 rescoring of saved predictions; the aggregate is the arithmetic mean of the two station RMSEs.

| Arm | Sili RMSE (W m^-2) | Zhujia RMSE (W m^-2) | Station-equal RMSE (W m^-2) | Difference from A |
|---|---:|---:|---:|---:|
| A | 135.241 | 110.625 | 122.933 | — |
| AC | 132.639 | 106.891 | 119.765 | -3.168 |
| AP | 135.098 | 110.345 | 122.721 | -0.212 |

AC is lower than A by 2.60 W m^-2 at Sili and 3.73 W m^-2 at Zhujia. The AP control is close to A, while AC is lower than AP at both stations. This is an encouraging single-seed validation signal for the local COT input beyond the added head capacity. It is not yet a stable result: seeds 43 and 44, date-block uncertainty, and the pre-registered three-seed decision rule remain outstanding. These numbers are validation metrics, not an independent-test or operational forecast claim.

The parameter counts were A 22,177, AC 24,489, and AP 24,613; AC and AP differ by about 0.5%.

All six saved prediction files were independently rescored in FP64 against the frozen validation rows. Each saved checkpoint was also reloaded on CPU and compared with its saved A800 prediction. CPU-vs-GPU differences were small in aggregate (maximum absolute point difference 0.452 W m^-2; station RMSE differences at most 0.0046 W m^-2), consistent with a device-level FP32 inference comparison. Checkpoint hashes and prediction-to-label lengths matched.

## Profile and implementation note

The single-GPU profile selected batch 2,000, the largest profiled batch that retained at least 12 updates per epoch on the smaller station; peak allocated memory was about 0.189 GiB. The assigned device was one NVIDIA A800-SXM4-80GB (GPU 3). This compact 3x3 head is compute-light; a low utilization snapshot does not imply a failed run, and there was no OOM. The profile was performed before the six runs.

The training log includes PyTorch's warning for constructing a tensor from the read-only geometry memmap before copying it to CUDA. The code only reads that tensor, and the full run, saved-prediction audit, and checkpoint reload completed. Before seeds 43 and 44, an explicit writable geometry copy was added to remove this warning source and each run now records its training/head source hashes.

Seed 42 used training source SHA-256 `f633d1c290ebb6081a4dd0913d9cede973991d6f476c2a59187989ab8edab4b4` and head source SHA-256 `eab8a098541b3eab47b1972e0413a82b82746039b768e3479fa1c4f75732730b`. The replica source hash is `ece9abfb04f01a1e89bf613ba76be0c0c32d753841ec60de12f942ca52c86f58`. This is a data-safety and provenance correction, not a model or objective change.

## Evidence files

- `ghi_head_pack_manifest.json`: frozen data, R provenance, sample counts, and array hashes.
- `profile.json`: single-GPU batch/throughput profile.
- `pilot_fp64_audit.json`: independent FP64 validation metrics and checkpoint hashes.
- `checkpoint_reload_audit.json`: CPU checkpoint reload comparison to saved A800 predictions.
- `sili/`, `zhujia/`: six checkpoints, saved validation predictions, validation summaries, run metadata, and epoch metrics.

Next: fix the read-only-array warning without changing the model or loss, record the revised source hash, then run seeds 43 and 44 on the same pack and frozen training contract. Do not open the test payload.
