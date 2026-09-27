# Sili pilot readiness correction — 2026-09-27

## Verified execution state

At approximately 23:21 BJT, PBS `210963.tc6000` was running on node22 GPU 7 and building its train bank (352/512 sequences), not training. Two account jobs were unfinished. At deployment of the correction below, the builder had reached train 408/512; the model run directory did not yet exist. The validation bank and training were still pending. These are dated snapshots, not completion evidence.

## Loss tensor correction

`prepare()` returns CPP labels and masks as `[B,16,64,64]`. The original training loop sliced `y[:,0]` and `m[:,0]`, dropping the target-time axis. Predictions after removing their singleton channel are `[B,16,64,64]`; with batch 8 the original loss would fail on incompatible dimensions. No training had reached this code yet.

The shared `training_loss()` now retains every target frame and rejects mismatched prediction, target, or mask shapes. Loss weights, optimizer, architecture, splits, and forecast inputs are unchanged. The smoke now exercises the actual loss and backward pass, checks finite nonzero output-layer gradients, verifies that later valid leads affect loss, verifies masked leads do not, and rejects dropped-time-axis labels. Zero-initialized residual equality with pure transport is also retained.

The corrected files passed Python compilation and the expanded CPU smoke in the remote `swc` environment (`PASS_SILI_TRANSPORT_THICKNESS_SMOKE`). They were copied into the job's script paths only after confirming that its training output directory did not exist. The original scripts are preserved in `audits/sili_loss_shape_20260927` on the server. This is readiness verification, not training or model-quality evidence.

Deployed SHA-256:

- Trainer: `b71b85659cf73b2f5f39cd0f3072ae94fd876345f8a30fda012c2dbdfb40c3b5`
- Smoke: `92ea7c3a2553a63485f9890c72f5556b41e7f891853a84dd03ac06d01560c6f0`

## Interpretation and next decision

The current network directly uses the last two historical COT frames (last field and difference), transported last field, frozen SimVP→R forecast, lead, and history-derived motion. Although eight frames are stored and motion uses seven adjacent C13 pairs, the learned correction does **not** encode the full eight-frame COT history. Describe it as a two-frame-trend correction pilot. A full-history encoder requires a separately named candidate and matched comparison.

Current validation aggregates all valid local pixels and selects by thick-pixel RMSE. Before downstream use, independently evaluate station-pixel trajectories, peak magnitude/time/duration, thick misses, clear-reference false clouds, and spatial errors on identical validation samples/support, including +120 to +240 min. Keep overlapping windows grouped by day/event when assessing stability. A thick-pixel improvement alone is insufficient. Save per-sample predictions and reload the selected checkpoint for independent scoring. Test remains closed. Only after this audit should a full-history candidate or controlled GHI retraining be considered.
