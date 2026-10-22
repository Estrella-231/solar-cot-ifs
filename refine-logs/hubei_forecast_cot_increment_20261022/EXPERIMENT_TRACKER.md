# Execution tracker — Hubei history-anchored future-COT fusion

## 2026-10-22 — COMPLETE, candidate closed

- User explicitly authorized execution. Reused corrected immutable GHI_pack_attempt12 and history_bank_attempt1; no S/R fit/inference rerun. New GHI-only F/C roles and train-only normalizations are saved in prepared_attempt1/PREPARATION.json. Official validation, test and buffer split assignments are unchanged.
- CPU PBS 215680.tc6000 wrote all three stations, then failed at final receipt with a str/Path hash argument mismatch. Original log/code preserved; --finalize-existing recomputed every saved array/row hash and completed metadata without altering numerical arrays.
- GPU PBS 215681.tc6000 completed successfully on ZJNU college node22 A800 0/1/3/4, exit0, 107 s walltime. Pre-submit account unfinished counts0→1; no other job was canceled. Input hashes, no pre-existing allocated-GPU compute process, conditional zero initialization/mask fallback/gradient witnesses passed. Used the existing unchanged swc runtime/cuDNN recipe; no packages installed.
- Profile1/2/4 independent workers, selected4. Nine station×seed trials each finished G0/G1 base fits plus G2/G3 conditional fits: 36 heads total. Matched initial states/orders/parameter counts/updates; base unchanged during conditional fitting; official validation never selected epochs. All correction heads selected epoch1 by C_select; no retrospective epoch0 selection.
- Saved predictions score: G0 105.595270, G1 103.669917, G2 103.782946, G3 103.808491 W/m² equal-station/equal-seed RMSE. G1−G0 −1.925352 (−1.823%), 95% date-block CI[-4.734324,+0.670444]. Nine of nine paired point directions improve. G3−G2 +0.025545, CI[-0.003134,+0.058625]. Gate fails; no stable or incremental future-COT gain claimed.
- 15,275 distinct station rows, 22 target dates; three-seed replication does not add dates. Separate local NPZ-based metric/CI/budget audit passed to <1e-10. test/buffer reads0 throughout. No manuscript changes.
- Follow predeclared stop: freeze this fusion candidate; no new threshold/width/epoch/seed search and no automatic test release. Preserve historical-COT potential separately from future-COT failure. New-date confirmation is conditional, not launched.

## Paths

Remote root: `/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_forecast_cot_increment_20261022/`.

- `code_attempt1/`: actual source and PBS recipes; `prepared_attempt1/`: derived train/val input tensors and date roles.
- `training_attempt1/pairs/<station>_seed<seed>/`: best/last checkpoints, normalizations and per-epoch logs, saved validation predictions.
- `scored_attempt1/`: metrics and paired date-block intervals.
- Local `evidence_attempt1/`: compact remote evidence including all36 prediction files, metadata, final PBS status, independent audit and figures. Heavy checkpoints remain remote with local SHA256 index.
