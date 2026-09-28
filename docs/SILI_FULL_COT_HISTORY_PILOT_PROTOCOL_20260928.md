# Sili full-history COT trajectory pilot (2026-09-28)

## Question

The current residual head receives only the last historical COT field and a two-frame difference, although the bank contains eight historical COT maps. Test whether exposing all eight maps helps the learned thickness correction recover CPP-reference thick-cloud amplitude after local C13 transport.

This is a single-factor validation pilot, not a physical-truth or GHI experiment. CPP remains a satellite retrieval reference; test remains closed.

## Matched arms

- **Short-history control:** current two-map representation (last COT and last-minus-previous trend), using the low-COT penalty-off loss from the completed loss ablation.
- **Full-history candidate:** replace those two features with all eight historical COT maps, each aligned to all 16 future leads. Keep transported history, frozen SimVP→R forecast, lead, motion, target, masks, network width after the input layer, objective, optimizer, and clipping fixed.
- Initialize common downstream weights identically and map the control input weights so the full-history model reproduces the control input function when the six older-frame channels are zero. Keep zero-residual/step-0 transport eligible in model selection.

Both arms use the same 512 training and 512 validation sequences, seed 42, batch size, 512-update maximum, minibatch order, early-stop rule, and selection criterion. Profile batch sizes on one A800 first; choose the largest stable batch that leaves at least 10 GiB free, then use the same batch for both arms. Evaluate at fixed 16-update intervals and retain the best thick-COT checkpoint plus per-day sufficient statistics for every evaluation.

## Evaluation and gate

Use CPP-valid pixels intersected with C13 transport source support for all methods. Report 64×64 patch and station `[32,32]` metrics for all COT, COT≥30, bins 0–5/5–10/10–30/30–60/60–100, misses below COT 10, COT≥30 IoU, and false-COT rate on CPP COT<5, by lead and in aggregate. Compare short-history and full-history checkpoints against step 0 and each other with paired BJT-initialization-day bootstrap intervals (5,000 resamples, seed 20260928). Windows overlap; report day counts and preserve the clustered-unit interpretation.

Advance the full-history correction only if its selected learned checkpoint beats step 0 on patch COT≥30 RMSE with a paired 95% interval below zero, does not show a supported station deterioration, and does not create a supported low-COT/clear regression. Otherwise retain pure transport and close this residual-head branch before any GHI-head training. Do not use the ablation's unselected epoch metrics as a seed-stability result.

## Execution gates

Verify the fixed bank manifest and validation sequence ordering against the station-day sidecar, run CPU shape/function-equivalence smoke, compile in the remote `swc` environment, check the full live `slfu` PBS queue, and use one GPU. Preserve the existing audit and experiment outputs. Record the actual batch profile and GPU state; do not infer quality from utilization. Test and GHI-head training remain closed.
