# AGRI Nowcaster refactor contract

This is an independent first-paper candidate under the 2026-10-08 user authorization. Preserve the original vendor tree and monthly-repair outputs. Use only the frozen monthly train/validation split, train-only normalization, and the formally completed monthly VAE/nowcaster as initialization. Never load test payloads.

Minimal paired treatment: same official checkpoint initialization, sample ordering, optimizer, learning rate, per-rank batch, eight epochs, and validation selection. Control trains latent L1; candidate trains decoded 13-channel masked AGRI L1 plus 0.1 latent L1. Both select using the same season-balanced decoded AGRI validation MAE. The VAE is frozen, but decoder input gradients must remain enabled. Save source/checkpoint hashes, per-sample sufficient statistics, and paired-date bootstrap. This comparison only supports AGRI validation claims; a COT claim additionally requires a lineage-audited R.

Use PBS on zjnu-hpc; at most three unfinished account jobs, eight GPUs total, check qstat immediately before each qsub. Profile actual data-loader throughput and preserve >=15% device memory headroom. No GPU process on login. Existing 2026-10-04 stages are complete; this new explicit authorization covers this isolated experiment only.
