# Retry05: scoring only

This is a scoring-only recovery of the monthly-validation COT diagnostic. It reuses the six frozen prediction NPZ files and JSON receipts produced by PBS job 215551.tc6000 under retry04/predictions. No model is reloaded for inference, no optimizer runs, and no GPU is requested.

The only scorer correction accepts an absent optional official_monthly_validation_payloads_read field as zero only when the training receipt explicitly states that official monthly validation, test, and buffer were not read. Test and buffer counters must still equal zero. The original checkpoints, prediction arrays, cache, support, score definitions, date-block bootstrap count and seed remain unchanged.

The output is a one-time retrospective monthly-validation diagnostic against CPP retrieval reference, not independent test, full-region physical truth, or a GHI result. The source inference job is 215551.tc6000.
