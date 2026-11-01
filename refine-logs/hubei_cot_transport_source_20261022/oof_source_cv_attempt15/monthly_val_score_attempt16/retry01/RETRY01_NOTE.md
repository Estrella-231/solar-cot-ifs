# Retry 01 note

The original one-time frozen-checkpoint validation inference was submitted as PBS job `215545.tc6000` and exited before opening the cache or running inference. PBS wrapped the `exec_gpus` value across lines; the allocation contained six GPUs, but the first script parsed only five and correctly stopped at its pre-inference allocation guard. No model, prediction, label, monthly validation payload, test payload, or buffer payload was read by that attempt, and no weights were changed.

This independent `retry01/` directory fixes only the folded-line parser by removing newlines from `qstat -f` output before extracting allocated GPU IDs. The frozen models, cache, sample contract, score, and resource request are unchanged. The retry keeps its own output and submission receipt so the failed first attempt remains intact.
