# Retry 04 note

PBS job `215550.tc6000` passed six-GPU allocation and cache/checkpoint-hash preflight, then produced four valid frozen-model validation predictions. It stopped before scoring when the inference script required an optional monthly-validation counter absent from two reconstructed seed-44 training receipts. The split field on all six receipts explicitly states that official monthly validation, test, and buffer were not read; test/buffer counts are zero, and the optional counter is zero wherever present.

This isolated retry fixes only that inference-receipt predicate, using a zero default for the optional counter while requiring the explicit train-date-only split statement. All six frozen models are rerun together to keep one uniform inference-code hash and a single clean six-model prediction set. The four earlier predictions remain intact under retry03 and are not substituted silently. Checkpoint weights, validation cache, model inputs, and scorer are unchanged.
