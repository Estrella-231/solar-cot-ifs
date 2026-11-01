# Frozen forecast-COT GHI-head inference counterfactual

## Question

The three-seed oracle-trained AC head improves validation GHI RMSE substantially, while the same three-seed head trained on five-fold OOF forecast COT has no reliable gain over history-only COT. This inference-only counterfactual asks whether the already-trained forecast-COT AC head can use correctly aligned future COT when it is supplied at validation time.

## Fixed inputs and treatment

- Use the exact same 66,590 validation `(station, init_time, target_time, lead)` rows, GHI labels, clear-sky values, forecast AGRI, geometry, history COT, and masks from the causal forecast pack.
- Use the already-trained forecast-COT AC checkpoints for seeds 42/43/44 at both Sili and Zhujia; do not retrain or select a new checkpoint.
- To reduce wall-clock time without changing the comparison, run the three frozen seeds concurrently on three PBS-allocated GPUs (one seed per GPU); each process evaluates both stations with the same fixed inputs and settings.
- Baseline inference uses the forecast-COT validation pack unchanged.
- Counterfactual inference replaces only future COT slots with observed-future-AGRI→frozen-R oracle COT. Historical COT slots remain byte-identical. Oracle COT is standardized with the forecast pack's train-only COT mean/std so that no validation-fitted normalization is introduced.
- Require byte-identical validation keys and equal non-COT validation arrays between packs. Recompute baseline inference and require it to reproduce the saved forecast-COT predictions before scoring the substitution.

## Evaluation and interpretation

Report paired GHI RMSE for each station and seed, equal station×seed mean delta, and a 2,000-replicate bootstrap over the 52 target BJT date blocks. Also report +15–60 and +120–240 min guards. Negative counterfactual-minus-baseline delta favors oracle substitution.

If the frozen forecast-COT head improves under oracle substitution, it demonstrably responds to future-COT inputs and the remaining gap is upstream forecast-COT fidelity/alignment or train-validation source shift. If it does not, the learned head's fusion/representation is the next bottleneck. This is an oracle-only validation diagnostic, not deployable forecast skill. Test and buffer payloads remain closed; no manuscript text is edited.
