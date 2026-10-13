# Frozen GHI-head COT input-dependence diagnostics

Date: 2026-10-13 BJT  
Data: frozen new-monthly-split validation cohort only; test remained closed.

## Method

Without retraining, each seed-42/43/44 AC checkpoint for Sili and Zhujia was reloaded on CPU and evaluated on the same validation rows. Four COT interventions were applied to the normalized COT map, then the physical GHI RMSE and prediction changes were recomputed:

1. **Physical zero:** set all nine COT pixels to the normalized representation of physical COT=0.
2. **Center only:** retain the center COT and replace the eight neighbors with physical COT=0.
3. **Neighbor permutation:** preserve the center and all eight neighbor values, but permute the eight neighbor locations with one frozen permutation.
4. **Same-station/month mismatch:** replace each map with a different validation sample's COT from the same station and BJT month, using a deterministic cyclic shift.

The baseline for every comparison is the saved AC prediction from that same checkpoint. The table summarizes the six station×seed diagnostics with equal weight. These are fixed-input sensitivity probes, not bootstrap significance tests or model retraining results.

## Results

| Intervention | Mean RMSE change vs baseline (W m⁻²) | Range across 6 station×seed checks | Mean absolute prediction change (W m⁻²) |
|---|---:|---:|---:|
| Physical-zero COT | +89.900 | +32.970 to +122.018 | 120.490 |
| Center-only COT | +73.012 | +25.218 to +102.818 | 106.789 |
| Neighbor permutation, center retained | +0.027 | +0.006 to +0.064 | 0.383 |
| Same-station/month COT mismatch | +37.126 | +10.600 to +79.499 | 56.180 |

All six fixed checkpoints show RMSE worsening after each intervention. The distribution-preserving neighbor permutation changes predictions very little: average absolute GHI change is 0.383 W m⁻² and average RMSE change is 0.027 W m⁻². By contrast, assigning another same-station/month sample's COT map produces a much larger change, consistent with the head using sample-specific COT values. Removing COT values also causes a large shift, although the physical-zero input may be outside the training distribution.

## Interpretation

The evidence supports that the AC head depends on the magnitude and sample alignment of its COT input. It does **not** show strong use of the exact arrangement of the eight neighboring COT values in this 3×3 head: a value-preserving neighbor permutation barely changes validation predictions. Therefore this diagnostic does not justify claiming that the current head learned local cloud texture or directional spatial structure. It is compatible with a learned response to local COT intensity and its correspondence with the AGRI sample.

The center-only intervention also changes many input values to zero and shifts the input distribution, so its large deterioration cannot isolate the causal value of surrounding pixels. The same-month mismatch is less extreme, but remains an intentional cross-sample substitution. These tests establish model sensitivity, not physical causation. R also embeds a larger native receptive field and CPP supervision; the AP capacity arm does not control those R pathways.

The useful next controlled check is to repeat several frozen, center-preserving neighborhood permutations and spatial transforms while preserving each sample's full 3×3 COT value distribution, and report per-case output sensitivity. Only if that sensitivity remains small should a spatially structured COT encoder be justified as a new, separately named experiment. Keep the current successful A/AC/AP weights unchanged.

## Reproduction

- Results: `ghi_heads_cot_dependence_validation.json`.
- Script: `../diagnose_ghi_cot_inputs.py`.
- The script verifies all six seed/station checkpoint identities, compares reloaded predictions with saved AC outputs, and records `test_payload_opened=false`.
