# T07 provisional 3×3 head design

Status: prototype only; not a frozen contract and not a training result.

## Candidate structure

- Per-pixel AGRI + geometry encoder: two 1×1 convolutions, preserving all nine spatial positions.
- A: flatten the encoded 3×3 field in fixed row-major order; MLP; softplus output for nonnegative `kt`.
- AC: append a per-pixel one-channel COT encoder output before the same style readout.
- AP: append a learned low-width per-pixel feature projection from the AGRI+geometry representation; this controls whether AC's gain can be explained by auxiliary learned capacity.
- No spatial/global pooling and no COT mean, percentile, or full-patch average.
- Candidate counts with two provisional geometry planes: A 22,177; AC 24,489; AP 24,613 trainable parameters. AC/AP differ by 0.51%.

## Witness and limits

`witness_head_candidate_v0.py` uses only synthetic CPU tensors. It checks output shape/finiteness/nonnegativity, A's COT independence, identical common trunk initialization, AC sensitivity and nonzero gradient to COT, AP gradients to AGRI/geometry, and AC/AP parameter matching. The witness passes. Its provisional shape is `AGRI [B,13,3,3]`, `geometry [B,2,3,3]`, and `COT [B,1,3,3]`.

Geometry channel count, exact geometry variables and normalization, COT normalization, mask/data contract, optimizer/update budget and site-specific data interface are not frozen until T03/T01 and the retrieval/cache dependencies are accepted. This prototype must not be used for GPU training or interpreted as evidence of COT value. Final T07 should bind the accepted data contract and R output manifest, then rerun this witness against the final configuration.
