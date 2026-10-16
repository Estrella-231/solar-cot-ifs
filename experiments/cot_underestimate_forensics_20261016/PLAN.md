# Fixed-model COT underestimation forensics

Scope approved by the user on 2026-10-16: keep the current 3×3 GHI inputs,
three observed frames (T−30/T−15/T), six frozen corrected-R-COT GHI models,
and all training/validation rows unchanged. No training, qsub, test reads, or
16×16 GHI experiment. A 16×16 RGB display is context from the existing R input,
not a change to the GHI model input.

1. Recompute AC-versus-A output changes on the 4,263 common validation rows
   and all three seeds. These are independently fitted heads; output differences
   alone do not identify a causal COT effect.
2. Define severe diagnostic anchors before opening cloud images: every seed
   underestimates GHI by >20 W/m² and AC lowers it by >5 W/m². Select the largest
   mean squared-error deterioration on three distinct BJT dates per station.
   Match each to an all-seed helpful upward-correction case, using station,
   solar zenith, center log1p COT, and season; report matching distances.
   These selected cases are not a representative performance estimate.
3. Use existing real AGRI, exact saved 3×3 COT histories, and NOMObsTime-derived
   scan offsets. Verify source frame/row/column and exact GHI interval means;
   compare native five-minute records only as timing diagnostics, never as
   interchangeable 15-minute evaluation targets. Future records are diagnosis
   only, never model inputs. Missing records remain missing.
4. On CPU, reload each frozen AC checkpoint and reproduce saved predictions.
   Compute local input gradients dGHI/dlog1p(COT) at the exact original input.
   A zero-standardized-COT intervention is separately labeled out-of-distribution
   sensitivity, not an AGRI-only trained baseline or physical causal effect.
5. Reuse the four already extracted validation CTH/parallax/shadow geometries
   only after an exact station/time/grid join with current predictions. Keep old
   model predictions out of the new table. Cloud-shadow geometry is an
   approximate product-based screen, not a cloud-shadow truth label.
6. Deliver row/date summaries, paired true-cloud figures, source receipts,
   frozen-model sensitivity, and an evidence-bounded recommendation about
   time representation, cloud-shadow alignment, or the learned GHI mapping.
   Keep the paper and shared tracker unchanged; upload only this diagnostic
   folder after analysis. Full input caches and checkpoints stay off Git.
