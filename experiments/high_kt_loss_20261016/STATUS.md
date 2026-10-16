# 2026-10-16 — completed, candidate rejected

24/24 fresh paired fits completed on one local RTX4060 Laptop; same 3x3,
three-frame head, split, initialization, row order, batch1792 and 30 epochs.
Only train high-transmittance loss weight changed (1 to 2).

High-GHI>=800 AC RMSE175.070->148.322 W/m², but all-validation
107.784->110.004 and low/middle-kt regress. Both weighted arms fail the
prespecified overall/low-regime protection criteria. No model replacement,
weight/threshold search, additional architecture experiment, or test access.

Independent FP64 scoring, second row-weighted date bootstrap, CPU checkpoint
reloads, and visual inspection completed. See REPORT.md and the persistent
issue record ../../docs/HIGH_GHI_UNDERESTIMATION.md. High-GHI underprediction
remains unresolved; this is an executed partial repair with unacceptable costs.
