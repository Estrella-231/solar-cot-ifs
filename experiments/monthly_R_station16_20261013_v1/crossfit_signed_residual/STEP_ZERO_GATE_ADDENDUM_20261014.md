# Step-zero selection gate for the OOF residual rerun

The first OOF residual run selected epoch 0 for all six station/seed pairs.
Epoch 0 was measured after one optimizer epoch; although exact zero-residual
identity was checked before training, that untrained identity was not included
as a checkpoint candidate. The result is retained unchanged as run v1.

Run v2 reuses the frozen v1 OOF A predictions/features and the same train/val
rows, fold assignment, model capacity, initialization, optimizer, permutations,
and training budget. The only change is to include the exact zero-residual
checkpoint (epoch -1) in validation checkpoint selection. If no trained epoch
beats A, the selected model remains exactly A. This is a predeclared guard
against a learned correction making the frozen baseline worse; it does not
change inputs or labels and does not open the test payload.
