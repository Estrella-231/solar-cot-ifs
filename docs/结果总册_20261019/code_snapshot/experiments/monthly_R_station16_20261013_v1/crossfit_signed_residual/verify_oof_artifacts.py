#!/usr/bin/env python3
"""Verify OOF train-row coverage and residual-head identity on synthetic tensors."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from signed_residual_head_oof import SignedResidualHeadOOF, trainable_parameter_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-run", type=Path, required=True)
    ap.add_argument("--status", type=Path, required=True)
    a = ap.parse_args()
    status = json.loads(a.status.read_text())
    if status.get("state") != "COMPLETE" or status.get("test_payload_opened") is not False:
        raise ValueError("OOF run status gate failed")
    idx = np.load(a.oof_run / "train_pack_row_indices.npy", allow_pickle=False)
    pred = np.load(a.oof_run / "A_oof_train_prediction_ghi_wm2.npy", allow_pickle=False)
    feat = np.load(a.oof_run / "A_oof_train_encoded_features.npy", allow_pickle=False)
    fold = np.load(a.oof_run / "oof_fold_id.npy", allow_pickle=False)
    if len(idx) != len(pred) or len(idx) != len(fold) or feat.shape != (len(idx), 32, 3, 3):
        raise AssertionError("OOF arrays have inconsistent row coverage")
    if len(np.unique(idx)) != len(idx) or not np.isfinite(pred).all() or not np.isfinite(feat).all():
        raise AssertionError("OOF rows are duplicated or outputs nonfinite")
    counts = []
    for arm in ("AGRI", "AGRI_COT"):
        model = SignedResidualHeadOOF(arm)
        with torch.no_grad():
            x = torch.zeros(2, 13, 3, 3)
            g = torch.zeros(2, 2, 3, 3)
            c = torch.zeros(2, 1, 3, 3)
            clear = torch.full((2,), 500.0)
            base = torch.full((2,), 321.0)
            enc = torch.zeros(2, 32, 3, 3)
            output, _, residual = model(x, g, c, clear,
                                        base_override=base,
                                        encoded_override=enc)
        if not torch.equal(output, base) or torch.count_nonzero(residual).item() != 0:
            raise AssertionError("OOF step-zero identity failed")
        counts.append(trainable_parameter_count(model))
    if len(set(counts)) != 1:
        raise AssertionError(f"arms are not capacity matched: {counts}")
    print(json.dumps({"state": "PASS", "n_rows": int(len(idx)),
                      "fold_counts": np.bincount(fold).tolist(),
                      "feature_shape": list(feat.shape),
                      "equal_trainable_parameters": counts[0],
                      "test_payload_opened": False}, indent=2))


if __name__ == "__main__":
    main()
