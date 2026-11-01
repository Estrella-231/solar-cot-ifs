"""CPU witness for the frozen SimVP train/validation contract loader."""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def main():
    p = argparse.ArgumentParser()
    for name in ("project", "source", "agri-root", "cache", "manifests", "checkpoint"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    # The frozen data-contract loader's model factory calls .cuda() directly.
    # Replace only module placement for this CPU contract witness.
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self
    supervision = a.project / "experiments/hunan_simvp_supervision_20261003_v2/code"
    r_scripts = a.project / "experiments/monthly_R_station16_20261013_v1/scripts"
    sys.path[:0] = [str(supervision), str(r_scripts)]
    from train_simvp_supervision_20261003 import load_contract

    train, val, norm, checkpoint, model = load_contract(SimpleNamespace(
        source=a.source, root=a.agri_root, cache=a.cache,
        manifests=a.manifests, checkpoint=a.checkpoint))
    assert len(train) == 35_320 and len(val) == 3_140
    assert model is not None and checkpoint.get("epoch") == 10
    assert norm["validation_test_used"] is False
    print("TRAIN_VAL_CONTRACT_PASS", {
        "train": len(train), "validation": len(val), "test_payloads_read": 0,
        "model_device": next(model.parameters()).device.type,
    })


if __name__ == "__main__":
    main()
