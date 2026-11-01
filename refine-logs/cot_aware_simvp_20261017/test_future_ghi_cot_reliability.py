"""CPU witness for fold-routed, future-slot-only COT calibration."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def main():
    path = Path(__file__).with_name("train_future_ghi_heads_cot_reliability.py")
    spec = importlib.util.spec_from_file_location("future_ghi_reliability", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    params_excluded = {str(fold): {"sili": {"15": {"intercept": 10., "slope": .5},
                                               "30": {"intercept": 20., "slope": .25},
                                               "45": {"intercept": 30., "slope": .1}}}
                       for fold in range(5)}
    params_full = {"sili": {"15": {"intercept": 1., "slope": .5},
                            "30": {"intercept": 2., "slope": .25},
                            "45": {"intercept": 3., "slope": .1}}}
    calibration = {"fold_by_sequence_index": {"7": 2, "99": 3},
                   "parameters_excluding_each_fold": params_excluded,
                   "parameters_full_train": params_full}
    cot = np.ones((2, 3, 1, 3, 3), dtype=np.float32) * 4.
    keys = [{"station": "sili", "lead_minutes": "30", "sequence_index_within_split": "7"},
            {"station": "sili", "lead_minutes": "30", "sequence_index_within_split": "99"}]
    train = module.apply_reliability_calibration(cot, keys, "train", calibration)
    # For lead 30, context slots refer to +0 (history), +15, and +30.
    assert np.all(train[0, 0] == 4.)
    assert np.all(train[0, 1] == 12.)  # 10 + .5*4
    assert np.all(train[0, 2] == 21.)  # 20 + .25*4
    assert np.all(train[1, 1] == 12.) and np.all(train[1, 2] == 21.)
    val = module.apply_reliability_calibration(cot[:1], keys[:1], "val", calibration)
    assert np.all(val[0, 1] == 1. + .5 * 4.)
    assert np.all(val[0, 2] == 2. + .25 * 4.)
    try:
        module.apply_reliability_calibration(cot[:1], [{**keys[1], "sequence_index_within_split": "100"}], "train", calibration)
    except ValueError as exc:
        assert "fold assignment" in str(exc)
    else:
        raise AssertionError("missing fold assignment must fail closed")
    print("FUTURE_GHI_COT_RELIABILITY_TRANSFORM_WITNESS_PASS")


if __name__ == "__main__":
    main()
