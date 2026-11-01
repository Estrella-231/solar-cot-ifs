"""CPU witness for the v2 scorer's independent station/time CPP re-indexing."""
import importlib.util
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCORER = (HERE.parents[1] / "refine-logs" / "cot_value_first_20261013" /
          "score_supervision_cot_allval_20261016_v2.py")
spec = importlib.util.spec_from_file_location("v2_scorer", SCORER)
scorer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scorer)

ref_keys, source = scorer.reference_key_map(
    HERE / "sources" / "rows.csv", HERE / "sources" / "validation_predictions.npz")
bank = np.load(HERE / "repaired_existing_271_windows" / "forecast_cot_val_bank.npz",
               allow_pickle=False)
refs, mask, pixels = scorer.reindex_reference(bank["target_time_utc"].astype(str), ref_keys)
assert np.array_equal(refs, bank["cpp_reference_cot"], equal_nan=True)
assert np.array_equal(mask, bank["cpp_valid_mask"].astype(bool))
assert pixels == int(mask.sum()) == 668672
assert source["key_rule"] == "(station,target_time_utc), full SUPERVISED index"

# Verify the CI helper emits a finite, correctly signed point CSI delta.
truth = np.zeros((3, 2, 16, 16, 16), dtype=np.float32)
mask_small = np.ones_like(truth, dtype=bool)
pred = {"original_simvp": np.zeros_like(truth), "mse_finetune": np.zeros_like(truth),
        "temporal_delta": np.zeros_like(truth), "spatial_gradient": np.zeros_like(truth)}
truth[:, :, :, 7:10, 7:10] = 35.0
for method in ("temporal_delta", "spatial_gradient"):
    pred[method][:, :, :, 7:10, 7:10] = 35.0
dates = np.tile(np.asarray([["2026-01-01"] * 16,
                            ["2026-01-02"] * 16,
                            ["2026-01-03"] * 16]), (1, 1))
lead_macro = {name: {scale: {f"s{si}_l{li}": 0.0 for si in range(2) for li in range(16)}
                     for scale in scorer.SCALES} for name in pred}
pair = scorer.bootstrap_pair_vectorized(pred, dates, truth, mask_small,
                                        np.random.default_rng(7), 30)
delta = pair["temporal_delta"]["original_simvp"]["local_3x3_pooled_CSI_delta"]
assert delta["point"] == 1.0
assert delta["ci95"] == [1.0, 1.0]
rmse_delta = pair["temporal_delta"]["original_simvp"]["local_3x3_macro_rmse_delta"]
assert rmse_delta["point"] == -35.0
assert rmse_delta["ci95"] == [-35.0, -35.0]
print("PASS: 271-window CPP key join exact; 668672 pixels; CSI point and paired bootstrap verified")
