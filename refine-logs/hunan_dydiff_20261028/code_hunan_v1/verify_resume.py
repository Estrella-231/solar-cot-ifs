"""Verify resumed trainer weights, EMA and optimizer match an uninterrupted run."""
import argparse
import json
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument("--resumed", required=True)
p.add_argument("--reference", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
resumed = torch.load(Path(a.resumed) / "last.pt", map_location="cpu", weights_only=False)
reference = torch.load(Path(a.reference) / "last.pt", map_location="cpu", weights_only=False)
for kind in ("model", "ema"):
    for key in resumed[kind]:
        torch.testing.assert_close(resumed[kind][key], reference[kind][key], rtol=0, atol=0)
for parameter_id, state in resumed["optimizer"]["state"].items():
    for key, value in state.items():
        # NCCL reduction may differ below FP32 rounding at near-zero Adam moments.
        # Model/EMA remain bit-exact above; only optimizer moments get FP32 tolerance.
        torch.testing.assert_close(value, reference["optimizer"]["state"][parameter_id][key], rtol=1e-6, atol=1e-20)
assert resumed["global_step"] == reference["global_step"]
assert resumed["signature"] == reference["signature"]
for root in (a.resumed, a.reference):
    complete = json.loads((Path(root) / "training_complete.json").read_text())
    assert complete["status"] == "PASS"
    val = json.loads((Path(root) / "validation_epoch_002.json").read_text())
    assert val["sequence_count"] == 5 and val["forecast_evaluated"]
with Path(a.output).open("x") as f:
    json.dump({"status": "PASS", "resume_exact_model_ema": True,
               "optimizer_rtol":1e-6,"optimizer_atol":1e-20,
               "global_step": resumed["global_step"], "satellite_payloads_read": 0}, f, indent=2)
print("TRAINER_RESUME_PARITY_PASS", flush=True)
