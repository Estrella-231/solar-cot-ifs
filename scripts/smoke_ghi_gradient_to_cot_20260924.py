"""Show a frozen GHI head can return gradients to causal forecast-COT inputs."""
import argparse
import json
from pathlib import Path

import torch
from causal_cot_trajectory_solarresnet_v5 import CausalCOTTrajectorySolarResNetV5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("refuse to overwrite gradient witness")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = CausalCOTTrajectorySolarResNetV5().eval().requires_grad_(False)
    image = torch.randn(2, 16, 16, 16)
    meta = torch.randn(2, 5)
    history = torch.randn(2, 8, 1, 16, 16)
    forecast = torch.randn(2, 16, 1, 16, 16, requires_grad=True)
    lead = torch.tensor([3, 10])
    pred = model(image, meta, history, forecast, lead)
    pred.sum().backward()
    grad = forecast.grad.detach()
    causal = []
    for row, k in enumerate(lead.tolist()):
        before = float(grad[row, :k+1].abs().sum())
        after = float(grad[row, k+1:].abs().max())
        if not (before > 0 and after == 0):
            raise RuntimeError("GHI gradient missing or post-target gradient leakage")
        causal.append({"lead_index":k, "up_to_target_l1":before, "post_target_max_abs":after})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"state":"PASS", "test_used":False,
                                       "head_parameters_frozen":True,
                                       "forecast_cot_requires_grad":True,
                                       "causal_gradients":causal}, indent=2))
    print("GHI_TO_COT_CAUSAL_GRADIENT_WITNESS_PASS", flush=True)


if __name__ == "__main__":
    main()
