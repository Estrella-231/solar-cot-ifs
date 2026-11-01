"""CPU causality, shape and gradient witness for CausalCOTTrajectoryHead."""
import json
from pathlib import Path

import torch

from causal_cot_trajectory_head_v4 import CausalCOTTrajectoryHead


def main(output):
    torch.manual_seed(42)
    model = CausalCOTTrajectoryHead().train()
    n = 3
    x = torch.randn(n, 13, 16, 16)
    history = torch.randn(n, 8, 1, 16, 16, requires_grad=True)
    forecast = torch.randn(n, 16, 1, 16, 16, requires_grad=True)
    geometry = torch.randn(n, 3)
    station = torch.tensor([0, 1, 0])
    lead = torch.tensor([0, 5, 15])
    baseline = model(x, history, forecast, geometry, station, lead)
    baseline.square().mean().backward()
    grad = float(forecast.grad.abs().sum())
    # Future COT strictly after lead 5 must not affect this sample's output.
    changed = forecast.detach().clone()
    changed[1, 6:] += 1000
    altered = model(x, history.detach(), changed, geometry, station, lead)
    causal_error = float((baseline.detach()[1] - altered.detach()[1]).abs())
    if not torch.isfinite(baseline).all() or grad <= 0 or causal_error != 0:
        raise RuntimeError({"finite": bool(torch.isfinite(baseline).all()),
                            "forecast_gradient_l1": grad, "causal_error": causal_error})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "state": "PASS_CPU_CAUSAL_TRAJECTORY_WITNESS",
        "test_used": False,
        "history_steps": 8,
        "forecast_steps": 16,
        "station_readout": "feature at [8,8] after causal dilated 3D convolutions; no global spatial average",
        "future_after_target_abs_change": causal_error,
        "forecast_gradient_l1": grad,
        "output_shape": list(baseline.shape),
    }, indent=2) + "\n")
    print("CAUSAL_COT_TRAJECTORY_CPU_WITNESS_PASS", baseline.tolist())


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1] / "audits" / "causal_cot_trajectory_cpu_witness_20260918.json")
