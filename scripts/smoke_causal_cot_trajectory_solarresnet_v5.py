"""CPU contract witness for the SolarResNet-v5 causal trajectory head."""
import json
from pathlib import Path
import torch
from causal_cot_trajectory_solarresnet_v5 import CausalCOTTrajectorySolarResNetV5

torch.manual_seed(42)
model = CausalCOTTrajectorySolarResNetV5().eval()
image, meta = torch.randn(2, 16, 16, 16), torch.randn(2, 5)
history, forecast = torch.randn(2, 8, 1, 16, 16), torch.randn(2, 16, 1, 16, 16)
lead = torch.tensor([2, 9])
with torch.no_grad():
    first = model(image, meta, history, forecast, lead)
    changed = forecast.clone()
    for row, value in enumerate(lead.tolist()): changed[row, value + 1:] += 1000
    second = model(image, meta, history, changed, lead)
assert torch.equal(first, second), 'post-target COT affected output'
out = Path(__file__).resolve().parents[1] / 'audits' / 'causal_cot_trajectory_solarresnet_v5_cpu_witness.json'
out.write_text(json.dumps({'state': 'PASS', 'post_target_max_abs_change': float((first-second).abs().max()), 'test_used': False}, indent=2))
print('CAUSAL_COT_TRAJECTORY_SOLARRESNET_V5_CPU_WITNESS_PASS')
