"""Verify unchanged architecture with seeded random tensors, no checkpoints."""
import argparse, importlib.util, json
from pathlib import Path
import torch
from regional_model import COTUNet, loss_parts

ap=argparse.ArgumentParser();ap.add_argument('--original-source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args();assert not a.out.exists();torch.set_num_threads(1);torch.manual_seed(42)
spec=importlib.util.spec_from_file_location('original_architecture_only',a.original_source)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
model=COTUNet();original=module.COTUNet();original.load_state_dict(model.state_dict(),strict=True)
model.eval();original.eval();x=torch.randn(2,16,64,64)
with torch.no_grad():difference=float((model(x)-original(x)).abs().max())
assert difference==0
model.train();p=model(x);target=torch.ones_like(p);mask=torch.ones_like(p,dtype=torch.bool)
mask[0,:,0,:]=False;target[~mask]=float('nan')
num,count=loss_parts(p,target,mask);(num/count).backward()
assert torch.isfinite(num) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
r=dict(state='PASS_CPU_ARCHITECTURE_AND_MASKED_NAN_BACKWARD',parameter_count=sum(v.numel() for v in model.parameters()),
       random_weight_strict_load=True,forward_max_abs_difference=difference,scientific_payloads_read=0,
       original_checkpoint_loaded=False,gpu_tested=False)
a.out.write_text(json.dumps(r,indent=2));print(json.dumps(r))
