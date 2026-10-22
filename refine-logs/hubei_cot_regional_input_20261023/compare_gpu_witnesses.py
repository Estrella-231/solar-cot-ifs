"""Compare independently reset 2/4-GPU global-step witnesses, before fit."""
import argparse,json
from pathlib import Path
import torch

ap=argparse.ArgumentParser();ap.add_argument('--two',type=Path,required=True);ap.add_argument('--four',type=Path,required=True)
ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();assert not a.out.exists()
two=json.loads((a.two/'WITNESS.json').read_text());four=json.loads((a.four/'WITNESS.json').read_text())
assert two['state']==four['state']=='PASS_GPU_GLOBAL_STEP_AND_CHECKPOINT_REPLAY'
assert two['world']==2 and four['world']==4
assert two['global_valid_pixels']==four['global_valid_pixels'] and two['global_batch']==four['global_batch']==128
x=torch.load(a.two/'witness.pt',map_location='cpu');y=torch.load(a.four/'witness.pt',map_location='cpu')
assert x['binding']==y['binding']
sq=base=0.;maximum=0.
for k,g in x['gradients'].items():
    h=y['gradients'][k]; diff=(g-h).double(); sq+=float((diff*diff).sum());base+=float(g.double().square().sum());maximum=max(maximum,float(diff.abs().max()))
relative=(sq/max(base,1e-30))**.5
loss_relative=abs(two['global_huber']-four['global_huber'])/max(abs(two['global_huber']),1e-12)
passed=relative<1e-3 and maximum<1e-4 and loss_relative<1e-4
result=dict(state='PASS_2_4_GPU_GLOBAL_BATCH_EQUIVALENCE' if passed else 'FAIL_PRESERVE_WITNESSES',
            gradient_relative_l2=relative,gradient_max_abs_difference=maximum,loss_relative_difference=loss_relative,
            tolerances=dict(gradient_relative_l2=1e-3,gradient_max_abs=1e-4,loss_relative=1e-4),
            monthly_val_scored=False,training_started=False)
a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result));assert passed,result
