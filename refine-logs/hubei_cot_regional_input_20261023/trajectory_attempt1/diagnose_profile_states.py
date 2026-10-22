import json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
a=torch.load(P/'profile2_fp32_attempt1/checkpoint.pt',map_location='cpu')
b=torch.load(P/'profile4_fp32_attempt1/checkpoint.pt',map_location='cpu')
maximum=max(float((v-b['model'][k]).abs().max()) for k,v in a['model'].items())
result=dict(state='PROFILE1_COMPARED_SEPARATELY_UPDATED_PARAMETER_STATES',global_steps=[a['global_step'],b['global_step']],
    maximum_parameter_difference=maximum,prior_failed_gradient_relative_difference=.001467001181515241,
    thresholds_unchanged=dict(gradient_relative_l2=.001,gradient_max_abs=.0001,loss_relative=.0001),
    remedy='both2/4 load the same nonzero-gradient canonical checkpoint, compare one shared global update',
    training_started=False)
(P/'PROFILE1_DIAGNOSIS.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
