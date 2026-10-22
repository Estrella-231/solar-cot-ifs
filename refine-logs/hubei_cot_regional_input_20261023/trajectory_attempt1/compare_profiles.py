import json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
two=json.loads((P/'profile2_fp32_attempt1/PROFILE.json').read_text());four=json.loads((P/'profile4_fp32_attempt1/PROFILE.json').read_text())
assert two['state']==four['state']=='PASS_PROFILE_AND_FULL_STATE_REPLAY'
assert two['global_count']==four['global_count'] and two['global_batch']==four['global_batch']==128
g2=torch.load(P/'profile2_fp32_attempt1/gradients.pt',map_location='cpu')
g4=torch.load(P/'profile4_fp32_attempt1/gradients.pt',map_location='cpu');assert set(g2)==set(g4)
aa=torch.cat([v.flatten().double() for _,v in sorted(g2.items())]);bb=torch.cat([v.flatten().double() for _,v in sorted(g4.items())])
relative=float(torch.linalg.vector_norm(aa-bb)/torch.linalg.vector_norm(aa).clamp_min(1e-15));maximum=float(abs(aa-bb).max())
loss_relative=abs(two['global_loss']-four['global_loss'])/max(abs(two['global_loss']),1e-15)
assert relative<1e-3 and maximum<1e-4 and loss_relative<1e-4,(relative,maximum,loss_relative)
amp={n:json.loads((P/f'profile{n}_amp_attempt1/PROFILE.json').read_text()) for n in [2,4]}
assert all(r['state']=='PASS_PROFILE_AND_FULL_STATE_REPLAY' and r['resume_max_abs']<1e-7 for r in amp.values())
speedup=amp[4]['frames_per_second']/amp[2]['frames_per_second']
result=dict(state='PASS_TRAJECTORY_2_4_GPU_CORRECTNESS_AND_SPEED',gradient_relative_l2=relative,gradient_max_abs=maximum,
    loss_relative_difference=loss_relative,profiles=amp,speedup_4_vs_2=speedup,selected_world=4 if speedup>=1.15 else 2,
    global_batch=128,selection_rule='4 GPUs if at least15 percent faster, else2',formal_training_started=False)
assert not (P/'PROFILE_DECISION_attempt1.json').exists()
(P/'PROFILE_DECISION_attempt1.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
