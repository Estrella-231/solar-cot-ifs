"""Fixed-head future-COT mean/packet-shuffle controls; no training."""
import csv,hashlib,json
from pathlib import Path
import sys
import numpy as np
import torch
B=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/HunanDyDiff_20261028/cot_ghi_diagnostic')
H=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/cloud_probability_input_20261020')
sys.path.insert(0,str(H))
from train_four_arm import FourArmIntervalHead
torch.set_num_threads(8);torch.set_num_interop_threads(1)
with np.load(B/'audit_features1/features.npz') as z:x=z['dydiff'];y=z['truth'];clear=z['clear'];observed_baseline=z['observed']
with np.load(B/'audit_canary1/features.npz') as z:
 assert np.array_equal(z['dydiff'],x),'future real AGRI or QA leaked into main input'
 assert not np.array_equal(z['observed'],observed_baseline),'canary did not change observed branch'
 canary={'status':'PASS_NO_FUTURE_OBSERVATION_OR_QA_EFFECT_ON_MAIN_INPUT','exact_max_abs_delta':float(np.max(abs(z['dydiff']-x))),'observed_branch_changed':True}
rows=list(csv.DictReader((B/'audit_features1/ghi_rows.csv').open()))
oldrows=list(csv.DictReader((B/'attempt1/ghi_rows.csv').open()));assert rows==oldrows
sites=np.array([r['station'] for r in rows]);leads=np.array([int(r['lead_minutes']) for r in rows]);k=leads//15-1
future=k[:,None]+np.arange(3)[None]>=2
conditions={'real_predicted_COT':x.copy(),'future_COT_train_mean':x.copy(),'all_COT_train_mean':x.copy()}
conditions['future_COT_train_mean'][:,:,15][future]=0
conditions['all_COT_train_mean'][:,:,15]=0
# Shuffle only the future COT sequence packet within the SAME station and lead.
# Preserve marginal values, the3x3 layout, and available actual historical COT.
rng=np.random.default_rng(20261028);permutations=[]
for repeat in range(20):
 donor=np.arange(len(rows))
 for s in ('sili','zhujia'):
  for lead in range(15,241,15):
   ids=np.flatnonzero((sites==s)&(leads==lead));assert len(ids)>1
   donor[ids]=np.roll(ids,int(rng.integers(1,len(ids))))
 assert np.all(donor!=np.arange(len(rows)))
 xx=x.copy()
 for j in range(3):
  ids=np.flatnonzero(future[:,j]);xx[ids,j,15]=x[donor[ids],j,15]
 conditions[f'shuffled_future_COT_{repeat:02d}']=xx;permutations.append(donor.tolist())
out={};replays=[]
with np.load(B/'attempt1/ghi_predictions.npz') as prior:
 for s in ('sili','zhujia'):
  ids=np.flatnonzero(sites==s)
  for seed in (42,43,44):
   p=H/f'results/four_arm_paired_v1_attempt1/seed{seed}'/s/'M10/best.pt'
   head=FourArmIntervalHead().eval().requires_grad_(False);head.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['model'])
   for name,xx in conditions.items():
    with torch.inference_mode():pred=(head(torch.from_numpy(xx[ids]))*torch.from_numpy(clear[ids])).numpy().astype(np.float64)
    arr=np.full(len(rows),np.nan);arr[ids]=pred;out[f'{s}_seed{seed}_{name}']=arr
    if name=='real_predicted_COT':
     diff=float(abs(pred-prior[f'{s}_seed{seed}_M10_dydiff'][ids]).max());assert diff<.01
     replays.append({'station':s,'seed':seed,'replay_max_abs_Wm2':diff})
dest=B/'audit_counterfactuals1';dest.mkdir(exist_ok=False)
np.savez_compressed(dest/'predictions.npz',truth=y,clear=clear,**out)
(dest/'receipt.json').write_text(json.dumps({'status':'PASS','canary':canary,'replay':replays,'shuffle_seed':20261028,'shuffles':20,'shuffle_rule':'same station/lead, nonself cyclic shift of whole temporal/spatial COT packet; future slots only','zero_mean_rule':'standardized COT zero means frozen training COT mean, not physical clear sky','donor_indices':permutations,'heads_frozen':True,'test_payloads_read':0},indent=2))
print('FIXED_HEAD_COUNTERFACTUALS_COMPLETE',flush=True)
