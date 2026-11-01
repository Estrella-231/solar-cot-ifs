"""Independent central-difference witness for the saved exact-input COT gradient."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--diagnostic',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
 exp=a.root/'experiments/r_geometry_downstream_ghi_20261016';sys.path.insert(0,str(exp));from train_three_arm_ghi import IntervalHead
 run=exp/'results/paired_three_arm_v1';z=np.load(run/'validation_inputs.npz');norm=json.loads((a.root/'experiments/cot_interval_history_20261015_v1/data_attempt2/pack_manifest.json').read_text());setup=json.loads((run/'setup.json').read_text());cases=json.loads((a.diagnostic/'selected_cases.json').read_text())['cases'];probe=np.load(a.diagnostic/'results_v1/selected_frozen_probes.npz');cs=setup['COT_pool_train_std'];cm=setup['COT_pool_train_mean'];checks=[]
 for station in ('sili','zhujia'):
  ci=[j for j,c in enumerate(cases) if c['station']==station];ids=[cases[j]['validation_index'] for j in ci]
  raw=z['agri_raw'][ids];x=np.where(z['agri_valid'][ids],(raw-np.asarray(norm['AGRI_norm_mean'],np.float32)[None,None,:,None,None])/np.asarray(norm['AGRI_norm_std'],np.float32)[None,None,:,None,None],0).astype(np.float32)
  X=torch.tensor(x);G=torch.tensor(z['geometry'][ids].copy());C=torch.tensor(((z['cot_corrected_log1p'][ids]-cm)/cs).astype(np.float32));cl=torch.tensor([cases[j]['clear'] for j in ci],dtype=torch.float32);h=.001
  for si,seed in enumerate((42,43,44)):
   model=IntervalHead();model.load_state_dict(torch.load(run/f'seed{seed}'/station/'AC_history_corrected_R/best.pt',map_location='cpu',weights_only=False)['model']);model.eval().requires_grad_(False)
   with torch.inference_mode():fd=((model(X,G,C+h/cs)-model(X,G,C-h/cs))*cl/(2*h)).numpy()
   expected=probe['gradient_dGHI_dlog1pCOT'][si,ci].sum((1,2,3))
   for k,j in enumerate(ci):
    err=float(abs(fd[k]-expected[k]));allowed=max(.25,.005*abs(float(expected[k])));assert err<=allowed,(seed,j,fd[k],expected[k],err,allowed)
    checks.append(dict(seed=seed,case_index=j,autograd=float(expected[k]),central_difference=float(fd[k]),absolute_difference=err,tolerance=allowed))
 out=dict(state='PASS_INDEPENDENT_CENTRAL_DIFFERENCE',h_log1p_COT=h,checks=checks,max_abs_difference=max(r['absolute_difference'] for r in checks),weights_modified=False,test_payloads_read=0)
 (a.diagnostic/'results_v1/finite_difference_check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='checks'}))
if __name__=='__main__':main()
