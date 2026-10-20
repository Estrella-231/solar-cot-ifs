"""CPU-only R replay for source-time audit; never reports forecast scores."""
import sys,csv,json,importlib.util
from pathlib import Path
import numpy as np,torch
b=Path(sys.argv[1]);d=b/'diagnosis_v1/source_events';report=json.loads((d/'report.json').read_text());r=b.parent/'hubei_cloud_probability_20261020';torch.set_num_threads(2)
sp=importlib.util.spec_from_file_location('r',b/'code_attempt2/train_cot_repaired.py');mod=importlib.util.module_from_spec(sp);sp.loader.exec_module(mod);model=mod.COTUNet().eval();model.load_state_dict(torch.load(r/'R_correctutc_attempt3/best.pt',map_location='cpu')['model']);norm=json.loads((r/'R_pack_correctutc_attempt2/norm.json').read_text());mean=np.array(norm['mean'],np.float32)[None,:,None,None];std=np.array(norm['std'],np.float32)[None,:,None,None]
def invert(ag,ge):
 valid=np.isfinite(ag).all((1,2,3))&np.isfinite(ge).all((1,2,3));o=np.full((len(ag),1,16,16),np.nan,np.float32);raw=np.concatenate((ag[valid],ge[valid]),1);day=raw[:,15:16]>.5;v=(raw-mean)/std;v[:,15:16]=np.where(day,v[:,15:16],0.)
 with torch.inference_mode():cot=torch.expm1(model(torch.from_numpy(v))).numpy();cot[~day]=0
 o[valid]=cot;return o
for e in report['events']:
 init=e['init'];z=np.load(d/(init+'.npz'));bundle={k:z[k] for k in z.files}
 for name,a,g in [('canonical',z['canonical_agri'],z['canonical_geometry']),('saved_truth',z['saved_truth'],z['saved_geometry']),('saved_pred',z['saved_pred'],z['saved_geometry'])]:bundle[name+'_cot']=invert(a,g)
 for st in ('Niushou','Wangzhai','Mizhuang'):
  for arm in ('FA','FAC'):
   pred=[]
   for seed in (42,43,44):
    p=b/f'runtime_attempt2/paired_seed{seed}_attempt1'/st/f'{arm}_validation_predictions_with_keys.npz'
    with np.load(p,allow_pickle=False) as saved:
     take=np.flatnonzero(saved['init_time_bjt']==init);arr=np.full(16,np.nan);truth=np.full(16,np.nan)
     for j in take:
      l=int(saved['lead_minutes'][j])//15-1;arr[l]=saved['pred_ghi_wm2'][j];truth[l]=saved['truth_ghi_wm2'][j]
    pred.append(arr)
   bundle[st+'_'+arm]=np.stack(pred);bundle[st+'_truth']=truth
 np.savez_compressed(d/(init+'_with_cot_ghi.npz'),**bundle)
print('SOURCE_TIME_EVENT_REPLAY_COMPLETE_NO_FORECAST_SCORES',flush=True)
