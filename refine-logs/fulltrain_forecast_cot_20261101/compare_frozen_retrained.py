"""Compare saved predictions on canonical common station/init/target keys."""
import csv,json
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent
OLD=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/HunanDyDiff_20261028/hunan_full_validation_20261030/evaluation_attempt1')
OUT=B/'common_sample_diagnosis';OUT.mkdir(exist_ok=True)
def key(r):return (r['station'],str(datetime.fromisoformat(r['init_time_utc'])),str(datetime.fromisoformat(r['target_time_utc'])))
def main():
 rows=list(csv.DictReader((OLD/'ghi_rows.csv').open()))
 old=np.load(OLD/'ghi_predictions.npz',allow_pickle=False)
 oi={key(r):i for i,r in enumerate(rows)};assert len(oi)==len(rows)
 allruns=[];keysets=[];newkeys={};checks=[]
 for st in ('sili','zhujia'):
  for seed in (42,43,44):
   d=B/'heads'/f'{st}_seed{seed}'/st/f'seed{seed}'
   aa=np.load(d/'A/validation_predictions.npz',allow_pickle=False);cc=np.load(d/'AC/validation_predictions.npz',allow_pickle=False)
   assert np.array_equal(aa['keys'],cc['keys']) and np.array_equal(aa['truth_ghi_wm2'],cc['truth_ghi_wm2'])
   nr=[json.loads(k) for k in aa['keys']];ni={key(r):i for i,r in enumerate(nr)};assert len(ni)==len(nr)
   common=sorted(set(oi)&set(ni));keysets.append(set(common));newkeys[st]=set(ni)
   ix=np.array([oi[k] for k in common]);jx=np.array([ni[k] for k in common])
   yold=old['truth'][ix];yn=aa['truth_ghi_wm2'][jx];delta=yn-yold
   checks.append({'station':st,'seed':seed,'rows':len(common),'truth_max_abs':float(abs(delta).max()),'truth_rmse_difference':float(np.sqrt(np.mean(delta**2))),'truth_mismatch_gt_1e3':int(sum(abs(delta)>.001))})
   allruns.append({'station':st,'seed':seed,'keys':common,'old_y':yold,'new_y':yn,'frozen_A':old[f'{st}_seed{seed}_M00_dydiff'][ix],'frozen_AC':old[f'{st}_seed{seed}_M10_dydiff'][ix],'new_A':aa['prediction_ghi_wm2'][jx],'new_AC':cc['prediction_ghi_wm2'][jx],'full_new_A':aa['prediction_ghi_wm2'],'full_new_AC':cc['prediction_ghi_wm2'],'full_new_y':aa['truth_ghi_wm2']})
 print('JOIN',json.dumps(checks),flush=True)
 assert all(c['truth_mismatch_gt_1e3']==0 for c in checks),'Different labels: do not conflate native scores'
 dates=sorted({(datetime.fromisoformat(k[2])+timedelta(hours=8)).date().isoformat() for r in allruns for k in r['keys']})
 rng=np.random.default_rng(20261102);w=rng.multinomial(len(dates),np.full(len(dates),1/len(dates)),size=5000).astype(float)
 boot_cache={}
 def calc(scope,field1,field2):
  runs=[r for r in allruns if scope not in ('sili','zhujia') or r['station']==scope];points=[];boots=[];details=[]
  for r in runs:
   lead=np.array([int((datetime.fromisoformat(k[2])-datetime.fromisoformat(k[1])).total_seconds()/60) for k in r['keys']]);keep=(lead>=45) if scope=='lead45_240' else np.ones(len(lead),bool)
   if scope=='lead15_30':keep=lead<=30
   y=r['old_y'];p=np.stack([r[field1],r[field2]]);e=(p-y)**2;v=np.sqrt(e[:,keep].mean(axis=1));points.append(v)
   dt=np.array([(datetime.fromisoformat(k[2])+timedelta(hours=8)).date().isoformat() for k in r['keys']]);n=np.array([sum(keep&(dt==d)) for d in dates]);s=np.array([e[:,keep&(dt==d)].sum(axis=1) for d in dates]).T
   den=w@n;boots.append(np.sqrt((w@s.T)/np.maximum(den[:,None],1)))
   details.append({'station':r['station'],'seed':r['seed'],'before':float(v[0]),'after':float(v[1]),'bias_before':float((p[0]-y)[keep].mean()),'bias_after':float((p[1]-y)[keep].mean()),'rows':int(keep.sum())})
  v=np.mean(points,axis=0);bs=np.mean(boots,axis=0);boot_cache[(scope,field1,field2)]=bs
  return {'before':float(v[0]),'after':float(v[1]),'reduction_percent':float(100*(1-v[1]/v[0])),'delta_ci95':np.quantile(bs[:,1]-bs[:,0],[.025,.975]).tolist(),'runs':details}
 result={'label_checks':checks,'old_rows':len(rows),'new_rows':sum(len(x) for x in newkeys.values()),'common_rows':len(set.union(*keysets)),'comparisons':{}}
 for scope in ('all','sili','zhujia','lead15_30','lead45_240'):
  result['comparisons'][scope]={name:calc(scope,a,c) for name,a,c in [('frozen_COT_gain','frozen_A','frozen_AC'),('retrained_COT_gain','new_A','new_AC'),('no_COT_retraining','frozen_A','new_A'),('COT_retraining','frozen_AC','new_AC')]}
  oldb=boot_cache[(scope,'frozen_A','frozen_AC')];newb=boot_cache[(scope,'new_A','new_AC')]
  result['comparisons'][scope]['gain_change_delta_ci95']=np.quantile((newb[:,1]-newb[:,0])-(oldb[:,1]-oldb[:,0]),[.025,.975]).tolist()
 leadrows=[]
 for lead in range(15,241,15):
  vals=[]
  for r in allruns:
   keep=np.array([int((datetime.fromisoformat(k[2])-datetime.fromisoformat(k[1])).total_seconds()/60)==lead for k in r['keys']])
   vals.append([np.sqrt(np.mean((r[f][keep]-r['old_y'][keep])**2)) for f in ('frozen_A','frozen_AC','new_A','new_AC')])
  v=np.mean(vals,axis=0);leadrows.append({'lead_minutes':lead,**dict(zip(('frozen_A','frozen_AC','new_A','new_AC'),map(float,v)))})
 result['lead_metrics']=leadrows
 (OUT/'COMMON_SAMPLE_RESULTS.json').write_text(json.dumps(result,indent=2))
 with (OUT/'lead_metrics.csv').open('w',newline='') as f:wr=csv.DictWriter(f,fieldnames=list(leadrows[0]));wr.writeheader();wr.writerows(leadrows)
 print('RESULT_WRITTEN',str(OUT/'COMMON_SAMPLE_RESULTS.json'),flush=True)
if __name__=='__main__':main()
