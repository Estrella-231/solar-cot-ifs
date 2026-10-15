#!/usr/bin/env python3
"""Independent FP64 scoring and paired target-date bootstrap for three GHI arms."""
import argparse,csv,hashlib,json
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np

ARMS=('A_history','AC_history_old_R','AC_history_corrected_R')
STATIONS=('sili','zhujia')
PAIRS=(('AC_history_old_R','A_history'),('AC_history_corrected_R','A_history'),('AC_history_corrected_R','AC_history_old_R'))

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<22),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--pack',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=2000);a=p.parse_args()
 rows=list(csv.DictReader((a.run/'validation_rows.csv').open(newline='',encoding='utf-8')));packrows=list(csv.DictReader((a.pack/'rows.csv').open(newline='',encoding='utf-8')))
 if any(r['split']!='val' for r in rows) or any(r['split']=='test' for r in packrows):raise ValueError('test/nonvalidation row found')
 valids=np.asarray([i for i,r in enumerate(packrows) if r['split']=='val'],np.int64)
 if len(rows)!=len(valids) or len(rows)!=4263:raise ValueError('validation row coverage mismatch')
 for x,y in zip(rows, (packrows[i] for i in valids)):
  if (x['station'],x['target_time_utc'])!=(y['station'],y['target_time_utc']):raise ValueError('validation sample key/order differs from fixed pack')
 station=np.asarray([r['station'] for r in rows]);dates=np.asarray([(datetime.fromisoformat(r['target_time_utc'])+timedelta(hours=8)).date().isoformat() for r in rows]);udays=np.unique(dates);di=np.searchsorted(udays,dates);D=len(udays)
 pred=np.full((3,3,len(rows)),np.nan,np.float64);truth=np.full(len(rows),np.nan,np.float64);metric_by={};hashes={}
 completion=json.loads((a.run/'completion.json').read_text());
 if completion.get('state')!='COMPLETE_18_PAIRED_RUNS' or completion.get('test_payloads_read')!=0:raise ValueError('training completion gate failed')
 for si,seed in enumerate((42,43,44)):
  for st in STATIONS:
   ids=np.flatnonzero(station==st);expected=valids[ids]
   for ai,arm in enumerate(ARMS):
    zpath=a.run/f'seed{seed}'/st/arm/'validation_predictions.npz';z=np.load(zpath,allow_pickle=False)
    if not np.array_equal(z['pack_index'],expected):raise ValueError(f'prediction row mismatch: seed{seed}/{st}/{arm}')
    if not np.isfinite(z['prediction']).all() or not np.isfinite(z['truth']).all():raise ValueError('nonfinite saved predictions')
    if np.isfinite(truth[ids]).all():np.testing.assert_array_equal(truth[ids],z['truth'])
    truth[ids]=z['truth'];pred[si,ai,ids]=z['prediction']
    rmse=float(np.sqrt(np.mean((z['prediction'].astype(np.float64)-z['truth'].astype(np.float64))**2)))
    mae=float(np.mean(np.abs(z['prediction'].astype(np.float64)-z['truth'].astype(np.float64))))
    mbe=float(np.mean(z['prediction'].astype(np.float64)-z['truth'].astype(np.float64)))
    m=json.loads((zpath.parent/'metrics.json').read_text())
    if max(abs(rmse-m['rmse']),abs(mae-m['mae']),abs(mbe-m['mbe']))>1e-8:raise ValueError('independent metric differs from run receipt')
    metric_by[(seed,st,arm)]=dict(rmse=rmse,mae=mae,mbe=mbe,n=len(ids))
    hashes[str(zpath)]=sha(zpath)
 if not np.isfinite(pred).all() or not np.isfinite(truth).all():raise ValueError('incomplete common predictions')
 for i,r in enumerate(rows):
  t=datetime.fromisoformat(r['target_time_utc']);
  if datetime.fromisoformat(r['previous_time_utc'])!=t-timedelta(minutes=15) or datetime.fromisoformat(r['earlier_time_utc'])!=t-timedelta(minutes=30):raise ValueError('history timestamp contract failed')
  # Training materializes CSV labels as float32 before saving FP64 copies.
  # Compare against that exact trained-label representation, not raw decimal FP64.
  if float(np.float32(float(r['ghi'])))!=truth[i]:raise ValueError('prediction truth does not match fixed label row after FP32 label conversion')
  if (t+timedelta(hours=8)).date()!=(datetime.fromisoformat(r['earlier_time_utc'])+timedelta(hours=8)).date():raise ValueError('history crosses BJT date')
 def summarize(mask,seed_value=20261016):
  e2=(pred-truth[None,None,:])**2;per=[];boot=[];n_dates=len(np.unique(dates[mask]))
  if not mask.any():raise ValueError('empty score group')
  rng=np.random.default_rng(seed_value);w=rng.multinomial(D,np.full(D,1/D),a.bootstrap)
  for st in STATIONS:
   ids=np.flatnonzero(mask&(station==st))
   if not len(ids):continue
   cnt=np.bincount(di[ids],minlength=D).astype(np.float64)
   sums=np.stack([[np.bincount(di[ids],weights=e2[s,arm,ids],minlength=D) for arm in range(3)] for s in range(3)])
   point=np.sqrt(sums.sum(-1)/cnt.sum());per.append(point)
   denom=w@cnt
   if np.any(denom<=0):raise ValueError('empty bootstrap site resample')
   site=np.sqrt(np.einsum('bd,sad->bsa',w,sums)/denom[:,None,None]);boot.append(site)
  means=np.stack(per).mean((0,1));bmean=np.stack(boot).mean((0,2))
  contrasts={}
  for cand,ctrl in PAIRS:
   ci=ARMS.index(cand);ri=ARMS.index(ctrl);delta=float(means[ci]-means[ri]);bd=bmean[:,ci]-bmean[:,ri]
   contrasts[f'{cand} - {ctrl}']=dict(delta_RMSE=delta,relative_change_percent=float(100*delta/means[ri]),date_cluster_CI95=np.quantile(bd,[.025,.975]).tolist())
  site_metrics={}
  for st in STATIONS:
   ids=np.flatnonzero(mask&(station==st))
   if not len(ids):continue
   site_metrics[st]={arm:dict(RMSE=float(np.mean([np.sqrt(np.mean(e2[s,ai,ids])) for s in range(3)])),
                                    MAE=float(np.mean([np.mean(np.abs(pred[s,ai,ids]-truth[ids])) for s in range(3)])),
                                    MBE=float(np.mean([np.mean(pred[s,ai,ids]-truth[ids]) for s in range(3)]))) for ai,arm in enumerate(ARMS)}
  mae_equal={};mbe_equal={}
  for ai,arm in enumerate(ARMS):
   site_mae=[];site_bias=[]
   for st in STATIONS:
    ids=np.flatnonzero(mask&(station==st))
    if len(ids):
     site_mae.append(np.mean([np.mean(np.abs(pred[s,ai,ids]-truth[ids])) for s in range(3)]))
     site_bias.append(np.mean([np.mean(pred[s,ai,ids]-truth[ids]) for s in range(3)]))
   mae_equal[arm]=float(np.mean(site_mae));mbe_equal[arm]=float(np.mean(site_bias))
  return dict(n=int(mask.sum()),target_BJT_dates=int(n_dates),rmse_station_seed_mean={arm:float(means[i]) for i,arm in enumerate(ARMS)},mae_station_seed_mean=mae_equal,mbe_station_seed_mean=mbe_equal,contrasts=contrasts,per_station=site_metrics)
 zero=np.asarray([r['suspect_daylight_long_zero']=='1' for r in rows]);groups={'all_validation':np.ones(len(rows),bool),'score_only_excluding_source_zero_suspect':~zero}
 summaries={k:summarize(v) for k,v in groups.items()}
 seed_deltas={}
 seed_metrics=[]
 for seed in (42,43,44):
  site={}
  for st in STATIONS:
   site[st]={arm:metric_by[(seed,st,arm)] for arm in ARMS}
  seed_deltas[str(seed)]={f'{ca} - {ct}':float(np.mean([site[st][ca]['rmse']-site[st][ct]['rmse'] for st in STATIONS])) for ca,ct in PAIRS}
  for st in STATIONS:
   for arm in ARMS:seed_metrics.append(dict(seed=seed,station=st,arm=arm,**metric_by[(seed,st,arm)]))
 result=dict(state='PASS_INDEPENDENT_FP64_THREE_ARM_RECOMPUTATION',rows=len(rows),train_rows=47666,validation_rows=4263,validation_dates=len(udays),arms=list(ARMS),main_estimator='mean of site-specific RMSE across two stations and three paired seeds',bootstrap=f'{a.bootstrap} paired BJT target-date multinomial cluster resamples; fixed seed set; conditional on validation checkpoint selection',groups=summaries,per_seed_station_equal_delta_RMSE=seed_deltas,test_payloads_read=0,scientific_scope='same-time observed three-frame local AGRI reconstruction of unchanged 15-minute GHI; not causal forecasting; R frozen; CPP is retrieval reference only',source_hashes=hashes)
 a.output.mkdir(parents=True,exist_ok=True);(a.output/'independent_score.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 with (a.output/'seed_metrics.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(seed_metrics[0]));w.writeheader();w.writerows(seed_metrics)
 np.savez_compressed(a.output/'paired_validation_scoring.npz',prediction=pred,truth=truth,station=station,dates=dates,pack_index=valids)
 print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},indent=2))
if __name__=='__main__':main()
