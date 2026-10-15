#!/usr/bin/env python3
"""Second, row-weighted FP64 verification of the paired date-bootstrap result."""
import argparse,json
from pathlib import Path
import numpy as np

ARMS=('A_history','AC_history_old_R','AC_history_corrected_R')
PAIRS=(('AC_history_old_R','A_history'),('AC_history_corrected_R','A_history'),('AC_history_corrected_R','AC_history_old_R'))

def main():
 p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--score',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--bootstrap',type=int,default=2000);a=p.parse_args()
 z=np.load(a.archive,allow_pickle=False);pred=np.asarray(z['prediction'],np.float64);truth=np.asarray(z['truth'],np.float64);station=z['station'].astype(str);dates=z['dates'].astype(str)
 if pred.shape!=(3,3,len(truth)) or not np.isfinite(pred).all() or not np.isfinite(truth).all():raise ValueError('invalid saved prediction archive')
 days=np.unique(dates);di=np.searchsorted(days,dates);rng=np.random.default_rng(20261016);weights=rng.multinomial(len(days),np.full(len(days),1/len(days)),a.bootstrap)
 sites=[];site_boot=[];e2=(pred-truth[None,None,:])**2
 for st in ('sili','zhujia'):
  ids=np.flatnonzero(station==st);err=e2[:,:,ids];point=np.sqrt(err.mean(axis=2));sites.append(point)
  rw=weights[:,di[ids]];den=rw.sum(axis=1)
  if np.any(den<=0):raise ValueError('empty bootstrap row-weight denominator')
  b=np.empty((a.bootstrap,3,3))
  for seed in range(3):
   for arm in range(3):b[:,seed,arm]=np.sqrt((rw@err[seed,arm])/den)
  site_boot.append(b)
 means=np.stack(sites).mean((0,1));bm=np.stack(site_boot).mean((0,2));ref=json.loads(a.score.read_text())['groups']['all_validation']
 deltas={}
 for ca,ct in PAIRS:
  i=ARMS.index(ca);j=ARMS.index(ct);ci=np.quantile(bm[:,i]-bm[:,j],[.025,.975]).tolist();d=float(means[i]-means[j])
  recorded=ref['contrasts'][f'{ca} - {ct}']
  if abs(d-recorded['delta_RMSE'])>1e-8 or np.max(np.abs(np.asarray(ci)-recorded['date_cluster_CI95']))>1e-8:raise AssertionError(f'independent row-weighted recomputation mismatch: {ca}-{ct}')
  deltas[f'{ca} - {ct}']=dict(delta_RMSE=d,CI95=ci)
 out=dict(state='PASS_SECOND_ROW_WEIGHTED_FP64_RECOMPUTATION',arms=list(ARMS),equal_station_seed_mean_RMSE={arm:float(means[i]) for i,arm in enumerate(ARMS)},contrasts=deltas,validation_rows=len(truth),BJT_dates=len(days),bootstrap=a.bootstrap,test_payloads_read=0)
 a.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
