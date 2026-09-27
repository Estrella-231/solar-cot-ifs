"""Paired, validation-only audit for the Zhujia AGRI/COT residual capacity control."""
import argparse,json
from pathlib import Path
import numpy as np

def met(y,p):
 e=p-y;return {'n':int(len(y)),'rmse':float(np.mean(e**2)**.5),'mae':float(np.mean(abs(e))),'bias':float(e.mean())}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--bank',type=Path,required=True);ap.add_argument('--base',type=Path,required=True);ap.add_argument('--source-run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 arms=('agri_residual','cot_residual');arr={k:np.load(a.run/f'{k}_val_predictions.npz') for k in arms};ref=arr[arms[0]]
 for k in arms[1:]:
  for key in ('row_ids','lead','true_ghi'): assert np.array_equal(arr[k][key],ref[key]),f'{key} mismatch {k}'
 row=ref['row_ids']; station=np.load(a.base/'station.npy',mmap_mode='r');seq=np.load(a.base/'sequence.npy',mmap_mode='r')
 assert np.all(station[row]==1);days=json.loads((a.bank/'complete.json').read_text())['initialization_BJT_days']['val']; day=np.asarray([days[str(int(s))] for s in seq[row]])
 unique=np.unique(day);y=ref['true_ghi'];pred={k:arr[k]['pred_ghi'] for k in arms}
 result={k:met(y,pred[k]) for k in arms};lead={}
 for li in np.unique(ref['lead']):
  m=ref['lead']==li;lead[str(int(li)*15+15)]={k:met(y[m],pred[k][m])['rmse'] for k in arms}
 rng=np.random.default_rng(20260928); ixday=[np.flatnonzero(day==d) for d in unique];boot=[]
 for _ in range(5000):
  draw=rng.integers(0,len(ixday),len(ixday));ix=np.concatenate([ixday[i] for i in draw]);boot.append(met(y[ix],pred['cot_residual'][ix])['rmse']-met(y[ix],pred['agri_residual'][ix])['rmse'])
 source=np.load(a.source_run/'zero_val_predictions.npz');lookup={int(r):i for i,r in enumerate(source['row_ids'])};ix=np.asarray([lookup[int(r)] for r in row]);assert all(int(r) in lookup for r in row);assert np.array_equal(source['lead'][ix],ref['lead']);assert np.array_equal(source['true_ghi'][ix],ref['true_ghi']);anchor=source['pred_ghi'][ix];max_abs_anchor_diff=float(max(np.max(np.abs(pred[k]-anchor)) for k in arms));fallback=max_abs_anchor_diff<=1e-4
 out={'state':'PAIRED_ZHUJIA_CAPACITY_CONTROL_AUDIT_COMPLETE','test_used':False,'rows':int(len(row)),'initialization_days':int(len(unique)),'row_truth_lead_keys_identical':True,'metrics':result,'cot_minus_agri_rmse':result['cot_residual']['rmse']-result['agri_residual']['rmse'],'paired_initialization_day_bootstrap':{'repetitions':5000,'seed':20260928,'rmse_delta_cot_minus_agri_95pct':[float(x) for x in np.quantile(boot,[.025,.975])],'fraction_cot_better':float(np.mean(np.asarray(boot)<0))},'lead_rmse':lead,'both_best_predictions_within_1e-4_of_zero_anchor':bool(fallback),'max_abs_difference_from_zero_anchor_W_m2':max_abs_anchor_diff,'largest_cot_minus_agri_squared_error_rows':[{'row_id':int(row[i]),'lead_min':int(ref['lead'][i]*15+15),'truth':float(y[i]),'agri_prediction':float(pred['agri_residual'][i]),'cot_prediction':float(pred['cot_residual'][i]),'delta_squared_error':float((pred['cot_residual'][i]-y[i])**2-(pred['agri_residual'][i]-y[i])**2)} for i in np.argsort((pred['cot_residual']-y)**2-(pred['agri_residual']-y)**2)[-12:][::-1]],'source_run':str(a.source_run)}
 a.out.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
