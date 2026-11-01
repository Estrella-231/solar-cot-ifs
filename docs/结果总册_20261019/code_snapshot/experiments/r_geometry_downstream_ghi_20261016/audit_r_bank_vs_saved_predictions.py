#!/usr/bin/env python3
"""Check bank inference against saved paired-R validation outputs on shared keys."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<22),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--r-experiment',type=Path,required=True);p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tolerance',type=float,default=.05);a=p.parse_args()
 rows=list(csv.DictReader((a.r_experiment/'data/paired_common_v1/rows_supervised.csv').open(newline='',encoding='utf-8')))
 keys=list(csv.DictReader((a.bank/'frame_keys.csv').open(newline='',encoding='utf-8')));lookup={(r['station'],r['target_time_utc']):int(r['index']) for r in keys}
 bank=np.load(a.bank/'frame_cot_log1p.npy',mmap_mode='r',allow_pickle=False);outs=[];common=0
 for ai,arm in enumerate(('old','corrected')):
  z=np.load(a.r_experiment/f'results/paired_v1/{arm}/validation_predictions.npz',allow_pickle=False);ids=z['array_indices'].astype(int);pred=z['pred_log1p_cot']
  for j,i in enumerate(ids):
   r=rows[i];k=(r['station'],r['target_time_utc'])
   if k not in lookup:continue
   common+=int(ai==0);d=np.abs(np.asarray(bank[lookup[k],ai,0])-np.asarray(pred[j,0,7:10,7:10],np.float32))
   outs.append((arm,float(d.max()),float(d.mean()),r['split']))
  if np.any(~np.isfinite(pred)):raise ValueError(f'nonfinite saved R validation output: {arm}')
 summaries={}
 for arm in ('old','corrected'):
  vals=np.asarray([x[1] for x in outs if x[0]==arm],np.float64)
  if not len(vals):raise ValueError(f'no common bank/validation R keys for {arm}')
  summaries[arm]=dict(shared_validation_rows=int(len(vals)),max_abs_center3x3=float(vals.max()),p99_abs_center3x3=float(np.quantile(vals,.99)),mean_abs_center3x3=float(np.mean([x[2] for x in outs if x[0]==arm])))
 if any(x['max_abs_center3x3']>a.tolerance for x in summaries.values()):raise AssertionError(f'bank outputs exceed FP16-vs-FP32 tolerance: {summaries}')
 report=dict(state='PASS_R_BANK_MATCHES_SAVED_R_VALIDATION',reason='same checkpoint and raw input reconstruction; validation outputs saved under autocast, bank runs FP32',tolerance=a.tolerance,bank_shape=list(bank.shape),summary=summaries,split_counts={s:sum(r['split']==s for r in keys) for s in ('train','val')},test_payloads_read=0,source_hashes={str(a.bank/'bank_manifest.json'):sha(a.bank/'bank_manifest.json'),str(a.bank/'frame_keys.csv'):sha(a.bank/'frame_keys.csv')})
 a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
