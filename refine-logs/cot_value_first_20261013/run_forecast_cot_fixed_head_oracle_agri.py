#!/usr/bin/env python3
"""Offline attribution: fixed seed42 AC head, oracle target AGRI, swap same-time R COT for SimVP->R forecast COT."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
EXP=ROOT/'experiments/monthly_R_station16_20261013_v1'
PACK=EXP/'data/ghi_head_pack_new_r_v1'
BANK=ROOT/'experiments/cot_value_first_20261013/results/forecast_cot_val_bank_20261014_v1/forecast_cot_val_bank.npz'
OUT=ROOT/'experiments/cot_value_first_20261013/results/forecast_cot_fixed_head_oracle_agri_20261014_v1'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<22),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--batch-size',type=int,default=512);a=ap.parse_args()
 if OUT.exists(): raise FileExistsError(OUT)
 OUT.mkdir(parents=True)
 meta=json.loads((PACK/'pack_manifest.json').read_text());assert meta['state']=='COMPLETE' and meta['test_payload_opened'] is False
 rows=list(csv.DictReader((PACK/'rows.csv').open(newline='',encoding='utf-8')))
 lookup={(r['station'],r['target_time_utc']):i for i,r in enumerate(rows) if r['split']=='val'}
 assert len(lookup)==sum(r['split']=='val' for r in rows)
 z=np.load(BANK,allow_pickle=False);fc=z['forecast_log1p_cot'];times=z['target_time_utc'];seq=z['sequence_ids'];cpp=z['cpp_valid_mask']
 raw=np.load(PACK/'agri_raw.npy',mmap_mode='r',allow_pickle=False);valid=np.load(PACK/'agri_valid.npy',mmap_mode='r',allow_pickle=False);geo=np.load(PACK/'geometry.npy',mmap_mode='r',allow_pickle=False);cot=np.load(PACK/'cot_log1p.npy',mmap_mode='r',allow_pickle=False)
 am=np.asarray(meta['AGRI_norm_mean'],np.float32)[None,:,None,None];ast=np.asarray(meta['AGRI_norm_std'],np.float32)[None,:,None,None];cm=np.float32(meta['COT_norm_mean']);cs=np.float32(meta['COT_norm_std'])
 torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');records=[]
 for si,station in enumerate(('sili','zhujia')):
  ck=EXP/f'results/ghi_heads_new_r_seed42_v1/{station}/AC_best.pt';payload=torch.load(ck,map_location='cpu',weights_only=False)
  import sys;sys.path.insert(0,str(EXP))
  from head_candidate_v0 import LocalGHIHead
  model=LocalGHIHead('AC',2).to(device);model.load_state_dict(payload['model']);model.eval()
  idxs=[];fcrops=[];inst=[]
  for w in range(len(seq)):
   for l in range(16):
    t=str(times[w,l]);i=lookup.get((station,t))
    if i is None:continue
    idxs.append(i);fcrops.append(fc[w,si,l,7:10,7:10]);inst.append((int(w),str(seq[w]),t,(l+1)*15,float(np.mean(cpp[w,si,l,7:10,7:10]))))
  preds_true=[];preds_fore=[];truth=[];clear=[]
  for st in range(0,len(idxs),a.batch_size):
   ix=np.asarray(idxs[st:st+a.batch_size],np.int64);sl=slice(st,st+len(ix))
   x=np.where(np.asarray(valid[ix],bool), (np.asarray(raw[ix],np.float32)-am)/ast,0).astype(np.float32)
   g=np.asarray(geo[ix],np.float32);ct=((np.asarray(cot[ix],np.float32)-cm)/cs).astype(np.float32)
   cf=((np.asarray(fcrops[sl],np.float32)[:,None,:,:]-cm)/cs).astype(np.float32)
   with torch.inference_mode():
    tx=torch.from_numpy(x).to(device);tg=torch.from_numpy(g).to(device)
    pt=model(tx,tg,torch.from_numpy(ct).to(device)).cpu().numpy()
    pf=model(tx,tg,torch.from_numpy(cf).to(device)).cpu().numpy()
   preds_true.extend(pt.tolist());preds_fore.extend(pf.tolist())
   truth.extend([float(rows[i]['ghi']) for i in ix]);clear.extend([float(rows[i]['clear']) for i in ix])
  ptrue=np.asarray(preds_true,np.float64)*np.asarray(clear,np.float64);pfore=np.asarray(preds_fore,np.float64)*np.asarray(clear,np.float64);y=np.asarray(truth,np.float64)
  for j,(i,meta_i) in enumerate(zip(idxs,inst)):
   w,sid,t,lead,sup=meta_i
   records.append({'station':station,'sequence_id':sid,'window_index':w,'target_time_utc':t,'lead_min':lead,'pack_index':i,'ghi_obs_wm2':y[j],'ghi_same_time_R_COT_wm2':ptrue[j],'ghi_forecast_SimVP_R_COT_oracle_AGRI_wm2':pfore[j],'prediction_shift_wm2':pfore[j]-ptrue[j],'cpp_support_3x3_fraction':sup})
 with (OUT/'paired_predictions.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 by={}
 for station in ('sili','zhujia'):
  q=[r for r in records if r['station']==station];y=np.array([r['ghi_obs_wm2'] for r in q]);p=np.array([r['ghi_same_time_R_COT_wm2'] for r in q]);f=np.array([r['ghi_forecast_SimVP_R_COT_oracle_AGRI_wm2'] for r in q])
  by[station]={'n_forecast_instances':len(q),'unique_target_rows':len({r['target_time_utc'] for r in q}),'unique_dates':len({r['target_time_utc'][:10] for r in q}),'rmse_same_time_R_COT':float(np.sqrt(np.mean((p-y)**2))),'rmse_forecast_COT_oracle_AGRI':float(np.sqrt(np.mean((f-y)**2))),'delta_rmse_forecast_minus_same_time_COT':float(np.sqrt(np.mean((f-y)**2))-np.sqrt(np.mean((p-y)**2))),'mean_abs_GHI_shift':float(np.mean(np.abs(f-p)))}
 summary={'state':'COMPLETE_OFFLINE_COUNTERFACTUAL','claim_scope':'fixed seed42 AC head; actual target-time AGRI and geometry held fixed; only COT input swapped from same-time frozen-R output to SimVP-forecast-AGRI passed through the same frozen R','not_deployable_forecast_score':True,'test_payload_opened':False,'cpp_support_used_as_filter':False,'overlapping_forecast_instances_retained':True,'n_forecast_instances':len(records),'unique_target_keys':len({(r['station'],r['target_time_utc']) for r in records}),'unique_forecast_windows':len({r['sequence_id'] for r in records}),'by_station':by,'source_hashes':{'pack_manifest':sha(PACK/'pack_manifest.json'),'rows':sha(PACK/'rows.csv'),'forecast_bank':sha(BANK),'AC_head':{s:sha(EXP/f'results/ghi_heads_new_r_seed42_v1/{s}/AC_best.pt') for s in ('sili','zhujia')},'head_code':sha(EXP/'head_candidate_v0.py')},'interpretation':'Difference estimates the effect of forecast-COT substitution under oracle target AGRI. It cannot show operational end-to-end GHI forecast skill; it does not substitute forecast AGRI.'}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()


