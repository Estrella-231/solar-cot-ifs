#!/usr/bin/env python3
"""Frozen AC head 2x2 offline attribution: true/predicted target AGRI x true/predicted COT."""
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
EXP=ROOT/'experiments/monthly_R_station16_20261013_v1'
PACK=EXP/'data/ghi_head_pack_new_r_v1'
BANK=ROOT/'experiments/cot_value_first_20261013/results/forecast_agri3x3_bank_20261014_v1/forecast_cot_val_bank.npz'
OUT=ROOT/'experiments/cot_value_first_20261013/results/fixed_head_2x2_agri_cot_attribution_20261014_v1'
sys.path.insert(0,str(EXP))
from head_candidate_v0 import LocalGHIHead
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<22),b''):h.update(b)
 return h.hexdigest()
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--batch-size',type=int,default=512);a=ap.parse_args()
 if OUT.exists():raise FileExistsError(OUT)
 meta=json.loads((PACK/'pack_manifest.json').read_text());assert meta['state']=='COMPLETE' and meta['test_payload_opened'] is False
 rows=list(csv.DictReader((PACK/'rows.csv').open(newline='',encoding='utf-8')))
 lookup={(r['station'],r['target_time_utc']):i for i,r in enumerate(rows) if r['split']=='val'}
 assert len(lookup)==sum(r['split']=='val' for r in rows)
 z=np.load(BANK,allow_pickle=False);fcot=z['forecast_log1p_cot'];fagri=z['forecast_agri_station_3x3'];times=z['target_time_utc'];seq=z['sequence_ids'];dsidx=z['dataset_indices']
 with np.load(ROOT/'experiments/cot_value_first_20261013/results/forecast_cot_val_bank_20261014_v1/forecast_cot_val_bank.npz',allow_pickle=False) as ref:
  for k in ('sequence_ids','dataset_indices','target_time_utc'):assert np.array_equal(z[k],ref[k]),k
 raw=np.load(PACK/'agri_raw.npy',mmap_mode='r',allow_pickle=False);valid=np.load(PACK/'agri_valid.npy',mmap_mode='r',allow_pickle=False);geo=np.load(PACK/'geometry.npy',mmap_mode='r',allow_pickle=False);cot=np.load(PACK/'cot_log1p.npy',mmap_mode='r',allow_pickle=False)
 am=np.asarray(meta['AGRI_norm_mean'],np.float32)[None,:,None,None];ast=np.asarray(meta['AGRI_norm_std'],np.float32)[None,:,None,None];cm=np.float32(meta['COT_norm_mean']);cs=np.float32(meta['COT_norm_std'])
 torch.set_num_threads(4);device=torch.device('cuda');records=[]
 for si,station in enumerate(('sili','zhujia')):
  ck=EXP/f'results/ghi_heads_new_r_seed42_v1/{station}/AC_best.pt';payload=torch.load(ck,map_location='cpu',weights_only=False)
  model=LocalGHIHead('AC',2).to(device);model.load_state_dict(payload['model']);model.eval()
  idxs=[];fag=[];fc=[];instances=[]
  for w in range(len(seq)):
   for l in range(16):
    t=str(times[w,l]);idx=lookup.get((station,t))
    if idx is None:continue
    idxs.append(idx);fag.append(fagri[w,si,l]);fc.append(fcot[w,si,l,7:10,7:10]);instances.append((str(seq[w]),t,(l+1)*15,int(w)))
  outputs={k:[] for k in ('TT','FT','TF','FF')};ys=[];clears=[]
  for st in range(0,len(idxs),a.batch_size):
   sl=slice(st,st+a.batch_size);ix=np.asarray(idxs[sl],np.int64)
   v=np.asarray(valid[ix],bool);assert v.all(), 'one or more common rows has invalid target AGRI in 3x3'
   xt=((np.asarray(raw[ix],np.float32)-am)/ast).astype(np.float32)
   xf=((np.asarray(fag[sl],np.float32)-am)/ast).astype(np.float32)
   g=np.asarray(geo[ix],np.float32)
   ct=((np.asarray(cot[ix],np.float32)-cm)/cs).astype(np.float32)
   cf=((np.asarray(fc[sl],np.float32)[:,None,:,:]-cm)/cs).astype(np.float32)
   tx=torch.from_numpy(xt).to(device);fx=torch.from_numpy(xf).to(device);tg=torch.from_numpy(g).to(device);tc=torch.from_numpy(ct).to(device);fc_t=torch.from_numpy(cf).to(device)
   with torch.inference_mode():
    out={'TT':model(tx,tg,tc),'FT':model(fx,tg,tc),'TF':model(tx,tg,fc_t),'FF':model(fx,tg,fc_t)}
   for k in out:outputs[k].extend(out[k].cpu().numpy().tolist())
   ys.extend(float(rows[i]['ghi']) for i in ix);clears.extend(float(rows[i]['clear']) for i in ix)
  y=np.asarray(ys,np.float64);cl=np.asarray(clears,np.float64)
  pred={k:np.asarray(v,np.float64)*cl for k,v in outputs.items()}
  for j,(i,inst) in enumerate(zip(idxs,instances)):
   sid,t,lead,w=inst
   records.append({'station':station,'sequence_id':sid,'window_index':w,'target_time_utc':t,'lead_min':lead,'pack_index':i,'ghi_obs_wm2':y[j],**{f'pred_{k}_wm2':pred[k][j] for k in pred}})
 OUT.mkdir(parents=True)
 with (OUT/'paired_predictions.csv').open('w',newline='',encoding='utf-8') as f:
  wr=csv.DictWriter(f,fieldnames=list(records[0]));wr.writeheader();wr.writerows(records)
 receipt={'state':'COMPLETE_FROZEN_AC_2X2_OFFLINE_ATTRIBUTION','test_payload_opened':False,'forecast_bank_sha256':sha(BANK),'pack_manifest_sha256':sha(PACK/'pack_manifest.json'),'pack_rows_sha256':sha(PACK/'rows.csv'),'head_sha256':{s:sha(EXP/f'results/ghi_heads_new_r_seed42_v1/{s}/AC_best.pt') for s in ('sili','zhujia')},'head_code_sha256':sha(EXP/'head_candidate_v0.py'),'instances':len(records),'unique_sequences':len(set(r['sequence_id'] for r in records)),'design':{'TT':'actual target AGRI, actual-time frozen-R COT','FT':'SimVP forecast AGRI, actual-time frozen-R COT','TF':'actual target AGRI, SimVP forecast AGRI -> frozen R COT','FF':'SimVP forecast AGRI and its frozen-R COT'},'interpretation':'All arms use the same fixed seed42 AC head; TT is oracle same-time reference, FF is a fixed-head operational-input replay but not a trained forecast-GHI head or publication score.'}
 (OUT/'inference_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
