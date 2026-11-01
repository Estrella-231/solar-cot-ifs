"""Bounded CPU export of existing validation references and saved COT only."""
import csv,json,hashlib
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np

ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
OUT=ROOT/'refine-logs/forecast_thick_cloud_events_20261102'
OUT.mkdir(parents=True,exist_ok=True)
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def bjt(x):return (datetime.fromisoformat(x)+timedelta(hours=8)).strftime('%Y%m%d%H%M')
def write(name,rr):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
refs=[];pred=[];meta={'test_payloads_read':0,'new_inference':False,'sources':{}}
pack=ROOT/'refine-logs/fulltrain_forecast_cot_20261101/pack/val'
hn=rows(pack/'keys.csv');assert len(hn)==66086 and all(r['split']=='val' for r in hn)
hc=np.load(pack/'cot_log1p.npy',mmap_mode='r');assert hc.shape==(len(hn),3,1,3,3)
for i,r in enumerate(hn):pred.append({'region':'Hunan','station':r['station'],'init':bjt(r['init_time_utc']),'target':bjt(r['target_time_utc']),'cot':float(np.expm1(hc[i,-1,0,1,1]))})
wanted={(r['station'],r['target_time_utc']) for r in hn}
cp=ROOT/'experiments/r_solar_geometry_corrected_20261015/data/paired_common_v1'
rr=rows(cp/'rows_supervised.csv');a=np.load(cp/'cot.npy',mmap_mode='r');m=np.load(cp/'common_mask.npy',mmap_mode='r')
for i,r in enumerate(rr):
 if r['split']!='val' or (r['station'],r['target_time_utc']) not in wanted:continue
 valid=bool(m[i,0,8,8]);v=float(a[i,0,8,8]);valid=valid and np.isfinite(v) and 0<=v<=100
 refs.append({'region':'Hunan','station':r['station'],'target':bjt(r['target_time_utc']),'cpp':v if valid else '', 'valid':int(valid),'source_index':i})
meta['sources']['Hunan_CPP']={'root':str(cp),'rows_sha256':hashlib.sha256((cp/'rows_supervised.csv').read_bytes()).hexdigest(),'val_rows_exported':sum(r['region']=='Hunan' for r in refs)}
hbbase=ROOT/'refine-logs/hubei_threeframe_dydiff_forecast_20261102';hh=rows(hbbase/'evaluation/keys.csv');assert len(hh)==24451
stpos={'Niushou':(8,6),'Wangzhai':(9,8),'Mizhuang':(7,10)}
want={r['target_time_bjt'] for r in hh}
for k in range(4):
 pp=hbbase/f'rank{k}';rk=rows(pp/'keys.csv');f=np.load(pp/'features_raw.npy',mmap_mode='r');assert f.shape==(len(rk),3,17,3,3)
 for i,r in enumerate(rk):pred.append({'region':'Hubei','station':r['station'],'init':r['init_time_bjt'],'target':r['target_time_bjt'],'cot':float(np.expm1(f[i,-1,15,1,1]))})
cp=ROOT/'experiments/hubei_solar_geometry_repair_20261021/prepared_attempt12/R_packs/full'
rr=rows(cp/'rows.csv');a=np.load(cp/'target.npy',mmap_mode='r');m=np.load(cp/'mask.npy',mmap_mode='r')
for i,r in enumerate(rr):
 if r['split']!='val' or r['time_bjt'] not in want:continue
 for st,(y,x) in stpos.items():
  valid=bool(m[i,0,y,x]);v=float(a[i,0,y,x])*100.0;valid=valid and np.isfinite(v) and 0<=v<=100.00001
  refs.append({'region':'Hubei','station':st,'target':r['time_bjt'],'cpp':v if valid else '', 'valid':int(valid),'source_index':i})
meta['sources']['Hubei_CPP']={'root':str(cp),'rows_sha256':hashlib.sha256((cp/'rows.csv').read_bytes()).hexdigest(),'target_units':'packed COT/100; multiply 100 exactly as train_corrected_r.py before log1p','val_rows_exported':sum(r['region']=='Hubei' for r in refs)}
assert len({(r['region'],r['station'],r['target']) for r in refs})==len(refs)
assert len({(r['region'],r['station'],r['init'],r['target']) for r in pred})==90537
write('reference_centers.csv',refs);write('forecast_cot_centers.csv',pred)
meta.update({'reference_rows':len(refs),'forecast_rows':len(pred),'files':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['reference_centers.csv','forecast_cot_centers.csv']}})
(OUT/'EXPORT_CENTERS.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta))
