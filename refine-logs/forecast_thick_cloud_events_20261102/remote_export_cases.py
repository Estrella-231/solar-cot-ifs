"""Extract real observed/forecast images and already saved R outputs, validation only."""
import csv,json,hashlib,sys
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np

ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907');P=ROOT/'refine-logs/forecast_thick_cloud_events_20261102';OUT=P/'cases';OUT.mkdir(exist_ok=True)
def rows(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def bjt(x):return (datetime.fromisoformat(x)+timedelta(hours=8)).strftime('%Y%m%d%H%M')
def compact(x):return datetime.fromisoformat(x).strftime('%Y%m%d%H%M')
cases=json.loads((P/'CASE_SELECTION.json').read_text())['cases'];sources=[]
hb=ROOT/'refine-logs/hubei_threeframe_dydiff_forecast_20261102';hbo=ROOT/'refine-logs/hubei_same_time_geometry_20261102'
hbsel=json.loads((hb/'selection.json').read_text())['windows'];obskeys=rows(hbo/'frames/keys.csv');obsix={r['time_bjt']:i for i,r in enumerate(obskeys)}
obsraw=np.load(hbo/'frames/raw.npy',mmap_mode='r');obscot=np.load(hbo/'frames/cot_log1p.npy',mmap_mode='r')
cpproot=ROOT/'experiments/hubei_solar_geometry_repair_20261021/prepared_attempt12/R_packs/full';rr=rows(cpproot/'rows.csv');hbref={r['time_bjt']:i for i,r in enumerate(rr) if r['split']=='val'};hbcp=np.load(cpproot/'target.npy',mmap_mode='r');hbcm=np.load(cpproot/'mask.npy',mmap_mode='r')
hcot={}
for rank in range(4):
 k=rows(hb/f'rank{rank}/cot_keys.csv');a=np.load(hb/f'rank{rank}/cot_log1p.npy',mmap_mode='r')
 for i,r in enumerate(k):hcot[int(r['window']),r['time_bjt']]=(a,i)
hn=ROOT/'refine-logs/fulltrain_forecast_cot_20261101';sn=ROOT/'experiments/hunan_monthly_block_20261002';rawroot=Path('/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed')
seq=rows(sn/'manifests_v1/hunan_8to16.csv');seqval=[r for r in seq if r['split']=='val'];idx=rows(hn/'bank_merged/index.csv');bankix={int(r['sequence_index_within_split']):i for i,r in enumerate(idx) if r['split']=='val'}
bankag=np.load(hn/'bank_merged/forecast_agri_raw.npy',mmap_mode='r');bankcot=np.load(hn/'bank_merged/forecast_cot_log1p.npy',mmap_mode='r');hnkeys=rows(hn/'pack/val/keys.csv')
cp=ROOT/'experiments/r_solar_geometry_corrected_20261015/data/paired_common_v1';cprows=rows(cp/'rows_supervised.csv');hnref={(r['station'],bjt(r['target_time_utc'])):i for i,r in enumerate(cprows) if r['split']=='val'};hncp=np.load(cp/'cot.npy',mmap_mode='r');hncm=np.load(cp/'common_mask.npy',mmap_mode='r')
old=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/HunanDyDiff_20261028/hunan_full_validation_20261030/evaluation_attempt1');oldrows=rows(old/'cot_rows.csv');oldix={}
for i,r in enumerate(oldrows):oldix.setdefault((r['station'],bjt(r['target_time_utc'])),i)
with np.load(old/'cot_predictions.npz') as z:oldrc=z['cot_observed']
for c in cases:
 region=c['region'];st=c['station'];ini=compact(c['init']);init=datetime.strptime(ini,'%Y%m%d%H%M');times=[(init+timedelta(minutes=15*k)).strftime('%Y%m%d%H%M') for k in range(1,17)]
 oa=[];pa=[];oc=[];pc=[];cc=[];mm=[];source=[]
 if region=='Hubei':
  found=[s for s in hbsel if s['init_time_bjt']==ini];assert len(found)==1;s=found[0]
  if s['old_prediction_index'] is not None:ag=np.load(hb/'assets/old_dydiff_agri_patch.npy',mmap_mode='r')[s['old_prediction_index']]
  else:ag=np.load(hb/f"forecast_patches/val_{s['original_val_index']:05d}.npy",mmap_mode='r')
  assert ag.shape==(16,13,16,16)
  for k,t in enumerate(times):
   j=obsix.get(t)
   if j is not None:assert obskeys[j]['split']=='val'
   oa.append(np.array(obsraw[j,:13]) if j is not None else np.full((13,16,16),np.nan));pa.append(np.array(ag[k]));oc.append(np.expm1(np.asarray(obscot[j]).squeeze()) if j is not None else np.full((16,16),np.nan))
   ci=hcot.get((s['index'],t));pc.append(np.expm1(np.asarray(ci[0][ci[1]]).squeeze()) if ci is not None else np.full((16,16),np.nan))
   j=hbref.get(t);cc.append(np.array(hbcp[j,0])*100 if j is not None else np.full((16,16),np.nan));mm.append(np.array(hbcm[j,0],bool) if j is not None else np.zeros((16,16),bool));source.append({'time_bjt':t,'observed_frame_index':obsix.get(t),'cpp_index':j})
  center={'Niushou':(8,6),'Wangzhai':(9,8),'Mizhuang':(7,10)}[st]
 else:
  keys=[r for r in hnkeys if r['station']==st and bjt(r['init_time_utc'])==ini];assert keys;seqi=int(keys[0]['sequence_index_within_split']);r=seqval[seqi];assert r['split']=='val';paths=r['data_relpaths'].split('|');assert len(paths)==24;bi=bankix[seqi];si={'sili':0,'zhujia':1}[st];rr0,cc0=[(163,148),(73,150)][si];center=(8,8)
  for k,t in enumerate(times):
   rel=paths[k+8];stamp=datetime.strptime(Path(rel).stem,'%Y%m%d%H%M')+timedelta(hours=8);assert stamp.strftime('%Y%m%d%H%M')==t
   raw=np.load(rawroot/'data'/rel,mmap_mode='r');patch=np.array(raw[[0,1,2,3,4,5,8,9,10,11,12,13,14],rr0-8:rr0+8,cc0-8:cc0+8],np.float32);oa.append(patch);pa.append(np.array(bankag[bi,si,k]));pc.append(np.expm1(np.asarray(bankcot[bi,si,k]).squeeze()));j=oldix.get((st,t));oc.append(np.expm1(oldrc[j]) if j is not None else np.full((16,16),np.nan))
   j=hnref.get((st,t));cc.append(np.array(hncp[j,0]) if j is not None else np.full((16,16),np.nan));mm.append(np.array(hncm[j,0],bool) if j is not None else np.zeros((16,16),bool));source.append({'time_bjt':t,'observed_path':str(rawroot/'data'/rel),'cpp_index':j,'observed_R_source':'same corrected R; saved old full validation response'})
 fc=center if region=='Hubei' else (1,1)
 data={'times_bjt':np.array(times),'observed_agri':np.stack(oa),'forecast_agri':np.stack(pa),'observed_cot':np.stack(oc),'forecast_cot':np.stack(pc),'cpp':np.stack(cc),'cpp_mask':np.stack(mm),'center':np.array(center),'forecast_center':np.array(fc)}
 assert data['observed_agri'].shape==(16,13,16,16) and data['forecast_agri'].shape==((16,13,16,16) if region=='Hubei' else (16,13,3,3))
 assert np.isfinite(data['forecast_agri']).all() and np.isfinite(data['forecast_cot'][:,fc[0],fc[1]]).sum()>=8
 target=OUT/(c['event_id']+'.npz');np.savez_compressed(target,**data);sources.append({'case':c,'sources':source,'center':center,'file':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()});print('EXPORTED',target.name,flush=True)
(OUT/'SOURCE_MANIFEST.json').write_text(json.dumps({'state':'COMPLETE_VALIDATION_CASE_EXPORT','new_inference':False,'test_payloads_read':0,'future_observed_images':'offline visualization only','cases':sources},indent=2))
