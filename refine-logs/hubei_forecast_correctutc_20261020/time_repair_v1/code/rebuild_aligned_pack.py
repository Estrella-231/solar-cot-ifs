"""Rejoin existing OOF forecasts to actual BJT labels; independent full audit."""
import sys,csv,json,hashlib,tarfile
from pathlib import Path
from datetime import datetime,timedelta
from collections import Counter
import numpy as np
ST={'Niushou':(8,6),'Wangzhai':(9,8),'Mizhuang':(7,10)}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
def write(p,r):
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
def main():
 base,local,out=map(Path,sys.argv[1:4]);meta=local/'metadata';inp=local/'inputs';rpack=local/'full_R_pack';foldp=base/'runtime_attempt2/R_fold_packs_attempt1'
 audit=json.loads((meta/'audit.json').read_text());assert audit['state']=='ALL_TRAIN_VAL_TIME_LINEAGE_AUDIT_PASS'
 for n,v in audit['files'].items():assert sha(meta/n)==v['sha256'],('metadata hash',n)
 dates={r['date_BJT'].replace('-',''):r['split'] for r in read(meta/'day_assignment.csv')};folds={r['date_BJT'].replace('-',''):int(r['heldout_fold']) for r in read(meta/'train_date_folds.csv')}
 norm=json.loads((rpack/'norm.json').read_text());rr=read(rpack/'rows.csv');rx=np.load(rpack/'x_norm.npy',mmap_mode='r');rl={r['time_bjt']:j for j,r in enumerate(rr)};mu=np.array(norm['mean'],np.float32)[:,None,None];sd=np.array(norm['std'],np.float32)[:,None,None]
 labels={}
 for r in read(meta/'ghi15_train_val_daylight_clear.csv'):
  assert r['split'] in ('train','val');t=r['target_time_BJT'].replace('-','').replace(':','').replace(' ','')[:12];assert dates[t[:8]]==r['split']
  if r['valid_daylight_clear_sky_label'] not in ('1','1.0','True','true'):continue
  assert r['mask'] in ('1','1.0') and r['reason']=='complete' and int(r['source_records'])>=15
  key=(r['station'],t);assert key not in labels
  y=float(r['ghi15_raw']);cl=float(r['clear_sky_ghi_15min_wm2']);assert np.isfinite(y) and y>=0 and np.isfinite(cl) and cl>0
  labels[key]=(r['split'],y,cl,float(r['solar_cosine']))
 original_report=json.loads((base/'runtime_attempt2/forecast_pack_attempt1/pack_report.json').read_text());original_R={r['role']:r for r in original_report['source_correctutc_Rs']}
 rows=[];ag=[];ge=[];ct=[];ys=[];cls=[];source=[];geomcheck=[];coscheck=[];seen=set();missing=Counter();normalizer_reused=True
 for f in range(6):
  role=f'fold_{f}' if f<5 else 'validation';split='train' if f<5 else 'val';keys=read(meta/f'{role}_keys.csv');oldkeys=read(inp/'forecast'/role/'keys.csv');assert len(keys)==len(oldkeys)
  # Independently recheck actual source-index dates and OOF train exclusions.
  si=read(meta/f'{role}_source_index.csv');lookup={r['data_paths']:r for r in si if r['split']=='val'}
  if f<5:
   trained=set()
   for r in read(meta/f'{role}_training_index.csv'):
    ds=[Path(x).stem[:12] for x in r['data_paths'].split('|')];bs=[Path(x).stem[:12] for x in r['aux_paths'].split('|')];assert all(datetime.strptime(g,'%Y%m%d%H%M')-datetime.strptime(u,'%Y%m%d%H%M')==timedelta(hours=8) for u,g in zip(ds,bs))
    assert all(dates[t[:8]]==('train' if r['split']=='train' else 'val') for t in bs)
    if r['split']=='train':
     assert all(folds[t[:8]]!=f for t in bs);trained.update(t[:8] for t in bs)
   for r in read(foldp/f'fold_{f}/rows.csv'):
    if r['split']=='train':assert dates[r['time_bjt'][:8]]=='train' and folds[r['time_bjt'][:8]]!=f
   assert trained
  pred=np.load(inp/'forecast'/role/'pred_agri_patch.npy',mmap_mode='r');geo=np.load(inp/'forecast'/role/'future_geometry_patch.npy',mmap_mode='r');cot=np.load(base/f'runtime_attempt2/forecast_pack_attempt1/cot_banks/{role}.npy',mmap_mode='r')
  assert pred.shape==(len(keys),16,13,16,16) and geo.shape==(len(keys),16,3,16,16) and cot.shape==(len(keys),16,1,16,16)
  assert np.isfinite(pred).all() and np.isfinite(geo).all() and np.isfinite(cot).all()
  for i,(k,old) in enumerate(zip(keys,oldkeys)):
   index=lookup[old['data_paths']];u=[Path(x).stem[:12] for x in index['data_paths'].split('|')];b=[Path(x).stem[:12] for x in index['aux_paths'].split('|')]
   assert len(u)==len(b)==24 and all(datetime.strptime(g,'%Y%m%d%H%M')-datetime.strptime(v,'%Y%m%d%H%M')==timedelta(hours=8) for v,g in zip(u,b));assert k['init_time_bjt']==b[7] and k['target_times_bjt']=='|'.join(b[8:])
   assert all(dates[t[:8]]==split for t in b)
   if f<5:assert all(folds[t[:8]]==f for t in b) and not {t[:8] for t in b}&trained
   init=datetime.strptime(k['init_time_bjt'],'%Y%m%d%H%M');assert k['source_aux_bjt_times']=='|'.join(b)
   for l,t in enumerate(b[8:]):
    assert datetime.strptime(t,'%Y%m%d%H%M')==init+timedelta(minutes=15*(l+1))
    if t in rl:
     j=rl[t];graw=rx[j]*sd+mu;geomcheck.append(float(abs(geo[i,l]-graw[13:16]).max()))
    for st,(y,x) in ST.items():
     lab=labels.get((st,t))
     if lab is None:missing[split+'/'+st]+=1;continue
     ls,ghi,clear,cosine=lab;assert ls==split;key=(st,k['init_time_bjt'],t);assert key not in seen;seen.add(key)
     sl=(slice(y-1,y+2),slice(x-1,x+2));r=dict(index=len(rows),split=split,station=st,init_time_bjt=k['init_time_bjt'],target_time_bjt=t,lead_minutes=15*(l+1),ghi_wm2=ghi,clear_sky_wm2=clear,source_sequence_index=i,source_role=role)
     rows.append(r);ag.append(np.array(pred[i,l][:,sl[0],sl[1]]));ge.append(np.array(geo[i,l][:2,sl[0],sl[1]]));ct.append(np.array(cot[i,l][:,sl[0],sl[1]]));ys.append(ghi);cls.append(clear);coscheck.append(dict(split=split,station=st,target=t,stored_cosSOZ=float(geo[i,l,0,y,x]),label_noaa_cosSOZ=cosine,difference=float(geo[i,l,0,y,x])-cosine))
  ph=sha(inp/'forecast'/role/'pred_agri_patch.npy');gh=sha(inp/'forecast'/role/'future_geometry_patch.npy');ch=sha(base/f'runtime_attempt2/forecast_pack_attempt1/cot_banks/{role}.npy')
  assert ph==original_R[role]['forecast_pred_sha256'] and gh==original_R[role]['forecast_geometry_sha256'] and ch==original_R[role]['COT_sha256']
  source.append(dict(role=role,sequences=len(keys),canonical_keys_sha256=sha(meta/f'{role}_keys.csv'),pred_sha256=ph,geometry_sha256=gh,cot_sha256=ch,correctutc_R_source=original_R[role]))
 assert geomcheck and max(geomcheck)<1e-3,('R-pack geometry mismatch',max(geomcheck))
 cosdiff=np.array([r['difference'] for r in coscheck]);write(local/'solar_geometry_checks.csv',coscheck)
 if '--geometry-only' in sys.argv:
  print(json.dumps(dict(samples=len(rows),by_split_station=dict(Counter(r['split']+'/'+r['station'] for r in rows)),R_geometry_max_abs=max(geomcheck),worst=sorted(coscheck,key=lambda r:abs(r['difference']),reverse=True)[:20],solar_abs_mean=float(abs(cosdiff).mean()),solar_abs_p99=float(np.quantile(abs(cosdiff),.99)))),flush=True);return
 # Check the actual frozen formula, rather than relaxing a failed approximate-NOAA threshold.
 from legacy_solar_formula import compute_solar_angles
 station_ll={'Niushou':(32.08,112.04),'Wangzhai':(32.04,112.12),'Mizhuang':(32.12,112.20)}
 legacy_cache={};legacy_diff=[]
 for r in coscheck:
  key=(r['station'],r['target'])
  if key not in legacy_cache:
   lat,lon=station_ll[r['station']];utc=datetime.strptime(r['target'],'%Y%m%d%H%M')-timedelta(hours=8)
   legacy_cache[key]=float(np.cos(np.deg2rad(compute_solar_angles(lat,lon,utc)[0])))
  legacy_diff.append(r['stored_cosSOZ']-legacy_cache[key])
 assert max(abs(np.asarray(legacy_diff)))<1e-4,('frozen-source solar geometry mismatch',max(abs(np.asarray(legacy_diff))))
 assert {r['split'] for r in rows}=={'train','val'}
 out.mkdir(exist_ok=False)
 for name,a in [('agri_raw.npy',ag),('geometry.npy',ge),('cot.npy',ct),('ghi_wm2.npy',ys),('clear_sky_wm2.npy',cls)]:np.save(out/name,np.asarray(a,np.float32),allow_pickle=False)
 write(out/'rows.csv',rows);write(out/'solar_cosine_checks.csv',coscheck)
 report=dict(state='HUBEI_GHI_FORECAST_HEADS_TRAINVAL_PACK_TEST_CLOSED',samples=len(rows),samples_by_split_station=dict(Counter(r['split']+'/'+r['station'] for r in rows)),missing_or_invalid_label_candidates=dict(missing),canonical_time_binding_repaired=True,time_rule='BJT = original full-scene UTC +8h = source auxiliary BJT; init history end; target +15..240min',labels='existing independent train/val minute workbook labels, (target-15min,target] complete; existing matching clear-sky15mean',split_sha256=sha(meta/'day_assignment.csv'),train_forecast_inputs_are_oof=True,oof_fold_count=5,oof_train_bank_report_sha256=sha(meta/'audit.json'),validation_forecast_source='frozen_full_train_S_R',correctutc_cot_rebuilt=True,cot_values_reused_without_change=True,source_correctutc_Rs=source,R_pack_geometry_comparisons=len(geomcheck),R_pack_geometry_max_abs=max(geomcheck),solar_cosine_label_check_max_abs=float(max(abs(cosdiff))),solar_cosine_label_check_mean_abs=float(abs(cosdiff).mean()),frozen_source_solar_cosine_max_abs=float(max(abs(np.asarray(legacy_diff)))),known_solar_limitation='Frozen S/R geometry retains source EOT missing radians-to-degrees conversion; max zenith difference 3.236 degrees vs corrected formula, not an 8-hour clock error. This experiment repairs label binding only.',normalization_fit='refit GHI normalization on corrected train support only',test_and_buffer_values_read=0,test_payloads_read=0,buffer_payloads_read=0)
 report['arrays_sha256']={p.name:sha(p) for p in out.glob('*.npy')};(out/'pack_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
