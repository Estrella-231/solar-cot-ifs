"""Build immutable local corrected S data and full/five-fold R packs."""
import argparse,csv,hashlib,json,os,subprocess,sys,tarfile
from pathlib import Path
from datetime import datetime,timedelta
from collections import Counter
import numpy as np
from solar_geometry_noaa import compute_solar_angles
AG=[0,1,2,3,4,5,8,9,10,11,12,13,14]
def rows(p):return list(csv.DictReader(Path(p).open(encoding='utf-8-sig')))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def csvout(p,r):
 with Path(p).open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)
def atomic_npy(p,x):
 t=p.with_suffix('.part')
 with t.open('wb') as f:np.save(f,x,allow_pickle=False)
 os.replace(t,p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--preparation',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--old-stage',type=Path,required=True);p.add_argument('--rpack',type=Path,required=True);p.add_argument('--cpp-sidecar',type=Path,required=True);p.add_argument('--restored',type=Path,required=True);p.add_argument('--grid',type=Path,required=True);a=p.parse_args();prep=a.preparation;out=a.base/'prepared_attempt2';out.mkdir(exist_ok=False);assert not a.cache.exists();a.cache.mkdir();(a.cache/'data').mkdir();(a.cache/'aux').mkdir();meta=prep/'metadata'
 independent=json.loads((a.base/'NOAA_ALL_TRAIN_VAL_REFERENCE_AUDIT.json').read_text());assert independent['state']=='PASS_AGAINST_OFFICIAL_NOAA_JAVASCRIPT_GEOMETRIC_ANGLES' and independent['by_split']['train']>0 and independent['by_split']['val']>0
 report=json.loads((prep/'GEOMETRY_CALCULATION_REPORT.json').read_text());rest=json.loads((a.restored/'RESTORATION_REPORT.json').read_text());assert rest['state']=='EXACT_UTC_HDF_RADIANCE_RESTORATION_COMPLETE' and rest['frames']==report['restoration_frames'];restlut={r['utc']:r for r in rest['records']};witness=json.loads((prep/'RESTORATION_REQUEST_WITNESS.json').read_text());witnesslut={r['utc']:r for r in witness['records']}
 for u,r in restlut.items():assert r['source_compatibility_checked'] and r['request_sha256']==witnesslut[u]['augmented_request_sha256'] and r['restored_sha256']==sha(a.restored/f'{u}.npz')
 # Independently enforce the unchanged fold/date separation using actual 24-frame keys.
 allowed={q['date_BJT'].replace('-',''):q['split'] for q in rows(meta/'day_assignment.csv')}
 for role in ['full']+[f'fold_{k}' for k in range(5)]:
  rs=rows(meta/('validation_source_index.csv' if role=='full' else f'{role}_training_index.csv'));seen={split:set() for split in ('train','val')}
  for q in rs:
   ds=q['data_paths'].split('|');gs=q['aux_paths'].split('|');assert len(ds)==len(gs)==24
   for dp,gp in zip(ds,gs):
    utc=datetime.strptime(Path(dp).stem[:12],'%Y%m%d%H%M');bjt=datetime.strptime(Path(gp).stem[:12],'%Y%m%d%H%M');assert bjt-utc==timedelta(hours=8) and allowed[bjt.strftime('%Y%m%d')]==q['split'];seen[q['split']].add(bjt.strftime('%Y%m%d'))
  assert not seen['train']&seen['val']
  if role!='full':
   held={Path(g).stem[:8] for q in rows(meta/f'{role}_source_index.csv') for g in q['aux_paths'].split('|')};assert not held&seen['train'] and not held&seen['val']
 grid=dict(np.load(a.grid));frames=rows(prep/'frames_manifest.csv');manifest=[];max_auxerr=0.;allbyUTC={}
 from concurrent.futures import ThreadPoolExecutor
 def correct_frame(jf):
  j,f=jf
  u=f['utc'];source=a.old_stage/'data'/f['data_name'];raw=np.load(source).copy();old_ir=raw[6:17].copy();gz=np.load(prep/'aux'/f['aux_name']);g=np.stack([gz[k] for k in ('cosSOZ','cosRAA','day_mask')]);z,az,day=compute_solar_angles(grid['lat'],grid['lon'],datetime.strptime(u,'%Y%m%d%H%M'));assert np.array_equal(g[2],day)
  if u in restlut:
   restored=np.load(a.restored/f'{u}.npz');yy,xx=restored['y'].astype(int),restored['x'].astype(int);assert (g[2,yy,xx]>.5).all()
   for ch in range(6):raw[ch,yy,xx]=restored['reflectance'][ch]
  raw[:6,g[2]<.5]=0.;raw[17]=np.where(day>.5,az,0.).astype(np.float32);raw[18]=z.astype(np.float32);raw[19]=day
  assert np.array_equal(raw[6:17],old_ir) and np.isfinite(raw).all();err=float(abs(g[0]-np.cos(np.deg2rad(raw[18]))).max());assert err<1e-6
  dp=a.cache/'data'/f['data_name'];gp=a.cache/'aux'/f['aux_name'];atomic_npy(dp,raw);np.savez(gp,cosSOZ=g[0],cosRAA=g[1],day_mask=g[2]);record=dict(utc=u,bjt=f['bjt'],split=f['split'],data_path=str(dp),aux_path=str(gp),data_sha256=sha(dp),aux_sha256=sha(gp))
  return record,err
 with ThreadPoolExecutor(max_workers=4) as pool:
  for j,(record,err) in enumerate(pool.map(correct_frame,enumerate(frames))):
   manifest.append(record);allbyUTC[record['utc']]=Path(record['data_path']);max_auxerr=max(max_auxerr,err)
   if j%256==0:print('LOCAL_CORRECTED_FRAMES',j,len(frames),flush=True)
 csvout(out/'stage_manifest.csv',manifest);(out/'indexes').mkdir();(out/'norms').mkdir();roles=['full']+[f'fold_{i}' for i in range(5)]
 for role in roles:
  src=meta/('validation_source_index.csv' if role=='full' else f'{role}_training_index.csv');r=rows(src);new=[]
  for q in r:
   assert q['split'] in ('train','val');new.append(dict(split=q['split'],data_paths='|'.join(str(a.cache/'data'/Path(f).name) for f in q['data_paths'].split('|')),aux_paths='|'.join(str(a.cache/'aux'/Path(f).name) for f in q['aux_paths'].split('|'))))
  dest=out/'indexes'/f'{role}.csv';csvout(dest,new)
  cmd=[sys.executable,str(Path(__file__).parent/'fit_hubei_simvp_train_norm.py'),'--index-csv',str(dest),'--output-npz',str(out/'norms'/f'{role}.npz'),'--report-json',str(out/'norms'/f'{role}.json'),'--sampled-frame-csv',str(out/'norms'/f'{role}_sample.csv'),'--max-frames','2048']
  with (out/'norms'/f'{role}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
  if role!='full':
   oo=rows(meta/f'{role}_source_index.csv');csvout(out/'indexes'/f'{role}_oof.csv',[dict(split=q['split'],data_paths='|'.join(str(a.cache/'data'/Path(f).name) for f in q['data_paths'].split('|')),aux_paths='|'.join(str(a.cache/'aux'/Path(f).name) for f in q['aux_paths'].split('|'))) for q in oo]);assert all(q['split']=='val' for q in oo)
  print('S_NORM_READY',role,flush=True)
 rr=rows(a.rpack/'rows.csv');raw=np.load(prep/'R_raw_pending_restoration.npy');cp=np.load(a.cpp_sidecar);lut={str(k):i for i,k in enumerate(cp['time_bjt'])};assert len(lut)==len(rr);cot=np.stack([cp['cot'][lut[r['time_bjt']]] for r in rr]);valid=np.stack([cp['cot_valid_mask'][lut[r['time_bjt']]] for r in rr]).astype(bool);max_reconstruction=0.
 for i,r in enumerate(rr):
  u=(datetime.strptime(r['time_bjt'],'%Y%m%d%H%M')-timedelta(hours=8)).strftime('%Y%m%d%H%M');assert str(cp['split'][lut[r['time_bjt']]])==r['split']
  if u in restlut:
   z=np.load(a.restored/f'{u}.npz');yy,xx=z['y'].astype(int),z['x'].astype(int);take=(yy>=103)&(yy<119)&(xx>=118)&(xx<134)
   for ch in range(6):raw[i,ch,yy[take]-103,xx[take]-118]=z['reflectance'][ch,take]
  raw[i,:6,raw[i,15]<.5]=0
  if u in allbyUTC:
   s=np.load(allbyUTC[u],mmap_mode='r')[AG,103:119,118:134];err=float(abs(raw[i,:13]-s).max());assert err<1e-3,('R/S physical alignment',u,err);max_reconstruction=max(max_reconstruction,err)
 valid &= (raw[:,15]>.5)&(raw[:,13]>np.cos(np.deg2rad(80.)));target=np.where(valid,cot/100.,0.).astype(np.float32)[:,None];assert np.isfinite(target).all() and (target>=0).all() and (target<=1).all()
 folds={q['date_BJT'].replace('-',''):int(q['heldout_fold']) for q in rows(meta/'train_date_folds.csv')};(out/'R_packs').mkdir();rn0=json.loads((a.rpack/'norm.json').read_text());sourceRstatus=json.loads((a.rpack/'pack_status.json').read_text());rresults=[]
 for role in roles:
  keep=np.array([r['split']=='val' or role=='full' or folds[r['time_bjt'][:8]]!=int(role[-1]) for r in rr]);ii=np.flatnonzero(keep);qraw=raw[ii];new=[dict(r,index=j,valid_cot_pixels=int(valid[i].sum())) for j,(i,r) in enumerate((i,rr[i]) for i in ii)];train=np.array([r['split']=='train' for r in new]);assert train.any();v=qraw[train].astype(np.float64);mu=v.mean(axis=(0,2,3)).astype(np.float32);sd=np.maximum(v.std(axis=(0,2,3)),1e-6).astype(np.float32);norm=dict(input_channels=rn0['input_channels'],mean=mu.tolist(),std=sd.tolist(),fit_split='train only',train_samples=int(train.sum()),test_used=False,solar_geometry_corrected=True);xn=((qraw-mu[None,:,None,None])/sd[None,:,None,None]).astype(np.float32);assert np.isfinite(xn).all();rp=out/'R_packs'/role;rp.mkdir();np.save(rp/'x_norm.npy',xn);np.save(rp/'target.npy',target[ii]);np.save(rp/'mask.npy',valid[ii,None]);csvout(rp/'rows.csv',new);(rp/'norm.json').write_text(json.dumps(norm,indent=2))
  counts={s:dict(rows=sum(r['split']==s for r in new),supervised_rows=int(sum(r['split']==s and valid[i].any() for i,r in zip(ii,new))),valid_pixels=int(valid[ii][np.array([r['split']==s for r in new])].sum())) for s in ('train','val')};status=dict(state='HUBEI_R_TRAIN_VAL_CORRECT_UTC_PACK_BUILT_TEST_CLOSED',rows=len(new),counts_by_split=counts,time_rule='CPP UTC = BJT - 8 hours',fold=role,solar_geometry_corrected=True,solar_geometry_source_sha256=sha(Path(__file__).parent/'solar_geometry_noaa.py'),corrected_SZA80_mask=True,test_payloads_read=0,buffer_payloads_read=0,source_correctutc_pack_sha256=sha(a.rpack/'pack_status.json'),source_CPP_sidecar_sha256=sha(a.cpp_sidecar));(rp/'pack_status.json').write_text(json.dumps(status,indent=2));(rp/'SHA256.json').write_text(json.dumps({n:sha(rp/n) for n in ('x_norm.npy','target.npy','mask.npy','rows.csv','norm.json')},indent=2));rresults.append(dict(role=role,**counts));print('R_PACK_READY',role,flush=True)
 final=dict(solar_geometry_source_sha256=sha(Path(__file__).parent/'solar_geometry_noaa.py'),state='CORRECTED_SOLAR_TRAIN_VAL_DATA_READY',S_cache=str(a.cache),S_frames=len(frames),S_sequences=report['S_sequences'],S_norms={role:sha(out/'norms'/f'{role}.npz') for role in roles},R_packs=rresults,full_grid_aux_roundtrip_max_abs=max_auxerr,R_S_physical_reconstruction_max_abs=max_reconstruction,raw_radiance_restoration_report_sha256=sha(a.restored/'RESTORATION_REPORT.json'),independent_NOAA_report_sha256=sha(a.base/'NOAA_ALL_TRAIN_VAL_REFERENCE_AUDIT.json'),test_payloads_read=0,buffer_payloads_read=0);(out/'DATA_READY.json').write_text(json.dumps(final,indent=2));print(json.dumps(final),flush=True)
if __name__=='__main__':main()
