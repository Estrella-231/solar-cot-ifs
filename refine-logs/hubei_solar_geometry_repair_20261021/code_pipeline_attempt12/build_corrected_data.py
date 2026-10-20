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
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--preparation',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--old-stage',type=Path,required=True);p.add_argument('--rpack',type=Path,required=True);p.add_argument('--cpp-sidecar',type=Path,required=True);p.add_argument('--restored',type=Path,required=True);p.add_argument('--grid',type=Path,required=True);p.add_argument('--canonical-data-root',type=Path,required=True);p.add_argument('--force-rebuild',action='store_true');a=p.parse_args();prep=a.preparation;out=a.base/'prepared_attempt12';out.mkdir(exist_ok=False);meta=prep/'metadata'
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
 grid=dict(np.load(a.grid));frames=rows(prep/'frames_manifest.csv');manifest=[];max_auxerr=0.;allbyUTC={};auxbyUTC={};prior_manifest=a.base/'prepared_attempt10'/'stage_manifest.csv';reused=False
 if prior_manifest.is_file() and not a.force_rebuild:
  candidate=rows(prior_manifest)
  if len(candidate)==len(frames)==10473 and len({r['utc'] for r in candidate})==10473:
   valid_cache=True
   for r in candidate:
    dp=Path(r['data_path']);gp=Path(r['aux_path'])
    if not dp.is_file() or not gp.is_file() or sha(dp)!=r['data_sha256'] or sha(gp)!=r['aux_sha256']:
     valid_cache=False;break
   if valid_cache:
    manifest=candidate;reused=True
    for r in manifest:
     allbyUTC[r['utc']]=Path(r['data_path']);auxbyUTC[r['utc']]=Path(r['aux_path'])
     raw=np.load(r['data_path'],mmap_mode='r',allow_pickle=False);aux=np.load(r['aux_path'])
     errs=(float(np.abs(aux['cosSOZ']-np.cos(np.deg2rad(raw[18]))).max()),float(np.abs(aux['cosRAA']-np.cos(np.deg2rad(raw[17]-raw[15]))).max()),float(np.abs(aux['day_mask']-raw[19]).max()));cache_geom_err=max(errs);max_auxerr=max(max_auxerr,cache_geom_err);assert cache_geom_err<1e-6,('reused cache geometry invalid',r['utc'],errs)
    print('REUSED_ATTEMPT10_CACHE',len(manifest),'sha256_and_geometry_verified',flush=True)
 if not reused:
  if a.cache.exists():raise RuntimeError(f'Attempt5 cache path already exists and cache reuse failed: {a.cache}')
  a.cache.mkdir();(a.cache/'data').mkdir();(a.cache/'aux').mkdir()
  manifest=[];allbyUTC={};auxbyUTC={};max_auxerr=0.
  from concurrent.futures import ThreadPoolExecutor
  def correct_frame(jf):
   j,f=jf
   u=f['utc'];source=a.old_stage/'data'/f['data_name'];raw=np.load(source).copy();old_ir=raw[6:17].copy();gz=np.load(prep/'aux'/f['aux_name']);g=np.stack([gz[k] for k in ('cosSOZ','cosRAA','day_mask')]);z,az,day=compute_solar_angles(grid['lat'],grid['lon'],datetime.strptime(u,'%Y%m%d%H%M'));az=np.where(day>.5,az,0.);g_expected=np.stack((np.cos(np.deg2rad(z)),np.cos(np.deg2rad(az-grid['saa'])),day)).astype(np.float32);gerr=float(np.abs(g-g_expected).max());assert gerr<1e-6,('prepared solar aux mismatch',u,gerr);g=g_expected
   if u in restlut:
    restored=np.load(a.restored/f'{u}.npz');yy,xx=restored['y'].astype(int),restored['x'].astype(int);assert (g[2,yy,xx]>.5).all()
    for ch in range(6):raw[ch,yy,xx]=restored['reflectance'][ch]
   raw[:6,g[2]<.5]=0.;raw[17]=np.where(day>.5,az,0.).astype(np.float32);raw[18]=z.astype(np.float32);raw[19]=day
   assert np.array_equal(raw[6:17],old_ir) and np.isfinite(raw).all();err=float(abs(g[0]-np.cos(np.deg2rad(raw[18]))).max());assert err<1e-6
   dp=a.cache/'data'/f['data_name'];gp=a.cache/'aux'/f['aux_name'];atomic_npy(dp,raw);np.savez(gp,cosSOZ=g[0],cosRAA=g[1],day_mask=g[2]);record=dict(utc=u,bjt=f['bjt'],split=f['split'],data_path=str(dp),aux_path=str(gp),data_sha256=sha(dp),aux_sha256=sha(gp))
   return record,err
  with ThreadPoolExecutor(max_workers=4) as pool:
   for j,(record,err) in enumerate(pool.map(correct_frame,enumerate(frames))):
    manifest.append(record);allbyUTC[record['utc']]=Path(record['data_path']);auxbyUTC[record['utc']]=Path(record['aux_path']);max_auxerr=max(max_auxerr,err)
    if j%256==0:print('LOCAL_CORRECTED_FRAMES',j,len(frames),flush=True)
  csvout(out/'stage_manifest.csv',manifest)
 else:
  csvout(out/'stage_manifest.csv',manifest)
 assert len(manifest)==10473 and len(allbyUTC)==len(auxbyUTC)==10473
 dataByUTC={r['utc']:Path(r['data_path']) for r in manifest};auxByBJT={r['bjt']:Path(r['aux_path']) for r in manifest}
 (out/'indexes').mkdir();(out/'norms').mkdir();roles=['full']+[f'fold_{i}' for i in range(5)]
 for role in roles:
  src=meta/('validation_source_index.csv' if role=='full' else f'{role}_training_index.csv');r=rows(src);new=[]
  for q in r:
   assert q['split'] in ('train','val');new.append(dict(split=q['split'],data_paths='|'.join(str(dataByUTC[Path(f).stem[:12]]) for f in q['data_paths'].split('|')),aux_paths='|'.join(str(auxByBJT[Path(f).stem[:12]]) for f in q['aux_paths'].split('|'))))
  dest=out/'indexes'/f'{role}.csv';csvout(dest,new)
  cmd=[sys.executable,str(Path(__file__).parent/'fit_hubei_simvp_train_norm.py'),'--index-csv',str(dest),'--output-npz',str(out/'norms'/f'{role}.npz'),'--report-json',str(out/'norms'/f'{role}.json'),'--sampled-frame-csv',str(out/'norms'/f'{role}_sample.csv'),'--max-frames','2048']
  with (out/'norms'/f'{role}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
  if role!='full':
   oo=rows(meta/f'{role}_source_index.csv');csvout(out/'indexes'/f'{role}_oof.csv',[dict(split=q['split'],data_paths='|'.join(str(dataByUTC[Path(f).stem[:12]]) for f in q['data_paths'].split('|')),aux_paths='|'.join(str(auxByBJT[Path(f).stem[:12]]) for f in q['aux_paths'].split('|'))) for q in oo]);assert all(q['split']=='val' for q in oo)
  print('S_NORM_READY',role,flush=True)
 rr=rows(a.rpack/'rows.csv');raw=np.load(prep/'R_raw_pending_restoration.npy').copy();assert raw.shape==(len(rr),16,16,16) and len(rr)==3479
 cp=np.load(a.cpp_sidecar);lut={str(k):i for i,k in enumerate(cp['time_bjt'])};assert len(lut)==len(rr)
 assert set(r['time_bjt'] for r in rr)==set(map(str,cp['time_bjt'])) and len(set(r['time_bjt'] for r in rr))==len(rr), 'R/CPP key sets differ; joining by key is required'
 cot=np.stack([cp['cot'][lut[r['time_bjt']]] for r in rr]);valid=np.stack([cp['cot_valid_mask'][lut[r['time_bjt']]] for r in rr]).astype(bool);max_reconstruction=0.;source_audit=[]
 source_receipt=json.loads((a.base/'R_NON_S_CACHE_SOURCE_AUDIT_attempt12.json').read_text())
 transfer_receipt=json.loads((a.base/'TRANSFER_AUDIT_attempt6.json').read_text())
 assert source_receipt['state']=='PASS_R_NON_S_CACHE_SOURCES_TRAIN_VAL_SHA256' and source_receipt['count']==560
 assert transfer_receipt['state']=='PASS_TRANSFERRED_R_ONLY_TRAIN_VAL_SHA256' and transfer_receipt['count']==202 and transfer_receipt['all_hashes_match'] and transfer_receipt['source_audit_sha256']==sha(a.base/'MISSING_R_SOURCE_AUDIT_attempt6.json')
 assert source_receipt['R_only_transfer_audit_sha256']==sha(a.base/'TRANSFER_AUDIT_attempt6.json')
 assert source_receipt['restoration_witness_audit_sha256']==sha(a.base/'RESTORATION_WITNESS_ARCHIVE_AUDIT_attempt7.json')
 canonical_lut={x['utc']:x for x in source_receipt['records']};assert len(canonical_lut)==560
 witness_audit=json.loads((a.base/'RESTORATION_WITNESS_ARCHIVE_AUDIT_attempt7.json').read_text())
 assert witness_audit['state']=='PASS_RESTORATION_WITNESS_ARCHIVE_MATCHES_RESTORED_REPORT' and witness_audit['witness_frames']==255
 assert witness_audit['witness_archive_sha256']==sha(a.base/'restoration_requests_attempt2.tar.gz')
 # Rebuild every R AGRI patch from the same canonical full-scene source used by S.
 # Old data16_label NPZ patches are not trusted as training inputs after the
 # demonstrated row-level discrepancy; labels and CPP masks remain unchanged.
 for i,r in enumerate(rr):
  bjt=datetime.strptime(r['time_bjt'],'%Y%m%d%H%M');utc=bjt-timedelta(hours=8);u=utc.strftime('%Y%m%d%H%M');assert str(cp['split'][lut[r['time_bjt']]])==r['split']
  before=raw[i,:13].copy();source_kind='canonical_full_scene';source_path_used=''
  if u in allbyUTC:
   source_path_used=str(allbyUTC[u]);full=np.load(allbyUTC[u],mmap_mode='r');patch=np.asarray(full[AG,103:119,118:134],dtype=np.float32).copy();source_kind='corrected_S_scene_cache'
  else:
   rel=Path(utc.strftime('%Y/%Y%m/%Y%m%d'))/f'{u}.npy';full_path=a.canonical_data_root/rel
   src=canonical_lut.get(u)
   if src is None or src['bjt']!=r['time_bjt'] or src['split']!=r['split'] or src['relative']!=rel.as_posix():raise ValueError(f'non-S-cache source receipt does not bind row {r["time_bjt"]}: {rel}')
   full_path=Path(src['source_path'])
   expected_path=(a.old_stage/'data'/f'{u}.npy') if src['source_kind']=='old_stage_flat' else (a.base/'r_only_data_attempt6'/rel)
   if full_path!=expected_path:raise ValueError(f'Non-S-cache source root mismatch for {r["time_bjt"]}: {full_path}')
   source_kind=src['source_kind']
   if not full_path.is_file():raise FileNotFoundError(f'R train/val AGRI source missing for row {r["time_bjt"]}: {full_path}')
   if sha(full_path)!=src['sha256']:raise ValueError(f'R source SHA-256 mismatch for {r["time_bjt"]}: {full_path}')
   source_path_used=str(full_path)
   full=np.load(full_path,mmap_mode='r',allow_pickle=False)
   if full.shape!=(20,256,256):raise ValueError(f'Canonical full AGRI shape mismatch at {u}: {full.shape}')
   patch=np.asarray(full[AG,103:119,118:134],dtype=np.float32).copy()
  z,az,day=compute_solar_angles(grid['lat'][103:119,118:134],grid['lon'][103:119,118:134],utc)
  az=np.where(day>.5,az,0.);geo=np.stack((np.cos(np.deg2rad(z)),np.cos(np.deg2rad(az-grid['saa'][103:119,118:134])),day)).astype(np.float32)
  if u not in allbyUTC and u in restlut:
   restored=np.load(a.restored/f'{u}.npz');yy,xx=restored['y'].astype(int),restored['x'].astype(int);take=(yy>=103)&(yy<119)&(xx>=118)&(xx<134)
   if take.any():patch[:6,yy[take]-103,xx[take]-118]=restored['reflectance'][:6,take]
  patch[:6,day<.5]=0.
  raw[i,:13]=patch;raw[i,13:16]=geo
  if u in allbyUTC:
   s=np.load(allbyUTC[u],mmap_mode='r');s_patch=np.asarray(s[AG,103:119,118:134],dtype=np.float32);err=float(np.abs(patch.astype(np.float64)-s_patch.astype(np.float64)).max());assert err==0.,('R/S physical alignment',u,err);max_reconstruction=max(max_reconstruction,err)
   s_aux=np.load(auxbyUTC[u]);station_slice=(slice(103,119),slice(118,134));s_geo=np.stack((s_aux['cosSOZ'][station_slice],s_aux['cosRAA'][station_slice],s_aux['day_mask'][station_slice]));s_err=float(np.abs(geo-s_geo).max());assert np.array_equal(geo[2],s_geo[2]) and s_err<1e-6,('R/S station-patch geometry mismatch',r['time_bjt'],u,geo.shape,s_geo.shape,s_err)
  if not np.isfinite(raw[i]).all():raise ValueError(f'Nonfinite corrected R input at {r["time_bjt"]}')
  source_audit.append(dict(index=i,time_bjt=r['time_bjt'],time_utc=u,split=r['split'],source_kind=source_kind,source_path_used=source_path_used,source_npz_original=r.get('path_agri',''),old_vs_canonical_max_abs=float(np.abs(before.astype(np.float64)-patch.astype(np.float64)).max()),restored_crop_pixels=int(take.sum()) if u not in allbyUTC and u in restlut else 0,overlap_with_corrected_S=(u in allbyUTC),R_S_max_abs=float(err) if u in allbyUTC else None))
 csvout(out/'R_input_source_alignment.csv',source_audit);np.save(out/'R_raw_corrected_canonical_attempt12.npy',raw,allow_pickle=False)
 valid &= (raw[:,15]>.5)&(raw[:,13]>np.cos(np.deg2rad(80.)));target=np.where(valid,cot/100.,0.).astype(np.float32)[:,None];assert np.isfinite(target).all() and (target>=0).all() and (target<=1).all()
 folds={q['date_BJT'].replace('-',''):int(q['heldout_fold']) for q in rows(meta/'train_date_folds.csv')};(out/'R_packs').mkdir();rn0=json.loads((a.rpack/'norm.json').read_text());sourceRstatus=json.loads((a.rpack/'pack_status.json').read_text());rresults=[]
 for role in roles:
  keep=np.array([r['split']=='val' or role=='full' or folds[r['time_bjt'][:8]]!=int(role[-1]) for r in rr]);ii=np.flatnonzero(keep);qraw=raw[ii];train=np.array([rr[i]['split']=='train' for i in ii]);new=[]
  for j,i in enumerate(ii):
   q=dict(rr[i]);q['source_npz_original']=q.get('path_agri','');q['path_agri']=source_audit[i]['source_path_used'];q['source_kind']=source_audit[i]['source_kind'];q['index']=j;q['valid_cot_pixels']=int(valid[i].sum());new.append(q)
  assert train.any();v=qraw[train].astype(np.float64);mu=v.mean(axis=(0,2,3)).astype(np.float32);sd=np.maximum(v.std(axis=(0,2,3)),1e-6).astype(np.float32);norm=dict(input_channels=rn0['input_channels'],mean=mu.tolist(),std=sd.tolist(),fit_split='train only',train_samples=int(train.sum()),test_used=False,solar_geometry_corrected=True);xn=((qraw-mu[None,:,None,None])/sd[None,:,None,None]).astype(np.float32);assert np.isfinite(xn).all();rp=out/'R_packs'/role;rp.mkdir();np.save(rp/'x_norm.npy',xn);np.save(rp/'target.npy',target[ii]);np.save(rp/'mask.npy',valid[ii,None]);csvout(rp/'rows.csv',new);(rp/'norm.json').write_text(json.dumps(norm,indent=2))
  counts={s:dict(rows=sum(r['split']==s for r in new),supervised_rows=int(sum(r['split']==s and valid[i].any() for i,r in zip(ii,new))),valid_pixels=int(valid[ii][np.array([r['split']==s for r in new])].sum())) for s in ('train','val')};status=dict(state='HUBEI_R_TRAIN_VAL_CORRECT_UTC_PACK_BUILT_TEST_CLOSED',rows=len(new),counts_by_split=counts,time_rule='CPP UTC = BJT - 8 hours',fold=role,solar_geometry_corrected=True,solar_geometry_source_sha256=sha(Path(__file__).parent/'solar_geometry_noaa.py'),corrected_SZA80_mask=True,test_payloads_read=0,buffer_payloads_read=0,source_correctutc_pack_sha256=sha(a.rpack/'pack_status.json'),source_CPP_sidecar_sha256=sha(a.cpp_sidecar));(rp/'pack_status.json').write_text(json.dumps(status,indent=2));(rp/'SHA256.json').write_text(json.dumps({n:sha(rp/n) for n in ('x_norm.npy','target.npy','mask.npy','rows.csv','norm.json')},indent=2));rresults.append(dict(role=role,**counts));print('R_PACK_READY',role,flush=True)
 actual_cache_root=str(Path(manifest[0]['data_path']).parent.parent);final=dict(solar_geometry_source_sha256=sha(Path(__file__).parent/'solar_geometry_noaa.py'),state='CORRECTED_SOLAR_TRAIN_VAL_DATA_READY',S_cache=actual_cache_root,S_frames=len(frames),S_sequences=report['S_sequences'],S_norms={role:sha(out/'norms'/f'{role}.npz') for role in roles},R_packs=rresults,R_input_source_policy='canonical full AGRI scenes for all train/val R rows; corrected S cache crop where overlapping; labels and CPP masks preserved',R_input_source_alignment_sha256=sha(out/'R_input_source_alignment.csv'),R_input_source_alignment_rows=len(source_audit),R_non_S_cache_source_audit_sha256=sha(a.base/'R_NON_S_CACHE_SOURCE_AUDIT_attempt12.json'),R_non_S_cache_source_count=len(canonical_lut),R_non_S_cache_source_counts=source_receipt['source_counts'],R_only_transfer_audit_sha256=sha(a.base/'TRANSFER_AUDIT_attempt6.json'),restoration_witness_archive_audit_sha256=sha(a.base/'RESTORATION_WITNESS_ARCHIVE_AUDIT_attempt7.json'),R_raw_corrected_sha256=sha(out/'R_raw_corrected_canonical_attempt12.npy'),full_grid_aux_roundtrip_max_abs=max_auxerr,R_S_physical_reconstruction_max_abs=max_reconstruction,raw_radiance_restoration_report_sha256=sha(a.restored/'RESTORATION_REPORT.json'),independent_NOAA_report_sha256=sha(a.base/'NOAA_ALL_TRAIN_VAL_REFERENCE_AUDIT.json'),test_payloads_read=0,buffer_payloads_read=0);(out/'DATA_READY.json').write_text(json.dumps(final,indent=2));print(json.dumps(final),flush=True)
if __name__=='__main__':main()
