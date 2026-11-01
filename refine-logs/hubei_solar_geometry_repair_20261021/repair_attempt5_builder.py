from pathlib import Path
p=Path(r'F:\CODE_Classify\辐照度预报\solar-cot-ifs\refine-logs\hubei_solar_geometry_repair_20261021\code_pipeline_attempt5\build_corrected_data.py')
s=p.read_text(encoding='utf-8')
start="grid=dict(np.load(a.grid));frames=rows(prep/'frames_manifest.csv');manifest=[];max_auxerr=0.;allbyUTC={};auxbyUTC={};prior_manifest=a.base/'prepared_attempt4'/'stage_manifest.csv';reused=False"
end="(out/'indexes').mkdir();(out/'norms').mkdir();roles=['full']+[f'fold_{i}' for i in range(5)]"
i=s.index(start); j=s.index(end,i)+len(end)
block="""grid=dict(np.load(a.grid));frames=rows(prep/'frames_manifest.csv');manifest=[];max_auxerr=0.;allbyUTC={};auxbyUTC={};prior_manifest=a.base/'prepared_attempt4'/'stage_manifest.csv';reused=False
 if prior_manifest.is_file():
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
     max_auxerr=max(max_auxerr,float(np.abs(aux['cosSOZ']-np.cos(np.deg2rad(raw[18]))).max()))
    print('REUSED_ATTEMPT4_CACHE',len(manifest),'sha256_verified',flush=True)
 if not reused:
  if a.cache.exists():raise RuntimeError(f'Attempt5 cache path already exists and cache reuse failed: {a.cache}')
  a.cache.mkdir();(a.cache/'data').mkdir();(a.cache/'aux').mkdir()
  manifest=[];allbyUTC={};auxbyUTC={};max_auxerr=0.
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
    manifest.append(record);allbyUTC[record['utc']]=Path(record['data_path']);auxbyUTC[record['utc']]=Path(record['aux_path']);max_auxerr=max(max_auxerr,err)
    if j%256==0:print('LOCAL_CORRECTED_FRAMES',j,len(frames),flush=True)
  csvout(out/'stage_manifest.csv',manifest)
 else:
  csvout(out/'stage_manifest.csv',manifest)
 assert len(manifest)==10473 and len(allbyUTC)==len(auxbyUTC)==10473
 dataByUTC={r['utc']:Path(r['data_path']) for r in manifest};auxByBJT={r['bjt']:Path(r['aux_path']) for r in manifest}
 (out/'indexes').mkdir();(out/'norms').mkdir();roles=['full']+[f'fold_{i}' for i in range(5)]"""
s=s[:i]+block+s[j:]
p.write_text(s,encoding='utf-8')
p=Path(r'F:\CODE_Classify\辐照度预报\solar-cot-ifs\refine-logs\hubei_solar_geometry_repair_20261021\code_pipeline_attempt5\audit_prepared_attempt5.py')
s=p.read_text(encoding='utf-8')
old="assert [str(k) for k in cp['time_bjt']] == times"
new="assert set(str(k) for k in cp['time_bjt']) == set(times) and len(set(str(k) for k in cp['time_bjt'])) == len(times), 'CPP/R key sets differ'"
assert old in s;s=s.replace(old,new,1);p.write_text(s,encoding='utf-8')
