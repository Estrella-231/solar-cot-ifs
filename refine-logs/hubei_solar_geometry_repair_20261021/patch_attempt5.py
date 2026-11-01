from pathlib import Path
import re, hashlib, json, tarfile, subprocess
root=Path(r'F:\CODE_Classify\辐照度预报\solar-cot-ifs\refine-logs\hubei_solar_geometry_repair_20261021')
code=root/'code_pipeline_attempt5'
for p in code.glob('*.py'):
    s=p.read_text(encoding='utf-8')
    s2=s.replace('attempt4','attempt5')
    p.write_text(s2,encoding='utf-8')

p=code/'build_corrected_data.py'; s=p.read_text(encoding='utf-8')
old="a=p.parse_args();prep=a.preparation;out=a.base/'prepared_attempt5';out.mkdir(exist_ok=False);assert not a.cache.exists();a.cache.mkdir();(a.cache/'data').mkdir();(a.cache/'aux').mkdir();meta=prep/'metadata'"
new="a=p.parse_args();prep=a.preparation;out=a.base/'prepared_attempt5';out.mkdir(exist_ok=False);meta=prep/'metadata'"
assert old in s, 'init replacement anchor missing'
s=s.replace(old,new,1)
start="grid=dict(np.load(a.grid));frames=rows(prep/'frames_manifest.csv');manifest=[];max_auxerr=0.;allbyUTC={};auxbyUTC={}"
end="csvout(out/'stage_manifest.csv',manifest);(out/'indexes').mkdir();(out/'norms').mkdir();roles=['full']+[f'fold_{i}' for i in range(5)]"
assert start in s and end in s, 'frame replacement anchors missing'
# preserve frame creation function from current source but make execution conditional by reusing valid attempt4 cache manifest
idx1=s.index(start); idx2=s.index(end,idx1)+len(end)
oldblock=s[idx1:idx2]
# splice the existing correct_frame implementation into the else branch, then rebuild mappings from manifest
func_start=oldblock.index(' def correct_frame(jf):')
func_end=oldblock.index(" csvout(out/'stage_manifest.csv',manifest);",func_start)
func=oldblock[func_start:func_end]
# indent function and executor block by two spaces, retaining its internal indentation
indented='\n'.join(('  '+line if line else line) for line in func.splitlines())
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
"""+indented+"\n  with ThreadPoolExecutor(max_workers=4) as pool:\n   for j,(record,err) in enumerate(pool.map(correct_frame,enumerate(frames))):\n    manifest.append(record);allbyUTC[record['utc']]=Path(record['data_path']);auxbyUTC[record['utc']]=Path(record['aux_path']);max_auxerr=max(max_auxerr,err)\n    if j%256==0:print('LOCAL_CORRECTED_FRAMES',j,len(frames),flush=True)\n csvout(out/'stage_manifest.csv',manifest)\nassert len(manifest)==10473 and len(allbyUTC)==len(auxbyUTC)==10473\ndataByUTC={r['utc']:Path(r['data_path']) for r in manifest};auxByBJT={r['bjt']:Path(r['aux_path']) for r in manifest}\n(out/'indexes').mkdir();(out/'norms').mkdir();roles=['full']+[f'fold_{i}' for i in range(5)]"
# the function is included, but currently its original indentation starts one space; fix accidental nested def/body indent to match else
# reindent generated function block robustly: function body should be under if not reused (3 spaces def, 4+ body)
# The function text in old code had one leading space; generated block added two, yielding 3-space def and original 1-space body+2=3; body needs 4.
lines=block.splitlines()
for i,line in enumerate(lines):
    if line.startswith('   def correct_frame(jf):'):
        for j in range(i+1,len(lines)):
            if lines[j].startswith('  with ThreadPoolExecutor'):
                break
            if lines[j].strip():
                lines[j]=' '+lines[j]
        break
block='\n'.join(lines)
s=s[:idx1]+block+s[idx2:]
old="new.append(dict(split=q['split'],data_paths='|'.join(str(a.cache/'data'/Path(f).name) for f in q['data_paths'].split('|')),aux_paths='|'.join(str(a.cache/'aux'/Path(f).name) for f in q['aux_paths'].split('|'))))"
new="new.append(dict(split=q['split'],data_paths='|'.join(str(dataByUTC[Path(f).stem[:12]]) for f in q['data_paths'].split('|')),aux_paths='|'.join(str(auxByBJT[Path(f).stem[:12]]) for f in q['aux_paths'].split('|'))))"
assert old in s, 'index path replacement anchor missing'
s=s.replace(old,new,1)
old="assert [r['time_bjt'] for r in rr]==list(map(str,cp['time_bjt']))"
new="assert set(r['time_bjt'] for r in rr)==set(map(str,cp['time_bjt'])) and len(set(r['time_bjt'] for r in rr))==len(rr), 'R/CPP key sets differ; joining by key is required'"
assert old in s, 'R CPP assertion anchor missing'
s=s.replace(old,new,1)
p.write_text(s,encoding='utf-8')

p=code/'audit_prepared_attempt5.py'; s=p.read_text(encoding='utf-8')
s=s.replace("assert [str(k) for k in cp['time_bjt']] == times", "assert set(str(k) for k in cp['time_bjt']) == set(times) and len(set(str(k) for k in cp['time_bjt'])) == len(times), 'CPP/R key sets differ'")
p.write_text(s,encoding='utf-8')

# A new immutable PBS entry point. Reuse attempt4's complete on-node cache when hash verification passes;
# otherwise build a new attempt5 cache on node22 /dev/shm.
base='/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021'
script=f'''#!/bin/bash
#PBS -N hb_solar_data5
#PBS -q gpu
#PBS -l nodes=node22:ppn=8
#PBS -l walltime=04:00:00
#PBS -j oe
#PBS -o {base}/prepare_data_attempt5.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
/public/home/slfu/miniconda3/envs/swc/bin/python -u {base}/code_pipeline_attempt5/build_corrected_data.py --base {base} --preparation /tmp/hubei_solar_audit_20261021_attempt5 --cache /dev/shm/hubei_solar_corrected_20261021_attempt5 --old-stage /public/home/slfu/ttzhou/swc/irradiance_forecast/hubei_independent_trainval_stage_20261017 --rpack /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cloud_probability_20261020/R_pack_correctutc_attempt2 --cpp-sidecar /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cloud_probability_20261020/cpp_attempt2/cpp_supervision_trainval.npz --restored {base}/restored_attempt3 --grid {base}/grid_static256.npz --canonical-data-root /home/Data_Pool_3/chenyi/Auxiliary_data/HuBei_AGRI_Preprocessed/data
/public/home/slfu/miniconda3/envs/swc/bin/python -u {base}/code_pipeline_attempt5/audit_prepared_attempt5.py --base {base} --preparation /tmp/hubei_solar_audit_20261021_attempt5 --rpack /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cloud_probability_20261020/R_pack_correctutc_attempt2 --cpp-sidecar /public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cloud_probability_20261020/cpp_attempt2/cpp_supervision_trainval.npz --grid {base}/grid_static256.npz
'''
(root/'prepare_data_attempt5.pbs').write_text(script,encoding='utf-8')

# Run local syntax compile before packing; hash and archive are attempt-specific.
files=sorted(code.glob('*.py'))
for f in files: compile(f.read_text(encoding='utf-8'),str(f),'exec')
subprocess.run(['tar','-czf',str(root/'code_pipeline_attempt5.tar.gz'),'-C',str(root),'code_pipeline_attempt5'],check=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
manifest={'attempt':'attempt5','reason':'fix CPP/R key-order assumption; require exact unique-key set and join by time key; reuse only SHA-verified complete attempt4 node22 cache','files':[{'file':f.name,'sha256':sha(f)} for f in files],'archive_sha256':sha(root/'code_pipeline_attempt5.tar.gz'),'prep_pbs_sha256':sha(root/'prepare_data_attempt5.pbs')}
(root/'CODE_SHA256_attempt5.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
