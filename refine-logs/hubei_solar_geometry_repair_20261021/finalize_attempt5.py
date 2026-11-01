from pathlib import Path
import hashlib,json,subprocess
root=Path(r'F:\CODE_Classify\辐照度预报\solar-cot-ifs\refine-logs\hubei_solar_geometry_repair_20261021');code=root/'code_pipeline_attempt5'
base='/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021'
pbs=f'''#!/bin/bash
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
(root/'prepare_data_attempt5.pbs').write_bytes(pbs.encode('ascii'))
helper='''import subprocess,sys,json\nfrom pathlib import Path\nBASE=Path(%r)\nsys.path.insert(0,str(BASE/'code_pipeline_attempt5'))\nimport queue_controller as q\nassert (BASE/'prepare_data_attempt5.pbs').is_file()\nassert not (BASE/'prepared_attempt5').exists()\npre=q.queue()\nprint('QUEUE_BEFORE',json.dumps(pre,sort_keys=True),flush=True)\nif len(pre)>=3: raise RuntimeError('three unfinished account jobs; refusing submission')\nif any(v.get('name')=='hb_solar_data5' for v in pre.values()): raise RuntimeError('attempt5 data job already active')\np=subprocess.run([q.QSUB,str(BASE/'prepare_data_attempt5.pbs')],capture_output=True,text=True)\nprint('QSUB_RESULT',p.returncode,p.stdout.strip(),p.stderr.strip(),flush=True)\nif p.returncode: raise SystemExit(p.returncode)\njid=p.stdout.strip()\nif not jid or not jid[0].isdigit(): raise RuntimeError('unrecognized qsub result; inspect queue before any retry')\nafter=q.queue()\nprint('SUBMITTED',jid,'QUEUE_AFTER',json.dumps(after,sort_keys=True),flush=True)\nif jid not in after: print('JOB_NOT_IN_IMMEDIATE_QSTAT; inspect qstat -f before any retry',flush=True)\n'''%('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021')
(root/'submit_prepare_attempt5.py').write_text(helper,encoding='utf-8')
files=sorted(code.glob('*.py'))
for f in files: compile(f.read_text(encoding='utf-8'),str(f),'exec')
for f in (root/'prepare_data_attempt5.pbs',root/'submit_prepare_attempt5.py'):
 if f.suffix=='.py': compile(f.read_text(encoding='utf-8'),str(f),'exec')
import shutil
shutil.rmtree(code/'__pycache__',ignore_errors=True)
subprocess.run(['tar','-czf',str(root/'code_pipeline_attempt5.tar.gz'),'--exclude=__pycache__','-C',str(root),'code_pipeline_attempt5'],check=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
manifest={'attempt':'attempt5','reason':'accept identical unique CPP/R key sets independent of row ordering; join by BJT key; reuse only hash-verified attempt4 cache on node22','files':[{'file':f.name,'sha256':sha(f)} for f in files],'archive_sha256':sha(root/'code_pipeline_attempt5.tar.gz'),'prep_pbs_sha256':sha(root/'prepare_data_attempt5.pbs'),'submit_helper_sha256':sha(root/'submit_prepare_attempt5.py')}
(root/'CODE_SHA256_attempt5.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))



