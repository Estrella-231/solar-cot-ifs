"""Deterministic PBS dependency runner for the authorized solar correction only."""
import json,os,re,subprocess,time,hashlib
from pathlib import Path
BASE=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021');CODE=Path(__file__).resolve().parent
ROLES=['full']+[f'fold_{i}' for i in range(5)]
QSTAT='/opt/gridview/pbs/dispatcher/bin/qstat';QSUB='/opt/gridview/pbs/dispatcher/bin/qsub'
BASEENV='/public/home/slfu/miniconda3/envs/swc';SENV='/public/home/slfu/ttzhou/swc/irradiance_forecast/hubei_cot_independent_20261017/.venv_simvp'
STATE=BASE/'QUEUE_STATE_attempt2.json'
def save(s):
 p=STATE.with_suffix('.tmp');p.write_text(json.dumps(s,indent=2));p.replace(STATE)
def queue():
 p=subprocess.run([QSTAT,'-u','slfu'],capture_output=True,text=True,check=True);jobs={}
 for ln in p.stdout.splitlines():
  x=ln.split()
  if len(x)<10 or not re.match(r'^\d+\.tc6000$',x[0]) or x[1]!='slfu' or x[9]=='C':continue
  f=subprocess.run([QSTAT,'-f',x[0]],capture_output=True,text=True);g=re.findall(r'gpus=(\d+)',f.stdout);jobs[x[0]]={'state':x[9],'gpus':sum(map(int,g)) if g else 0,'name':x[3]}
 return jobs

def complete(k):
 p=BASE/({'profile':'S_PROFILE_READY_attempt2.json','R':'R_READY_attempt2.json','GHI':'ALL_PAIRED_COMPLETE_attempt2.json'}.get(k,f'S_runs_attempt2/{k}/S_BANK_READY.json'))
 if not p.exists():return False
 try:d=json.loads(p.read_text())
 except json.JSONDecodeError:return False
 return bool(d.get('state'))
def deps(k):
 if not (BASE/'prepared_attempt2/DATA_READY.json').exists():return False
 if k in ('profile','R'):return True
 if k=='GHI':return complete('R') and all(complete(r) for r in ROLES)
 return complete('profile')
def shape(k):
 if k in ('profile','R','GHI'):return 1
 d=json.loads((BASE/'S_PROFILE_READY_attempt2.json').read_text());return d['choice'][k]['gpus']
def pbs(k,g):
 mode=k if k in ('profile','R','GHI') else 'S';role=k if mode=='S' else 'full';env=SENV if mode in ('profile','S') else BASEENV;name='hb_sol_'+k.replace('_','');out=BASE/'pbs_scripts'/f'{k}.pbs';out.parent.mkdir(exist_ok=True)
 text=f'''#!/bin/bash
#PBS -N {name}
#PBS -q gpu
#PBS -l nodes=node22:ppn=8:gpus={g}
#PBS -l walltime=36:00:00
#PBS -j oe
#PBS -o {BASE}/{k}.job.log
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
GPU_IDS=$({QSTAT} -f "$PBS_JOBID" | tr -d '[:space:]' | grep -oE 'node22-gpu/[0-9]+' | cut -d/ -f2 | sort -nu | paste -sd, -)
test "$(awk -F, '{{print NF}}' <<< "$GPU_IDS")" -eq {g}
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
IFS=, read -r -a devices <<< "$GPU_IDS"
for dev in "${{devices[@]}}"; do test -z "$(nvidia-smi -i "$dev" --query-compute-apps=pid --format=csv,noheader)"; done
echo "PBS=$PBS_JOBID NODE=$(hostname) GPU=$GPU_IDS MODE={mode} ROLE={role}"
CUDNN_LIB={BASEENV}/lib/python3.10/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH="$CUDNN_LIB:${{LD_LIBRARY_PATH:-}}"
export LD_PRELOAD="$CUDNN_LIB/libcudnn_ops_infer.so.8:$CUDNN_LIB/libcudnn_ops_train.so.8:$CUDNN_LIB/libcudnn_cnn_infer.so.8:$CUDNN_LIB/libcudnn_cnn_train.so.8:$CUDNN_LIB/libcudnn_adv_infer.so.8:$CUDNN_LIB/libcudnn_adv_train.so.8:$CUDNN_LIB/libcudnn.so.8"
{env}/bin/python -u {CODE}/run_stages.py {mode} --role {role}
echo "STAGE_SUCCESS {k} test_payloads_read=0"
''';out.write_text(text);return out

def main():
 import fcntl
 lock=(BASE/'queue_controller_attempt2.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=json.loads(STATE.read_text()) if STATE.exists() else {'state':'WAITING_DATA','jobs':{},'completed':[],'failed':{},'owner':'hubei_solar_geometry_repair_20261021','test_payloads_read':0};keys=['profile','R']+ROLES+['GHI']
 while True:
  if (BASE/'STOP_QUEUE_attempt2').exists():state['state']='STOP_REQUESTED_NO_NEW_SUBMISSIONS';save(state);return
  active=queue()
  for k,info in list(state['jobs'].items()):
   if complete(k):
    if k not in state['completed']:state['completed'].append(k);print('COMPLETED',k,info['pbs'],flush=True)
   elif info['pbs'] not in active:
    # Completed PBS entries may disappear; only a real completion artifact confirms success.
    if time.time()-info['submitted_unix']>120:state['failed'][k]=info;state['state']='FAILED_STAGE_REQUIRES_REPAIR';save(state);print('FAILED',k,info['pbs'],flush=True);return
  for k in keys:
   if k in state['jobs'] or not deps(k):continue
   active=queue();ng=shape(k)
   if len(active)>=3 or sum(j['gpus'] for j in active.values())+ng>8:break
   script=pbs(k,ng);state['state']='SUBMITTING';state['submitting']=k;save(state)
   # Recheck immediately before every individual qsub; never batch submissions.
   before=queue()
   if len(before)>=3 or sum(j['gpus'] for j in before.values())+ng>8:continue
   p=subprocess.run([QSUB,str(script)],capture_output=True,text=True)
   if p.returncode or not re.fullmatch(r'\d+\.tc6000',p.stdout.strip()):state['state']='SUBMISSION_OUTCOME_REQUIRES_MANUAL_CHECK';state['submit_output']=p.stdout+p.stderr;save(state);raise RuntimeError(state['submit_output'])
   jid=p.stdout.strip();state['jobs'][k]={'pbs':jid,'gpus':ng,'script':str(script),'submitted_unix':time.time()};save(state);state['queue_after_last_submit']=queue();state.pop('submitting',None);state['state']='RUNNING_AUTHORIZED_STAGES';save(state);print('SUBMITTED',k,jid,ng,flush=True)
  if complete('GHI'):state['state']='ALL_CORRECTED_SOLAR_PAIRED_COMPLETE';save(state);return
  save(state);time.sleep(60)
if __name__=='__main__':main()
