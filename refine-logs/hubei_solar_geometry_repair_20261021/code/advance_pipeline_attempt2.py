"""Local dependency dispatcher. No data synthesis, no retries after unknown qsub outcomes."""
import subprocess,time,json,shlex,hashlib,sys,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent.parent
BASE='/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021'
CODE=BASE+'/code_pipeline_attempt2';PREP='/tmp/hubei_solar_audit_20261021_attempt5';FD='/tmp/hubei_solar_restore_20261021_attempt2';PY='/public/home/slfu/miniconda3/envs/swc/bin/python';FDPY='/home/Data_Pool/zjnu/.conda/envs/swc/bin/python'
STATE=HERE/'PIPELINE_STATE_ATTEMPT2.json'
def run(args):
 p=subprocess.run(list(map(str,args)),capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
 if p.returncode:raise RuntimeError(str(args)+'\n'+p.stdout+'\n'+p.stderr)
 return p.stdout

def ssh(command,host='zjnu-hpc'):return run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,command])
def node(command):return ssh('ssh node22 '+shlex.quote(command))
def save(stage,**extra):
 p=dict(stage=stage,time_unix=time.time(),test_payloads_read=0,**extra);STATE.write_text(json.dumps(p,indent=2),encoding='utf-8');print(json.dumps(p),flush=True)
def cpdown(remote,name,host='zjnu-hpc'):run(['scp',host+':'+remote,HERE/name])
def cpup(name,remote,host='zjnu-hpc'):run(['scp',HERE/name,host+':'+remote])
def waitfile(path,host='zjnu-hpc',job=None):
 while True:
  if ssh('test -f '+shlex.quote(path)+' && echo READY || true',host).strip()=='READY':return
  if job:
   text=ssh('qstat -f '+shlex.quote(job)+' 2>/dev/null || true')
   if 'job_state = C' in text:raise RuntimeError('Job ended without expected artifact: '+job+' '+text[-1500:])
  time.sleep(60)
def qsubmit(script):
 # Server controller helper checks account-wide queue immediately before/after qsub.
 cmd=f"import sys;sys.path.insert(0,{CODE!r});import queue_controller as q;import subprocess,json;v=q.queue();assert len(v)<3,v;p=subprocess.run([q.QSUB,{script!r}],capture_output=True,text=True,check=True);print(p.stdout.strip(),flush=True);print(json.dumps(q.queue()),flush=True)"
 # Save unknown outcomes before invoking external submission; never blindly resubmit.
 save('SUBMITTING_DATA_PREPARATION_NO_AUTORETRY',script=script)
 while True:
  available=ssh(PY+' -c '+shlex.quote(f'import sys;sys.path.insert(0,{CODE!r});import queue_controller as q;print(len(q.queue()))')).strip()
  if int(available)<3:break
  time.sleep(60)
 text=ssh(PY+' -c '+shlex.quote(cmd));save('DATA_PREPARATION_SUBMITTED',receipt=text);return text.splitlines()[0]
def main():
 if STATE.exists():raise RuntimeError('Existing pipeline state: inspect before manual restart; no blind duplicate jobs')
 save('WAITING_ALL_TRAIN_VAL_GEOMETRY_AUDIT',audit_pbs='215288.tc6000')
 waitfile(BASE+'/GEOMETRY_CALCULATION_REPORT_attempt5.json',job='215288.tc6000')
 while 'job_state = C' not in ssh('qstat -f 215288.tc6000 2>/dev/null || true'):time.sleep(60)
 cpdown(BASE+'/GEOMETRY_CALCULATION_REPORT_attempt5.json','GEOMETRY_CALCULATION_REPORT_attempt5.json');cpdown(BASE+'/reference_cases_attempt5.jsonl','reference_cases_attempt5.jsonl')
 run(['node',HERE/'validate_noaa_reference.js',HERE/'noaa_main_reference.js',HERE/'reference_cases_attempt5.jsonl',HERE/'NOAA_ALL_TRAIN_VAL_REFERENCE_AUDIT.json']);cpup('NOAA_ALL_TRAIN_VAL_REFERENCE_AUDIT.json',BASE+'/NOAA_ALL_TRAIN_VAL_REFERENCE_AUDIT.json');save('ALL_TRAIN_VAL_OFFICIAL_NOAA_REFERENCE_PASS')
 node(PY+' '+CODE+'/augment_restoration_requests.py '+PREP+' /public/home/slfu/ttzhou/swc/irradiance_forecast/hubei_independent_trainval_stage_20261017')
 node('tar -czf '+BASE+'/restoration_requests_attempt2.tar.gz -C '+PREP+' restore_requests_with_witness RESTORATION_REQUEST_WITNESS.json')
 cpdown(BASE+'/restoration_requests_attempt2.tar.gz','restoration_requests_attempt2.tar.gz');ssh('mkdir '+FD,'FD-201');cpup('restoration_requests_attempt2.tar.gz',FD+'/requests.tar.gz','FD-201');cpup('grid_static256.npz',FD+'/grid_static256.npz','FD-201');run(['scp',HERE/'code/restore_daylit_reflectance.py','FD-201:'+FD+'/restore_daylit_reflectance.py'])
 ssh('tar -xzf '+FD+'/requests.tar.gz -C '+FD+'; nohup '+FDPY+' -u '+FD+'/restore_daylit_reflectance.py --requests '+FD+'/restore_requests_with_witness --grid '+FD+'/grid_static256.npz --output '+FD+'/restored --workers 6 > '+FD+'/restore.log 2>&1 < /dev/null &','FD-201');save('RESTORING_RAW_VISIBLE_CHANNELS_ON_FD201_CPU')
 while True:
  v=ssh('test -f '+FD+'/restored_visible.tar.gz && echo READY || true','FD-201')
  if v.strip()=='READY':break
  text=ssh('tail -n 12 '+FD+'/restore.log','FD-201')
  if 'Traceback' in text or 'AssertionError' in text or 'RuntimeError' in text:raise RuntimeError(text)
  time.sleep(60)
 cpdown(FD+'/restored_visible.tar.gz','restored_visible.tar.gz','FD-201');cpup('restored_visible.tar.gz',BASE+'/restored_visible.tar.gz');node('mkdir '+BASE+'/restored_attempt2; tar -xzf '+BASE+'/restored_visible.tar.gz -C '+BASE+'/restored_attempt2');save('RAW_REFLECTANCE_RESTORATION_COMPLETE')
 jid=qsubmit(BASE+'/prepare_data_attempt2.pbs');waitfile(BASE+'/prepared_attempt2/DATA_READY.json',job=jid);cpdown(BASE+'/prepared_attempt2/DATA_READY.json','DATA_READY.json');save('CORRECTED_TRAIN_VAL_DATA_READY',preparation_pbs=jid)
 ssh('cd '+BASE+'; nohup '+PY+' -u '+CODE+'/queue_controller.py > queue_controller_attempt2.log 2>&1 < /dev/null &');save('AUTHORIZED_TRAINING_CONTROLLER_STARTED',controller=BASE+'/QUEUE_STATE_attempt2.json')
if __name__=='__main__':
 try:main()
 except Exception as e:save('FAILED_REQUIRES_REPAIR',error=repr(e));raise
