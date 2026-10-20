"""Resume authorized Hubei solar-geometry repair after transient pre-auth SSH closure."""
import subprocess, time, json, shlex, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent.parent
BASE='/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021'
CODE=BASE+'/code_pipeline_attempt2'; PREP='/tmp/hubei_solar_audit_20261021_attempt5'; FD='/tmp/hubei_solar_restore_20261021_attempt2'
PY='/public/home/slfu/miniconda3/envs/swc/bin/python'; FDPY='/home/Data_Pool/zjnu/.conda/envs/swc/bin/python'
STATE=HERE/'PIPELINE_STATE_ATTEMPT3.json'

def retryable_preauth(detail):
    return 'kex_exchange_identification: Connection closed by remote host' in detail

def run(args):
    p=subprocess.run(list(map(str,args)),capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
    if p.returncode:
        raise RuntimeError(str(args)+'\\n'+p.stdout+'\\n'+p.stderr)
    return p.stdout

def ssh(command,host='zjnu-hpc'):
    while True:
        p=subprocess.run(['ssh','-vv','-o','BatchMode=yes','-o','ConnectTimeout=15',host,command],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
        if p.returncode==0:
            return p.stdout
        detail=p.stdout+'\\n'+p.stderr
        if retryable_preauth(detail):
            print('SSH rejected before authentication; safe read/write retry in 60 s for host='+host,flush=True)
            time.sleep(60)
            continue
        raise RuntimeError(str(command)+'\\n'+detail[-3000:])

def node(command):
    return ssh('ssh node22 '+shlex.quote(command))

def save(stage,**extra):
    rec=dict(stage=stage,time_unix=time.time(),test_payloads_read=0,**extra)
    STATE.write_text(json.dumps(rec,indent=2),encoding='utf-8')
    print(json.dumps(rec),flush=True)

def scp_retry(args):
    while True:
        p=subprocess.run(['scp','-vv','-o','BatchMode=yes','-o','ConnectTimeout=20',*map(str,args)],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=600)
        if p.returncode==0:
            return p.stdout
        detail=p.stdout+'\\n'+p.stderr
        if retryable_preauth(detail):
            print('SCP rejected before authentication; safe transfer retry in 60 s',flush=True)
            time.sleep(60)
            continue
        raise RuntimeError(str(args)+'\\n'+detail[-3000:])

def cpdown(remote,name,host='zjnu-hpc'):
    return scp_retry([host+':'+remote,HERE/name])
def cpup(name,remote,host='zjnu-hpc'):
    return scp_retry([HERE/name,host+':'+remote])

def waitfile(path,host='zjnu-hpc',job=None):
    while True:
        status=ssh('if test -f '+shlex.quote(path)+'; then echo READY; else echo WAIT; fi'+('; qstat -f '+shlex.quote(job)+' 2>/dev/null | grep -E "job_state|exit_status" || true' if job else ''),host)
        if status.startswith('READY'):
            return
        if job and 'job_state = C' in status:
            raise RuntimeError('Job ended without expected artifact: '+job+' '+status[-1500:])
        time.sleep(60)

def qsubmit(script):
    cmd=f"import sys;sys.path.insert(0,{CODE!r});import queue_controller as q;import subprocess,json;v=q.queue();assert len(v)<3,v;p=subprocess.run([q.QSUB,{script!r}],capture_output=True,text=True,check=True);print(p.stdout.strip(),flush=True);print(json.dumps(q.queue()),flush=True)"
    save('SUBMITTING_DATA_PREPARATION_NO_AUTORETRY',script=script)
    while True:
        available=int(ssh(PY+' -c '+shlex.quote(f'import sys;sys.path.insert(0,{CODE!r});import queue_controller as q;print(len(q.queue()))')).strip())
        if available<3:
            break
        print('PBS account has 3 unfinished jobs; waiting 60 s before qsub',flush=True)
        time.sleep(60)
    text=ssh(PY+' -c '+shlex.quote(cmd))
    save('DATA_PREPARATION_SUBMITTED',receipt=text)
    return text.splitlines()[0]

def main():
    if STATE.exists():
        raise RuntimeError('Attempt3 state already exists; inspect before restart, never duplicate jobs blindly')
    prior=HERE/'PIPELINE_STATE_ATTEMPT2.json'
    prev=json.loads(prior.read_text(encoding='utf-8')) if prior.exists() else {}
    err=prev.get('error','')
    expected_resume=('GEOMETRY_CALCULATION_REPORT_attempt5.json' in err and 'Connection closed by 10.65.129.39 port 22' in err)
    if prev and not expected_resume:
        raise RuntimeError('Previous attempt did not fail solely while waiting for the already-running geometry report; manual provenance review required')
    save('RESUMING_WAIT_FOR_EXISTING_AUDIT_NO_RESUBMIT',prior_state=prev.get('stage'),audit_pbs='215288.tc6000',previous_error=err[-500:] if err else None)
    waitfile(BASE+'/GEOMETRY_CALCULATION_REPORT_attempt5.json',job='215288.tc6000')
    while 'job_state = C' not in ssh('qstat -f 215288.tc6000 2>/dev/null || true'):
        time.sleep(60)
    cpdown(BASE+'/GEOMETRY_CALCULATION_REPORT_attempt5.json','GEOMETRY_CALCULATION_REPORT_attempt5.json')
    cpdown(BASE+'/reference_cases_attempt5.jsonl','reference_cases_attempt5.jsonl')
    run(['node',HERE/'validate_noaa_reference.js',HERE/'noaa_main_reference.js',HERE/'reference_cases_attempt5.jsonl',HERE/'NOAA_ALL_TRAIN_VAL_REFERENCE_AUDIT.json'])
    cpup('NOAA_ALL_TRAIN_VAL_REFERENCE_AUDIT.json',BASE+'/NOAA_ALL_TRAIN_VAL_REFERENCE_AUDIT.json')
    save('ALL_TRAIN_VAL_OFFICIAL_NOAA_REFERENCE_PASS')
    node(PY+' '+CODE+'/augment_restoration_requests.py /tmp/hubei_solar_audit_20261021_attempt5 /public/home/slfu/ttzhou/swc/irradiance_forecast/hubei_independent_trainval_stage_20261017')
    node('tar -czf '+BASE+'/restoration_requests_attempt2.tar.gz -C '+PREP+' restore_requests_with_witness RESTORATION_REQUEST_WITNESS.json')
    cpdown(BASE+'/restoration_requests_attempt2.tar.gz','restoration_requests_attempt2.tar.gz')
    ssh('mkdir -p '+FD,'FD-201')
    cpup('restoration_requests_attempt2.tar.gz',FD+'/requests.tar.gz','FD-201')
    cpup('grid_static256.npz',FD+'/grid_static256.npz','FD-201')
    cpup('code/restore_daylit_reflectance.py',FD+'/restore_daylit_reflectance.py','FD-201')
    ssh('tar -xzf '+FD+'/requests.tar.gz -C '+FD+'; nohup '+FDPY+' -u '+FD+'/restore_daylit_reflectance.py --requests '+FD+'/restore_requests_with_witness --grid '+FD+'/grid_static256.npz --output '+FD+'/restored --workers 6 > '+FD+'/restore.log 2>&1 < /dev/null &','FD-201')
    save('RESTORING_RAW_VISIBLE_CHANNELS_ON_FD201_CPU')
    while True:
        v=ssh('test -f '+FD+'/restored_visible.tar.gz && echo READY || echo WAIT','FD-201')
        if v.strip()=='READY':
            break
        text=ssh('tail -n 12 '+FD+'/restore.log','FD-201')
        if 'Traceback' in text or 'AssertionError' in text or 'RuntimeError' in text:
            raise RuntimeError(text)
        time.sleep(60)
    cpdown(FD+'/restored_visible.tar.gz','restored_visible.tar.gz','FD-201')
    cpup('restored_visible.tar.gz',BASE+'/restored_visible.tar.gz')
    node('mkdir -p '+BASE+'/restored_attempt2; tar -xzf '+BASE+'/restored_visible.tar.gz -C '+BASE+'/restored_attempt2')
    save('RAW_REFLECTANCE_RESTORATION_COMPLETE')
    jid=qsubmit(BASE+'/prepare_data_attempt2.pbs')
    waitfile(BASE+'/prepared_attempt2/DATA_READY.json',job=jid)
    cpdown(BASE+'/prepared_attempt2/DATA_READY.json','DATA_READY.json')
    save('CORRECTED_TRAIN_VAL_DATA_READY',preparation_pbs=jid)
    ssh('cd '+BASE+'; nohup '+PY+' -u '+CODE+'/queue_controller.py > queue_controller_attempt2.log 2>&1 < /dev/null &')
    save('AUTHORIZED_TRAINING_CONTROLLER_STARTED',controller=BASE+'/QUEUE_STATE_attempt2.json')

if __name__=='__main__':
    try:
        main()
    except Exception as e:
        save('FAILED_REQUIRES_REPAIR',error=repr(e))
        raise
