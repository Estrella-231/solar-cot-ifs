"""Acknowledge the verified latest backup, stop only the replaced job, release DDP."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--new-job',required=True);a=p.parse_args();root=a.root
def run(*args):return subprocess.check_output(args,text=True)
for _ in range(90):
    if (root/'PAUSED_BACKUP_READY.json').exists():break
    time.sleep(1)
else:raise TimeoutError('No paused complete backup; original job must remain running')
r=json.loads((root/'PAUSED_BACKUP_READY.json').read_text())
assert not (root/'CUTOVER_PAUSE_ROLLED_BACK.json').exists()
assert r['state']=='PAUSED_BETWEEN_EPOCHS_WITH_COMPLETE_BACKUP'
assert r['global_step']%256==0 and r['next_epoch_progress'][1]==0
assert (Path(r['backup'])/'BACKUP_RECEIPT.json').exists()
for job in ('215356.tc6000',a.new_job):
    info=run('qstat','-f',job)
    assert re.search(r'Job_Owner\s*=\s*slfu@',info) and re.search(r'job_state\s*=\s*R',info),info
    (root/('cutover_'+job+'.txt')).write_text(info)
status=run('ssh','node22','cat',f'/proc/{r["pid"]}/status')
assert re.search(r'State:\s*T',status),status
queue=run('qstat','-u','slfu');(root/'cutover_queue_before.txt').write_text(queue)
subprocess.run(['qdel','215356.tc6000'],check=True)
for _ in range(120):
    exists=subprocess.run(['ssh','node22','test','-d',f'/proc/{r["pid"]}']).returncode==0
    if not exists:break
    time.sleep(1)
else:raise RuntimeError('Original trainer still present; DDP full training has NOT been released')
stop=dict(state='SOURCE_TRAINER_EXIT_CONFIRMED_AFTER_BACKUP',source_job='215356.tc6000',
          new_job=a.new_job,backup=r['backup'],global_step=r['global_step'],
          completed_epochs=r['checkpoint_epoch']+1,original_pid=r['pid'])
(root/'CUTOVER_ORIGINAL_STOP_CONFIRMED.json').write_text(json.dumps(stop,indent=2))
tmp=root/'CUTOVER_READY.tmp';tmp.write_text(json.dumps(stop,indent=2));tmp.rename(root/'CUTOVER_READY.json')
print(json.dumps(stop),flush=True)
print(run('qstat','-u','slfu'),flush=True)
