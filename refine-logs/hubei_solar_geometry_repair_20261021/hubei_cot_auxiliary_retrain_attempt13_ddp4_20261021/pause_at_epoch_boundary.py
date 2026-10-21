"""Pause only our live COT trainer between epochs and capture a final backup.

The original trainer resumes automatically if the controller does not acknowledge
the cutover within 180 seconds. No in-flight optimizer updates are discarded.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import torch

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
p.add_argument('--pid',type=int,required=True);a=p.parse_args()
root=a.root;src=root.parent/'hubei_cot_auxiliary_retrain_attempt13_retry4'
probe=json.loads((root/'PROBE_PASS.json').read_text())
assert probe['state']=='PASS_DDP4_RESUME_PROBE'
proc=Path('/proc')/str(a.pid)
assert proc.stat().st_uid==os.getuid()
command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
assert str(src/'run_cot_aux_simvp_attempt13.py') in command
assert '--aux-mode cot_aux' in command or 'cot_aux' in command
log=src/'runs/cot_aux/training.log';cp=src/'runs/cot_aux/checkpoints/last.ckpt'
assert Path(os.readlink(proc/'fd/1')).resolve()==log.resolve()
def progress():
    with log.open('rb') as f:
        f.seek(max(0,log.stat().st_size-12000));s=f.read().decode(errors='replace')
    found=re.findall(r'Epoch (\d+):[^\r\n]*?\s(\d+)/282',s)
    return tuple(map(int,found[-1])) if found else None
start=time.monotonic();paused=False
try:
    while time.monotonic()-start<900:
        position=progress()
        if position and position[1]==0 and position[0]>probe['source_global_step']//256:
            os.kill(a.pid,signal.SIGSTOP);paused=True;time.sleep(.15)
            if progress()!=position:
                os.kill(a.pid,signal.SIGCONT);paused=False;time.sleep(.3);continue
            ck=torch.load(cp,map_location='cpu')
            if ck['global_step']!=position[0]*256 or ck['epoch']!=position[0]-1:
                os.kill(a.pid,signal.SIGCONT);paused=False;time.sleep(.3);continue
            subprocess.run([sys.executable,str(root/'capture_backup.py'),'--source',str(src),
                            '--root',str(root),'--label','cutover_boundary'],check=True)
            backup=json.loads((root/'LATEST_BACKUP.json').read_text())
            assert backup['global_step']==ck['global_step']
            r=dict(state='PAUSED_BETWEEN_EPOCHS_WITH_COMPLETE_BACKUP',pid=a.pid,
                   checkpoint_epoch=ck['epoch'],global_step=ck['global_step'],
                   next_epoch_progress=list(position),backup=backup['backup'],
                   previous_probe=probe,automatic_resume_timeout_seconds=180)
            (root/'PAUSED_BACKUP_READY.json').write_text(json.dumps(r,indent=2))
            print(json.dumps(r),flush=True)
            for _ in range(180):
                if not proc.exists() or (root/'CUTOVER_ORIGINAL_STOP_CONFIRMED.json').exists():
                    paused=False;break
                time.sleep(1)
            break
        time.sleep(.05)
    else: raise TimeoutError('No safely observable epoch boundary; original trainer unchanged')
finally:
    if paused and proc.exists():
        os.kill(a.pid,signal.SIGCONT)
        (root/'CUTOVER_PAUSE_ROLLED_BACK.json').write_text(json.dumps(dict(state='SOURCE_AUTO_RESUMED_AFTER_NO_ACK',pid=a.pid)))
