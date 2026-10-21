"""Stable, versioned backup of the live COT arm, including optimizer/scheduler."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--label',default='preflight')
    a=p.parse_args();src=a.source
    cp=src/'runs/cot_aux/checkpoints/last.ckpt'
    out=a.root/'backups'/(a.label+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True,exist_ok=False)
    before=cp.stat()
    files=[*list((src/'runs/cot_aux/checkpoints').glob('*.ckpt')),
           src/'runs/cot_aux/config.py',src/'runs/cot_aux/attempt13_init_receipt.json',
           src/'prepared_inputs.json',src/'lambda/train_only_lambda.json',src/'PAIR_STARTED_attempt13.json']
    files += list((src/'runs/cot_aux/cwd').glob('lightning_logs/version_*/metrics.csv'))
    files += list((src/'runs/cot_aux/cwd').glob('lightning_logs/version_*/events.out.tfevents.*'))
    records={}
    for path in files:
        dest=out/path.relative_to(src);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest)
        records[str(path.relative_to(src))]={'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'bytes':dest.stat().st_size}
    after=cp.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Retry backup: source checkpoint changed during copy'
    assert records[str(cp.relative_to(src))]['sha256']==hashlib.sha256(cp.read_bytes()).hexdigest()
    ck=torch.load(out/cp.relative_to(src),map_location='cpu')
    assert ck['global_step']==(ck['epoch']+1)*256
    assert len(ck['optimizer_states'])==len(ck['lr_schedulers'])==1
    r=dict(state='PASS_COMPLETE_CHECKPOINT_BACKUP',source_attempt=str(src),backup=str(out),
           epoch=ck['epoch'],global_step=ck['global_step'],optimizer_states=1,lr_schedulers=1,
           files=records,test_payloads_read=0,buffer_payloads_read=0)
    (out/'BACKUP_RECEIPT.json').write_text(json.dumps(r,indent=2))
    (a.root/'LATEST_BACKUP.json').write_text(json.dumps(r,indent=2))
    print(json.dumps(r),flush=True)


if __name__=='__main__':main()
