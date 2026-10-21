"""Compare complete single-GPU and DDP4 COT auxiliary training epochs."""
import datetime
import json
from pathlib import Path
import re
import statistics

root=Path(__file__).resolve().parent
src=root.parent/'hubei_cot_auxiliary_retrain_attempt13_retry4'
times={}
for e,m,s in re.findall(r'Epoch (\d+):[^\r\n]*?282/282 \[(\d+):(\d+)',
                       (root/'ddp4_retry4.log').read_text(errors='replace')):
    e=int(e)
    if e>=17: times[e]=max(times.get(e,0),int(m)*60+int(s))
serial=[]
for path in (src/'runs/cot_aux').glob('train_*.log'):
    for line in path.read_text(errors='replace').splitlines():
        if re.search(r' - Epoch \d+: Lr:',line):
            serial.append((datetime.datetime.strptime(line[:23],'%Y-%m-%d %H:%M:%S,%f'),
                           int(re.search(r'Epoch (\d+)',line).group(1))))
serial.sort()
single=[(b[0]-a[0]).total_seconds() for a,b in zip(serial,serial[1:]) if b[1]==a[1]+1]
ddp=list(times.values());assert len(ddp)==33
meta=json.loads((root/'runs/cot_aux/run_metadata.json').read_text())
assert meta['completed_epochs']==50 and meta['global_updates']==12800
result=dict(state='PASS_COMPLETE_TRAINING_SPEED_AUDIT',
    measured_at_bjt=datetime.datetime.now().isoformat(),completed_epochs=50,global_updates=12800,
    serial_epoch_intervals=len(single),serial_mean_seconds=statistics.mean(single),
    serial_median_seconds=statistics.median(single),ddp_epochs=len(ddp),
    ddp_mean_seconds=statistics.mean(ddp),ddp_median_seconds=statistics.median(ddp),
    speedup=statistics.mean(single)/statistics.mean(ddp),ddp_total_epoch_seconds=sum(ddp),
    estimated_serial_seconds_same_33_epochs=33*statistics.mean(single),
    estimated_seconds_saved=33*statistics.mean(single)-sum(ddp),ddp_epoch_seconds=times,
    timing_scope='complete training + validation epochs; excludes one-off migration and probe; serial uses successive validation-log timestamps, DDP uses final epoch progress elapsed',
    test_payloads_read=0,buffer_payloads_read=0)
(root/'TRAINING_SPEED_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
