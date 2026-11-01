"""Submit only the CPU metadata build, with live whole-account PBS limit."""
import subprocess,json
from pathlib import Path
run=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/monthly_R_station16_20261013_v1/interval_alignment_20261015')
receipt=run/'manifest_full_v1_submission.json'
assert not receipt.exists() and not (run/'manifest_full_v1').exists(),'already submitted/started'
before=subprocess.check_output(['qstat','-u','slfu'],text=True)
active=[]
for line in before.splitlines():
    p=line.split()
    if len(p)>2 and p[0][0].isdigit() and p[1]=='slfu' and p[-2] not in ('C','F'):active.append(line)
assert len(active)<3,active
job=subprocess.check_output(['qsub',str(run/'run_interval_manifest_20261015.pbs')],text=True).strip()
after=subprocess.check_output(['qstat','-u','slfu'],text=True)
data=dict(job=job,before=before,after=after,unfinished_before=len(active),purpose='CPU scan metadata only; no training/test/GPU')
receipt.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))
