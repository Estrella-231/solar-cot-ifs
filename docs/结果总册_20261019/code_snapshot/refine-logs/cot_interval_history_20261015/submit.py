"""One idempotent PBS submission after completed source/geometry audits."""
import json,subprocess
from pathlib import Path

root=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/cot_interval_history_20261015_v1')
receipt=root/'submission_attempt1.json'
assert not receipt.exists() and not (root/'results_attempt1').exists()
geo=json.loads((root/'geometry_audit.json').read_text());assert geo['state'].startswith('PASS_')
wb=json.loads((root/'workbook_audit.json').read_text());assert wb['test_payloads_read']==0
assert all(r['max_abs_cached_source_difference']==0 for r in wb['stations'].values())
pack=json.loads((root/'data_attempt2/pack_manifest.json').read_text());assert pack['state']=='COMPLETE' and pack['history_offsets_minutes']==[-30,-15,0]
before=subprocess.check_output(['qstat','-u','slfu'],text=True);active=[]
for line in before.splitlines():
    p=line.split()
    if len(p)>2 and p[0][0].isdigit() and p[1]=='slfu' and p[-2] not in ('C','F'):active.append(line)
if len(active)>=3:raise RuntimeError('PBS limit reached; no submission: '+str(active))
job=subprocess.check_output(['qsub',str(root/'code/run.pbs')],text=True).strip()
after=subprocess.check_output(['qstat','-u','slfu'],text=True)
data=dict(job=job,before=before,after=after,unfinished_before=len(active),purpose='two stations x two history arms x three seeds; GHI-only corrected geometry; test closed')
receipt.write_text(json.dumps(data,indent=2));print(json.dumps(data,indent=2))
