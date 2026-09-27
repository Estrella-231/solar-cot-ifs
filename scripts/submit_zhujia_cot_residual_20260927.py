"""Submit one Zhujia residual pilot after conservative PBS/resource checks."""
import re,subprocess
from pathlib import Path
PBS=Path('/opt/gridview/pbs/dispatcher/bin'); EXP=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/regional_station_cot_20260925')
def run(args): return subprocess.check_output(args,text=True)
listing=run([str(PBS/'qstat'),'-u','slfu']);print(listing,flush=True)
jobs=[line.split()[0] for line in listing.splitlines() if line.split() and re.match(r'^\d+\.',line.split()[0]) and line.split()[-2]!='C']
if len(jobs)>=3: raise SystemExit('blocked: account already has 3 unfinished jobs')
gpus=0
for job in jobs:
    detail=run([str(PBS/'qstat'),'-f',job]); m=re.search(r'Resource_List.nodes\s*=\s*([^\n]+)',detail)
    if not m: raise SystemExit('blocked: cannot audit GPU allocation for '+job)
    gpus+=sum(map(int,re.findall(r'gpus=(\d+)',m.group(1))))
if gpus+1>8: raise SystemExit('blocked: would exceed eight-GPU account limit')
node=run([str(PBS/'pbsnodes'),'-a','node21']);print(node[:5000],flush=True)
if 'state = free' not in node.lower(): raise SystemExit('blocked: node21 is not reported free')
script=EXP/'scripts/run_zhujia_cot_residual_20260927_node21.pbs'
subprocess.run(['bash','-n',str(script)],check=True)
out=EXP/'zhujia_cot_residual_seed42_20260927_v1'
if out.exists() and any(out.iterdir()): raise SystemExit('nonempty output exists; refusing duplicate')
job=run([str(PBS/'qsub'),str(script)]).strip();print('SUBMITTED',job);print(run([str(PBS/'qstat'),'-u','slfu']))
