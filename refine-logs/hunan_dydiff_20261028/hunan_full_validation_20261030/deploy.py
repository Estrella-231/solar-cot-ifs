"""Copy this named evaluation only; job submission is a separate action."""
from pathlib import Path
import ast,json,subprocess

HERE=Path(__file__).resolve().parent
REMOTE='/public/home/slfu/ttzhou/swc/irradiance_forecast/HunanDyDiff_20261028/hunan_full_validation_20261030'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','zjnu-hpc']
def run(args,**kwargs):
    r=subprocess.run(args,cwd=HERE,capture_output=True,text=True,encoding='utf-8',errors='replace',**kwargs)
    if r.returncode:raise RuntimeError(r.stdout+'\n'+r.stderr)
    return r.stdout
files=[p.name for p in HERE.glob('*.py')]+['run_four_gpu.pbs','EXPERIMENT_PLAN.md','CODE_SHA256.json']
for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
print(run(SSH+[f'mkdir -p {REMOTE} && test ! -e {REMOTE}/forecast_attempt1 && test ! -e {REMOTE}/evaluation_attempt1']))
print(run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15',*files,'zjnu-hpc:'+REMOTE+'/']))
# Normalize CRLF only for this new evaluation's PBS file, not training sources.
print(run(SSH+[f"sed -i 's/\\r$//' {REMOTE}/run_four_gpu.pbs && /public/home/slfu/miniconda3/envs/swc/bin/python -m py_compile {REMOTE}/generate_hunan.py {REMOTE}/evaluate_full_gpu.py {REMOTE}/merge_shards.py {REMOTE}/score_full.py {REMOTE}/score_saved.py && bash -n {REMOTE}/run_four_gpu.pbs && sha256sum {REMOTE}/generate_hunan.py {REMOTE}/evaluate_full_gpu.py {REMOTE}/run_four_gpu.pbs"]))
result={'status':'DEPLOYED_SYNTAX_PASS','remote_directory':REMOTE,'files':files,'submitted':False}
(HERE/'DEPLOYMENT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
