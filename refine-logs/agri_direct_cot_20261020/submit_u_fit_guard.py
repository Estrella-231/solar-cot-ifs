"""Submit exactly one CPU-only U sample extraction after full-account queue check."""
import hashlib,json,re,subprocess
from pathlib import Path

B=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/refine-logs/agri_direct_cot_20261020')
Q='/opt/gridview/pbs/dispatcher/bin/'

def main():
    target=B/'results/u_fit_submission_20261021.json';intent=B/'results/u_fit_submission_intent_20261021.json'
    assert not target.exists() and not intent.exists(),'inspect existing submission; no retry'
    assert not (B/'results/u_train_fit_sample_v1_20261021').exists()
    assert (B/'results/predictor_u_v1/u_teacher_cache/cache_manifest.json').is_file()
    expected=json.loads((B/'results/u_fit_sources_20261021.json').read_text())
    for n,h in expected.items():assert hashlib.sha256((B/n).read_bytes()).hexdigest()==h,n
    before=subprocess.run([Q+'qstat','-u','slfu'],capture_output=True,text=True,check=True)
    rr=[l.split() for l in before.stdout.splitlines() if re.match(r'^\s*\d+\.',l)]
    assert all(len(r)>=11 and len(r[-2])==1 for r in rr)
    count=sum(r[-2] not in ('C','F') for r in rr);assert count<3,before.stdout
    rec={'queue_before':before.stdout,'unfinished_before':count,'source_sha256':expected,
         'kind':'CPU read-only U diagnostic sample, no new training or GHI reads'}
    intent.write_text(json.dumps(rec,indent=2)+'\n')
    q=subprocess.run([Q+'qsub',str(B/'run_u_train_fit_sample_cpu_20261021.pbs')],capture_output=True,text=True,check=True)
    rec['job_id']=q.stdout.strip();assert re.fullmatch(r'\d+\.[\w.-]+',rec['job_id'])
    target.write_text(json.dumps(rec,indent=2)+'\n')
    after=subprocess.run([Q+'qstat','-u','slfu'],capture_output=True,text=True)
    rec['queue_after']=after.stdout;rec['queue_after_exit']=after.returncode
    target.write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rec,indent=2),flush=True)

if __name__=='__main__':main()
