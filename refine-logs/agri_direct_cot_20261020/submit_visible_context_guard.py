"""One specific CPU diagnostic submission; no automatic retries."""
import hashlib
import json
import re
import subprocess
from datetime import datetime,timezone
from pathlib import Path

BASE=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/refine-logs/agri_direct_cot_20261020')
QSTAT='/opt/gridview/pbs/dispatcher/bin/qstat'
QSUB='/opt/gridview/pbs/dispatcher/bin/qsub'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    out=BASE/'results/visible_context_submission_v2_20261021.json'
    intent=BASE/'results/visible_context_submission_v2_intent_20261021.json'
    if out.exists() or intent.exists():raise RuntimeError('Submission record exists; inspect authoritative queue, do not resubmit')
    if (BASE/'results/visible_neighbor_context_v1_20261021').exists():raise RuntimeError('Diagnostic output already exists')
    names=['extract_visible_neighbor_context.py','run_visible_neighbor_context_cpu_20261021.pbs','VISIBLE_NEIGHBOR_HISTORY_CONTEXT_PLAN_20261021.md']
    expected=json.loads((BASE/'results/visible_context_sources_20261021.json').read_text())
    for n in names:assert sha(BASE/n)==expected[n],n
    before=subprocess.run([QSTAT,'-u','slfu'],capture_output=True,text=True,check=True)
    print(before.stdout,flush=True)
    jobrows=[line.split() for line in before.stdout.splitlines() if re.match(r'^\s*\d+\.',line)]
    # tc6000 short table has state in penultimate column; unknown states count.
    assert all(len(r)>=11 and len(r[-2])==1 for r in jobrows), 'Queue layout changed'
    unfinished=sum(r[-2] not in ('C','F') for r in jobrows)
    assert unfinished<3, 'Full account queue has no slot'
    record={'time_utc':datetime.now(timezone.utc).isoformat(),'queue_before':before.stdout,
            'unfinished_job_rows_before':unfinished,'source_sha256':expected,
            'completed_rows_excluded':[r[0] for r in jobrows if r[-2] in ('C','F')],
            'prior_attempt':'215365 failed before output creation due to archive-only index path; now exact source index and SHA bound',
            'kind':'CPU-only offline V context; no training, GPU, GHI or test/buffer'}
    intent.write_text(json.dumps(record,indent=2)+'\n')
    submitted=subprocess.run([QSUB,str(BASE/'run_visible_neighbor_context_cpu_20261021.pbs')],capture_output=True,text=True,check=True)
    job=submitted.stdout.strip();assert re.fullmatch(r'\d+\.[\w.-]+',job),job
    record['job_id']=job;record['state']='SUBMITTED_CPU_CONTEXT';record['qsub_stderr']=submitted.stderr
    out.write_text(json.dumps(record,indent=2)+'\n')
    after=subprocess.run([QSTAT,'-u','slfu'],capture_output=True,text=True)
    record['queue_after']=after.stdout;record['queue_after_exit']=after.returncode
    out.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)

if __name__=='__main__':main()
