"""Replace only our held downstream jobs with a four-GPU migration probe."""
import json
from pathlib import Path
import re
import shutil
import subprocess

root=Path(__file__).resolve().parent
src=root.parent/'hubei_cot_auxiliary_retrain_attempt13_retry4'
assert (root/'probe_arm/RESUME_INPUT.json').is_file()
out=root/'queue_before_migration';out.mkdir(exist_ok=False)
def run(*args): return subprocess.check_output(args,text=True)
def queue():
    q=run('qstat','-u','slfu')
    jobs=[row for row in q.splitlines() if re.match(r'^\d+\.',row.strip())]
    unfinished=[row for row in jobs if row.split()[-2] not in ('C','E')]
    # Exiting still occupies the limit until the job disappears.
    unfinished += [row for row in jobs if row.split()[-2]=='E']
    assert len(unfinished)<=3,q
    return q,unfinished
q,jobs=queue();(out/'queue.txt').write_text(q)
for job,script in [('215359.tc6000','attempt13_infer_score_retry4.pbs'),('215424.tc6000','attempt13_agri_guard_retry4.pbs')]:
    info=run('qstat','-f',job)
    assert re.search(r'Job_Owner\s*=\s*slfu@',info) and re.search(r'job_state\s*=\s*H',info),info
    (out/(job+'.txt')).write_text(info)
    shutil.copy2(src/script,out/script)
for job in ('215424.tc6000','215359.tc6000'):
    print('CANCEL_OWN_HELD_DOWNSTREAM',job,flush=True)
    subprocess.run(['qdel',job],check=True)
    q,jobs=queue();(out/(job+'_after.txt')).write_text(q)
q,jobs=queue();assert len(jobs)<3,q
job=run('qsub',str(root/'attempt13_ddp4.pbs')).strip()
q,jobs=queue();(out/'after_submit.txt').write_text(q)
r=dict(state='DDP4_PROBE_SUBMITTED_SOURCE_TRAINING_UNTOUCHED',job=job,source_job='215356.tc6000',
       cancelled_only_held_jobs=['215359.tc6000','215424.tc6000'],source_training_cancelled=False)
(root/'PROBE_SUBMISSION.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
