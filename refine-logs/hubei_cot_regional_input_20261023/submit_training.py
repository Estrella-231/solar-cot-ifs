"""Launch the frozen regional-R fit after completed GPU correctness/profile."""
import argparse, hashlib, json, subprocess
from pathlib import Path
from submit_profile import queue


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    root=a.root;receipt=root/'TRAIN_SUBMISSION_attempt1.json';intent=root/'TRAIN_SUBMISSION_INTENT_attempt1.json'
    assert not receipt.exists() and not intent.exists(), 'Inspect previous submission before any retry'
    profile=json.loads((root/'PROFILE_DECISION_attempt1.json').read_text())
    assert profile['state']=='PROFILE_COMPLETE_NO_TRAINING'
    world=profile['selected_world'];assert world in (2,4)
    previous=json.loads((root/'PROFILE_SUBMISSION_attempt1.json').read_text())
    node=previous['node'];assert node=='node23'
    terminal=subprocess.check_output(['qstat','-f',previous['job_id']],text=True)
    assert 'job_state = C' in terminal and 'exit_status = 0' in terminal, terminal
    before=queue();assert len(before)<3 and sum(j['requested_gpus'] for j in before)+world<=8,before
    physical=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',node,
      'nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits'],text=True)
    free=sum(int(line.split(',')[-2])<500 and int(line.split(',')[-1])<5 for line in physical.strip().splitlines())
    assert free>=world,physical
    template=(root/'profile_node23_attempt1.pbs').read_text()
    header=template.split('for N in 2 4; do')[0]
    header=header.replace('rregionprof','rregionfit').replace('walltime=00:40:00','walltime=02:00:00')
    header=header.replace('profile_attempt1.log','train_attempt1.log')
    header=header.replace('gpus=4',f'gpus={world}').replace('-eq 4',f'-eq {world}')
    body=f'''python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node={world} "$BASE/train_regional_r.py" --pack "$PACK" --roles "$BASE/R_INTERNAL_ROLES.json" --out "$BASE/r_region_seed42_attempt1"
python "$BASE/score_regional_r.py" --run "$BASE/r_region_seed42_attempt1" --pack "$PACK" --roles "$BASE/R_INTERNAL_ROLES.json" --cpp-ready "$BASE/source_ready_attempt1" --out "$BASE/r_region_seed42_attempt1_scores"
'''
    script=root/'train_node23_attempt1.pbs';script.write_text(header+body)
    subprocess.run(['bash','-n',str(script)],check=True)
    provenance={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in
                ('regional_model.py','train_regional_r.py','score_regional_r.py','R_INTERNAL_ROLES.json','train_node23_attempt1.pbs')}
    record=dict(queue_before=before,physical_before=physical,world=world,node=node,source_sha256=provenance,
                profile_job=previous['job_id'],profile_terminal=terminal,purpose='random-init regional COT retrieval; not GHI benefit')
    intent.write_text(json.dumps(record,indent=2))
    job=subprocess.check_output(['qsub',str(script)],text=True).strip()
    record.update(job_id=job,queue_after=queue(),global_batch=128,max_updates=750,test_payloads_read=0)
    receipt.write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)


if __name__=='__main__':main()
