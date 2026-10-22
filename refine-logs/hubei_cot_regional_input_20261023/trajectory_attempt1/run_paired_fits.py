"""Six seed-matched fits; freeze arm choice on internal training holdout only."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--world',type=int,required=True);a=ap.parse_args()
decision=json.loads((a.root/'PROFILE_DECISION_attempt1.json').read_text())
assert decision['state']=='PASS_TRAJECTORY_2_4_GPU_CORRECTNESS_AND_SPEED' and decision['selected_world']==a.world
pack=json.loads((a.root/'PAIR_PACK_COMPLETE.json').read_text())['pack']
out=a.root/'fits_attempt1';out.mkdir(exist_ok=False);finished=[]
for seed in [42,43,44]:
    for arm in ['full','repeat_last']:
        destination=out/f'seed{seed}_{arm}'
        with (out/f'seed{seed}_{arm}.log').open('w') as log:
            cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',f'--nproc_per_node={a.world}',
                 a.root/'train_trajectory.py','--pack',pack,'--out',destination,'--arm',arm,'--seed',seed]
            subprocess.run(list(map(str,cmd)),stdout=log,stderr=subprocess.STDOUT,check=True)
        receipt=json.loads((destination/'COMPLETE.json').read_text());assert receipt['state']=='TRAJECTORY_FIT_INTERNAL_SELECTION_COMPLETE_NO_OUTER_VAL'
        best=destination/'best.pt';receipt['best_sha256']=hashlib.sha256(best.read_bytes()).hexdigest()
        receipt['run']=str(destination);finished.append(receipt)
        (out/'PROGRESS.json').write_text(json.dumps(dict(completed=finished,total_fits=6),indent=2))
means={arm:sum(r['best_select_mse_normalized'] for r in finished if r['arm']==arm)/3 for arm in ['full','repeat_last']}
selected='full' if means['full']<means['repeat_last']-1e-12 else 'repeat_last'
result=dict(state='SIX_FITS_COMPLETE_ARM_FROZEN_BEFORE_OUTER_VAL',selected_arm=selected,internal_mse_mean=means,
    rule='lower three-seed mean internal physical pixel MSE; tie within1e-12 chooses repeat_last',
    runs=finished,test_payloads_read=0,outer_validation_payloads_read=0,GHI_labels_read=0)
(out/'ARM_SELECTION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
