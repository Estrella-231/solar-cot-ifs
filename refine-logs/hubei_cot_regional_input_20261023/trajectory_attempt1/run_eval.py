"""Six frozen models in independent GPU waves after arm selection is fixed."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--gpus',required=True);a=ap.parse_args()
gpus=a.gpus.split(',');assert len(gpus)==4
selection=json.loads((a.root/'fits_attempt1/ARM_SELECTION.json').read_text());assert len(selection['runs'])==6
subprocess.run([sys.executable,str(a.root/'build_eval_pack.py')],check=True)
pack=json.loads((a.root/'EVAL_PACK_COMPLETE.json').read_text())['pack']
out=a.root/'eval_attempt1';out.mkdir(exist_ok=False)
for start in range(0,6,4):
    tasks=[]
    for i,spec in enumerate(selection['runs'][start:start+4]):
        name=f"seed{spec['seed']}_{spec['arm']}";log=(out/f'{name}.log').open('w')
        cmd=[sys.executable,a.root/'infer_eval.py','--run',spec['run'],'--pack',pack,'--out',out/name]
        tasks.append((subprocess.Popen(list(map(str,cmd)),env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus[i]),stdout=log,stderr=subprocess.STDOUT),log,name))
    failures=[]
    for p,log,name in tasks:
        rc=p.wait();log.close()
        if rc:failures.append(dict(name=name,exit=rc))
    assert not failures,failures
(out/'COMPLETE.json').write_text(json.dumps(dict(state='SIX_FROZEN_TRAJECTORY_VAL_PREDICTIONS_COMPLETE',
    arm_selection_frozen_before_val=True,selected_arm=selection['selected_arm'],test_payloads_read=0,GHI_labels_read=0),indent=2))
print('TRAJECTORY_VAL_INFERENCE_COMPLETE',flush=True)
