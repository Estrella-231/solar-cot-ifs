"""Prepare an isolated full-state resume; never modify the live source run."""
import argparse
import csv
import json
import pathlib
import shutil
import torch
from ddp_runtime import digest


def remap(x, old, new):
    if isinstance(x, str): return x.replace(old, new)
    if isinstance(x, dict): return {remap(k, old, new): remap(v, old, new) for k,v in x.items()}
    if isinstance(x, list): return [remap(v, old, new) for v in x]
    if isinstance(x, tuple): return tuple(remap(v, old, new) for v in x)
    return x


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=pathlib.Path,required=True)
    p.add_argument('--backup',type=pathlib.Path,required=True)
    p.add_argument('--mode',choices=['probe','full'],required=True)
    a=p.parse_args()
    origin=json.loads((a.backup/'BACKUP_RECEIPT.json').read_text())
    src=pathlib.Path(origin['source_attempt'])
    original_arm=src/'runs/cot_aux'
    arm=a.root/('probe_arm' if a.mode=='probe' else 'runs/cot_aux')
    arm.mkdir(parents=True,exist_ok=False)
    (arm/'checkpoints').mkdir()
    ck=torch.load(a.backup/'runs/cot_aux/checkpoints/last.ckpt',map_location='cpu')
    assert ck['global_step']==(ck['epoch']+1)*256, 'Require a completed-epoch checkpoint'
    expected={'model':digest(ck['state_dict']), 'optimizer':digest(ck['optimizer_states'][0]),
              'scheduler':digest(ck['lr_schedulers'][0]),'global_step':ck['global_step'],'epoch':ck['epoch']}
    # Preserve the previously selected best; remap every callback path away from the source.
    for state in ck.get('callbacks',{}).values():
        if isinstance(state,dict):
            for key in ('best_model_path','kth_best_model_path'):
                name=state.get(key)
                if name:
                    fp=pathlib.Path(name)
                    backed=a.backup/'runs/cot_aux/checkpoints'/fp.name
                    candidate=backed if backed.exists() else fp
                    if not candidate.exists(): raise FileNotFoundError(candidate)
                    shutil.copy2(candidate,arm/'checkpoints'/fp.name)
    ck['callbacks']=remap(ck['callbacks'],str(original_arm),str(arm))
    for state in ck['callbacks'].values():
        if isinstance(state,dict) and state.get('best_k_models'):
            assert state['kth_best_model_path'] in state['best_k_models']
    assert digest(ck['state_dict'])==expected['model']
    assert digest(ck['optimizer_states'][0])==expected['optimizer']
    assert digest(ck['lr_schedulers'][0])==expected['scheduler']
    torch.save(ck,arm/'resume.ckpt')
    shutil.copy2(original_arm/'attempt13_init_receipt.json',arm/'attempt13_init_receipt.json')
    shutil.copy2(original_arm/'config.py',arm/'source_config.py')
    # Exact serial validation value at this saved epoch is the real-model reduction witness.
    scores=[]
    for fp in (a.backup/'runs/cot_aux/cwd').glob('lightning_logs/version_*/metrics.csv'):
        with fp.open() as f:
            for row in csv.DictReader(f):
                if row.get('epoch') and int(float(row['epoch']))==ck['epoch'] and row.get('val_cot_rmse'):
                    scores.append(float(row['val_cot_rmse']))
    if not scores:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        for fp in (a.backup/'runs/cot_aux/cwd').glob('lightning_logs/version_*/events.out.tfevents.*'):
            event=EventAccumulator(str(fp),size_guidance={'scalars':0});event.Reload()
            if 'val_cot_rmse' in event.Tags()['scalars']:
                scores += [v.value for v in event.Scalars('val_cot_rmse') if v.step==ck['global_step']-1]
    if not scores: raise RuntimeError('No exact serial validation metric for saved epoch')
    expected['serial_val_cot_rmse']=scores[-1]
    expected['source_attempt']=str(src)
    expected['backup']=str(a.backup)
    expected['source_job_untouched']=215356
    expected['global_batch']=16
    expected['local_batch_regular']=4
    expected['steps_per_epoch']=256
    expected['target_updates']=12800
    (arm/'RESUME_INPUT.json').write_text(json.dumps(expected,indent=2))
    print(json.dumps(expected,indent=2),flush=True)


if __name__=='__main__': main()
