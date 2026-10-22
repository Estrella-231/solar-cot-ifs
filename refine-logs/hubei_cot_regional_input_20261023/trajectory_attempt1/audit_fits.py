"""Read six terminal states and full checkpoints before opening new val inference."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
selection=json.loads((P/'fits_attempt1/ARM_SELECTION.json').read_text())
assert selection['state']=='SIX_FITS_COMPLETE_ARM_FROZEN_BEFORE_OUTER_VAL'
assert len(selection['runs'])==6
checks=[]; bindings={};step0={}
for spec in selection['runs']:
    path=Path(spec['run']);complete=json.loads((path/'COMPLETE.json').read_text())
    assert complete['state']=='TRAJECTORY_FIT_INTERNAL_SELECTION_COMPLETE_NO_OUTER_VAL'
    assert all(complete[k]==0 for k in ['outer_validation_payloads_read','test_payloads_read','GHI_labels_read'])
    assert sha(path/'best.pt')==spec['best_sha256']
    best=torch.load(path/'best.pt',map_location='cpu');last=torch.load(path/'last.pt',map_location='cpu')
    binding=best['binding'];assert binding==last['binding']
    assert binding['model_source_sha256']==sha(P/'trajectory_model.py')
    assert binding['trainer_source_sha256']==sha(P/'train_trajectory.py')
    assert binding['arm']==spec['arm'] and binding['seed']==spec['seed']
    assert binding['global_batch']==128 and binding['max_updates']==2000
    assert best['epoch']==complete['best_epoch'] and last['global_step']==complete['global_step']
    assert last['scheduler']['last_epoch']==complete['global_step']<=2000
    assert len(last['optimizer']['state'])>0 and 'scale' in last['scaler']
    epochs=[json.loads(p.read_text()) for p in sorted(path.glob('epoch_*.json'))]
    assert len(epochs)==complete['final_epoch']+1
    assert all(np.isfinite(e['train_mse_normalized']) and np.isfinite(e['select_mse_normalized']) for e in epochs)
    assert complete['best_select_mse_normalized']==min([json.loads((path/'STEP0.json').read_text())['select_mse_normalized']]+[e['select_mse_normalized'] for e in epochs])
    for k,v in best['model'].items():assert torch.isfinite(v).all(),k
    bindings[spec['seed'],spec['arm']]=binding
    step0[spec['seed'],spec['arm']]=json.loads((path/'STEP0.json').read_text())['select_mse_normalized']
    checks.append(dict(seed=spec['seed'],arm=spec['arm'],best_epoch=best['epoch'],best_step=best['global_step'],
        final_step=last['global_step'],best_select_mse=complete['best_select_mse_normalized'],best_sha256=spec['best_sha256']))
for seed in [42,43,44]:
    left=dict(bindings[seed,'full']);right=dict(bindings[seed,'repeat_last']);left.pop('arm');right.pop('arm')
    assert left==right
    assert step0[seed,'full']==step0[seed,'repeat_last']
means={arm:float(np.mean([c['best_select_mse'] for c in checks if c['arm']==arm])) for arm in ['full','repeat_last']}
assert means==selection['internal_mse_mean']
chosen='full' if means['full']<means['repeat_last']-1e-12 else 'repeat_last'
assert chosen==selection['selected_arm']
report=dict(state='PASS_SIX_FROZEN_FITS_BUDGET_BINDING_AND_INTERNAL_SELECTION',checks=checks,
    selected_arm=chosen,internal_mse_mean=means,arm_selection_sha256=sha(P/'fits_attempt1/ARM_SELECTION.json'),
    actual_best_and_last_checkpoints_read=True,paired_step0_exact=True,paired_training_binding_equal=True,
    outer_validation_payloads_read=0,test_payloads_read=0,GHI_labels_read=0)
(P/'FITS_FINAL_AUDIT_attempt1.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
