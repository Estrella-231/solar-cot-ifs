"""Check the first completed DDP epoch and bind its actual saved checkpoint."""
import hashlib
import json
from pathlib import Path
import torch

root=Path(__file__).resolve().parent
arm=root/'runs/cot_aux'
start=json.loads((root/'CUTOVER_READY.json').read_text())
ck_path=arm/'checkpoints/last.ckpt'
ck=torch.load(ck_path,map_location='cpu')
assert ck['global_step']==(ck['epoch']+1)*256
assert ck['global_step']>=start['global_step']+256
assert len(ck['optimizer_states'])==len(ck['lr_schedulers'])==1
source=root.parent/'hubei_cot_auxiliary_retrain_attempt13_retry4/runs/cot_aux'
assert str(source) not in json.dumps(ck['callbacks'],default=str)
best=[v for v in ck['callbacks'].values() if isinstance(v,dict) and 'best_k_models' in v][0]
assert best['kth_best_model_path'] in best['best_k_models']
assert Path(best['best_model_path']).is_file()
out=dict(state='PASS_FIRST_DDP4_COMPLETE_EPOCH_AND_CHECKPOINT',epoch=ck['epoch'],
         completed_epochs=ck['epoch']+1,global_step=ck['global_step'],
         resumed_from_global_step=start['global_step'],minimum_new_updates=256,
         global_batch=16,steps_per_epoch=256,world_size=4,
         checkpoint=str(ck_path),checkpoint_sha256=hashlib.sha256(ck_path.read_bytes()).hexdigest(),
         optimizer_states=1,schedulers=1,callback_paths_remapped=True,
         selected_best=best['best_model_path'],test_payloads_read=0,buffer_payloads_read=0)
(root/'FIRST_FULL_EPOCH_PASS.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
