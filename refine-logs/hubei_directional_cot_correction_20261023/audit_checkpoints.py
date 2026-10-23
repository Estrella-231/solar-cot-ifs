"""CPU cold reload of all selected predictions and actual optimizer update counters."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from train_corrector import Corrector
P=Path(__file__).resolve().parent;root=Path(json.loads((P/'PACK_COMPLETE.json').read_text())['path']);torch.set_num_threads(1)
select=np.flatnonzero(np.load(root/'select.npy'));x=np.array(np.load(root/'x.npy',mmap_mode='r')[select]);base=np.array(np.load(root/'base.npy',mmap_mode='r')[select]);available=np.array(np.load(root/'forecast_mask.npy',mmap_mode='r')[select]);records=[];orders=[];inits=[];direction=np.array(np.load(json.loads((P/'GATE_COMPLETE.json').read_text())['path'],mmap_mode='r')[select])
for seed in [42,43,44]:
    out=P/'training_attempt1'/f'seed{seed}';done=json.loads((out/'COMPLETE.json').read_text());meta=json.loads((out/'run_metadata.json').read_text());curve=json.loads((out/'curve.json').read_text())
    assert [r['step'] for r in curve]==list(range(0,601,50));assert min(curve,key=lambda r:r['select_equal_station_lead_rmse'])['step']==done['best_step']
    orders.append(meta['schedule_sha256']);inits.append(meta['initialization_sha256']);assert meta['source_sha256']==hashlib.sha256((P/'train_corrector.py').read_bytes()).hexdigest()
    last=torch.load(out/'last.pt',map_location='cpu',weights_only=False);best=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    steps={int(v['step']) for v in last['optimizer']['state'].values()};assert steps=={600} and last['step']==600 and last['scheduler']['last_epoch']==600
    assert last['optimizer']['param_groups'][0]['lr']==0. and best['step']==done['best_step']
    assert hashlib.sha256((out/'best.pt').read_bytes()).hexdigest()==done['best_checkpoint_sha256']
    model=Corrector().eval();model.load_state_dict(best['model']);pred=[]
    with torch.inference_mode():
        for j in range(0,len(select),128):pred.append(model(torch.from_numpy(x[j:j+128]),torch.from_numpy(base[j:j+128]),torch.from_numpy(available[j:j+128]),torch.from_numpy(direction[j:j+128])).numpy())
    with np.load(out/'best_select_predictions.npz') as saved:
        stable=(direction==0)&available;assert np.array_equal(saved['prediction'][stable],base[stable])
        assert (saved['prediction'][(direction>0)&available]>=base[(direction>0)&available]-1e-4).all()
        assert (saved['prediction'][(direction<0)&available]<=base[(direction<0)&available]+1e-4).all()
        assert np.array_equal(saved['field_index'],select);err=float(np.max(abs(np.concatenate(pred)-saved['prediction'])))
    assert err<1e-3,err
    records.append(dict(seed=seed,actual_optimizer_steps=sorted(steps),scheduler_last_epoch=600,best_step=best['step'],CPU_cold_replay_max_abs_COT=err))
assert len(set(orders))==1 and len(set(inits))==3
report=dict(state='PASS_COLD_CHECKPOINT_UPDATES_ORDER_AND_PREDICTIONS',seeds=records,same_sampling_order=True,distinct_initializations=True,cold_replay_tolerance_COT=1e-3,outer_payloads_read=0)
(P/'CHECKPOINT_AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
