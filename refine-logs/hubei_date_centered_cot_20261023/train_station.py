import argparse,copy,csv,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
P=Path(__file__).resolve().parent;H=P.parent/'hubei_forecast_cot_increment_20261022';D=P.parent/'hubei_neural_cot_diagnosis_20261023'
sys.path.insert(0,str(H/'code_attempt1'))
from train import ConditionalHead
from history_base import sha256,array_sha256
ap=argparse.ArgumentParser();ap.add_argument('--station',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
assert not a.out.exists();a.out.mkdir()
p=H/'prepared_attempt1'/a.station;roles=np.load(p/'roles.npy');rows=list(csv.DictReader((p/'rows.csv').open()));take=np.flatnonzero(np.isin(roles,['C_fit','C_select']));rr=[rows[i] for i in take];role=roles[take]
ff=np.load(p/'fusion_features.npy',mmap_mode='r')[take].copy();future=np.load(p/'future.npy',mmap_mode='r')[take]
solar=(future[:,13].reshape(-1,9)>np.cos(np.deg2rad(80.)))&ff[:,297:306].astype(bool)
for lo in [279,288]:ff[:,lo:lo+9]=np.where(solar,ff[:,lo:lo+9],0)
ff[:,297:306]=solar.astype('float32');available=torch.from_numpy(solar.all(1).astype('float32')).cuda()
truth=np.load(p/'ghi.npy',mmap_mode='r')[take].astype('float64');clear=np.load(p/'clear.npy',mmap_mode='r')[take].astype('float64')
dates=np.array([r['target_time_bjt'][:8] for r in rr]);fit=np.flatnonzero(role=='C_fit');select=np.flatnonzero(role=='C_select')
assert set(dates[fit]).isdisjoint(set(dates[select]))
ttruth=torch.from_numpy(truth).cuda();tclear=torch.from_numpy(clear).cuda();tff=torch.from_numpy(ff).cuda();arms=['raw_no_future_COT','raw_future_COT','center_no_future_COT','center_future_COT'];receipts=[]
for seed in [42,43,44]:
    source=D/'replay'/a.station/f'seed{seed}_replay.npz'
    with np.load(source) as z:
        assert np.array_equal(z['truth'],truth) and np.array_equal(z['roles'],role)
        assert np.array_equal(z['target_time'],np.array([r['target_time_bjt'] for r in rr]))
        base_ghi=z['prediction'][0].astype('float64')
    # Original normalized predictions were float32; undo final physical multiplication.
    base_np=(base_ghi/clear).astype('float32');assert np.max(abs(base_np.astype('float64')*clear-base_ghi))<1e-8
    base=torch.from_numpy(base_np).cuda();features=torch.cat([tff,base[:,None]],1);zero=features.clone();zero[:,279:297]=0
    centered=truth.copy();offsets={}
    for day in sorted(set(dates[fit])):
        m=(role=='C_fit')&(dates==day);offset=float(np.mean(truth[m]-base_ghi[m]));centered[m]-=offset;offsets[day]=offset
    target_center=torch.from_numpy(centered).cuda();assert np.array_equal(centered[select],truth[select])
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);template=ConditionalHead(308).cuda();initial=copy.deepcopy(template.state_dict());ih=array_sha256(initial)
    rng=np.random.default_rng(seed);orders=[rng.permutation(fit) for _ in range(30)];oh=hashlib.sha256(np.concatenate(orders).astype('int64').tobytes()).hexdigest()
    results=[base_ghi[select]];run=[]
    for arm in arms:
        model=ConditionalHead(308).cuda();model.load_state_dict(initial);assert array_sha256(model.state_dict())==ih
        features_arm=zero if 'no_future' in arm else features;target=target_center if arm.startswith('center') else ttruth
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=30,eta_min=1e-5)
        def predict(ids):
            model.eval()
            with torch.no_grad():return torch.cat([model(features_arm[ii],base[ii],available[ii]).double()*tclear[ii] for ii in [torch.as_tensor(ids[j:j+1536],device='cuda') for j in range(0,len(ids),1536)]]).cpu().numpy()
        initial_pred=predict(select);assert np.array_equal(initial_pred,base_ghi[select])
        best=float(np.sqrt(np.mean((initial_pred-truth[select])**2)));best_epoch=0;best_state=copy.deepcopy(initial);updates=0;logs=[dict(epoch=0,select_rmse=best,updates=0)]
        for epoch,order in enumerate(orders,1):
            model.train();total=0.;seen=0
            for j in range(0,len(order),1536):
                ii=torch.as_tensor(order[j:j+1536],device='cuda');q=model(features_arm[ii],base[ii],available[ii])
                loss=(((q.double()*tclear[ii]-target[ii])/1000)**2).mean();assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();updates+=1;total+=float(loss.detach())*len(ii);seen+=len(ii)
            scheduler.step();q=predict(select);score=float(np.sqrt(np.mean((q-truth[select])**2)))
            logs.append(dict(epoch=epoch,select_rmse=score,fit_objective=total/seen,updates=updates))
            if score<best:best=score;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        assert updates==30*((len(fit)+1535)//1536)
        checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=30,updates=updates,initialization_sha256=ih,order_sha256=oh)
        torch.save(checkpoint,a.out/f'seed{seed}_{arm}_last.pt')
        torch.save(dict(model=best_state,epoch=best_epoch,seed=seed,station=a.station,initialization_sha256=ih,order_sha256=oh),a.out/f'seed{seed}_{arm}_best.pt')
        model.load_state_dict(best_state);q=predict(select);assert abs(float(np.sqrt(np.mean((q-truth[select])**2)))-best)<1e-8
        results.append(q);run.append(dict(arm=arm,best_epoch=best_epoch,select_rmse=best,updates=updates,initialization_sha256=ih,order_sha256=oh,parameters=sum(v.numel() for v in model.parameters()),best_sha256=sha256(a.out/f'seed{seed}_{arm}_best.pt')))
        (a.out/f'seed{seed}_{arm}_epochs.json').write_text(json.dumps(logs,indent=2));print(a.station,seed,arm,best_epoch,best,flush=True)
    np.savez_compressed(a.out/f'seed{seed}_Cselect.npz',prediction=np.array(results),truth=truth[select],clear=clear[select],date=dates[select],arms=['G1']+arms,
        init_time=np.array([rr[i]['init_time_bjt'] for i in select]),target_time=np.array([rr[i]['target_time_bjt'] for i in select]),lead=np.array([int(rr[i]['lead_minutes']) for i in select]))
    receipts.append(dict(seed=seed,arms=run,training_date_offsets=offsets,negative_centered_fit_targets=int((centered[fit]<0).sum()),source_prediction_sha256=sha256(source)))
(a.out/'COMPLETE.json').write_text(json.dumps(dict(state='DATE_CENTERED_FOUR_ARM_CSELECT_COMPLETE',receipts=receipts,fit_rows=len(fit),select_rows=len(select),new_upstream_training=False,official_val_new_period_test_buffer_payloads_read=0),indent=2))
