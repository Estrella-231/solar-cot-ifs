import argparse,copy,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
P=Path(__file__).resolve().parent;H=P.parent/'hubei_forecast_cot_increment_20261022';D=P.parent/'hubei_neural_cot_diagnosis_20261023'
sys.path.insert(0,str(H/'code_attempt1'))
from train import ConditionalHead
from history_base import HistoryCOTGHIHead,sha256,array_sha256
from regularization import response_loss,engineering_checks
G=P.parent/'hubei_cot_calibration_coverage_20261023'
ap=argparse.ArgumentParser();ap.add_argument('--station',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
assert not a.out.exists();a.out.mkdir()
p=H/'prepared_attempt1'/a.station;meta=json.loads((p/'meta.json').read_text());rows=list(csv.DictReader((p/'rows.csv').open()));roles=np.load(p/'roles.npy')
audit=json.loads((G/'METADATA_AUDIT.json').read_text())[a.station];assert sha256(p/'rows.csv')==audit['row_metadata_sha256']
wide_dates=set(audit['wide_dates']);take=np.array([i for i,(r,v) in enumerate(zip(rows,roles)) if (r['split']=='train' and r['target_time_bjt'][:8] in wide_dates) or v=='C_select'])
rr=[rows[i] for i in take];role=roles[take];dates=np.array([r['target_time_bjt'][:8] for r in rr]);small=np.flatnonzero(role=='C_fit');large=np.flatnonzero(np.isin(dates,list(wide_dates)));select=np.flatnonzero(role=='C_select')
assert len(small)==audit['small_rows'] and len(large)==audit['wide_rows'] and len(select)==audit['select_rows']
assert set(dates[large]).isdisjoint(set(dates[select])) and all(rr[i]['split']=='train' for i in large)
data={k:np.load(p/(k+'.npy'),mmap_mode='r')[take].copy() for k in ['future','history','cot','fusion_features','ghi','clear']}
assert all(np.isfinite(v).all() for v in data.values())
ff=data['fusion_features'];solar=(data['future'][:,13].reshape(-1,9)>np.cos(np.deg2rad(80.)))&ff[:,297:306].astype(bool)
for lo in [279,288]:ff[:,lo:lo+9]=np.where(solar,ff[:,lo:lo+9],0)
ff[:,297:306]=solar.astype('float32');available=torch.from_numpy(solar.all(1).astype('float32')).cuda()
truth=data['ghi'].astype('float64');clear=data['clear'].astype('float64');ttruth=torch.from_numpy(truth).cuda();tclear=torch.from_numpy(clear).cuda()
tensors={k:torch.from_numpy(data[k]).cuda() for k in ['future','history','cot','fusion_features']}
arms=['large_no_future_COT','large_future_COT','large_regularized_future_COT'];receipts=[]
for seed in [42,43,44]:
    ck=H/'training_attempt1/pairs'/f'{a.station}_seed{seed}'/'G1_best.pt';base_model=HistoryCOTGHIHead().cuda().eval();base_model.load_state_dict(torch.load(ck,map_location='cuda',weights_only=False)['model'])
    with torch.inference_mode():base=torch.cat([base_model(tensors['future'][j:j+1536],tensors['history'][j:j+1536],tensors['cot'][j:j+1536]) for j in range(0,len(take),1536)])
    base_np=base.cpu().numpy().astype('float64')*clear;del base_model
    source=D/'replay'/a.station/f'seed{seed}_replay.npz'
    with np.load(source) as old:
        keys=list(zip(old['init_time'],old['target_time'],old['lead']));lookup={key:i for i,key in enumerate(keys)}
        current=[(r['init_time_bjt'],r['target_time_bjt'],int(r['lead_minutes'])) for r in rr]
        ii=[i for i,k in enumerate(current) if k in lookup];jj=[lookup[current[i]] for i in ii]
        assert len(ii)==len(old['truth']) and np.array_equal(truth[ii],old['truth'][jj])
        replay_difference=float(abs(base_np[ii]-old['prediction'][0,jj]).max());assert replay_difference<=.001,replay_difference
    features=torch.cat([tensors['fusion_features'],base[:,None]],1).detach();zero=features.clone();zero[:,279:297]=0
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);template=ConditionalHead(308).cuda();initial=copy.deepcopy(template.state_dict());ih=array_sha256(initial)
    def stream(ids):
        rng=np.random.default_rng(seed);chunks=[];remaining=60*1536
        while remaining:
            order=rng.permutation(ids);v=order[:min(remaining,len(order))];chunks.append(v);remaining-=len(v)
        return np.concatenate(chunks).reshape(60,1536)
    orders={'large':stream(large)};oh={k:hashlib.sha256(v.astype('int64').tobytes()).hexdigest() for k,v in orders.items()}
    predictions=[base_np[select]];run=[]
    hv=torch.as_tensor(ff[:,270:279],device='cuda');sd=meta['future_cot_norm']['log1p_std']
    check=engineering_checks(template,features,base,available,hv,sd)
    (a.out/f'seed{seed}_ENGINEERING.json').write_text(json.dumps(check,indent=2))
    for arm in arms:
        scope=arm.split('_')[0];model=ConditionalHead(308).cuda();model.load_state_dict(initial);assert array_sha256(model.state_dict())==ih
        features_arm=zero if 'no_future' in arm else features
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60,eta_min=1e-5)
        def predict():
            model.eval()
            with torch.no_grad():return torch.cat([model(features_arm[ii],base[ii],available[ii]).double()*tclear[ii] for ii in [torch.as_tensor(select[j:j+1536],device='cuda') for j in range(0,len(select),1536)]]).cpu().numpy()
        q=predict();assert np.array_equal(q,base_np[select]);best=float(np.sqrt(np.mean((q-truth[select])**2)));best_step=0;best_state=copy.deepcopy(initial);logs=[dict(step=0,select_rmse=best)]
        for step,order in enumerate(orders[scope],1):
            model.train();ii=torch.as_tensor(order,device='cuda');omega=torch.zeros((),device='cuda')
            if 'regularized' in arm:
                x=features_arm[ii].detach().requires_grad_(True)
                q,omega,direction=response_loss(model,x,base[ii],available[ii],hv[ii],sd)
            else:q=model(features_arm[ii],base[ii],available[ii])
            primary=(((q.double()*tclear[ii]-ttruth[ii])/1000)**2).mean();loss=primary+omega;assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step();sch.step()
            q=predict();score=float(np.sqrt(np.mean((q-truth[select])**2)));logs.append(dict(step=step,select_rmse=score,fit_objective=float(primary.detach()),response_penalty=float(omega.detach())))
            if score<best:best=score;best_step=step;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sch.state_dict(),step=60,updates=60,initialization_sha256=ih,order_sha256=oh[scope]),a.out/f'seed{seed}_{arm}_last.pt')
        torch.save(dict(model=best_state,step=best_step,seed=seed,station=a.station,base_sha256=sha256(ck),initialization_sha256=ih,order_sha256=oh[scope]),a.out/f'seed{seed}_{arm}_best.pt')
        model.load_state_dict(best_state);q=predict();assert abs(float(np.sqrt(np.mean((q-truth[select])**2)))-best)<1e-8;predictions.append(q)
        if 'regularized' not in arm:
            old_logs=json.loads((G/'replay'/a.station/f'seed{seed}_{arm}_steps.json').read_text())
            score_diff=max(abs(x['select_rmse']-y['select_rmse']) for x,y in zip(logs,old_logs))
            loss_diff=max(abs(x['fit_objective']-y['fit_objective']) for x,y in zip(logs[1:],old_logs[1:]))
            assert score_diff<=.001 and loss_diff<=1e-8,(score_diff,loss_diff)
            with np.load(G/'replay'/a.station/f'seed{seed}_Cselect.npz') as old:
                old_index=3 if 'no_future' in arm else 4
                prediction_diff=float(abs(q-old['prediction'][old_index]).max());assert prediction_diff<=.001,prediction_diff
            (a.out/f'seed{seed}_{arm}_CONTROL_REPLAY.json').write_text(json.dumps(dict(score_curve_max=score_diff,loss_curve_max=loss_diff,prediction_max=prediction_diff)))
        run.append(dict(arm=arm,best_step=best_step,select_rmse=best,updates=60,sampled_examples=60*1536,unique_sampled_rows=len(set(orders[scope].ravel().tolist())),initialization_sha256=ih,order_sha256=oh[scope],parameters=sum(v.numel() for v in model.parameters()),best_sha256=sha256(a.out/f'seed{seed}_{arm}_best.pt')))
        (a.out/f'seed{seed}_{arm}_steps.json').write_text(json.dumps(logs,indent=2));print(a.station,seed,arm,best_step,best,flush=True)
    np.savez_compressed(a.out/f'seed{seed}_Cselect.npz',prediction=np.array(predictions),truth=truth[select],clear=clear[select],date=dates[select],arms=['G1']+arms,
        init_time=np.array([rr[i]['init_time_bjt'] for i in select]),target_time=np.array([rr[i]['target_time_bjt'] for i in select]),lead=np.array([int(rr[i]['lead_minutes']) for i in select]))
    receipts.append(dict(seed=seed,base_sha256=sha256(ck),old_C_prediction_max_difference=replay_difference,arms=run))
(a.out/'COMPLETE.json').write_text(json.dumps(dict(state='RESPONSE_REGULARIZATION_THREE_ARM_COMPLETE',receipts=receipts,metadata_audit_sha256=sha256(G/'METADATA_AUDIT.json'),small_rows=len(small),large_rows=len(large),select_rows=len(select),upstream_training=False,official_val_new_period_outer_test_buffer_payloads_read=0),indent=2))
