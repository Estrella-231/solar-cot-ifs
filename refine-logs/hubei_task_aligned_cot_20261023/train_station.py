"""Fixed 600-update date-OOF screen. One independent station per assigned GPU."""
import argparse, copy, hashlib, json, time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import torch
from model import Correction, DIRECT_COLUMNS, response, losses, inference, load_npz

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--station',required=True)
    parser.add_argument('--gpu',required=True); args=parser.parse_args()
    torch.set_num_threads(1); torch.cuda.set_device(int(args.gpu)); device=torch.device('cuda',int(args.gpu))
    torch.backends.cuda.matmul.allow_tf32=False
    P=Path(__file__).resolve().parent; Q=P.parent/'hubei_cot_conditional_information_20261023'
    assert json.loads((P/'ENGINEERING_CHECK.json').read_text())['state']=='CPU_ENGINEERING_PASS'
    d=load_npz(Q/'inputs_attempt2'/f'{args.station}.npz'); dates=np.array([str(t)[:8] for t in d['target_time']])
    expected=json.loads((Q/'inputs_attempt2/COMPLETE.json').read_text())
    receipt=next(r for r in expected['stations'] if r['station']==args.station)
    assert hashlib.sha256((Q/'inputs_attempt2'/f'{args.station}.npz').read_bytes()).hexdigest()==receipt['output_sha256']
    mapping=json.loads((Q/'probe_attempt1/DATE_FOLDS.json').read_text())['date_to_fold']
    folds=np.array([mapping[t] for t in dates]); arms=['G1','physical','task','direct_AGRI']
    out=P/'training_attempt1'/args.station; out.mkdir(parents=True,exist_ok=False); records=[]; start=time.perf_counter()
    for sj,seed in enumerate([42,43,44]):
        base=d['bases'][sj]; x=np.column_stack([d['features'],base/d['clear']]); pred=np.full((4,len(base)),np.nan); pred[0]=base
        cot_pred=np.full((2,len(base),9),np.nan)
        for fold in range(3):
            teacher=load_npz(Q/'probe_attempt1'/args.station/f'seed{seed}_observed_R_OFFLINE_fold{fold}.npz')
            stats=load_npz(Q/'probe_attempt1'/args.station/f'seed{seed}_forecast_COT_fold{fold}.npz')
            train=np.isin(dates,teacher['train_dates'])&d['available']; test=folds==fold
            held=set(teacher['test_dates']); excluded={(datetime.strptime(str(day),'%Y%m%d')+timedelta(days=off)).strftime('%Y%m%d') for day in held for off in [-1,0,1]}
            assert set(dates[train]).isdisjoint(excluded) and test.sum()>0
            assert np.array_equal(stats['train_dates'],teacher['train_dates'])
            assert np.allclose(x[train].mean(0),stats['mean'],atol=1e-12,rtol=0)
            def tensor(a,dtype=torch.float32):return torch.as_tensor(a,dtype=dtype,device=device)
            norm=(x-stats['mean'])/stats['scale']
            # AGRI-only branch has its own train-only stats; no numeric COT columns.
            dx=x[:,DIRECT_COLUMNS]; dm=dx[train].mean(0); ds=dx[train].std(0); ds=np.where(ds>1e-8,ds,1.)
            xn=tensor(norm); xd=tensor((dx-dm)/ds); original=tensor(x[:,279:288]); reference=tensor(d['observed_COT_features'][:,:9])
            sens=tensor(response(teacher['coefficient'],teacher['scale'],x[:,270:279].astype(bool)))
            bt=tensor(base,torch.float64); yt=tensor(d['truth'],torch.float64); av=tensor(d['available'],torch.bool); ix=tensor(np.flatnonzero(train),torch.long)
            rng=np.random.default_rng(seed+1000*fold); order=rng.integers(0,len(ix),size=(600,128)); order_sha=hashlib.sha256(order.tobytes()).hexdigest()
            torch.manual_seed(seed+1000*fold); initial=Correction().to(device).state_dict()
            for ai,arm in enumerate(arms[1:],1):
                torch.manual_seed(seed+1000*fold); net=Correction(arm=='direct_AGRI').to(device)
                if arm!='direct_AGRI':net.load_state_dict(copy.deepcopy(initial))
                optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
                scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,600,eta_min=1e-5)
                curves=[]; begin=time.perf_counter()
                for step in range(600):
                    idx=ix[tensor(order[step],torch.long)]; optimizer.zero_grad(set_to_none=True)
                    if arm=='direct_AGRI':
                        loss=(net(xd[idx]).squeeze(-1)-(yt[idx]-bt[idx])/1000).square().mean(); lc=loss; lh=loss*0
                    else:
                        lo=torch.relu(original[idx]+net(xn[idx])); lc,lh=losses(lo,reference[idx],sens[idx]); loss=lc+(lh if arm=='task' else 0)
                    assert torch.isfinite(loss), (args.station,seed,fold,arm,step)
                    loss.backward(); grad=torch.nn.utils.clip_grad_norm_(net.parameters(),5); assert torch.isfinite(grad)
                    optimizer.step(); scheduler.step()
                    if step%100==0 or step==599:curves.append([step+1,float(loss),float(lc),float(lh)])
                net.eval()
                with torch.no_grad():
                    if arm=='direct_AGRI':y=torch.where(av,torch.clamp_min(bt+1000*net(xd).squeeze(-1),0),bt)
                    else:
                        log_out=torch.relu(original+net(xn)); y=inference(bt,av,sens,log_out,original)
                        cot_pred[ai-1,test]=torch.expm1(log_out).cpu().numpy()[test]
                    assert torch.isfinite(y).all(); pred[ai,test]=y.cpu().numpy()[test]
                state=dict(model=net.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),step=600,
                    arm=arm,station=args.station,seed=seed,fold=fold,mean=dm if arm=='direct_AGRI' else stats['mean'],
                    scale=ds if arm=='direct_AGRI' else stats['scale'],train_dates=teacher['train_dates'],test_dates=teacher['test_dates'],
                    sample_order_sha256=order_sha,curves=curves,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
                torch.save(state,out/f'seed{seed}_fold{fold}_{arm}_last.pt')
                # Exact same-device cold reloading, no tolerance widened from another precision.
                cold=Correction(arm=='direct_AGRI').to(device); cold.load_state_dict(torch.load(out/f'seed{seed}_fold{fold}_{arm}_last.pt',map_location=device)['model'])
                with torch.no_grad():
                    example=xd[:32] if arm=='direct_AGRI' else xn[:32]
                    replay=float((cold(example)-net(example)).abs().max()); assert replay==0
                records.append(dict(seed=seed,fold=fold,arm=arm,updates=600,rows=int(train.sum()),test_rows=int(test.sum()),
                    parameters=sum(p.numel() for p in net.parameters()),sample_order_sha256=order_sha,losses=curves,
                    cold_replay_max=replay,seconds=time.perf_counter()-begin))
                print(args.station,seed,fold,arm,'600 updates',flush=True)
        assert np.isfinite(pred).all() and np.isfinite(cot_pred).all()
        # Unavailable rows always return the original stored float64 G1 exactly.
        pred[:,~d['available']]=base[~d['available']]
        np.savez_compressed(out/f'seed{seed}_OOF.npz',prediction=pred,arms=arms,truth=d['truth'],date=dates,
            init_time=d['init_time'],target_time=d['target_time'],lead=d['lead'],available=d['available'],
            cot_prediction=cot_pred,cot_raw=np.expm1(x[:,279:288]),cot_reference=np.expm1(d['observed_COT_features'][:,:9]),
            source_role=d['source_role'],fold=folds)
    (out/'COMPLETE.json').write_text(json.dumps(dict(state='FIXED_600_UPDATE_OOF_COMPLETE',station=args.station,
        records=records,seconds=time.perf_counter()-start,gpu=torch.cuda.get_device_name(device),C_select_scored_rows=0,test_payloads_read=0),indent=2))

if __name__=='__main__':main()
