"""Read-only error/loss decomposition; no training or test payload."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def metrics(e):
    bias=float(e.mean()); mse=float(np.mean(e*e))
    return dict(n=len(e),rmse=float(np.sqrt(mse)),mae=float(np.mean(abs(e))),bias=bias,centered_mse=float(np.mean((e-bias)**2)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    root=a.root;out=root/'audits/cot_failure_causes_20260908_v1.json'
    assert not out.exists()
    previous=root/'audits/head_paired_pilot_20260908.json'
    audit=json.loads(previous.read_text()); pack=root/'data/head_pack_trainval_20260908_v1'
    pa=json.loads((pack/'audit.json').read_text())
    hashes={v['name']:v['sha256'] for v in pa['arrays'].values()}
    arrays={}
    for k in ['kt','ghi','clear','station','lead','split','weight']:
        p=pack/(k+'.npy');assert sha(p)==hashes[p.name]
        arrays[k]=np.load(p,allow_pickle=False)
    assert np.max(abs(arrays['kt']*arrays['clear']-arrays['ghi']))<1e-8
    result=dict(state='COMPLETE_DESCRIPTIVE_CAUSE_DECOMPOSITION',test_used=False,training=False,
                protocol_sha256=sha(root/'docs/COT_FAILURE_CAUSE_PROTOCOL_20260908.md'),script_sha256=sha(Path(__file__)),
                source_prediction_audit_sha256=sha(previous),small_array_hashes={k:hashes[k+'.npy'] for k in arrays},
                candidates=[],paired={},zero_prediction_reference_loss={},limits=[
                    'Validation selected single-seed models; no formal inference or causal attribution.',
                    'Zero-prediction loss mass is not actual trained-model loss or gradient contribution.',
                    'Train loss logs are changing-model minibatch averages, not fixed-checkpoint train evaluation.'])
    edges=[0,20,100,300,600,float('inf')]
    loaded={}
    for cohort,groups in audit['candidates'].items():
        loaded[cohort]={}
        for group,candidates in groups.items():
            loaded[cohort][group]={}
            for lr,prior in candidates.items():
                run=Path(prior['run']);path=run/'validation_predictions.npz'
                assert sha(path)==prior['predictions_sha256']
                with np.load(path,allow_pickle=False) as z: v={k:z[k] for k in z.files}
                ids=v['pack_row'];assert len(ids)==99849 and np.all(arrays['split'][ids]==1)
                for k,pk in [('observed_ghi','ghi'),('clear_sky_ghi','clear'),('station','station'),('lead','lead')]:
                    assert np.array_equal(v[k],arrays[pk][ids])
                err=v['pred_kt'].astype('float64')*v['clear_sky_ghi']-v['observed_ghi']
                score=np.mean([metrics(err[v['station']==s])['rmse'] for s in [0,1]])
                assert abs(score-prior['station_equal_ghi_rmse'])<1e-8
                ep_paths=sorted(run.glob('epoch_*.json')); eps=[json.loads(p.read_text()) for p in ep_paths]
                be=int(prior['best_epoch']); best=next(e for e in eps if e['epoch']==be)
                assert abs(best['station_equal_ghi_rmse']-score)<1e-8
                item=dict(cohort=cohort,group=group,lr=lr,score=score,best_epoch=be,epochs=len(eps),
                          first=eps[0],best=best,last=eps[-1],epoch_records=eps,
                          epoch_hashes={p.name:sha(p) for p in ep_paths},predictions_sha256=sha(path))
                result['candidates'].append(item);loaded[cohort][group][lr]=(score,v,err)
        choices={g:min(c,key=lambda lr:c[lr][0]) for g,c in loaded[cohort].items()}
        _,av,ea=loaded[cohort]['A'][choices['A']];_,cv,ec=loaded[cohort]['AC'][choices['AC']]
        assert np.array_equal(av['pack_row'],cv['pack_row'])
        delta=ec-ea
        summaries=[]
        masks=[('all',np.ones(len(ea),bool))]
        masks += [('station_'+str(s),av['station']==s) for s in [0,1]]
        masks += [('lead_'+str(15*(l+1)),av['lead']==l) for l in range(16)]
        masks += [('clear_'+str(lo)+'_'+str(hi),(av['clear_sky_ghi']>=lo)&(av['clear_sky_ghi']<hi)) for lo,hi in zip(edges[:-1],edges[1:])]
        for name,m in masks:
            if not m.any(): summaries.append(dict(group=name,n=0));continue
            am,cm=metrics(ea[m]),metrics(ec[m]);cross=float(2*np.mean(ea[m]*delta[m]));square=float(np.mean(delta[m]**2))
            diff=float(np.mean(ec[m]**2-ea[m]**2));assert np.isclose(diff,cross+square,atol=1e-8)
            summaries.append(dict(group=name,A=am,AC=cm,delta_rmse=cm['rmse']-am['rmse'],
                delta_mse=diff,correction_cross_term=cross,correction_square_term=square,
                delta_bias_squared=cm['bias']**2-am['bias']**2,
                delta_centered_mse=cm['centered_mse']-am['centered_mse']))
        result['paired'][cohort]=dict(existing_selected_lr=choices,groups=summaries,
            same_lr_deltas={lr:loaded[cohort]['AC'][lr][0]-loaded[cohort]['A'][lr][0] for lr in loaded[cohort]['A']})
        train=arrays['split']==0
        if cohort=='qc1':
            vp=root/'data/head_qc_view_20260908_v1';view=json.loads((vp/'view.json').read_text())
            assert sha(vp/'keep.npy')==view['keep_sha256'];train &= np.load(vp/'keep.npy',allow_pickle=False)
        n=int(train.sum());st=arrays['station'][train];le=arrays['lead'][train]
        counts=np.bincount(st*16+le,minlength=32);w=(n/(32*counts[st*16+le])).astype(np.float32).astype(np.float64)
        if cohort=='original': assert np.array_equal(w,arrays['weight'][train].astype('float64'))
        kt=arrays['kt'][train];clear=arrays['clear'][train];energy=w*kt**2;total=float(energy.sum())
        bins=[]
        masks=[('clear_'+str(lo)+'_'+str(hi),(clear>=lo)&(clear<hi)) for lo,hi in zip(edges[:-1],edges[1:])]
        masks += [('kt_gt_'+str(t),kt>t) for t in [2,5,10,100]]
        for name,m in masks:bins.append(dict(group=name,n=int(m.sum()),row_percent=100*float(m.mean()),zero_prediction_loss_percent=100*float(energy[m].sum())/total))
        result['zero_prediction_reference_loss'][cohort]=dict(train_rows=n,weighted_kt_mse=total/n,groups=bins)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'state':result['state'],'output':str(out),'cohorts':list(result['paired'])}))


if __name__=='__main__':main()
