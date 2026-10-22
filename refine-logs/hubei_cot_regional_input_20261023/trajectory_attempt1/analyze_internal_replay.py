"""Independent CPU checks and scope/thickness decomposition of saved replays."""
import hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;root=P/'internal_replay_attempt1';assert json.loads((root/'COMPLETE.json').read_text())['state']=='SIX_TRAIN_INTERNAL_FROZEN_REPLAYS_COMPLETE'
out=root/'analysis_attempt1';assert not out.exists();runs=[];checks=[];reference=None
for arm in ['full','repeat_last']:
    for seed in [42,43,44]:
        path=root/f'seed{seed}_{arm}'/'predictions.npz';receipt=json.loads(path.with_name('COMPLETE.json').read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['predictions_sha256']
        with np.load(path) as z:data={k:z[k] for k in z.files}
        if reference is None:reference=data
        for k in ['init','target','lead','reference','target_valid','common']:
            assert np.array_equal(data[k],reference[k],equal_nan=True) if data[k].dtype.kind not in 'US' else np.array_equal(data[k],reference[k])
        sums=data['frame_sums'].sum(0);assert np.allclose(sums,np.array(receipt['sums']),rtol=1e-12,atol=1e-6)
        # Recompute station-center sums directly from physical saved predictions.
        mask=data['common'][:,:,1,1];truth=data['reference'][:,:,1,1]
        for method,j in [('persistence',1),('transport',2),('learned',3)]:
            e=data[method][:,:,1,1]-truth
            value=float(np.sum(np.where(mask,e*e,0.)))
            assert abs(value-sums[2,0,j])<1e-6,(method,value,sums[2,0,j])
        scopes={}
        for index,scope in [(1,'region_common'),(2,'station_common')]:
            cloud={}
            for k,label in enumerate(['all','near_zero','thin','moderate','thick']):
                row=sums[index,k];n=row[0]
                cloud[label]=dict(n=int(n),rmse={m:float(np.sqrt(row[j]/n)) if n else None for m,j in [('persistence',1),('transport',2),('learned',3)]},
                    bias={m:float(row[j]/n) if n else None for m,j in [('persistence',4),('transport',5),('learned',6)]},
                    mean_learned_minus_transport=float(row[7]/n) if n else None)
            scopes[scope]=cloud
        runs.append(dict(arm=arm,seed=seed,scopes=scopes,regional_training_objective=receipt['full_domain_objective_normalized'],
                         objective_replay_abs_difference=receipt['objective_abs_difference']))
        checks.append(dict(name=path.parent.name,station_sse_recomputed=True,frame_sums_match_receipt=True,prediction_sha256=receipt['predictions_sha256']))
summary={}
for arm in ['full','repeat_last']:
    selected=[r for r in runs if r['arm']==arm];summary[arm]={}
    for scope in ['region_common','station_common']:
        summary[arm][scope]={}
        for label in ['all','near_zero','thin','moderate','thick']:
            cells=[r['scopes'][scope][label] for r in selected]
            assert len({r['n'] for r in cells})==1
            summary[arm][scope][label]=dict(n_per_seed=cells[0]['n'],
                rmse_seed_mean={m:float(np.mean([c['rmse'][m] for c in cells])) if cells[0]['n'] else None for m in ['persistence','transport','learned']},
                bias_seed_mean={m:float(np.mean([c['bias'][m] for c in cells])) if cells[0]['n'] else None for m in ['persistence','transport','learned']},
                mean_learned_minus_transport=float(np.mean([c['mean_learned_minus_transport'] for c in cells])) if cells[0]['n'] else None)
report=dict(state='PASS_INDEPENDENT_TRAIN_INTERNAL_REPLAY_ANALYSIS',checks=checks,seed_mean=summary,runs=runs,
    scope='train-internal select, not outer-validation/test or GHI evidence',new_training_runs=0,test_buffer_payloads_read=0)
out.mkdir();(out/'SUMMARY.json').write_text(json.dumps(report,indent=2));print(json.dumps(summary,indent=2))
