"""Independent NumPy audit of all eight saved pilot predictions, no test."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def metrics(pred,truth,clear,station,lead):
    e=pred.astype('float64')*clear-truth
    groups={}
    for s in (0,1):
        v=e[station==s]
        groups[str(s)]=dict(n=len(v),rmse=float(np.sqrt(np.mean(v*v))),mae=float(np.abs(v).mean()),mbe=float(v.mean()))
    return dict(station_equal_ghi_rmse=float(np.mean([g['rmse'] for g in groups.values()])),stations=groups,
        lead_rmse={str((l+1)*15):float(np.sqrt(np.mean(e[lead==l]**2))) for l in range(16)})

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    pack=a.root/'data/head_pack_trainval_20260908_v1'
    audit=json.loads((pack/'audit.json').read_text());assert audit['state']=='COMPLETE'
    arrays={}
    for k in ('split','ghi','clear','station','lead'):
        item=audit['arrays'][k];assert sha(pack/item['name'])==item['sha256']
        arrays[k]=np.load(pack/item['name'])
    ids=np.flatnonzero(arrays['split']==1);assert len(ids)==99849
    roots={'qc1':a.root/'runs/head_A_AC_qc1_seed42_20260908_v1',
           'original':a.root/'runs/head_A_AC_seed42_20260908_v1'}
    results={};selected={}
    for cohort,root in roots.items():
        marker=json.loads((root/'complete.json').read_text());assert marker['test_used'] is False
        results[cohort]={};selected[cohort]={}
        for group in ('A','AC'):
            candidates={}
            for lr in (0.0003,0.001):
                run=root/f'{group}_lr{lr:g}_seed42'
                report=json.loads((run/'complete.json').read_text())
                assert report['state']=='COMPLETE_VALIDATION_PILOT' and report['test_used'] is False
                assert sha(run/'best.pt')==report['checkpoint_sha256']
                assert sha(run/'validation_predictions.npz')==report['predictions_sha256']
                with np.load(run/'validation_predictions.npz') as z:
                    assert np.array_equal(z['pack_row'],ids)
                    for name,key in (('observed_ghi','ghi'),('clear_sky_ghi','clear'),('station','station'),('lead','lead')):
                        assert np.array_equal(z[name],arrays[key][ids]),(cohort,group,lr,name)
                    assert np.isfinite(z['pred_kt']).all() and (z['pred_kt']>=0).all()
                    m=metrics(z['pred_kt'],arrays['ghi'][ids],arrays['clear'][ids],arrays['station'][ids],arrays['lead'][ids])
                assert abs(m['station_equal_ghi_rmse']-report['station_equal_ghi_rmse'])<1e-8
                candidates[str(lr)]=dict(**m,learning_rate=lr,run=str(run),best_epoch=report['best_epoch_0based'],
                    checkpoint_sha256=report['checkpoint_sha256'],predictions_sha256=report['predictions_sha256'])
            results[cohort][group]=candidates
            selected[cohort][group]=min(candidates.values(),key=lambda v:(v['station_equal_ghi_rmse'],v['learning_rate']))
    contrasts={}
    for name,s in selected.items():
        base=s['A']['station_equal_ghi_rmse'];full=s['AC']['station_equal_ghi_rmse']
        contrasts[name]=dict(A=base,AC=full,AC_minus_A=full-base,relative_rmse_reduction_percent=100*(base-full)/base,
            station_AC_minus_A={k:s['AC']['stations'][k]['rmse']-s['A']['stations'][k]['rmse'] for k in ('0','1')})
    report=dict(state='PASS_ALL_8_PREDICTION_AUDIT',test_used=False,validation_rows=len(ids),same_keys_truth_masks=True,
        candidates=results,selected=selected,contrasts=contrasts,source_pack_sha256=sha(pack/'audit.json'),
        script_sha256=sha(__file__),scope='validation pilot after LR selection, no statistical significance or paper gain claim',
        model_reload_witness='PENDING_INDEPENDENT_CHECKPOINT_FORWARD',next='independent checkpoint witness and go/no-go across BOTH cohorts')
    output=a.root/'audits/head_paired_pilot_20260908.json';output.parent.mkdir(exist_ok=True)
    tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False));tmp.replace(output)
    print(json.dumps(dict(state=report['state'],contrasts=contrasts)),flush=True)

if __name__=='__main__':main()
