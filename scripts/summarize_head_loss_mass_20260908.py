"""NumPy-only independent aggregation of frozen-head saved predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);r=ap.parse_args().root
    directory=r/'audits/head_loss_mass_20260908_v1';output=directory/'loss_decomposition.json';assert not output.exists()
    report=json.loads((directory/'report.json').read_text());assert report['state']=='COMPLETE_FROZEN_HEAD_TRAIN_VAL_FORWARD'
    pack=r/'data/head_pack_trainval_20260908_v1';a={}
    for k in ['kt','ghi','clear','station','lead','split']:
        p=pack/(k+'.npy');assert sha(p)==report['source_hashes'][k];a[k]=np.load(p,allow_pickle=False)
    view=r/'data/head_qc_view_20260908_v1';meta=json.loads((view/'view.json').read_text())
    assert sha(view/'keep.npy')==meta['keep_sha256'];keep=np.load(view/'keep.npy',allow_pickle=False)
    result=dict(state='PASS_INDEPENDENT_FROZEN_LOSS_AGGREGATION',script_sha256=sha(Path(__file__)),
        forward_report_sha256=sha(directory/'report.json'),models=[],training=False,test_used=False,
        scope='Saved fixed-checkpoint actual errors; not optimizer trajectory or causal intervention.')
    edges=[0,20,100,300,600,float('inf')]
    for item in report['models']:
        path=directory/item['predictions_file'];assert sha(path)==item['predictions_sha256']
        with np.load(path,allow_pickle=False) as z: ids=z['pack_row'];pred=z['pred_kt'].astype('float64')
        expected=np.flatnonzero((a['split']==1)|((a['split']==0)&(keep if item['cohort']=='qc1' else True)))
        assert np.array_equal(ids,expected);assert len(np.unique(ids))==len(ids)
        kt=a['kt'][ids];cl=a['clear'][ids];ghi=a['ghi'][ids];st=a['station'][ids];lead=a['lead'][ids]
        assert np.max(abs(kt*cl-ghi))<1e-8
        erkt=pred-kt;erghi=pred*cl-ghi
        assert np.allclose(erkt*cl,erghi,rtol=1e-10,atol=1e-9)
        parts={}
        for splitcode,name in [(0,'train'),(1,'validation')]:
            sel=a['split'][ids]==splitcode;n=int(sel.sum());weights=np.ones(n)
            if name=='train':
                group=st[sel]*16+lead[sel];cnt=np.bincount(group,minlength=32)
                weights=(n/(32*cnt[group])).astype(np.float32).astype(np.float64)
            losses=weights*erkt[sel]**2;gsq=erghi[sel]**2
            # Independently verify GHI-residual-to-kt objective identity.
            assert np.allclose(losses,weights*gsq/cl[sel]**2,rtol=1e-8,atol=1e-8)
            bins=[]
            masks=[('clear_'+str(lo)+'_'+str(hi),(cl[sel]>=lo)&(cl[sel]<hi)) for lo,hi in zip(edges[:-1],edges[1:])]
            masks += [('kt_gt_'+str(t),kt[sel]>t) for t in [2,5,10,100]]
            for label,m in masks:
                bins.append(dict(group=label,n=int(m.sum()),row_percent=100*float(m.mean()),
                    actual_kt_loss_percent=100*float(losses[m].sum()/losses.sum()),
                    actual_ghi_sse_percent=100*float(gsq[m].sum()/gsq.sum()),
                    ghi_rmse=float(np.sqrt(gsq[m].mean())) if m.any() else None))
            assert np.isclose(sum(b['actual_kt_loss_percent'] for b in bins[:5]),100)
            sr=[float(np.sqrt(np.mean(erghi[sel][st[sel]==s]**2))) for s in [0,1]]
            parts[name]=dict(n=n,weighted_kt_mse=float(losses.mean()),station_equal_ghi_rmse=float(np.mean(sr)),station_rmse=sr,groups=bins)
        result['models'].append(dict(cohort=item['cohort'],group=item['group'],selected_lr=item['selected_lr'],
            checkpoint_sha256=item['checkpoint_sha256'],prediction_sha256=item['predictions_sha256'],splits=parts))
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'state':result['state'],'models':len(result['models'])}))


if __name__=='__main__':main()
