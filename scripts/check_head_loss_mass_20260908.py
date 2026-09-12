"""Frozen selected heads: complete train/val inference, never update weights."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import train_head_pilot as base


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root
    out=r/'audits/head_loss_mass_20260908_v1';assert not out.exists();out.mkdir()
    started=time.monotonic();torch.set_num_threads(2);base.seed(42)
    assert base.sha(Path(base.__file__))=='a610b06222f667d57d3646eeb986a2783f1d239b4578fc36093ff217d7274148'
    torch.backends.cudnn.benchmark=True
    pack=r/'data/head_pack_trainval_20260908_v1';audit=json.loads((pack/'audit.json').read_text())
    arrays={};tensors={};hashes={}
    for k in ['x','c','g','kt','ghi','clear','station','lead','split']:
        spec=audit['arrays'][k];p=pack/spec['name'];assert base.sha(p)==spec['sha256'];hashes[k]=spec['sha256']
        arrays[k]=np.load(p,allow_pickle=False)
        if k in ['x','c','g','station','lead']:tensors[k]=torch.from_numpy(arrays[k]).cuda()
    assert np.isin(arrays['split'],[0,1]).all()
    val=np.flatnonzero(arrays['split']==1);assert len(val)==99849
    candidates=json.loads((r/'audits/head_paired_pilot_20260908.json').read_text())['candidates']
    viewpath=r/'data/head_qc_view_20260908_v1/view.json';view=json.loads(viewpath.read_text())
    keepath=viewpath.parent/'keep.npy';assert base.sha(keepath)==view['keep_sha256'];keep=np.load(keepath)
    report=dict(state='RUNNING',script_sha256=base.sha(Path(__file__)),source_hashes=hashes,
        protocol_sha256=base.sha(r/'docs/COT_FAILURE_CAUSE_PROTOCOL_20260908.md'),training=False,test_used=False,
        batch=2048,precision='BF16 forward, saved float32 pred_kt, float64 residual aggregation',models=[])
    base.atomic(out/'report.json',report)
    for cohort,groups in candidates.items():
        train=np.flatnonzero((arrays['split']==0)&(keep if cohort=='qc1' else True))
        for group,choices in groups.items():
            lr=min(choices,key=lambda k:choices[k]['station_equal_ghi_rmse']);record=choices[lr];run=Path(record['run'])
            assert base.sha(run/'best.pt')==record['checkpoint_sha256']
            ck=torch.load(run/'best.pt',map_location='cpu');assert ck['group']==group and ck['batch']==2048
            assert ck['metadata']['pack_audit_sha256']==base.sha(pack/'audit.json')
            if cohort=='qc1':
                assert ck['metadata']['view_sha256']==base.sha(viewpath)
                assert ck['metadata']['input_transform']==view['transforms_in_base_pack_normalized_coordinates']
            model=base.Head(group=='AC').cuda().eval();model.load_state_dict(ck['model'],strict=True)
            for p in model.parameters():p.requires_grad_(False)
            ids=np.sort(np.concatenate([train,val]));pred=np.empty(len(ids),np.float32)
            transforms={}
            if cohort=='qc1':
                for key,spec in ck['metadata']['input_transform'].items():
                    transforms[key]=(torch.tensor(spec['mean'],device='cuda',dtype=torch.float32),torch.tensor(spec['std'],device='cuda',dtype=torch.float32))
            for start in range(0,len(ids),2048):
                ix=torch.from_numpy(ids[start:start+2048]).cuda()
                batch={k:v[ix] for k,v in tensors.items()}
                for k,(m,s) in transforms.items():
                    if k=='x':batch[k].sub_(m[None,:,None,None]).div_(s[None,:,None,None])
                    elif k=='g':batch[k][:,:2].sub_(m).div_(s)
                    else:batch[k].sub_(m).div_(s)
                with torch.autocast('cuda',dtype=torch.bfloat16):p=model(*(batch[k] for k in ['x','c','g','station','lead'])).float()
                pred[start:start+len(ix)]=p.cpu().numpy()
            assert np.isfinite(pred).all()
            assert base.sha(run/'validation_predictions.npz')==record['predictions_sha256']
            with np.load(run/'validation_predictions.npz') as z:
                assert np.array_equal(z['pack_row'],val)
                vp=pred[arrays['split'][ids]==1];ref=z['pred_kt'];err=abs(vp-ref)
                passed=bool(np.all(err<=.02*np.maximum(1,abs(ref))))
            path=out/(cohort+'_'+group+'_predictions.npz')
            np.savez(path,pack_row=ids,pred_kt=pred)
            entry=dict(cohort=cohort,group=group,selected_lr=lr,checkpoint_sha256=record['checkpoint_sha256'],
                predictions_file=path.name,predictions_sha256=base.sha(path),train_rows=len(train),validation_rows=len(val),
                validation_binding_pass=passed,validation_max_kt_difference=float(err.max()))
            report['models'].append(entry);base.atomic(out/'report.json',report)
            assert passed,'validation binding failed; preserve evidence and stop'
            print(json.dumps(entry),flush=True)
            del model
    report.update(state='COMPLETE_FROZEN_HEAD_TRAIN_VAL_FORWARD',seconds=time.monotonic()-started,
        peak_GiB=torch.cuda.max_memory_allocated()/2**30,torch_version=torch.__version__,gpu=torch.cuda.get_device_name())
    base.atomic(out/'report.json',report)
    print('FROZEN_HEAD_LOSS_MASS_FORWARD_COMPLETE',flush=True)


if __name__=='__main__':main()
