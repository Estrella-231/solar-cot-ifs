"""Independent saved-output/CPU reload audit of all 12 fixed-LR seed candidates."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import train_head_pilot as original
from audit_head_paired_pilot import metrics, sha

def summarize(values):
    a=np.asarray(values,dtype=np.float64)
    return dict(values=a.tolist(),mean=float(a.mean()),sample_std=float(a.std(ddof=1)),
                negative_count=int((a<0).sum()),n=len(a))

def run(root, report):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert sha(original.__file__)=='a610b06222f667d57d3646eeb986a2783f1d239b4578fc36093ff217d7274148'
    pack=root/'data/head_pack_trainval_20260908_v1'
    audit=json.loads((pack/'audit.json').read_text())
    assert audit['state']=='COMPLETE' and audit['test_used'] is False
    arrays={}
    for key in ('x','c','g','split','ghi','clear','station','lead'):
        item=audit['arrays'][key]
        assert sha(pack/item['name'])==item['sha256'],key
        arrays[key]=np.load(pack/item['name'],mmap_mode='r',allow_pickle=False)
    assert np.isin(arrays['split'],[0,1]).all()
    ids=np.flatnonzero(arrays['split']==1)
    assert len(ids)==99849
    positions=[int(np.flatnonzero((arrays['station'][ids]==s)&(arrays['lead'][ids]==l))[0])
               for s in (0,1) for l in (0,5,10,15)]
    witness_ids=ids[positions]
    view_path=root/'data/head_qc_view_20260908_v1/view.json'
    view=json.loads(view_path.read_text())
    assert view['pack_audit_sha256']==sha(pack/'audit.json')
    candidates={};paired={}
    for cohort,lr in (('original',0.0003),('qc1',0.001)):
        candidates[cohort]={};paired[cohort]={}
        for run_seed in (42,43,44):
            name=f'head_ghi_{cohort}_seed42_20260910_v1' if run_seed==42 else f'head_ghi_multiseed_{cohort}_seed{run_seed}_20260910_v1'
            out=root/'runs'/name
            assert json.loads((out/'complete.json').read_text())['test_used'] is False
            contract=json.loads((out/'execution_contract.json').read_text())
            assert contract['seed']==run_seed and contract['batch']==2048
            assert contract['pack_audit_sha256']==sha(pack/'audit.json') and contract['test_used'] is False
            assert contract['loss_scale_W_m2']==1000.
            assert contract['epochs_max']==50 and contract['patience']==8
            assert contract['train_rows']==(538053 if cohort=='original' else 537605)
            if run_seed!=42:
                assert contract['learning_rates']==[lr]
                assert contract['code_sha256']==sha(root/'scripts/train_head_ghi_multiseed.py')
                assert contract['imported_head_sha256']==sha(root/'scripts/train_head_ghi_multiseed_core.py')
                assert contract['protocol_sha256']==sha(root/'docs/HEAD_GHI_MULTISEED_PROTOCOL_20260910.md')
            if cohort=='qc1':assert contract['view_sha256']==sha(view_path)
            arms={}
            for group in ('A','AC'):
                path=out/f'{group}_lr{lr:g}_seed{run_seed}'
                done=json.loads((path/'complete.json').read_text())
                assert done['state']=='COMPLETE_VALIDATION_PILOT' and done['test_used'] is False
                assert sha(path/'best.pt')==done['checkpoint_sha256']
                assert sha(path/'validation_predictions.npz')==done['predictions_sha256']
                with np.load(path/'validation_predictions.npz',allow_pickle=False) as z:
                    assert np.array_equal(z['pack_row'],ids)
                    for name,key in (('observed_ghi','ghi'),('clear_sky_ghi','clear'),('station','station'),('lead','lead')):
                        assert np.array_equal(z[name],arrays[key][ids]),name
                    assert np.isfinite(z['pred_kt']).all() and (z['pred_kt']>=0).all()
                    score=metrics(z['pred_kt'],arrays['ghi'][ids],arrays['clear'][ids],arrays['station'][ids],arrays['lead'][ids])
                    reference=z['pred_kt'][positions].astype(np.float64)
                assert abs(score['station_equal_ghi_rmse']-done['station_equal_ghi_rmse'])<1e-8
                ck=torch.load(path/'best.pt',map_location='cpu')
                assert ck['metadata']==contract and ck['seed']==run_seed and ck['lr']==lr and ck['group']==group
                assert ck['batch']==2048 and ck['next_epoch']-1==done['best_epoch_0based']
                tensors={k:torch.from_numpy(np.array(arrays[k][witness_ids],copy=True))
                         for k in ('x','c','g','station','lead')}
                for k in ('x','c','g'):tensors[k]=tensors[k].float()
                if cohort=='qc1':
                    for k,spec in view['transforms_in_base_pack_normalized_coordinates'].items():
                        m=torch.tensor(spec['mean'],dtype=torch.float32);s=torch.tensor(spec['std'],dtype=torch.float32)
                        if k=='x':tensors[k].sub_(m[None,:,None,None]).div_(s[None,:,None,None])
                        elif k=='g':tensors[k][:,:2].sub_(m).div_(s)
                        else:tensors[k].sub_(m).div_(s)
                model=original.Head(group=='AC').float().eval()
                model.load_state_dict(ck['model'],strict=True)
                with torch.no_grad():pred=model(*(tensors[k] for k in ('x','c','g','station','lead'))).numpy()
                errors=np.abs(pred.astype(np.float64)-reference)
                tolerance=0.02*np.maximum(1.,np.abs(reference))
                assert np.isfinite(pred).all() and (errors<=tolerance).all(),(cohort,run_seed,group)
                arms[group]=dict(**score,run=str(path),seed=run_seed,lr=lr,best_epoch=done['best_epoch_0based'],
                    checkpoint_sha256=done['checkpoint_sha256'],predictions_sha256=done['predictions_sha256'],
                    cpu_reload_pass=True,cpu_max_abs_error=float(errors.max()))
                print('CANDIDATE_VERIFIED',cohort,run_seed,group,score['station_equal_ghi_rmse'],flush=True)
            candidates[cohort][str(run_seed)]=arms
            paired[cohort][str(run_seed)]=dict(
                AC_minus_A=arms['AC']['station_equal_ghi_rmse']-arms['A']['station_equal_ghi_rmse'],
                stations={s:arms['AC']['stations'][s]['rmse']-arms['A']['stations'][s]['rmse'] for s in ('0','1')},
                leads={l:arms['AC']['lead_rmse'][l]-arms['A']['lead_rmse'][l] for l in arms['A']['lead_rmse']})
    summaries={}
    for cohort in candidates:
        summaries[cohort]=dict(
            A=summarize([candidates[cohort][str(s)]['A']['station_equal_ghi_rmse'] for s in (42,43,44)]),
            AC=summarize([candidates[cohort][str(s)]['AC']['station_equal_ghi_rmse'] for s in (42,43,44)]),
            paired_delta=summarize([paired[cohort][str(s)]['AC_minus_A'] for s in (42,43,44)]),
            new_seed_delta=summarize([paired[cohort][str(s)]['AC_minus_A'] for s in (43,44)]),
            station_delta={st:summarize([paired[cohort][str(s)]['stations'][st] for s in (42,43,44)]) for st in ('0','1')},
            lead_delta={l:summarize([paired[cohort][str(s)]['leads'][l] for s in (42,43,44)]) for l in paired[cohort]['42']['leads']})
    report.update(state='PASS_ALL_12_FIXED_LR_PREDICTIONS_AND_CPU_RELOADS',validation_rows=len(ids),
        source_pack_sha256=sha(pack/'audit.json'),witness_pack_rows=witness_ids.tolist(),
        candidates=candidates,paired=paired,summaries=summaries,
        scope='validation seed stability only; seed42 used to select config; no test, significance, or iid-row inference')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    path=args.root/'audits/head_ghi_multiseed_20260910.json'
    assert not path.exists(),'refuse to overwrite an audit'
    report=dict(state='IN_PROGRESS',test_used=False,script_sha256=sha(__file__))
    try:run(args.root,report)
    except Exception as error:
        report.update(state='FAIL',error=repr(error))
        path.write_text(json.dumps(report,indent=2,allow_nan=False));raise
    path.write_text(json.dumps(report,indent=2,allow_nan=False))
    print('MULTISEED_AUDIT_PASS',json.dumps({k:v['paired_delta'] for k,v in report['summaries'].items()}),flush=True)

if __name__=='__main__':main()
