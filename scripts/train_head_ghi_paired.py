"""Single-factor GHI-objective A/AC experiment; immutable original and QC1 cohorts."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import torch
import train_head_ghi_core as core
import train_head_pilot as original

EXPECTED_ORIGINAL = 'a610b06222f667d57d3646eeb986a2783f1d239b4578fc36093ff217d7274148'

def objective_witness():
    # Deliberately include tiny clear sky and large kt; no clipping or exclusions.
    p = torch.tensor([0.2, 0.5, 1.2, 2.0], requires_grad=True)
    clear = torch.tensor([0.01, 10., 500., 1000.])
    truth = torch.tensor([4.78, 30., 300., 100.])
    weight = torch.tensor([0.8, 1.2, 0.5, 1.5])
    loss = core.ghi_objective(p, truth, clear, weight)
    loss.backward()
    pn, gn, yn, wn = [x.detach().double().numpy() for x in (p, clear, truth, weight)]
    expected = np.mean(wn*((pn*gn-yn)/1000.)**2)
    gradient = 2*wn*(pn*gn-yn)*gn/(len(pn)*1000.**2)
    np.testing.assert_allclose(float(loss), expected, rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(p.grad.numpy(), gradient, rtol=2e-6, atol=1e-8)
    kt = yn/gn
    np.testing.assert_allclose(expected, np.mean(wn*(gn/1000.)**2*(pn-kt)**2), rtol=1e-12)
    assert core.sha(original.__file__) == EXPECTED_ORIGINAL
    for use in (False, True):
        core.seed(); a = core.Head(use)
        core.seed(); b = original.Head(use)
        assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    return dict(state='PASS_OBJECTIVE_VALUE_GRADIENT_AND_ARCHITECTURE', scale_W_m2=1000., loss=float(loss))

def run(root, cohort):
    pack = root/'data/head_pack_trainval_20260908_v1'
    old_dir = 'head_A_AC_seed42_20260908_v1' if cohort == 'original' else 'head_A_AC_qc1_seed42_20260908_v1'
    old = root/'runs'/old_dir
    out = root/'runs'/f'head_ghi_{cohort}_seed42_20260910_v1'
    out.mkdir(parents=True, exist_ok=True)
    old_contract = json.loads((old/'execution_contract.json').read_text())
    assert old_contract['batch'] == 2048 and old_contract['test_used'] is False
    old_binding = old_contract.get('imported_head_sha256', old_contract['code_sha256'])
    assert old_binding == EXPECTED_ORIGINAL
    audit = json.loads((pack/'audit.json').read_text())
    assert audit['state'] == 'COMPLETE' and audit['test_used'] is False
    assert core.sha(pack/'audit.json') == old_contract['pack_audit_sha256']
    for spec in audit['arrays'].values():
        assert core.sha(pack/spec['name']) == spec['sha256'], spec['name']
    for name, key in [('rows.csv','rows_csv_sha256'), ('norm.json','norm_sha256'), ('source_shards.json','source_shards_sha256')]:
        assert core.sha(pack/name) == audit[key]
    data = {k: torch.from_numpy(np.load(pack/f'{k}.npy', allow_pickle=False)).cuda()
            for k in ('x','c','g','kt','ghi','clear','station','lead','split','sequence','weight')}
    assert all(torch.isfinite(v).all() for v in data.values())
    assert torch.isin(data['split'], torch.tensor([0,1],device='cuda')).all()
    assert data['x'].shape == (637902,13,16,16)
    assert (data['clear']>0).all() and (data['kt']>=0).all()
    assert torch.max(torch.abs(data['kt']*data['clear']-data['ghi']))<1e-8
    train = torch.where(data['split']==0)[0]
    val = torch.where(data['split']==1)[0]
    assert len(train)==538053 and len(val)==99849
    if cohort == 'qc1':
        view_dir=root/'data/head_qc_view_20260908_v1'
        view=json.loads((view_dir/'view.json').read_text())
        assert core.sha(view_dir/'view.json') == old_contract['view_sha256']
        assert view['pack_audit_sha256']==core.sha(pack/'audit.json')
        assert core.sha(view_dir/'keep.npy')==view['keep_sha256']
        assert core.sha(view_dir/'excluded_rows.csv')==view['excluded_rows_sha256']
        assert core.sha(root/'docs/QC_PILOT_DECISION_20260908.md')==view['decision_sha256']
        keep=torch.from_numpy(np.load(view_dir/'keep.npy')).cuda()
        assert keep[val].all()
        train=train[keep[train]]
        assert len(train)==537605
        for k,spec in view['transforms_in_base_pack_normalized_coordinates'].items():
            m=torch.tensor(spec['mean'],device='cuda',dtype=torch.float32)
            s=torch.tensor(spec['std'],device='cuda',dtype=torch.float32)
            if k=='x': data[k].sub_(m[None,:,None,None]).div_(s[None,:,None,None])
            elif k=='g': data[k][:,:2].sub_(m).div_(s)
            else: data[k].sub_(m).div_(s)
            assert torch.isfinite(data[k]).all()
        counts=torch.bincount(data['station'][train]*16+data['lead'][train],minlength=32)
        assert counts.cpu().tolist()==view['train_group_counts']
        data['weight'][train]=(len(train)/(32*counts[data['station'][train]*16+data['lead'][train]])).float()
    metadata=dict(old_contract, code_sha256=core.sha(__file__), imported_head_sha256=core.sha(core.__file__),
                  original_head_sha256=EXPECTED_ORIGINAL, cohort=cohort,
                  loss='mean(w*((pred_kt*interval_clear_GHI-observed_interval_GHI)/1000)^2)',
                  loss_scale_W_m2=1000., train_rows=len(train), val_rows=len(val),
                  baseline_execution_sha256=core.sha(old/'execution_contract.json'),
                  baseline_profile_sha256=old_contract['profile_sha256'],
                  profile_reused_from_original=True,
                  protocol_sha256=core.sha(root/'docs/HEAD_GHI_LOSS_PROTOCOL_20260910.md'))
    contract=out/'execution_contract.json'
    if contract.exists(): assert json.loads(contract.read_text())==metadata
    else: core.atomic(contract,metadata)
    core.atomic(out/'objective_witness.json',objective_witness())
    print('PACK_READY',cohort,len(train),len(val),flush=True)
    torch.backends.cudnn.benchmark=True
    assert torch.cuda.is_bf16_supported()
    core.seed(); model=core.Head(True).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3): core.update(model,opt,data,train[:2048])
    peak=torch.cuda.max_memory_allocated()/2**30
    assert peak<0.8*torch.cuda.get_device_properties(0).total_memory/2**30
    core.atomic(out/'batch_witness.json',dict(batch=2048,peak_GiB=peak,finite=True,
        rule='reuse measured original batch for single-factor parity; 20 percent memory reserve',
        torch_version=torch.__version__,cuda_version=torch.version.cuda,device=torch.cuda.get_device_name(0)))
    del model,opt;gc.collect();torch.cuda.empty_cache()
    core.overfit(data,train,out)
    for group in ('A','AC'):
        for lr in (0.0003,0.001): core.fit(data,train,val,out,group,lr,2048,metadata)
    core.atomic(out/'complete.json',dict(state='COMPLETE_GHI_LOSS_VALIDATION_PILOT',test_used=False,
        next='independent matched-output audit; no automatic test or additional configurations'))
    del data;gc.collect();torch.cuda.empty_cache()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path)
    parser.add_argument('--cpu-check',action='store_true')
    args=parser.parse_args()
    print(json.dumps(objective_witness()),flush=True)
    if args.cpu_check:return
    assert args.root is not None
    for cohort in ('original','qc1'):run(args.root,cohort)
    print('GHI_LOSS_PAIRED_TRAINING_COMPLETE',flush=True)

if __name__=='__main__':main()
