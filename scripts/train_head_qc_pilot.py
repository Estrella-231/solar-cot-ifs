"""Exploratory suspected-flatline QC view; original paired pilot is mandatory."""
import argparse,gc,json
from pathlib import Path
import numpy as np
import torch
import train_head_pilot as base

def main():
    p=argparse.ArgumentParser();p.add_argument('--pack',type=Path,required=True)
    p.add_argument('--view',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--original-run',type=Path,required=True);p.add_argument('--decision',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    view=json.loads((a.view/'view.json').read_text());audit=json.loads((a.pack/'audit.json').read_text())
    assert view['state']=='COMPLETE_QC_SENSITIVITY_VIEW' and audit['state']=='COMPLETE'
    assert base.sha(a.pack/'audit.json')==view['pack_audit_sha256']
    assert base.sha(a.view/'keep.npy')==view['keep_sha256']
    assert base.sha(a.view/'excluded_rows.csv')==view['excluded_rows_sha256']
    assert base.sha(a.decision)==view['decision_sha256']
    original=json.loads((a.original_run/'execution_contract.json').read_text())
    assert original['code_sha256']==base.sha(base.__file__)
    assert base.sha(a.original_run/'profile.json')==original['profile_sha256']
    assert original['batch']==2048
    for v in audit['arrays'].values():assert base.sha(a.pack/v['name'])==v['sha256']
    data={k:torch.from_numpy(np.load(a.pack/(k+'.npy'),allow_pickle=False)).cuda()
        for k in ('x','c','g','kt','ghi','clear','station','lead','split','sequence','weight')}
    keep=torch.from_numpy(np.load(a.view/'keep.npy')).cuda()
    train=torch.where((data['split']==0)&keep)[0];val=torch.where((data['split']==1)&keep)[0]
    assert len(train)==537605 and len(val)==99849 and len(keep)==637902
    assert torch.equal(val,torch.where(data['split']==1)[0])
    assert torch.max(torch.abs(data['kt']*data['clear']-data['ghi']))<1e-8
    for k,spec in view['transforms_in_base_pack_normalized_coordinates'].items():
        m=torch.tensor(spec['mean'],device='cuda',dtype=torch.float32)
        s=torch.tensor(spec['std'],device='cuda',dtype=torch.float32)
        if k=='x':
            data[k].sub_(m[None,:,None,None]).div_(s[None,:,None,None])
        elif k=='g':data[k][:,:2].sub_(m).div_(s)
        else:data[k].sub_(m).div_(s)
        assert torch.isfinite(data[k]).all()
    counts=torch.bincount(data['station'][train]*16+data['lead'][train],minlength=32)
    assert counts.cpu().tolist()==view['train_group_counts']
    data['weight'][train]=(len(train)/(32*counts[data['station'][train]*16+data['lead'][train]])).float()
    metadata=dict(original,code_sha256=base.sha(__file__),imported_head_sha256=base.sha(base.__file__),
        cohort='suspected_flatline_qc1_exploratory',view_sha256=base.sha(a.view/'view.json'),
        decision_sha256=base.sha(a.decision),train_rows=len(train),val_rows=len(val),
        input_transform=view['transforms_in_base_pack_normalized_coordinates'],
        original_cohort_paired_pilot_required=True,source_pack_row_indices_preserved=True)
    contract=a.output/'execution_contract.json'
    if contract.exists():assert json.loads(contract.read_text())==metadata
    else:base.atomic(contract,metadata)
    print('QC_PACK_LOADED',json.dumps(metadata),flush=True)
    torch.backends.cudnn.benchmark=True;base.seed()
    model=base.Head(True).cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):base.update(model,opt,data,train[:2048])
    peak=torch.cuda.max_memory_allocated()/2**30
    assert peak<0.8*torch.cuda.get_device_properties(0).total_memory/2**30
    base.atomic(a.output/'batch_witness.json',dict(batch=2048,peak_GiB=peak,finite=True,
        selection='frozen original profile batch; QC shape and dtype unchanged'))
    del model,opt;gc.collect();torch.cuda.empty_cache()
    base.overfit(data,train,a.output)
    for group in ('A','AC'):
        for lr in (0.0003,0.001):base.fit(data,train,val,a.output,group,lr,2048,metadata)
    base.atomic(a.output/'complete.json',dict(state='COMPLETE_QC1_VALIDATION_PILOT',test_used=False,
        original_cohort_paired_pilot_required=True,next='complete original paired pilot; independent joint audit'))

if __name__=='__main__':main()
