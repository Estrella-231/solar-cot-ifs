"""Paired AGRI-only vs AGRI+COT residual-capacity control on the fixed ZhuJia pilot."""
import argparse, copy, hashlib, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_regional_cot_ghi_pilot_20260927 import Rows, RegionalStationHead, seed_all, save
from train_zhujia_cot_residual_20260927 import ZhuRows, Residual


@torch.no_grad()
def evaluate_arm(model, loader, use_cot):
    model.eval(); rid=[]; pred=[]; truth=[]; leads=[]
    for b in loader:
        v=[x.cuda(non_blocking=True) for x in b]
        if not use_cot: v[2].zero_();v[3].zero_()
        z,_=model(*v[:6])
        rid.append(v[9].cpu().numpy());pred.append((z*v[7]).cpu().numpy())
        truth.append((v[6]*v[7]).cpu().numpy());leads.append(v[5].cpu().numpy())
    rid=np.concatenate(rid);p=np.concatenate(pred);y=np.concatenate(truth);ld=np.concatenate(leads);e=p-y
    metrics={'n':len(y),'rmse':float(np.mean(e**2)**.5),'mae':float(np.mean(abs(e))),'bias':float(e.mean()),
      'lead_rmse':{str(15*(i+1)):float(np.mean(e[ld==i]**2)**.5) for i in range(16)}}
    return metrics,{'row_ids':rid,'pred_ghi':p,'true_ghi':y,'lead':ld}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bank',type=Path,required=True);ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--source-run',type=Path,required=True);ap.add_argument('--sili-checkpoint',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--updates',type=int,default=600);ap.add_argument('--batch',type=int,default=512)
    ap.add_argument('--seed',type=int,default=42);ap.add_argument('--lr',type=float,default=3e-4)
    ap.add_argument('--cot-limit',type=float,default=.15);ap.add_argument('--delta-penalty',type=float,default=.05)
    a=ap.parse_args()
    if a.out.exists(): raise RuntimeError(f'refuse to overwrite {a.out}')
    a.out.mkdir(parents=True)
    torch.set_num_threads(2); seed_all(a.seed)
    tr=ZhuRows(Rows(a.bank,a.base,'train','forecast'))
    va=ZhuRows(Rows(a.bank,a.base,'val','forecast'))
    assert len(tr)>0 and len(va)>0
    # Both candidates use the exact same selected frozen no-COT base and same
    # COT-encoder initialization. Their trainable parameter counts are identical.
    expected_sili='4d243d0c8c3be33cff83ad8307c6d2e1bcde59d8a9bae6cac17bfd8e87ea7d4c'
    sili_hash=hashlib.sha256(a.sili_checkpoint.read_bytes()).hexdigest()
    assert sili_hash==expected_sili, f'approved Sili checkpoint changed: {sili_hash}'
    bank_contract=json.loads((a.bank/'complete.json').read_text())
    assert bank_contract.get('test_used') is False
    zero_ck=torch.load(a.source_run/'zero_best.pt',map_location='cpu',weights_only=False)['model']
    cot_ck=torch.load(a.source_run/'fusion_best.pt',map_location='cpu',weights_only=False)['model']
    seed_all(a.seed+991); base_a=RegionalStationHead();base_a.load_state_dict(zero_ck)
    seed_all(a.seed+991); base_c=RegionalStationHead();base_c.load_state_dict(zero_ck)
    cot_a=RegionalStationHead();cot_a.load_state_dict(cot_ck)
    cot_c=RegionalStationHead();cot_c.load_state_dict(cot_ck)
    seed_all(a.seed+1701); agri=Residual(base_a,cot_a,a.cot_limit).cuda()
    seed_all(a.seed+1701); cot=Residual(base_c,cot_c,a.cot_limit).cuda()
    n_agri=sum(p.numel() for p in agri.parameters() if p.requires_grad)
    n_cot=sum(p.numel() for p in cot.parameters() if p.requires_grad)
    assert n_agri==n_cot, (n_agri,n_cot)
    trainable={k:sum(p.numel() for p in m.parameters() if p.requires_grad) for k,m in [('agri_residual',agri),('cot_residual',cot)]}
    init_equal=all(torch.equal(agri.state_dict()[k],cot.state_dict()[k]) for k in agri.state_dict())
    assert init_equal
    # Verify both zero-initialized corrections exactly reproduce the same frozen baseline.
    witness_batch=next(iter(DataLoader(va,batch_size=4,shuffle=False,num_workers=0)))
    w=[x.cuda() for x in witness_batch]
    agri.eval();cot.eval()
    with torch.no_grad():
        base0=base_a(w[0],w[1],torch.zeros_like(w[2]),torch.zeros_like(w[3]),w[4],w[5])
        pa,da=agri(w[0],w[1],torch.zeros_like(w[2]),torch.zeros_like(w[3]),w[4],w[5])
        pc,dc=cot(*w[:6])
    assert torch.equal(da,torch.zeros_like(da)) and torch.equal(dc,torch.zeros_like(dc))
    assert torch.allclose(pa,base0,atol=1e-7,rtol=0) and torch.allclose(pc,base0,atol=1e-7,rtol=0)
    def run(model,name,use_cot):
        g=torch.Generator().manual_seed(a.seed+7)
        loader=DataLoader(tr,batch_size=a.batch,shuffle=True,num_workers=2,pin_memory=True,generator=g)
        valid=DataLoader(va,batch_size=512,shuffle=False,num_workers=2,pin_memory=True)
        # Step 0 is an exact fallback to the same frozen baseline.
        model.eval(); best,_=evaluate_arm(model,valid,use_cot); best=best['rmse']; beststep=0
        torch.save({'model':model.state_dict(),'step':0,'arm':name,'use_cot':use_cot},a.out/f'{name}_best.pt')
        curve=a.out/f'{name}_validation_curve.jsonl'
        curve.write_text(json.dumps({'arm':name,'step':0,'val_rmse':best,'best_step':0})+'\n')
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=1e-4)
        updates=0;epoch=0;losses=[]
        while updates<a.updates:
            model.train()
            for b in loader:
                if updates>=a.updates: break
                v=[x.cuda(non_blocking=True) for x in b]
                if not use_cot: v[2].zero_();v[3].zero_()
                opt.zero_grad(set_to_none=True)
                pred,delta=model(*v[:6])
                loss=(((pred-v[6])**2+a.delta_penalty*delta**2)*v[8]).sum()/v[8].sum().clamp_min(1)
                if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
                loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.,error_if_nonfinite=True);opt.step()
                updates+=1;losses.append(float(loss.detach().cpu()))
            epoch+=1
            met,_=evaluate_arm(model,valid,use_cot)
            key=met['rmse']
            if key<best:
                best=key;beststep=updates
                torch.save({'model':model.state_dict(),'step':updates,'epoch':epoch,'arm':name,'use_cot':use_cot},a.out/f'{name}_best.pt')
            rec={'arm':name,'step':updates,'epoch':epoch,'val_rmse':key,'best_val_rmse':best,'best_step':beststep,
                 'train_loss_mean_last100':float(np.mean(losses[-100:]))}
            with curve.open('a') as f:f.write(json.dumps(rec)+'\n')
            save(a.out/f'{name}_status.json',rec)
            print(json.dumps(rec),flush=True)
        ck=torch.load(a.out/f'{name}_best.pt',map_location='cuda',weights_only=False);model.load_state_dict(ck['model'])
        met,arr=evaluate_arm(model,valid,use_cot)
        np.savez_compressed(a.out/f'{name}_val_predictions.npz',**arr)
        save(a.out/f'{name}_metrics.json',met)
        return {'updates':updates,'best_step':int(ck['step']),'metrics':met,'final_loss_mean_last100':float(np.mean(losses[-100:]))}
    ra=run(agri,'agri_residual',False)
    rc=run(cot,'cot_residual',True)
    ar={k:np.load(a.out/f'{k}_val_predictions.npz') for k in ('agri_residual','cot_residual')}
    assert np.array_equal(ar['agri_residual']['row_ids'],ar['cot_residual']['row_ids'])
    assert np.array_equal(ar['agri_residual']['true_ghi'],ar['cot_residual']['true_ghi'])
    assert np.array_equal(ar['agri_residual']['lead'],ar['cot_residual']['lead'])
    # step-0 witness, common cohort, unchanged Sili anchor.
    witness={'same_trainable_parameters':n_agri==n_cot,'trainable_parameter_count':trainable,
      'same_frozen_zero_baseline_initialization':True,'same_COT_branch_initialization':bool(init_equal),
      'agri_control_COT_values_zeroed_but_causal_validity_mask_retained':True,
      'step0_exact_no_COT_fallback':True,'step0_agri_max_abs_difference_kt':float(torch.max(torch.abs(pa-base0)).cpu()),'step0_cot_max_abs_difference_kt':float(torch.max(torch.abs(pc-base0)).cpu()),'same_validation_rows_truth_and_leads':bool(np.array_equal(ar['agri_residual']['row_ids'],ar['cot_residual']['row_ids']) and np.array_equal(ar['agri_residual']['true_ghi'],ar['cot_residual']['true_ghi']) and np.array_equal(ar['agri_residual']['lead'],ar['cot_residual']['lead'])),
      'validation_rows':int(len(ar['agri_residual']['row_ids'])),'test_used':False,
      'Sili_checkpoint_sha256':__import__('hashlib').sha256(a.sili_checkpoint.read_bytes()).hexdigest()}
    save(a.out/'capacity_control_witness.json',witness)
    report={'state':'COMPLETE_ZHUJIA_RESIDUAL_CAPACITY_CONTROL_PILOT','station':'Zhujia','seed':a.seed,
      'train_sequences':int(len(np.unique(tr.rows.labels['sequence'][tr.rows.ids[tr.rows.labels['station'][tr.rows.ids]==1]]))),'val_sequences':int(len(np.unique(va.rows.labels['sequence'][va.rows.ids[va.rows.labels['station'][va.rows.ids]==1]]))),'actual_updates_per_arm':{'agri_residual':ra['updates'],'cot_residual':rc['updates']},'source_zero_checkpoint_sha256':__import__('hashlib').sha256((a.source_run/'zero_best.pt').read_bytes()).hexdigest(),'source_fusion_checkpoint_sha256':__import__('hashlib').sha256((a.source_run/'fusion_best.pt').read_bytes()).hexdigest(),'sili_checkpoint_sha256':__import__('hashlib').sha256(a.sili_checkpoint.read_bytes()).hexdigest(),'batch':a.batch,'lr':a.lr,
      'cot_limit_kt':a.cot_limit,'delta_penalty':a.delta_penalty,'arms':{'agri_residual':ra,'cot_residual':rc},
      'capacity_control':'Identical residual architecture/trainable parameter count and shared initial weights; AGri arm zeros COT values while retaining temporal validity masks; both consume the same frozen AGRI/geometry features.',
      'test_used':False,'station_id':1,'pilot_bank_contract':bank_contract.get('state'),'pilot_bank_complete_sha256':hashlib.sha256((a.bank/'complete.json').read_bytes()).hexdigest(),'sili_model_hash_preserved':sili_hash==expected_sili,'frozen_sili_checkpoint_sha256':sili_hash,'output_sha256':{name:hashlib.sha256((a.out/name).read_bytes()).hexdigest() for name in ['agri_residual_best.pt','cot_residual_best.pt','agri_residual_val_predictions.npz','cot_residual_val_predictions.npz']}}
    save(a.out/'complete.json',report);print('COMPLETE_ZHUJIA_RESIDUAL_CAPACITY_CONTROL_PILOT',flush=True)
if __name__=='__main__':main()
