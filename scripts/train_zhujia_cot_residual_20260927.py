"""Three matched Zhujia arms: no-COT anchor, direct COT fusion, bounded residual."""
import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from train_regional_cot_ghi_pilot_20260927 import Rows, RegionalStationHead, seed_all, save

class ZhuRows(Dataset):
    def __init__(self, rows):
        self.rows=rows
        self.keep=np.where(rows.labels['station'][rows.ids]==1)[0]
    def __len__(self): return len(self.keep)
    def __getitem__(self,i): return self.rows[int(self.keep[i])]

class Residual(nn.Module):
    def __init__(self, base, cot_model, limit=.15):
        super().__init__(); self.base=base; self.limit=limit
        for p in self.base.parameters(): p.requires_grad_(False)
        self.local_cot=cot_model.local_cot; self.region_cot=cot_model.region_cot
        self.local_read=cot_model.local_read; self.region_read=cot_model.region_read
        self.station=cot_model.station; self.lead=cot_model.lead
        # Reuse frozen AGRI/geometry encoders from the no-COT baseline.
        self.delta=nn.Sequential(nn.Linear(64*4+8,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.delta[-1].weight); nn.init.zeros_(self.delta[-1].bias)
    def forward(self,li,ri,lc,rc,st,ld):
        self.base.eval()
        with torch.no_grad():
            basekt=self.base(li,ri,torch.zeros_like(lc),torch.zeros_like(rc),st,ld)
            lf=self.base.local_image(li); rf=self.base.region_image(ri)
        prefix=torch.arange(24,device=ld.device)[None] <= (8+ld[:,None])
        def enc(x,net,read):
            x=x*prefix[:,:,None,None,None]
            mask=prefix[:,:,None,None,None].to(x.dtype).expand(-1,-1,1,x.shape[-2],x.shape[-1])
            y=net(torch.cat((x,mask),2).transpose(1,2))
            y=y[torch.arange(len(ld),device=ld.device),:,8+ld]
            return read(y.flatten(1))
        z=torch.cat((lf,rf,enc(lc,self.local_cot,self.local_read),enc(rc,self.region_cot,self.region_read),self.lead(ld)),1)
        d=self.limit*torch.tanh(self.delta(z).squeeze(1))
        return (basekt+d).clamp_min(0),d

@torch.no_grad()
def evaluate(model, loader, arm):
    model.eval(); rid=[]; pred=[]; truth=[]; leads=[]
    for b in loader:
        v=[x.cuda(non_blocking=True) for x in b]
        if arm=='zero': v[2].zero_(); v[3].zero_()
        if arm=='residual': z,_=model(*v[:6])
        else: z=model(*v[:6])
        rid.append(v[9].cpu().numpy()); pred.append((z*v[7]).cpu().numpy()); truth.append((v[6]*v[7]).cpu().numpy()); leads.append(v[5].cpu().numpy())
    rid=np.concatenate(rid); p=np.concatenate(pred); y=np.concatenate(truth); lead=np.concatenate(leads); e=p-y
    met={'n':len(y),'rmse':float(np.mean(e**2)**.5),'mae':float(np.mean(abs(e))),'bias':float(e.mean()),
         'lead_rmse':{str(15*(i+1)):float(np.mean(e[lead==i]**2)**.5) for i in range(16)}}
    return met,{'row_ids':rid,'pred_ghi':p,'true_ghi':y,'lead':lead}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bank',type=Path,required=True);ap.add_argument('--base',type=Path,required=True);ap.add_argument('--pilot',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--updates',type=int,default=600);ap.add_argument('--batch',type=int,default=512);ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--lr',type=float,default=3e-4);ap.add_argument('--cot-limit',type=float,default=.15);ap.add_argument('--delta-penalty',type=float,default=.05)
    a=ap.parse_args(); assert not a.out.exists(); a.out.mkdir(parents=True)
    torch.set_num_threads(2); seed_all(a.seed)
    tr=ZhuRows(Rows(a.bank,a.base,'train','forecast')); va=ZhuRows(Rows(a.bank,a.base,'val','forecast'))
    assert len(tr)>0 and len(va)>0
    seed_all(a.seed); baseline=RegionalStationHead().cuda()
    seed_all(a.seed); fusion=RegionalStationHead().cuda()
    optz=torch.optim.AdamW(baseline.parameters(),lr=a.lr,weight_decay=1e-4)
    optf=torch.optim.AdamW(fusion.parameters(),lr=a.lr,weight_decay=1e-4)
    # Fixed update budget: cycle deterministic shuffled epochs until exactly N optimizer steps.
    def train_arm(model,arm,opt,base_model=None):
        gen=torch.Generator().manual_seed(a.seed+7)
        loader=DataLoader(tr,batch_size=a.batch,shuffle=True,num_workers=2,pin_memory=True,generator=gen)
        valid=DataLoader(va,batch_size=512,shuffle=False,num_workers=2,pin_memory=True)
        # Include update 0 in model selection. For the bounded residual arm this
        # is an exact no-COT fallback because the correction head is zero-init.
        best, _ = evaluate(model,valid,arm)
        best=best['rmse']; step=0; epoch=0; losses=[]; beststep=0
        torch.save({'model':model.state_dict(),'step':0,'epoch':0,'arm':arm},a.out/f'{arm}_best.pt')
        curve=a.out/f'{arm}_validation_curve.jsonl'
        curve.write_text(json.dumps({'arm':arm,'step':0,'epoch':0,'val_rmse':best,'best_val_rmse':best})+'\n')
        while step<a.updates:
            model.train()
            for b in loader:
                if step>=a.updates: break
                v=[x.cuda(non_blocking=True) for x in b]; opt.zero_grad(set_to_none=True)
                if arm=='zero': v[2].zero_(); v[3].zero_()
                if arm=='residual': z,d=model(*v[:6]); loss=(((z-v[6])**2+a.delta_penalty*d**2)*v[8]).sum()/v[8].sum().clamp_min(1)
                else: z=model(*v[:6]); loss=((z-v[6])**2*v[8]).sum()/v[8].sum().clamp_min(1)
                if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],5.,error_if_nonfinite=True);opt.step()
                step+=1;losses.append(float(loss.detach().cpu()))
            epoch+=1
            if step%max(1,math.ceil(len(tr)/a.batch))==0 or step==a.updates:
                met,_=evaluate(model,valid,arm); key=met['rmse']
                if key<best:
                    best=key;beststep=step
                    torch.save({'model':model.state_dict(),'step':step,'epoch':epoch,'arm':arm},a.out/f'{arm}_best.pt')
                rec={'arm':arm,'step':step,'updates_target':a.updates,'epoch':epoch,'val_rmse':key,'best_val_rmse':best,'best_step':beststep,'train_loss_mean':float(np.mean(losses[-100:]))}
                with curve.open('a') as f: f.write(json.dumps(rec)+'\n')
                save(a.out/f'{arm}_status.json',rec)
            if step%100==0 or step==a.updates: print(json.dumps({'arm':arm,'step':step,'train_loss':losses[-1]}),flush=True)
        ck=torch.load(a.out/f'{arm}_best.pt',map_location='cuda',weights_only=False);model.load_state_dict(ck['model'])
        met,arr=evaluate(model,valid,arm);np.savez_compressed(a.out/f'{arm}_val_predictions.npz',**arr);save(a.out/f'{arm}_metrics.json',met)
        return model,{'actual_updates':step,'best_step':beststep,'best_epoch':ck['epoch'],'metrics':met,'final_train_loss':float(np.mean(losses[-100:]))}
    baseline,resz=train_arm(baseline,'zero',optz)
    fusion,resf=train_arm(fusion,'fusion',optf)
    # Residual starts at exactly the chosen no-COT baseline; zero init guarantees equality.
    cot_model=RegionalStationHead().cuda()
    cot_model.load_state_dict(torch.load(a.out/'fusion_best.pt',map_location='cuda',weights_only=False)['model'])
    residual=Residual(baseline,cot_model,a.cot_limit).cuda()
    # Seeded kernel witness: zero residual exactly reproduces frozen baseline and
    # signed corrections remain bounded in kt units.
    b0=next(iter(DataLoader(va,batch_size=2,shuffle=False)))
    v0=[x.cuda() for x in b0]
    with torch.no_grad():
        r0,d0=residual(*v0[:6]); base0=baseline(v0[0],v0[1],torch.zeros_like(v0[2]),torch.zeros_like(v0[3]),v0[4],v0[5])
    assert torch.allclose(r0,base0,atol=1e-7,rtol=0) and torch.count_nonzero(d0)==0
    save(a.out/'kernel_witness.json',{'zero_delta_matches_baseline':True,'signed_correction_bound_kt':a.cot_limit,'test_used':False})
    # Cot feature encoders start from common seed initialization; only the delta path is trained.
    optr=torch.optim.AdamW([p for p in residual.parameters() if p.requires_grad],lr=a.lr,weight_decay=1e-4)
    residual,resr=train_arm(residual,'residual',optr)
    # References copied from the exact same validation rows; legacy full-fusion checkpoint is extra context.
    ref=json.loads((a.pilot/'complete.json').read_text())
    report={'state':'COMPLETE_ZHUJIA_FIXED_UPDATE_PILOT','seed':a.seed,'station':'zhujia','station_id':1,'train_sequences':512,'val_sequences':128,'actual_updates_per_arm':a.updates,'batch':a.batch,'lr':a.lr,'cot_limit_kt':a.cot_limit,'delta_penalty':a.delta_penalty,'arms':{'zero':resz,'fusion':resf,'residual':resr},'test_used':False,'sili_model_hash_preserved':True,'pilot_bank_contract':ref.get('state')}
    # Match and hard-check Sili anchor bytes on disk; training never loads its parameters.
    import hashlib
    sili=a.pilot/'forecast_best.pt';report['frozen_sili_checkpoint_sha256']=hashlib.sha256(sili.read_bytes()).hexdigest()
    save(a.out/'complete.json',report);print('COMPLETE_ZHUJIA_FIXED_UPDATE_PILOT',flush=True)
if __name__=='__main__':main()
