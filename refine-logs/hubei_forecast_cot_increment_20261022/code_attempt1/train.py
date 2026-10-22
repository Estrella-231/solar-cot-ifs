"""Four-arm matched GHI experiment; official validation never selects epochs."""
import argparse,csv,json,hashlib,time,os,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from history_base import HistoryCOTGHIHead,sha256,array_sha256

class ConditionalHead(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.gate=nn.Sequential(nn.Linear(dim,16),nn.SiLU(),nn.Linear(16,1))
        self.delta=nn.Sequential(nn.Linear(dim,16),nn.SiLU(),nn.Linear(16,1))
        nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)
    def forward(self,z,base,available):
        delta=torch.sigmoid(self.gate(z)).squeeze(-1)*self.delta(z).squeeze(-1)
        return torch.clamp(base+available*delta,min=0)

def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepared',type=Path,required=True);ap.add_argument('--station',required=True);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--out',type=Path);ap.add_argument('--profile',type=Path);a=ap.parse_args()
    assert torch.cuda.is_available();torch.cuda.set_device(0);torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);np.random.seed(a.seed)
    prep=json.loads((a.prepared/'PREPARATION.json').read_text());assert prep['state']=='COMPLETE_F_C_SPLIT_AND_FROZEN_INPUTS'
    root=a.prepared/a.station;meta=json.loads((root/'meta.json').read_text());rows=list(csv.DictReader(open(root/'rows.csv')))
    roles=np.load(root/'roles.npy');data={}
    for k in ['future','history','cot','ghi','clear','fusion_features','forecast_available']:
        v=np.load(root/(k+'.npy'));assert np.isfinite(v).all();data[k]=torch.from_numpy(v).to('cuda')
    ids={k:np.where(roles==k)[0] for k in ['F_fit','F_select','C_fit','C_select','val']}
    BATCH=1536;EPOCHS=30
    def base_forward(model,ii,usecot):return model(data['future'][ii],data['history'][ii],data['cot'][ii] if usecot else torch.zeros_like(data['cot'][ii]))
    if a.profile:
        model=HistoryCOTGHIHead().cuda();opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        ix=torch.tensor(np.resize(ids['F_fit'],BATCH),device='cuda')
        torch.cuda.synchronize();start=None
        for k in range(76):
            if k==12:torch.cuda.synchronize();start=time.perf_counter()
            pred=base_forward(model,ix,True);loss=(((pred*data['clear'][ix]-data['ghi'][ix])/1000)**2).mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        torch.cuda.synchronize();seconds=time.perf_counter()-start
        write(a.profile,{'examples_per_second':BATCH*64/seconds,'updates':64,'batch':BATCH,'gpu':torch.cuda.get_device_name(0),'seconds':seconds,'peak_allocated':torch.cuda.max_memory_allocated(),'kernel_witness':'finite forward/backward on actual F_fit inputs','test_payloads_read':0,'buffer_payloads_read':0});return
    a.out.mkdir(parents=True,exist_ok=False)
    write(a.out/'run_config.json',{'station':a.station,'seed':a.seed,'epochs':EPOCHS,'batch':BATCH,'lr':1e-3,'weight_decay':1e-4,'eta_min':1e-5,'selection':'G0/G1 F_select; G2/G3 C_select only','data_meta':meta,'trainer_sha256':sha256(__file__),'gpu':torch.cuda.get_device_name(0),'physical_gpu':os.getenv('CUDA_VISIBLE_DEVICES'),'test_payloads_read':0,'buffer_payloads_read':0})
    rng=np.random.default_rng(a.seed)
    orders={phase:[rng.permutation(ids[phase]) for _ in range(EPOCHS)] for phase in ['F_fit','C_fit']}
    oh={p:hashlib.sha256(np.concatenate(v).astype(np.int64).tobytes()).hexdigest() for p,v in orders.items()}
    models={};predictions={};receipts={}
    def predict(model,forward,index):
        model.eval();pred=[]
        with torch.no_grad():
            for start in range(0,len(index),BATCH):
                ix=torch.tensor(index[start:start+BATCH],dtype=torch.long,device='cuda');pred.append(forward(model,ix).detach())
        return torch.cat(pred)
    def fit(arm,model,forward,phase,select):
        init_hash=array_sha256(model.state_dict());opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS,eta_min=1e-5)
        best=float('inf');best_state=None;best_epoch=0;updates=0;logs=[];started=time.time()
        for epoch,order in enumerate(orders[phase],1):
            model.train();sse=0.;seen=0
            for start in range(0,len(order),BATCH):
                ix=torch.tensor(order[start:start+BATCH],dtype=torch.long,device='cuda');pred=forward(model,ix)
                loss=(((pred*data['clear'][ix]-data['ghi'][ix])/1000)**2).mean()
                assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
                updates+=1;sse+=float(loss.detach())*len(ix);seen+=len(ix)
            sch.step();q=predict(model,forward,ids[select]);truth=data['ghi'][ids[select]];cl=data['clear'][ids[select]]
            score=float(torch.sqrt(((q.double()*cl.double()-truth.double())**2).mean()))
            logs.append({'epoch':epoch,'fit_mse_scaled':sse/seen,'inner_rmse_wm2':score,'inner_role':select,'updates':updates})
            if score<best:best=score;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sch.state_dict(),'epoch':epoch,'updates':updates,'order_sha256':oh[phase],'init_sha256':init_hash,'best_model':best_state,'best_epoch':best_epoch},a.out/(arm+'_last.pt'))
            (a.out/(arm+'_epochs.jsonl')).write_text('\n'.join(json.dumps(x) for x in logs)+'\n')
            if epoch==1 or epoch%5==0:print(json.dumps({'arm':arm,**logs[-1]}),flush=True)
        assert updates==EPOCHS*((len(ids[phase])+BATCH-1)//BATCH)
        model.load_state_dict(best_state);torch.save({'model':best_state,'epoch':best_epoch,'seed':a.seed,'station':a.station,'inner_role':select,'data_meta':meta},a.out/(arm+'_best.pt'))
        # Checkpoint parity on inner data; official validation is not touched here.
        witness=predict(model,forward,ids[select][:BATCH]);reload_model=copy.deepcopy(model)
        reload_model.load_state_dict(torch.load(a.out/(arm+'_best.pt'),map_location='cuda',weights_only=False)['model'])
        replay=predict(reload_model,forward,ids[select][:BATCH]);assert torch.equal(witness,replay)
        receipts[arm]={'updates':updates,'order_sha256':oh[phase],'initialization_sha256':init_hash,'best_epoch':best_epoch,'inner_role':select,'inner_rmse_wm2':best,'checkpoint_sha256':sha256(a.out/(arm+'_best.pt')),'parameters':sum(p.numel() for p in model.parameters()),'checkpoint_reload_max_delta':float(abs(witness-replay).max()),'elapsed_seconds':time.time()-started}
        model.eval();return model
    template=HistoryCOTGHIHead().cuda();initial=copy.deepcopy(template.state_dict());del template
    for arm,usecot in [('G0',False),('G1',True)]:
        model=HistoryCOTGHIHead().cuda();model.load_state_dict(initial)
        fn=lambda m,ix,usecot=usecot:base_forward(m,ix,usecot)
        models[arm]=fit(arm,model,fn,'F_fit','F_select')
        predictions[arm]=predict(models[arm],fn,np.arange(len(rows)))
    assert receipts['G0']['initialization_sha256']==receipts['G1']['initialization_sha256']
    frozen_hash=array_sha256(models['G1'].state_dict())
    for m in models.values():
        for p in m.parameters():p.requires_grad_(False)
    base=predictions['G1'].detach();features=torch.cat([data['fusion_features'],base[:,None]],1)
    z0=features.clone();lo,hi=meta['cot_feature_slot'];z0[:,lo:hi]=0
    c0=ConditionalHead(features.shape[1]).cuda();ci=copy.deepcopy(c0.state_dict());del c0
    witness={}
    for arm,z in [('G2',z0),('G3',features)]:
        model=ConditionalHead(z.shape[1]).cuda();model.load_state_dict(ci)
        fn=lambda m,ix,z=z:m(z[ix],base[ix],data['forecast_available'][ix])
        wi=torch.tensor(ids['C_fit'][:BATCH],device='cuda')
        with torch.no_grad():delta=float(abs(fn(model,wi)-base[wi]).max())
        assert delta==0;witness[arm+'_zero_init_delta']=delta
        # Mask fallback must be exactly the frozen base even after a nonzero residual.
        probe=copy.deepcopy(model)
        with torch.no_grad():probe.delta[-1].bias.fill_(.2);back=probe(z[wi],base[wi],torch.zeros_like(base[wi]));assert torch.equal(back,base[wi])
        models[arm]=fit(arm,model,fn,'C_fit','C_select')
        predictions[arm]=predict(models[arm],fn,np.arange(len(rows)))
    assert array_sha256(models['G1'].state_dict())==frozen_hash
    assert receipts['G2']['initialization_sha256']==receipts['G3']['initialization_sha256']
    assert receipts['G2']['parameters']==receipts['G3']['parameters']
    assert receipts['G2']['order_sha256']==receipts['G3']['order_sha256']
    assert receipts['G2']['updates']==receipts['G3']['updates']
    # Export validation once all checkpoints are frozen; scoring is separate CPU code.
    vi=ids['val'];vr=[rows[i] for i in vi]
    common={k:np.array([r[k] for r in vr]) for k in ['station','init_time_bjt','target_time_bjt','lead_minutes','source_role']}
    common.update(truth_ghi_wm2=data['ghi'][vi].cpu().numpy().astype(np.float64),clear_sky_wm2=data['clear'][vi].cpu().numpy().astype(np.float64))
    for arm,pred in predictions.items():
        out=(pred[vi].double()*data['clear'][vi].double()).cpu().numpy();assert np.isfinite(out).all()
        np.savez_compressed(a.out/(arm+'_validation_predictions.npz'),**common,pred_ghi_wm2=out)
    write(a.out/'COMPLETE.json',{'state':'FOUR_ARM_COMPLETE','station':a.station,'seed':a.seed,'arms':receipts,'witness':witness,'base_unchanged':True,'test_payloads_read':0,'buffer_payloads_read':0})
    print('FOUR_ARM_COMPLETE',a.station,a.seed,flush=True)
if __name__=='__main__':main()
