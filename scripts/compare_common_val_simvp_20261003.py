"""Physical-unit old/new SimVP comparison on train-disjoint common validation.

No test payloads. R is shared/frozen; real-AGRI->R is a retrieval proxy.
Cases are selected from observations, never from candidate improvement.
"""
import argparse,csv,json,hashlib,sys,time
from pathlib import Path
from datetime import datetime,timedelta
from functools import lru_cache
import numpy as np
import torch

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--solar',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();a.output.mkdir(exist_ok=False)
    solar=a.solar;base=solar.parent
    new=solar/'experiments/hunan_monthly_block_20261002'
    old_manifest=base/'HunanSimVPFull_20260906/manifests_v1/hunan_8to16.csv'
    new_manifest=new/'manifests_v1/hunan_8to16.csv'
    old_rows=list(csv.DictReader(old_manifest.open()))
    new_rows=list(csv.DictReader(new_manifest.open()))
    train={x for rows in (old_rows,new_rows) for r in rows if r['split']=='train' for x in r['data_relpaths'].split('|')}
    ov={r['data_relpaths'] for r in old_rows if r['split']=='val'}
    common=[r for r in new_rows if r['split']=='val' and r['data_relpaths'] in ov and not (set(r['data_relpaths'].split('|'))&train)]
    assert len(common)==242
    # Fixed six daytime init hours per common date, before any predictions.
    groups={}
    for r in common:
        init=datetime.fromisoformat(r['BJT_start'])+timedelta(minutes=105)
        groups.setdefault(init.strftime('%Y-%m-%d'),[]).append((init,r))
    selected=[]
    for day,values in sorted(groups.items()):
        used=set()
        for hour in (8,9,10,11,12,13):
            desired=datetime.fromisoformat(day)+timedelta(hours=hour)
            t,r=min(values,key=lambda item:abs((item[0]-desired).total_seconds()))
            if r['seq_id'] not in used and abs((t-desired).total_seconds())<=1800:
                used.add(r['seq_id']);selected.append(r)
    assert selected
    with (a.output/'selected_windows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=selected[0].keys());w.writeheader();w.writerows(selected)
    sys.path.insert(0,str(new/'code'))
    from triad_models import SimVPBackbone
    from hunan_data import load_frame
    sys.path.insert(0,str(solar/'scripts'))
    from train_cot_repaired import COTUNet
    data=Path('/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed')
    weights={'old':base/'HunanBackboneTriad_20260921/geometry24_v2/runs/simvp_geometry24_formal_e12_v1/best.pt',
             'new':new/'runs/simvp_seed42_v1/training/best.pt'}
    norms={'old':base/'HunanSimVPFull_20260906/manifests_v1/hunan_train_norm.json',
           'new':new/'manifests_v1/hunan_train_norm.json'}
    models={};stats={}
    for name in weights:
        ck=torch.load(weights[name],map_location='cpu',weights_only=False)
        assert ck['metadata']['geometry_contract']=='history_8_then_target_16_v2'
        assert ck['metadata']['normalization_sha256']==sha(norms[name])
        assert ck['metadata']['manifest_sha256']==sha(old_manifest if name=='old' else new_manifest)
        assert ck['metadata']['model_code_sha256']==sha(new/'code/triad_models.py')
        m=SimVPBackbone(new/'code').cuda().eval().requires_grad_(False)
        m.load_state_dict(ck['model'],strict=True);models[name]=m
        z=json.loads(norms[name].read_text())
        stats[name]=(np.array(z['mean'],np.float32)[:,None,None],np.array(z['std'],np.float32)[:,None,None])
    rdir=solar/'experiments/regional_station_cot_20260925/r256_full_exploratory_seed42_batch512_node22'
    pair=solar/'experiments/regional_station_cot_20260925/agri_cpp_pair_bank_v1'
    rc=torch.load(rdir/'best.pt',map_location='cpu',weights_only=False)
    assert rc['metadata']['normalization_sha256']==sha(pair/'norm.json')
    assert rc['metadata']['r_arch_sha256']==sha(solar/'scripts/train_cot_repaired.py')
    r=COTUNet().cuda().eval().requires_grad_(False);r.load_state_dict(rc['model'],strict=True)
    rn=json.loads((pair/'norm.json').read_text())
    rm=torch.tensor(rn['mean'],device='cuda')[:,None,None]
    rs=torch.tensor(rn['std'],device='cuda')[:,None,None]
    cpp_lookup={}
    for shard in json.loads((pair/'index.json').read_text())['shards']:
        d=pair/shard['directory']
        for rec in json.loads((d/'records.json').read_text()):
            if rec['split']=='val' and rec['included_for_R']:cpp_lookup[rec['utc']]=(str(d),rec['array_row'])
    @lru_cache(maxsize=4)
    def cpp_arrays(d):return np.load(Path(d)/'cot.npy',mmap_mode='r'),np.load(Path(d)/'mask.npy',mmap_mode='r')
    @lru_cache(maxsize=128)
    def frame(rel):return load_frame(data,rel)
    def retrieve(fields,geom):
        raw=np.concatenate((fields,geom),1)
        x=torch.tensor(raw,device='cuda');x=torch.nan_to_num((x-rm)/rs,nan=0.,posinf=0.,neginf=0.)
        return torch.cat([r(t).float().expm1() for t in x.split(8)]).cpu().numpy()[:,0]
    def rgb(fields):
        # Fixed C03/C02/C01 false-colour proxy, reflectance 0..0.8, gamma .6.
        z=np.transpose(fields[:,[2,1,0]],(0,2,3,1))
        return (np.clip(np.nan_to_num(z)/.8,0,1)**.6*255).astype(np.uint8)
    accum={name:{'sse':np.zeros(13),'n':np.zeros(13)} for name in models}
    stations={'sili':(163,148),'zhujia':(73,150)}
    records=[];best_cases={};start=time.monotonic()
    with torch.no_grad():
        for ix,row in enumerate(selected):
            paths=row['data_relpaths'].split('|');frames=[frame(p) for p in paths]
            fields=np.stack([x[0] for x in frames]);valid=np.stack([x[1] for x in frames]);geom=np.stack([x[2] for x in frames])
            preds={};gt=torch.tensor(geom[None],device='cuda')
            for name,m in models.items():
                mu,sd=stats[name];history=np.where(valid[:8],(fields[:8]-mu)/sd,0)
                with torch.autocast('cuda',dtype=torch.bfloat16):z=m(torch.tensor(history[None],device='cuda'),gt)
                preds[name]=z[0].float().cpu().numpy()*sd+mu
                take=valid[8:]&(geom[8:,2:3]>.5)
                e=np.where(take,preds[name]-fields[8:],0).astype(np.float64)
                accum[name]['sse']+=np.square(e).sum((0,2,3));accum[name]['n']+=take.sum((0,2,3))
            cot={'observed':retrieve(np.where(valid[8:],fields[8:],np.nan),geom[8:])}
            cot.update({name:retrieve(value,geom[8:]) for name,value in preds.items()})
            cp=np.full((16,256,256),np.nan,np.float32);cm=np.zeros_like(cp,bool)
            for ti,p in enumerate(paths[8:]):
                key=Path(p).stem
                if key in cpp_lookup:
                    d,j=cpp_lookup[key];ca,ma=cpp_arrays(d);cp[ti]=ca[j];cm[ti]=ma[j]
            init=datetime.fromisoformat(row['BJT_start'])+timedelta(minutes=105)
            for st,(sr,sc) in stations.items():
                sl=(slice(None),slice(sr-32,sr+32),slice(sc-32,sc+32))
                mask=valid[8:].all(1)&(geom[8:,0]>np.cos(np.deg2rad(80)))
                localmask=mask[sl]
                rec={'seq_id':row['seq_id'],'station':st,'init_bjt':str(init),'cot_reference':'same frozen R(real future AGRI); CPP separately where valid','series':{}}
                local={k:v[sl] for k,v in cot.items()}
                for name,v in local.items():
                    center=np.where(mask[:,sr,sc],v[:,32,32],np.nan)
                    near=np.where(localmask[:,30:35,30:35],v[:,30:35,30:35],np.nan)
                    maxima=np.array([float(np.nanmax(t)) if np.isfinite(t).any() else np.nan for t in near])
                    peaks=[]
                    for t,ok in zip(v,localmask):
                        if not ok.any():peaks.append([None,None]);continue
                        rr,cc=np.unravel_index(np.argmax(np.where(ok,t,-np.inf)),t.shape);peaks.append([int(rr-32),int(cc-32)])
                    hit=np.flatnonzero(np.isfinite(center)&(center>=10))
                    rec['series'][name]={'station_cot':center.tolist(),'peak5_cot':maxima.tolist(),'peak64_offset_rc':peaks,
                       'first_station_cot_ge10_min':int((hit[0]+1)*15) if len(hit) else None,
                       'peak5_value':float(np.nanmax(maxima)) if np.isfinite(maxima).any() else None,
                       'peak5_time_min':int((np.nanargmax(maxima)+1)*15) if np.isfinite(maxima).any() else None}
                    if name!='observed':
                        diff=v-local['observed'];rec[name+'_cot_proxy_rmse']=float(np.sqrt(np.mean(diff[localmask]**2))) if localmask.any() else None
                cppmask=cm[sl];localcpp=cp[sl]
                rec['series']['cpp']={'station_cot':np.where(cm[:,sr,sc],cp[:,sr,sc],np.nan).tolist()}
                for name,v in local.items():
                    rec[name+'_cpp_rmse']=float(np.sqrt(np.mean((v-localcpp)[cppmask]**2))) if cppmask.any() else None
                records.append(rec)
                # One illustrative case per date/station: maximum observed local 5x5 COT.
                score=rec['series']['observed']['peak5_value']
                key=(init.strftime('%Y-%m-%d'),st)
                if score is not None and (key not in best_cases or score>best_cases[key][0]):
                    payload={'observed_rgb':rgb(fields[8:]),'old_rgb':rgb(preds['old']),'new_rgb':rgb(preds['new']),
                       'observed_cot':local['observed'].copy(),'old_cot':local['old'].copy(),'new_cot':local['new'].copy(),
                       'cpp_cot':np.where(cppmask,localcpp,np.nan).copy(),'valid':localmask.copy()}
                    best_cases[key]=(score,rec,payload)
            print('PAIRED_WINDOW',ix+1,len(selected),str(init),round(time.monotonic()-start,1),flush=True)
    cases=[]
    for (day,st),(_,rec,payload) in sorted(best_cases.items()):
        name='case_'+day+'_'+st+'.npz';np.savez_compressed(a.output/name,**payload)
        cases.append({'file':name,'record':rec,'selection':'largest observed 5x5 COT among fixed-hour windows of this date; not selected by model difference'})
    report={'state':'COMPLETE_COMMON_VALIDATION_DIAGNOSTIC','common_windows':len(common),'evaluated_windows':len(selected),
       'dates':list(groups),'checkpoint_sha256':{k:sha(v) for k,v in weights.items()},'r_checkpoint_sha256':sha(rdir/'best.pt'),
       'normalization_sha256':{k:sha(v) for k,v in norms.items()},'model_code_sha256':sha(new/'code/triad_models.py'),
       'old_manifest_sha256':sha(old_manifest),'new_manifest_sha256':sha(new_manifest),'script_sha256':sha(__file__),
       'train_frame_overlap':0,'test_payloads_read':False,'device':torch.cuda.get_device_name(0),'elapsed_seconds':time.monotonic()-start,
       'rgb':'C03/C02/C01 false-colour proxy; fixed reflectance [0,.8], gamma .6; uint8 for display only',
       'channels':['C01','C02','C03','C04','C05','C06','C09','C10','C11','C12','C13','C14','C15'],
       'physical_pixel_rmse':{k:np.sqrt(v['sse']/v['n']).tolist() for k,v in accum.items()},'cases':cases,'records':records,
       'limitations':['common validation, not independent test','single seed; subset fixed before prediction','R(real AGRI) is retrieval proxy, not physical truth','peak location sequence is not identified cloud-object tracking','arrival threshold10 is diagnostic, initial-cloud cases are left-censored']}
    (a.output/'complete.json').write_text(json.dumps(report,indent=2,allow_nan=True)+'\n')
    print(report['state'],flush=True)

if __name__=='__main__':main()
