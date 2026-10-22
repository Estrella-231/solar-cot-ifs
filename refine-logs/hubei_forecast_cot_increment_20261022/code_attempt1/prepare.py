"""Build only new train-date roles and reuse immutable corrected forecast inputs."""
import argparse,csv,json,calendar,hashlib
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
from history_base import prep_data,sha256,STATIONS,CENTERS,OFFSETS

def date_range(a,b):
    return [(a+timedelta(days=k)).strftime('%Y%m%d') for k in range((b-a).days+1)]
def choose_roles(rows):
    days=sorted({r['target_time_bjt'][:8] for r in rows if r['split']=='train'})
    months=sorted({d[:6] for d in days});blocks=[];forbidden=set();cfit=set();csel=set();fsel=set()
    for month in months:
        ds=[d for d in days if d.startswith(month)];have=set(ds)
        pairs=[d for d in ds if (datetime.strptime(d,'%Y%m%d')+timedelta(days=1)).strftime('%Y%m%d') in have]
        if not pairs:raise ValueError('No consecutive C dates in '+month)
        first=min(pairs,key=lambda d:(abs(int(d[-2:])-15),d));t=datetime.strptime(first,'%Y%m%d')
        second=(t+timedelta(days=1)).strftime('%Y%m%d');cfit.add(first);csel.add(second)
        forbidden.update(date_range(t-timedelta(days=1),t+timedelta(days=2)))
        blocks.append({'month':month,'C_fit':first,'C_select':second})
    remaining=set(days)-forbidden
    for month in months:
        ds=sorted(d for d in remaining if d.startswith(month))
        if not ds:raise ValueError('No F selection date '+month)
        chosen=min(ds,key=lambda d:(abs(int(d[-2:])-7),d));fsel.add(chosen)
    fsel_buffer=set()
    for d in fsel:
        t=datetime.strptime(d,'%Y%m%d');fsel_buffer.update(date_range(t-timedelta(days=1),t+timedelta(days=1)))
    role=[]
    for r in rows:
        if r['split']=='val':role.append('val');continue
        assert r['split']=='train'
        init=datetime.strptime(r['init_time_bjt'],'%Y%m%d%H%M');target=datetime.strptime(r['target_time_bjt'],'%Y%m%d%H%M')
        footprint=set(date_range(init-timedelta(minutes=105),target));d=target.strftime('%Y%m%d')
        assert int((target-init).total_seconds()/60)==int(r['lead_minutes'])
        if len(footprint)==1 and d in cfit:role.append('C_fit')
        elif len(footprint)==1 and d in csel:role.append('C_select')
        elif len(footprint)==1 and d in fsel and not footprint&forbidden:role.append('F_select')
        elif not footprint&(forbidden|fsel_buffer):role.append('F_fit')
        else:role.append('purged')
    return np.array(role),{'monthly_C_blocks':blocks,'F_select_dates':sorted(fsel),'base_excluded_dates':sorted(forbidden|fsel_buffer),'selection':'metadata-only: C consecutive train dates nearest day15, then F selection train date nearest day7; one-day purges'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pack',type=Path,required=True);ap.add_argument('--bank',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--finalize-existing',action='store_true');a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=a.finalize_existing)
    rows=list(csv.DictReader(open(a.pack/'rows.csv')));roles,selection=choose_roles(rows)
    report={'state':'PREPARING','selection':selection,'source_pack_report_sha256':sha256(a.pack/'pack_report.json'),'source_bank_report_sha256':sha256(a.bank/'bank_report.json'),'stations':{},'test_payloads_read':0,'buffer_payloads_read':0,'upstream_note':'existing train forecasts are five-fold S/R OOF; validation uses final full S/R; historical R is full R on both. New dates isolate GHI stages only, not an unseen full chain.'}
    if a.finalize_existing:
        old=list(csv.DictReader(open(a.out/'row_roles.csv')))
        assert len(old)==len(rows) and [r['new_role'] for r in old]==roles.tolist()
        for st in STATIONS:
            path=a.out/st;meta=json.loads((path/'meta.json').read_text())
            for k,v in meta['arrays'].items():assert sha256(path/(k+'.npy'))==v
            assert sha256(path/'rows.csv')==meta['row_sha256'];report['stations'][st]=meta
        report.update(state='COMPLETE_F_C_SPLIT_AND_FROZEN_INPUTS',roles_sha256=sha256(a.out/'row_roles.csv'),script_sha256=sha256(__file__),base_source_sha256=sha256(Path(__file__).with_name('history_base.py')),finalized_after_failed_job='215680.tc6000: all arrays written; str-vs-Path final hash receipt error; values not rebuilt')
        (a.out/'PREPARATION.json').write_text(json.dumps(report,indent=2));print('PREPARATION_FINALIZED_EXISTING_HASHES_PASS',flush=True);return
    with (a.out/'row_roles.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])+['new_role']);w.writeheader();w.writerows([{**r,'new_role':str(role)} for r,role in zip(rows,roles)])
    keyrows=list(csv.DictReader(open(a.bank/'frame_keys.csv')));frame={r['bjt']:int(r['frame_index']) for r in keyrows}
    hv=np.load(a.bank/'history_cot_valid_16x16.npy',mmap_mode='r');fc=np.load(a.pack/'cot.npy',mmap_mode='r')
    for st in STATIONS:
        indices=np.array([i for i,r in enumerate(rows) if r['station']==st]);rr=roles[indices]
        d=prep_data(a.pack,a.bank,st,fit_rows=rr=='F_fit');sr=d['rows']
        for role in ['F_fit','F_select','C_fit','C_select','val']:
            count=int((rr==role).sum());dates=len({r['target_time_bjt'][:8] for r,v in zip(sr,rr==role) if v})
            if count<100 or dates<5:raise ValueError((st,role,count,dates))
        assert len(set(r['target_time_bjt'][:8] for r,v in zip(sr,rr=='C_fit') if v)&set(r['target_time_bjt'][:8] for r,v in zip(sr,rr=='C_select') if v))==0
        forecast=np.asarray(fc[indices],np.float32)
        if forecast.shape!=(len(sr),1,3,3):raise ValueError(forecast.shape)
        valid=np.isfinite(forecast)&(forecast>=0)&(forecast<=100)
        clog=np.log1p(np.where(valid,forecast,0).astype(np.float64))
        values=clog[(rr=='C_fit')[:,None,None,None]&valid];mu=float(values.mean());sd=max(float(values.std()),1e-6)
        fnorm=np.where(valid,(clog-mu)/sd,0).astype(np.float32)
        cy,cx=CENTERS[st];hm=np.empty((len(sr),8,1,3,3),bool)
        for i,r in enumerate(sr):
            init=datetime.strptime(r['init_time_bjt'],'%Y%m%d%H%M')
            ids=[frame[(init-timedelta(minutes=o)).strftime('%Y%m%d%H%M')] for o in OFFSETS]
            hm[i]=hv[ids,:,cy-1:cy+2,cx-1:cx+2]
        hist=d['cot'].numpy();latest=hist[:,-1]*d['norm']['history_cot_log1p_std']+d['norm']['history_cot_log1p_mean']
        diff=np.where(valid&hm[:,-1],(clog-latest)/sd,0).astype(np.float32)
        feature=np.concatenate([d['future'].numpy().reshape(len(sr),-1),hist.reshape(len(sr),-1),hm.reshape(len(sr),-1),fnorm.reshape(len(sr),-1),diff.reshape(len(sr),-1),valid.reshape(len(sr),-1),np.array([int(r['lead_minutes'])/240 for r in sr])[:,None]],1).astype(np.float32)
        # Only these 18 slots differ between G2 and G3. Masks are shared.
        cot_start=135+72+72;cot_stop=cot_start+18
        path=a.out/st;path.mkdir()
        arrays={k:d[k].numpy() for k in ['future','history','cot','ghi','clear']}
        arrays.update(fusion_features=feature,forecast_available=valid.reshape(len(sr),-1).all(1).astype(np.float32),roles=rr)
        for k,v in arrays.items():
            if v.dtype.kind=='f' and not np.isfinite(v).all():raise ValueError('nonfinite '+k)
            np.save(path/(k+'.npy'),v,allow_pickle=False)
        with (path/'rows.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(sr[0]));w.writeheader();w.writerows(sr)
        meta={'base_norm':d['norm'],'future_cot_norm':{'fit':'C_fit only','log1p_mean':mu,'log1p_std':sd},'role_counts':{v:int((rr==v).sum()) for v in sorted(set(rr))},'role_dates':{v:len({r['target_time_bjt'][:8] for r,b in zip(sr,rr==v) if b}) for v in sorted(set(rr))},'cot_feature_slot':[cot_start,cot_stop],'fusion_feature_dim':feature.shape[1]+1,'row_sha256':sha256(path/'rows.csv'),'arrays':{k:sha256(path/(k+'.npy')) for k in arrays},'test_payloads_read':0,'buffer_payloads_read':0}
        (path/'meta.json').write_text(json.dumps(meta,indent=2));report['stations'][st]=meta
        print(json.dumps({'station':st,'counts':meta['role_counts'],'dates':meta['role_dates']}),flush=True)
    report.update(state='COMPLETE_F_C_SPLIT_AND_FROZEN_INPUTS',roles_sha256=sha256(a.out/'row_roles.csv'),script_sha256=sha256(__file__),base_source_sha256=sha256(Path(__file__).with_name('history_base.py')))
    (a.out/'PREPARATION.json').write_text(json.dumps(report,indent=2));print('PREPARATION_COMPLETE',flush=True)
if __name__=='__main__':main()
