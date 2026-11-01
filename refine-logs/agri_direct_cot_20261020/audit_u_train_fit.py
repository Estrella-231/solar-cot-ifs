"""Frozen original P-C/P-XC on a preselected training sample, without optimization."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from train_future_cot_u import DirectCotForecaster,predict_target_layout
from audit_cot_event_timing import read_csv,csvout,sha,STATIONS
from audit_teacher_forecast_error_decomposition import met
from audit_dense_reference_events import changes,SCALES

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--batch',type=int,default=128)
    a=ap.parse_args();b=a.base
    source=b/'data/u_train_fit_sample_v1_20261021';rec=json.loads((source/'complete.json').read_text())
    assert rec['state']=='COMPLETE_FIXED_U_TRAIN_FIT_SAMPLE'
    assert rec['source']['trainer_sha256']==sha(b/'train_future_cot_u.py')
    for n,h in rec['artifact_sha256'].items():assert sha(source/n)==h,n
    cases=read_csv(source/'rows.csv');assert len(cases)==1024
    arrays={n:np.load(source/(n+'.npy'),mmap_mode='r') for n in ('x','g','y','m')}
    assert np.isfinite(arrays['x']).all() and np.isfinite(arrays['g']).all()
    assert np.isfinite(arrays['y'][arrays['m']]).all() and arrays['y'][arrays['m']].min()>=0
    out=b/'results/u_train_fit_diagnostic_v1_20261021'
    if out.exists():raise FileExistsError(out)
    out.mkdir();torch.set_num_threads(2)
    assert torch.cuda.is_available();device=torch.device('cuda')
    torch.cuda.reset_peak_memory_stats();began=time.monotonic()
    predictions={};provenance={}
    for arm,folder,h in [('cot_p_c','p_c_frozen_model_20261021','7258ea13ea210347f94a49174ec2e6ac88a241463ec74220aa9d30a2c7ba1b37'),
                         ('cot_p_xc','p_xc_frozen_model_20261021','4b926f3a94d2228505d9acce0a7d36adbd90cbd8ef221d9db02045b35fea50df')]:
        root=b/'data'/folder;ckpt=root/'epoch_012.pt';assert sha(ckpt)==h
        payload=torch.load(ckpt,map_location='cpu',weights_only=False)
        assert payload['epoch']==12
        assert payload['metadata']['p_code_sha256']==rec['source']['trainer_sha256']
        assert payload['metadata']['u_cache_manifest_sha256']==rec['source']['u_cache_manifest_sha256']
        model=DirectCotForecaster().to(device).eval().requires_grad_(False)
        model.load_state_dict(payload['model'],strict=True);del payload
        result=np.lib.format.open_memmap(out/(arm+'.npy'),mode='w+',dtype='float32',shape=(1024,16,1,16,16))
        for lo in range(0,1024,a.batch):
            hi=min(lo+a.batch,1024)
            xx=np.array(arrays['x'][lo:hi],copy=True)
            if arm=='cot_p_c':xx[:,:,:13]=0
            xt=torch.from_numpy(xx).to(device);gt=torch.from_numpy(np.array(arrays['g'][lo:hi],copy=True)).to(device)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                pp=predict_target_layout(model,xt,gt).float().cpu().numpy()
            assert pp.shape==(hi-lo,16,1,16,16) and np.isfinite(pp).all() and pp.min()>=0
            result[lo:hi]=pp
        result.flush();predictions[arm]=result;provenance[arm]=h;del model
        print('U_INFERENCE_COMPLETE',arm,'elapsed',time.monotonic()-began,flush=True)
    metricrows=[];change=[];bins=[]
    for st in (*STATIONS,'both'):
        selected=np.array([r['station']==st or st=='both' for r in cases])
        for scale,(rs,cs) in SCALES.items():
            yy=np.asarray(arrays['y'][selected,:,0,rs,cs]);mm=np.asarray(arrays['m'][selected,:,0,rs,cs],bool)
            for arm in predictions:
                pp=np.asarray(predictions[arm][selected,:,0,rs,cs])
                for li in [None]+list(range(16)):
                    p,y,m=(pp,yy,mm) if li is None else (pp[:,li],yy[:,li],mm[:,li])
                    e=np.log1p(p[m].astype(float))-np.log1p(y[m].astype(float))
                    huber=float(np.mean(np.where(np.abs(e)<.1,.5*e*e,.1*(np.abs(e)-.05)))) if len(e) else None
                    metricrows.append({'partition':'U_training_sample','scale':scale,'station':st,'lead_minutes':'all' if li is None else 15*(li+1),
                                       'arm':arm,**met(p,y,m),'log1p_huber_delta_0p1':huber})
                if scale in ('center1','station3'):
                    change.append({'partition':'U_training_sample','scale':scale,'station':st,'arm':arm,**changes(pp,yy,mm)})
                for name,lo,hi in [('nearzero',0,.1),('thin',.1,5),('moderate',5,30),('thick',30,float('inf'))]:
                    mask=mm&(yy>=lo)&(yy<hi)
                    bins.append({'partition':'U_training_sample','scale':scale,'station':st,'arm':arm,'reference_bin':name,**met(pp,yy,mask)})
    csvout(out/'u_field_metrics.csv',metricrows);csvout(out/'u_first_difference_metrics.csv',change);csvout(out/'u_reference_strata.csv',bins)
    # Keep V rows as previously computed descriptors, never as a new confirmation.
    v=[{'partition':'V_development','scale':r['scale'],'station':r['station'],'lead_minutes':r['lead_minutes'],
        'arm':r['arm'],**{k:v for k,v in r.items() if k not in ('reference','scale','station','lead_minutes','arm')}}
       for r in read_csv(b/'results/dense_r_field_event_diagnostic_v1_20261021/field_metrics.csv') if r['arm'] in predictions]
    csvout(out/'u_v_descriptive_comparison.csv',metricrows+v)
    result={'state':'COMPLETE_FROZEN_U_TRAIN_FIT_DIAGNOSTIC','station_windows':1024,'u_sequences':512,
            'batch':a.batch,'inference_dtype':'CUDA BF16 autocast; no optimizer or gradient',
            'device':torch.cuda.get_device_name(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'elapsed_seconds':time.monotonic()-began,'checkpoints_sha256':provenance,
            'source':{'sample_receipt_sha256':sha(source/'complete.json'),'script_sha256':sha(Path(__file__)),
                'plan_sha256':sha(b/'U_TRAIN_FIT_DIAGNOSTIC_PLAN_20261021.md'),
                'v_descriptors_sha256':sha(b/'results/dense_r_field_event_diagnostic_v1_20261021/field_metrics.csv')},
            'training_started':False,'ghi_labels_read':False,'test_payload_opened':False,'buffer_payload_opened':False,
            'ghi_gain_established':False,'scope':'training-fit diagnostic only; different sample distributions and FP16/FP32 R references prohibit unbiased U-V generalization or gain claims',
            'artifacts_sha256':{p.name:sha(p) for p in out.iterdir() if p.suffix in ('.npy','.csv')}}
    (out/'complete.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([r for r in metricrows if r['station']=='both' and r['scale']=='station3' and r['lead_minutes']=='all'],indent=2),flush=True)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
