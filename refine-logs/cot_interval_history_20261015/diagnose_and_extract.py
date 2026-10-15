"""Frozen-head history perturbation and four actual validation cloud cases; CPU only."""
import argparse,csv,json
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
import torch
from train_paired import IntervalHead

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--cases',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True)
    torch.set_num_threads(4);pack=a.root/'data_attempt2';run=a.root/'results_attempt1';v=np.load(run/'validation_inputs.npz');ids=v['pack_index'];rows=list(csv.DictReader((run/'validation_rows.csv').open()))
    valid=np.asarray(np.load(pack/'agri_valid.npy',mmap_mode='r')[ids]).astype(bool);meta=json.loads((pack/'pack_manifest.json').read_text())
    am=np.asarray(meta['AGRI_norm_mean'])[None,None,:,None,None];sd=np.asarray(meta['AGRI_norm_std'])[None,None,:,None,None]
    x=np.where(valid,(v['agri_raw']-am)/sd,0).astype(np.float32);g=v['geometry'].astype(np.float32);c=((v['cot_log1p'].astype(np.float32)-meta['COT_norm_mean'])/meta['COT_norm_std'])
    if c.ndim==4:c=c[:,:,None]
    stat=[];arrays={}
    for seed in (42,43,44):
        for st in ('sili','zhujia'):
            keep=np.array([r['station']==st for r in rows]);sub=np.flatnonzero(keep);model=IntervalHead();model.load_state_dict(torch.load(run/f'seed{seed}'/st/'AC_history/best.pt',map_location='cpu')['model']);model.eval()
            cc=torch.from_numpy(c[keep]);xx=torch.from_numpy(x[keep]);gg=torch.from_numpy(g[keep]);cl=torch.tensor([float(r['clear']) for r in np.array(rows,dtype=object)[keep]])
            with torch.inference_mode():
                actual=(model(xx,gg,cc)*cl).numpy().astype(float);repeated=(model(xx,gg,cc[:,-1:].repeat(1,3,1,1,1))*cl).numpy().astype(float)
            orig=np.load(run/f'seed{seed}'/st/'AC_history/validation_predictions.npz');np.testing.assert_allclose(actual,orig['cpu_prediction'],atol=.001,rtol=0)
            y=orig['truth'];stat.append(dict(seed=seed,station=st,actual_COT_rmse=float(np.sqrt(np.mean((actual-y)**2))),repeat_current_COT_rmse=float(np.sqrt(np.mean((repeated-y)**2))),mean_abs_GHI_change=float(np.mean(np.abs(actual-repeated))),p90_abs_GHI_change=float(np.quantile(np.abs(actual-repeated),.9))))
            arrays[f'{st}_{seed}_actual']=actual;arrays[f'{st}_{seed}_repeat_current']=repeated
    np.savez_compressed(a.output/'history_perturbation.npz',**arrays)
    (a.output/'history_perturbation.json').write_text(json.dumps(dict(state='COMPLETE_FROZEN_HEAD_HISTORY_INTERVENTION',runs=stat,note='Repeating current COT in older slots is an input intervention, not a separately trained static-COT baseline; same CPU baseline used',test_payloads_read=0),indent=2))
    base=a.root.parent/'monthly_R_station16_20261013_v1';bank=base/'results/new_monthly_R_seed42_cot_bank_v1';fr=list(csv.DictReader((bank/'frame_keys.csv').open()));lookup={(r['split'],r['target_time_utc']):int(r['index']) for r in fr};cot=np.load(bank/'cot_log1p.npy',mmap_mode='r')
    data=Path('/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed/data');scans=json.loads((base/'interval_alignment_20261015/manifest_full_v1/scan_offsets.json').read_text());cases=json.loads(a.cases.read_text());out={};receipt=[]
    for ci,case in enumerate(cases):
        st=case['station'];si=0 if st=='sili' else 1;rr,cc=(163,148) if st=='sili' else (73,150);t=datetime.fromisoformat(case['utc'])
        for k in (30,15,0):
            tt=t-timedelta(minutes=k);utc=tt.isoformat(sep=' ');bi=lookup[('val',utc)];path=data/tt.strftime('%Y/%Y%m/%Y%m%d')/(tt.strftime('%Y%m%d%H%M')+'.npy')
            xx=np.load(path,mmap_mode='r');out[f'c{ci}_m{k}_rgb_channels']=np.asarray(xx[[0,1,2],rr-8:rr+8,cc-8:cc+8]);out[f'c{ci}_m{k}_cot']=np.expm1(cot[bi,si,0])
            receipt.append(dict(case=ci,offset_minutes=-k,utc=utc,source=str(path),R_bank_index=bi,scan_offset_seconds=scans[utc][si]))
    np.savez_compressed(a.output/'actual_cloud_cases.npz',**out);(a.output/'case_source_manifest.json').write_text(json.dumps(dict(cases=cases,frames=receipt,test_payloads_read=0),indent=2));print(json.dumps(stat))

if __name__=='__main__':main()
