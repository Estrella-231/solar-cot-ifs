"""Recompute current no-IFS station metrics from saved validation predictions."""
import csv, hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
OUT=ROOT/'audits/current_station_table_no_ifs_20260912.json'
CSV=ROOT/'audits/current_station_table_no_ifs_20260912.csv'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def metric(pred,truth):
    err=pred-truth;mean=float(truth.mean())
    sst=float(np.sum((truth-mean)**2));assert mean>0 and sst>0
    rmse=float(np.sqrt(np.mean(err**2)));mae=float(np.mean(abs(err)))
    return dict(r2=float(1-np.sum(err**2)/sst),rmse_wm2=rmse,
        nrmse_pct=100*rmse/mean,nmae_pct=100*mae/mean,mae_wm2=mae,
        mean_observed_wm2=mean,n=int(len(truth)))

def main():
    assert not OUT.exists() and not CSV.exists()
    pack=ROOT/'data/head_pack_trainval_20260908_v1';audit=json.loads((pack/'audit.json').read_text())
    assert audit['state']=='COMPLETE' and audit['test_used'] is False
    arrays={}
    for key in ('split','station','ghi','clear'):
        spec=audit['arrays'][key];path=pack/spec['name'];assert sha(path)==spec['sha256']
        arrays[key]=np.load(path,allow_pickle=False)
    val=np.flatnonzero(arrays['split']==1);assert len(val)==99849
    rows=[];sources={}
    station_names=('sili','zhujia');station_cn=('四里','竺家')
    for arm,label in (('A','FY4B外推'),('AC','FY4B外推 + COT云属性')):
        for seed in (42,43,44):
            dirname='head_ghi_original_seed42_20260910_v1' if seed==42 else f'head_ghi_multiseed_original_seed{seed}_20260910_v1'
            run=ROOT/'runs'/dirname/f'{arm}_lr0.0003_seed{seed}'
            done=json.loads((run/'complete.json').read_text());path=run/'validation_predictions.npz'
            assert done['test_used'] is False and sha(path)==done['predictions_sha256']
            sources[f'{arm}_seed{seed}']=dict(predictions=str(path),sha256=sha(path),checkpoint_sha256=done['checkpoint_sha256'])
            with np.load(path,allow_pickle=False) as z:
                assert np.array_equal(z['pack_row'],val)
                for name,key in (('observed_ghi','ghi'),('clear_sky_ghi','clear'),('station','station')):
                    assert np.array_equal(z[name],arrays[key][val])
                pred=z['pred_kt'].astype(np.float64)*arrays['clear'][val]
            for sid,(station,cn) in enumerate(zip(station_names,station_cn)):
                mask=arrays['station'][val]==sid;m=metric(pred[mask],arrays['ghi'][val][mask])
                rows.append(dict(station=station,station_cn=cn,method=arm,method_label=label,seed=seed,**m))
    summary=[]
    for sid,(station,cn) in enumerate(zip(station_names,station_cn)):
        for arm,label in (('A','FY4B外推'),('AC','FY4B外推 + COT云属性')):
            chosen=[r for r in rows if r['station']==station and r['method']==arm]
            item=dict(station=station,station_cn=cn,method=arm,method_label=label,n=chosen[0]['n'],seeds=[42,43,44])
            for key in ('r2','rmse_wm2','nrmse_pct','nmae_pct','mae_wm2','mean_observed_wm2'):
                values=np.array([r[key] for r in chosen]);item[key]=dict(values=values.tolist(),mean=float(values.mean()),std_ddof1=float(values.std(ddof=1)))
            summary.append(item)
    report=dict(state='COMPLETE_CURRENT_NO_IFS_STATION_TABLE_METRICS',test_used=False,
        scope='Hunan validation; all canonical +15 to +240 min forecast rows; original cohort fixed lr0.0003; saved seeds42/43/44',
        methods={'A':'frozen SimVP forecast AGRI plus geometry/station/lead, no COT','AC':'same plus four predicted log1p(COT) patch statistics'},
        definitions={'r2':'1-SSE/SST within the same station rows','rmse_wm2':'sqrt(mean(error^2))',
            'nrmse_pct':'RMSE/mean(observed GHI in same station rows)*100','nmae_pct':'MAE/mean(observed GHI in same station rows)*100',
            'uncertainty':'mean and sample SD across model-training seeds42/43/44'},
        source_pack_sha256=sha(pack/'audit.json'),sources=sources,rows=rows,summary=summary,
        limitations=['validation not test','forecast AGRI, not future observed AGRI','IFS absent','three seeds; no significance test','original cohort only; QC1 is sensitivity analysis'])
    OUT.write_text(json.dumps(report,indent=2,allow_nan=False))
    fields=['station','station_cn','method','method_label','seed','n','mean_observed_wm2','r2','rmse_wm2','mae_wm2','nrmse_pct','nmae_pct']
    with CSV.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
    print(json.dumps({'state':report['state'],'summary':summary}))
if __name__=='__main__':main()
