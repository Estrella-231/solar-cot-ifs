"""Independent FP64 GHI scoring and shared target-date paired bootstrap."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
STATIONS=('Niushou','Wangzhai','Mizhuang');SEEDS=(42,43,44);ARMS=('FA','FAC')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def writecsv(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metrics(y,p):
    e=p-y;return {'n':len(y),'rmse':float(np.sqrt(np.mean(e*e,dtype=np.float64))),'mae':float(np.mean(np.abs(e),dtype=np.float64)),'bias':float(np.mean(e,dtype=np.float64))}
def main():
    p=argparse.ArgumentParser()
    for n in ('experiment','pack','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    pr=json.loads((a.pack/'pack_report.json').read_text());assert pr['correctutc_cot_rebuilt'] and pr['canonical_time_binding_repaired'] and pr['solar_geometry_corrected']
    rows=list(csv.DictReader((a.pack/'rows.csv').open()));y=np.load(a.pack/'ghi_wm2.npy',allow_pickle=False).astype(np.float64);cl=np.load(a.pack/'clear_sky_wm2.npy',allow_pickle=False).astype(np.float64)
    support={s:[i for i,r in enumerate(rows) if r['split']=='val' and r['station']==s] for s in STATIONS}
    dates=sorted({r['target_time_bjt'][:8] for r in rows if r['split']=='val'});assert len(dates)==22
    data={};records=[];leadrecords=[];checks=[]
    for seed in SEEDS:
        run=a.experiment/f'paired_seed{seed}_attempt1';meta=json.loads((run/'run_metadata.json').read_text());assert meta['state']=='HUBEI_GHI_HEADS_SEED_COMPLETE' and meta['epochs']==30 and meta['batch_size']==1536
        assert meta['pack_sha256']==sha(a.pack/'pack_report.json') and meta['test_payloads_read']==0
        for s in STATIONS:
            expected=[rows[i] for i in support[s]];pair=[]
            for arm in ARMS:
                d=run/s;report=json.loads((d/f'{arm}_validation.json').read_text());bundle=d/f'{arm}_validation_predictions_with_keys.npz'
                assert report['prediction_bundle_sha256']==sha(bundle) and report['checkpoint_sha256']==sha(d/f'{arm}_best.pt')
                assert report['checkpoint_reload_parity']=='PASS' and report['parameter_count']==24489
                train_n=sum(r['split']=='train' and r['station']==s for r in rows);assert report['actual_updates']==30*((train_n+1535)//1536)
                with np.load(bundle,allow_pickle=False) as z:b={k:z[k] for k in z.files}
                keys=list(zip(b['station'].tolist(),b['init_time_bjt'].tolist(),b['target_time_bjt'].tolist(),b['lead_minutes'].tolist()))
                assert keys==[(r['station'],r['init_time_bjt'],r['target_time_bjt'],int(r['lead_minutes'])) for r in expected]
                assert np.array_equal(b['truth_ghi_wm2'],y[support[s]])
                assert np.max(np.abs(b['clear_sky_ghi_wm2']-cl[support[s]]))<1e-4
                assert b['pred_ghi_wm2'].dtype==np.float64 and np.isfinite(b['pred_ghi_wm2']).all()
                m=metrics(b['truth_ghi_wm2'],b['pred_ghi_wm2']);assert abs(m['rmse']-report['rmse_wm2'])<1e-8
                records.append({'station':s,'seed':seed,'arm':arm,**m,'updates':report['actual_updates'],'best_epoch':report['best_epoch']})
                for lead in range(15,241,15):
                    take=b['lead_minutes']==lead;leadrecords.append({'station':s,'seed':seed,'arm':arm,'lead_minutes':lead,**metrics(b['truth_ghi_wm2'][take],b['pred_ghi_wm2'][take])})
                data[seed,s,arm]=b;pair.append(report)
            assert pair[0]['initial_state_sha256']==pair[1]['initial_state_sha256'] and pair[0]['epoch_order_sha256']==pair[1]['epoch_order_sha256']
            checks.append({'station':s,'seed':seed,'same_initial_weights':True,'same_epoch_orders':True,'updates_each':pair[0]['actual_updates'],'validation_rows':len(expected)})
    cells=[(seed,s) for seed in SEEDS for s in STATIONS];daily_n=np.zeros((len(dates),len(cells)));daily_sse={arm:np.zeros_like(daily_n) for arm in ARMS}
    for j,(seed,s) in enumerate(cells):
        for arm in ARMS:
            b=data[seed,s,arm];d=np.asarray([x[:8] for x in b['target_time_bjt']]);e=b['pred_ghi_wm2']-b['truth_ghi_wm2']
            for i,dt in enumerate(dates):
                take=d==dt;daily_n[i,j]=take.sum();daily_sse[arm][i,j]=np.sum(e[take]**2,dtype=np.float64)
    totals={arm:float(np.mean([r['rmse'] for r in records if r['arm']==arm])) for arm in ARMS}
    cis={}
    for mode in ('single_target_date','consecutive_two_day_blocks'):
        groups=[[i] for i in range(len(dates))] if mode=='single_target_date' else [list(range(i,min(i+2,len(dates)))) for i in range(0,len(dates),2)]
        if mode!='single_target_date':
            from datetime import datetime,timedelta
            assert all(datetime.strptime(dates[g[1]],'%Y%m%d')-datetime.strptime(dates[g[0]],'%Y%m%d')==timedelta(days=1) for g in groups)
        rng=np.random.default_rng(20261020);draw=rng.integers(0,len(groups),size=(10000,len(groups)));weights=np.zeros((10000,len(dates)))
        for j,g in enumerate(groups):weights[:,g]=np.sum(draw==j,axis=1)[:,None]
        count=weights@daily_n;valid=(count>0).all(axis=1);sc={arm:np.sqrt((weights@daily_sse[arm])[valid]/count[valid]).mean(axis=1) for arm in ARMS}
        delta=sc['FAC']-sc['FA'];gain=100*(sc['FA']-sc['FAC'])/sc['FA']
        cis[mode]={'resamples':10000,'valid_resamples':int(valid.sum()),'groups':[[dates[k] for k in g] for g in groups],'FAC_minus_FA_ci95':np.quantile(delta,[.025,.975]).tolist(),'relative_gain_percent_ci95':np.quantile(gain,[.025,.975]).tolist()}
    stationtable=[]
    for s in STATIONS:
        vals={arm:float(np.mean([r['rmse'] for r in records if r['station']==s and r['arm']==arm])) for arm in ARMS}
        stationtable.append({'station':s,**vals,'FAC_minus_FA':vals['FAC']-vals['FA'],'gain_percent':100*(vals['FA']-vals['FAC'])/vals['FA']})
    leadtable=[]
    for lead in range(15,241,15):
        vals={arm:float(np.mean([r['rmse'] for r in leadrecords if r['lead_minutes']==lead and r['arm']==arm])) for arm in ARMS}
        leadtable.append({'lead_minutes':lead,**vals,'FAC_minus_FA':vals['FAC']-vals['FA'],'gain_percent':100*(vals['FA']-vals['FAC'])/vals['FA']})
    summary={'state':'CORRECTUTC_FORECAST_PAIRED_FP64_RESCORING_COMPLETE','input_mode':'corrected-solar SimVP predicted future AGRI -> corrected-solar correct-UTC R -> forecast GHI','scope':'reused validation; not independent test','rmse_equal_station_equal_seed':totals,'FAC_minus_FA':totals['FAC']-totals['FA'],'relative_gain_percent':100*(totals['FA']-totals['FAC'])/totals['FA'],'stations':stationtable,'target_dates':dates,'bootstrap':cis,'pair_checks':checks,'pack_report_sha256':sha(a.pack/'pack_report.json'),'test_payloads_read':0,'buffer_payloads_read':0}
    writecsv(a.output/'per_station_seed_arm.csv',records);writecsv(a.output/'per_station.csv',stationtable);writecsv(a.output/'per_lead.csv',leadtable);writecsv(a.output/'per_station_seed_lead.csv',leadrecords)
    (a.output/'summary_fp64.json').write_text(json.dumps(summary,indent=2))
    lines=['# Corrected-UTC COT: paired future GHI validation','','Historical 8 AGRI frames -> frozen SimVP future AGRI -> corrected-UTC R COT -> 15-minute GHI. No future observed AGRI/CPP input; no test/buffer payloads.','','Same architecture (24489 parameters), same initial weights, sample orders and 30-epoch training budget; seed42/43/44. Reused validation selected best checkpoints; intervals are conditional validation evidence.','','| Station | Without COT | With forecast COT | Difference (W/m2) | Relative RMSE reduction |','|---|---:|---:|---:|---:|']
    for r in stationtable:lines.append(f"| {r['station']} | {r['FA']:.3f} | {r['FAC']:.3f} | {r['FAC_minus_FA']:+.3f} | {r['gain_percent']:.2f}% |")
    lines+=['',f"Equal-station/equal-seed RMSE: {totals['FA']:.6f} -> {totals['FAC']:.6f} W/m2; relative reduction {summary['relative_gain_percent']:.3f}%.",'',f"Paired target-date 95% difference interval: {cis['single_target_date']['FAC_minus_FA_ci95']}; consecutive-two-day interval: {cis['consecutive_two_day_blocks']['FAC_minus_FA_ci95']}.",'','Full lead results include all +15..+240 min, including exact +60/+120/+180/+240 points. Do not select only favorable stations/leads. This matched-capacity baseline differs from the historical smaller FA head; new-vs-old changes cannot be attributed solely to R repair.']
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
