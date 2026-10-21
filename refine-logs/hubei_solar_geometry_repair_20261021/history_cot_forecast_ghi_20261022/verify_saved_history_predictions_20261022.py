"""Independently bind saved predictions to source labels and recompute paired RMSE."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE/'receipts_heartbeat_20261022T0117'
STATIONS=('Niushou','Wangzhai','Mizhuang')
SEEDS=(42,43,44)
ARMS=('FA_HIST','FAC_HIST')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def keys(z):
    return list(zip(z['station'].astype(str),z['init_time_bjt'].astype(str),z['target_time_bjt'].astype(str),z['lead_minutes'].astype(int).tolist()))

def main():
    out=ROOT/'independent_saved_prediction_check.json'
    if out.exists():raise FileExistsError(out)
    pack=ROOT/'pack_truth_binding';pr=read(pack/'pack_report.json')
    assert pr['test_and_buffer_values_read']==pr['test_payloads_read']==pr['buffer_payloads_read']==0
    for name in ('ghi_wm2.npy','clear_sky_wm2.npy'):
        assert sha(pack/name)==pr['arrays_sha256'][name]
    with (pack/'rows.csv').open(encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f))
    assert {r['split'] for r in rr}=={'train','val'}
    truth=np.load(pack/'ghi_wm2.npy',allow_pickle=False)
    clear=np.load(pack/'clear_sky_wm2.npy',allow_pickle=False)
    valmap={ (r['station'],r['init_time_bjt'],r['target_time_bjt'],int(r['lead_minutes'])):i for i,r in enumerate(rr) if r['split']=='val'}
    assert len(valmap)==sum(r['split']=='val' for r in rr)
    source=read(ROOT/'score_receipt.json');published=read(ROOT/'global_summary.json')
    assert sha(ROOT/'pack_truth_binding/pack_report.json')==source['pack_report_sha256']
    assert sha(ROOT/'orchestrator_receipt.json')==source['source_training_receipt_sha256']
    pairs={};individual=[];hashes={}
    for st in STATIONS:
        for seed in SEEDS:
            base=ROOT/'pairs'/f'{st}_seed{seed}';m=read(base/'run_metadata.json')
            assert m['state']=='HISTORY_COT_PAIRED_TRAINING_COMPLETE'
            assert m['epochs']==30 and m['batch_size']==1536
            assert m['pair_initialization_equal'] and m['pair_row_order_equal'] and m['pair_updates_equal']
            assert m['test_payloads_read']==m['buffer_payloads_read']==0
            saved={}
            for arm in ARMS:
                result=read(base/(arm+'_result.json'));predpath=base/(arm+'_validation_predictions.npz')
                assert sha(predpath)==result['prediction_sha256']
                assert sha(base/(arm+'_best.pt'))==result['checkpoint_sha256']
                assert sha(base/(arm+'_last.pt'))==result['last_checkpoint_sha256']
                assert result['actual_updates']==m['actual_updates_per_arm']
                assert result['initialization_sha256']==m['initialization_sha256']
                assert result['row_order_sha256']==m['row_order_sha256']
                assert result['checkpoint_reload_parity']=='PASS'
                with np.load(predpath,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
                kk=keys(d);assert len(kk)==len(set(kk))
                expected={k for k in valmap if k[0]==st};assert set(kk)==expected
                assert set(d['split'].astype(str))=={'val'}
                ix=np.array([valmap[k] for k in kk],dtype=np.int64)
                # Trainer carries the source arrays to GPU as float32.
                assert np.array_equal(d['truth_ghi_wm2'].astype(np.float64),np.asarray(truth[ix],np.float32).astype(np.float64))
                assert np.array_equal(d['clear_sky_wm2'].astype(np.float64),np.asarray(clear[ix],np.float32).astype(np.float64))
                err=d['pred_ghi_wm2'].astype(np.float64)-d['truth_ghi_wm2'].astype(np.float64)
                assert np.isfinite(err).all()
                rmse=float(np.sqrt(np.mean(err**2)))
                assert abs(rmse-result['best_validation_fp64']['rmse_wm2'])<1e-10
                saved[arm]={'data':d,'keys':kk,'err':err,'rmse':rmse}
                hashes[str(predpath.relative_to(ROOT))]=sha(predpath)
            assert saved[ARMS[0]]['keys']==saved[ARMS[1]]['keys']
            individual.append({'station':st,'seed':seed,'n':len(kk),
                'baseline_rmse':saved[ARMS[0]]['rmse'],'history_cot_rmse':saved[ARMS[1]]['rmse'],
                'delta_cot_minus_baseline':saved[ARMS[1]]['rmse']-saved[ARMS[0]]['rmse'],
                'updates_per_arm':m['actual_updates_per_arm']})
            pairs[(st,seed)]=saved
    dates=sorted({str(t)[:8] for p in pairs.values() for t in p[ARMS[0]]['data']['target_time_bjt']})
    assert dates==published['bootstrap_blocks']
    sse=np.zeros((2,3,3,len(dates)),np.float64);counts=np.zeros_like(sse)
    for si,st in enumerate(STATIONS):
        for ri,seed in enumerate(SEEDS):
            for ai,arm in enumerate(ARMS):
                item=pairs[(st,seed)][arm]
                dd=np.array([str(x)[:8] for x in item['data']['target_time_bjt']])
                for di,date in enumerate(dates):
                    valid=dd==date
                    sse[ai,si,ri,di]=np.sum(item['err'][valid]**2)
                    counts[ai,si,ri,di]=valid.sum()
    assert np.array_equal(counts[0],counts[1])
    rng=np.random.default_rng(20261022);delta=[]
    for _ in range(5000):
        draw=rng.integers(len(dates),size=len(dates))
        multiplicity=np.bincount(draw,minlength=len(dates))
        denom=np.einsum('asrd,d->asr',counts,multiplicity)
        numer=np.einsum('asrd,d->asr',sse,multiplicity)
        assert (denom>0).all()
        met=np.sqrt(numer/denom).mean(axis=(1,2))
        delta.append(met[1]-met[0])
    ci=np.quantile(delta,[.025,.975])
    assert np.allclose(ci,published['paired_bjt_date_block_bootstrap_95pct_delta_ci_wm2'],atol=1e-10,rtol=0)
    base=float(np.mean([x['baseline_rmse'] for x in individual]));cot=float(np.mean([x['history_cot_rmse'] for x in individual]))
    assert abs(base-published['equal_station_equal_seed_mean_rmse_wm2'][ARMS[0]])<1e-10
    assert abs(cot-published['equal_station_equal_seed_mean_rmse_wm2'][ARMS[1]])<1e-10
    stations=[]
    for st in STATIONS:
        vv=[x for x in individual if x['station']==st]
        b=float(np.mean([x['baseline_rmse'] for x in vv]));c=float(np.mean([x['history_cot_rmse'] for x in vv]))
        stations.append({'station':st,'n_per_seed':vv[0]['n'],'baseline':b,'history_cot':c,
                         'relative_improvement_pct':100*(b-c)/b,'seeds_improving':sum(x['delta_cot_minus_baseline']<0 for x in vv)})
    report={'state':'PASS_INDEPENDENT_SAVED_PREDICTION_LABEL_BINDING_AND_FP64_RECOMPUTATION',
            'source_pack_labels_match':True,'eighteen_prediction_hashes_match':True,
            'best_and_last_checkpoint_hashes_match':True,'matched_initialization_order_updates':True,
            'raw_validation_rows_per_seed':len(valmap),'pairs':individual,'stations':stations,
            'baseline_rmse':base,'history_cot_rmse':cot,'delta':cot-base,'improvement_pct':100*(base-cot)/base,
            'date_block_ci95':ci.tolist(),'bootstrap_dates':len(dates),'all_pairs_improve':all(x['delta_cot_minus_baseline']<0 for x in individual),
            'stable_gain_proven':False,'forecast_cot_input_used':False,'history_cot_input_used':True,
            'independent_test_used':False,'test_payloads_read':0,'buffer_payloads_read':0,
            'prediction_sha256':hashes,'script_sha256':sha(Path(__file__))}
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['state','stations','baseline_rmse','history_cot_rmse','improvement_pct','date_block_ci95','all_pairs_improve']},ensure_ascii=False))

if __name__=='__main__':main()
