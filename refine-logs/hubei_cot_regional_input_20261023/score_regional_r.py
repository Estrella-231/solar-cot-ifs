"""Independent FP64 physical-COT scores from saved monthly-val outputs.

Reference is original stored CPP COT, not inverse-transformed training targets.
All reference strata are retained; this scores retrieval, not forecasting/GHI.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--pack',type=Path,required=True);ap.add_argument('--cpp-ready',type=Path,required=True)
    ap.add_argument('--roles',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();assert not a.out.exists()
    complete=json.loads((a.run/'training_complete.json').read_text())
    assert complete['state']=='R_REGION_TRAIN_COMPLETE_VAL_PREDICTIONS_SAVED'
    rows=list(csv.DictReader((a.pack/'rows.csv').open()))
    roles={r['time_bjt']:r['role'] for r in json.loads(a.roles.read_text())['roles']}
    mask=np.load(a.pack/'mask.npy',mmap_mode='r')
    expected={i for i,r in enumerate(rows) if roles[r['time_bjt']]=='monthly_val' and mask[i].any()}
    sources={r['time_bjt']:r for r in json.loads((a.cpp_ready/'frames.json').read_text())}
    scopes={'region256':(slice(None),slice(None))}
    for name,(r,c) in {'Niushou':(111,124),'Wangzhai':(112,126),'Mizhuang':(110,128)}.items():
        scopes[name+'_1x1']=(slice(r,r+1),slice(c,c+1))
        scopes[name+'_3x3']=(slice(r-1,r+2),slice(c-1,c+2))
        scopes[name+'_16x16']=(slice(r-8,r+8),slice(c-8,c+8))
    bins=('all','near_zero','thin','moderate','thick')
    records=[];seen=set();out_of_range=0;total_predicted=0
    for rank in range(complete['world']):
        ii=np.load(a.run/f'val_indices_rank{rank}.npy')
        pp=np.load(a.run/f'val_pred_log_rank{rank}.npy',mmap_mode='r')
        assert pp.shape==(len(ii),1,256,256)
        for j,i in enumerate(ii.tolist()):
            assert i in expected and i not in seen;seen.add(i);row=rows[i];key=row['time_bjt']
            assert row['split']=='val'
            source=a.cpp_ready/'payload/val'/(key+'.npz');assert sha(source)==sources[key]['output_sha256']
            with np.load(source,allow_pickle=False) as z:truth=z['cot'].astype(np.float64)
            pred=np.expm1(pp[j,0].astype(np.float64));assert np.isfinite(pred).all()
            valid=np.asarray(mask[i,0]);assert np.isfinite(truth[valid]).all()
            out_of_range+=int(((pred>100)&valid).sum());total_predicted+=int(valid.sum())
            strata={'all':valid,'near_zero':valid&(truth<.1),'thin':valid&(truth>=.1)&(truth<10),
                    'moderate':valid&(truth>=10)&(truth<30),'thick':valid&(truth>=30)}
            for scope,sl in scopes.items():
                e=(pred-truth)[sl]
                for label in bins:
                    mm=strata[label][sl];err=e[mm]
                    records.append(dict(time_bjt=key,date=key[:8],scope=scope,stratum=label,
                                        n=err.size,sse=float(np.dot(err,err)),sae=float(abs(err).sum()),se=float(err.sum())))
    assert seen==expected
    def metric(rr):
        n=sum(r['n'] for r in rr)
        return dict(n=n,rmse=float(np.sqrt(sum(r['sse'] for r in rr)/n)) if n else None,
                    mae=sum(r['sae'] for r in rr)/n if n else None,bias=sum(r['se'] for r in rr)/n if n else None)
    scores={s:{b:metric([r for r in records if r['scope']==s and r['stratum']==b]) for b in bins} for s in scopes}
    dates=sorted({r['date'] for r in records})
    a.out.mkdir(parents=True)
    with (a.out/'per_frame_error_sums.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    result=dict(state='SAVED_MONTHLY_VAL_R_PREDICTIONS_SCORED_FP64',frames=len(seen),dates=dates,
                units='dimensionless physical COT',reference='CPP satellite retrieval, not independent truth',
                scope='same-time observed AGRI retrieval, not forecast COT or GHI',scores=scores,
                reference_strata='[0,.1), [.1,10), [10,30), [30,100]; no reference-stratum row exclusion',
                station_neighborhoods='centered 1x1,3x3 and [r-8:r+8,c-8:c+8] 16x16, not shared patch origin',
                unclipped_valid_predictions_above100=out_of_range,valid_prediction_pixels=total_predicted,
                test_payloads_read=0,checkpoint_selected_without_monthly_val=True)
    (a.out/'REPORT.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='scores'}))


if __name__=='__main__':main()
