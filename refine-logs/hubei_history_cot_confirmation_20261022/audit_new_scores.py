"""Independent local NPZ checks, L2 RMSE and bincount-based date bootstrap."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--attempt',type=int,required=True);a=p.parse_args()
    expected=json.loads((a.evidence/f'scored_attempt{a.attempt}/SCORES.json').read_text());tests={};maxdiff=0.
    for scope in ['primary_shared_dates','two_station_extension']:
        stations=['Niushou','Wangzhai','Mizhuang'] if scope=='primary_shared_dates' else ['Niushou','Wangzhai'];data=[]
        per_site_rows={};seen_site_keys={}
        for st in stations:
            for seed in [42,43,44]:
                pair=[]
                for arm in ['G0','G1']:
                    with np.load(a.evidence/f'predictions_attempt{a.attempt}'/f'{st}_seed{seed}_{arm}.npz',allow_pickle=False) as z:
                        rr={k:z[k].copy() for k in z.files}
                    assert np.isfinite(rr['pred_ghi']).all() and np.isfinite(rr['ghi']).all()
                    pair.append(rr)
                x,y=pair
                for k in ['ghi','clear','init_time_BJT','target_time_BJT','lead_minutes','analysis']:assert np.array_equal(x[k],y[k])
                m=x['analysis']==scope;keys=list(zip(x['init_time_BJT'][m],x['target_time_BJT'][m],x['lead_minutes'][m]))
                assert len(set(keys))==len(keys)
                if st in seen_site_keys:assert keys==seen_site_keys[st]
                else:seen_site_keys[st]=keys;per_site_rows[st]=len(keys)
                for init,target,lead in keys:
                    assert int(lead) in range(15,241,15)
                    from datetime import datetime
                    assert (datetime.fromisoformat(target)-datetime.fromisoformat(init)).total_seconds()==int(lead)*60
                errors=np.array([x['pred_ghi'][m]-x['ghi'][m],y['pred_ghi'][m]-y['ghi'][m]],np.float64)
                data.append((errors,np.array([s[:10] for s in x['target_time_BJT'][m]])))
        days=np.array(sorted(set(s for _,d in data for s in d)));point=np.array([[np.linalg.norm(e)/np.sqrt(len(e)) for e in errors] for errors,_ in data]).mean(axis=0)
        rng=np.random.default_rng(20261022);w=rng.multinomial(len(days),np.ones(len(days))/len(days),5000)
        sse=[];den=[]
        for errors,d in data:
            idx=np.searchsorted(days,d);den.append(np.bincount(idx,minlength=len(days)))
            sse.append([np.bincount(idx,weights=e*e,minlength=len(days)) for e in errors])
        den=np.array(den);sse=np.array(sse);count=den@w.T;valid=(count>0).all(axis=0)
        per=np.sqrt(np.einsum('cad,bd->cab',sse,w[valid])/count[:,None,valid]);boot=per.mean(axis=0)
        delta=boot[1]-boot[0];ci=np.quantile(delta,[.025,.975]);gain=np.quantile(100*(boot[0]-boot[1])/boot[0],[.025,.975])
        ref=expected['scopes'][scope]
        got=np.r_[point,ci,gain];wanted=np.r_[[ref['no_cot_rmse'],ref['history_cot_rmse']],ref['difference_ci95'],ref['relative_reduction_ci95']]
        err=float(np.max(abs(got-wanted)));maxdiff=max(maxdiff,err);assert err<1e-9
        assert sum(per_site_rows.values())==ref['paired_forecast_rows_per_seed']
        tests[scope]=dict(points=point.tolist(),delta_ci95=ci.tolist(),relative_ci95=gain.tolist(),independent_max_abs_difference=err,rows_per_seed=sum(per_site_rows.values()),days=len(days))
    report=dict(state='PASS',scopes=tests,max_abs_difference=maxdiff,method='L2 RMSE; bincount date sums; paired date multinomial; NPZ alignment and canonical lead checks',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),old_test_payloads_read=0,old_buffer_payloads_read=0)
    (a.evidence/'INDEPENDENT_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
