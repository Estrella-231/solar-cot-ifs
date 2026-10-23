"""Independent saved-row confusion counts, folds and date-bootstrap check."""
import csv,hashlib,json
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
P=Path(__file__).resolve().parent;complete=json.loads((P/'COMPLETE.json').read_text());score=json.loads((P/'SCORES.json').read_text());rows=list(csv.DictReader((P/'OOF_predictions.csv').open()))
with np.load(P/'OOF_scores.npz') as saved:z={k:saved[k] for k in saved.files}
assert len(rows)==complete['rows'];assert len({(r['station'],r['init_time'],r['target_time']) for r in rows})==len(rows)
folds={r['fold']:r for r in complete['folds']}
for fold,record in folds.items():
    fn=P/'models'/f'fold_{fold}.npz';assert hashlib.sha256(fn.read_bytes()).hexdigest()==record['model_sha256']
    with np.load(fn) as model:
        fit=set(model['train_dates'].tolist());val=set(model['held_out_dates'].tolist())
        assert fit==set(record['train_dates']) and val==set(record['held_out_dates'])
        for day in val:
            dt=datetime.strptime(day,'%Y%m%d')
            assert not fit.intersection((dt+timedelta(days=k)).strftime('%Y%m%d') for k in [-1,0,1])
        assert model['coef'].shape==(307,3) and model['mean'].shape==(306,) and (model['std']>0).all()
dates=sorted({r['date'] for r in rows});di={d:i for i,d in enumerate(dates)};cm=np.zeros((len(dates),3,3),dtype='int64')
for i,r in enumerate(rows):
    label=int(r['label']);pred=int(r['prediction']);fold=int(r['fold']);assert r['date'] in folds[fold]['held_out_dates']
    delta=float(r['reference_minus_forecast']);assert label==(2 if delta>3 else 0 if delta< -3 else 1)
    assert label==z['truth'][i] and pred==z['prediction'][i] and r['date']==z['date'][i] and r['station']==z['station'][i]
    assert pred==int(np.argmax([float(r['score_down']),float(r['score_stable']),float(r['score_up'])]))
    cm[di[r['date']],label,pred]+=1
total=cm.sum(0);assert total.tolist()==score['overall']['confusion'];bacc=float(np.mean(np.diag(total)/total.sum(1)));assert abs(bacc-score['overall']['balanced_accuracy'])<1e-14
w=np.random.default_rng(20261023).multinomial(len(dates),np.ones(len(dates))/len(dates),5000)
boot=(w@cm.reshape(len(dates),9)).reshape(-1,3,3);ba=np.stack([boot[:,i,i]/boot[:,i,:].sum(1) for i in range(3)],1).mean(1);ci=np.quantile(ba,[.025,.975]);err=float(abs(ci-score['balanced_accuracy_ci95']).max());assert err<1e-12
report=dict(state='PASS_INDEPENDENT_OOF_KEYS_TEMPORAL_BUFFER_MODELS_CONFUSIONS_AND_DATE_CI',rows=len(rows),dates=len(dates),balanced_accuracy=bacc,ci95=ci.tolist(),ci_max_abs_difference=err,classes=['need_down','stable','need_up'],no_fit_date_in_any_held_out_plusminus_one_day=True,F_select_C_select_outer_payloads_read=0)
(P/'INDEPENDENT_AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
