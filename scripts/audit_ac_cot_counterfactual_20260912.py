"""Independent NumPy audit of saved AC COT counterfactual predictions."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def metric(pred, truth, clear, mask):
    err=pred[mask].astype(np.float64)*clear[mask]-truth[mask]
    return dict(n=int(mask.sum()),rmse=float(np.sqrt(np.mean(err**2))),
                mae=float(np.mean(abs(err))),bias=float(err.mean()),
                mean_predicted_ghi=float(np.mean(pred[mask].astype(np.float64)*clear[mask])))


def main(run, output):
    assert not output.exists()
    report=json.loads((run/'report.json').read_text())
    assert report['state']=='COMPLETE_AC_COT_COUNTERFACTUAL_FORWARD' and report['test_used'] is False
    pred_path=run/'counterfactual_predictions.npz';metrics_path=run/'metrics.csv'
    assert sha(pred_path)==report['predictions_sha256'] and sha(metrics_path)==report['metrics_sha256']
    saved=list(csv.DictReader(metrics_path.open()))
    lookup={(int(r['seed']),r['station'],r['group'],r['variant']):r for r in saved}
    with np.load(pred_path,allow_pickle=False) as z:
        seeds=z['seeds'].tolist();variants=z['variants'].tolist();pred=z['pred_kt']
        truth=z['observed_ghi'];clear=z['clear_sky_ghi'];station=z['station'];weather=z['predicted_weather']
        assert seeds==[42,43,44] and variants==report['variants']
        assert pred.shape==(3,7,99849) and np.isfinite(pred).all()
        groups={'all':np.ones(len(truth),bool),'predicted_sunny':weather==0,
                'predicted_partly_cloudy':weather==1,'predicted_overcast':weather==2}
        differences=[]
        for si,seed in enumerate(seeds):
            for vi,variant in enumerate(variants):
                for sid,name in enumerate(('sili','zhujia')):
                    for group,gmask in groups.items():
                        mask=(station==sid)&gmask
                        got=metric(pred[si,vi],truth,clear,mask)
                        row=lookup[(seed,name,group,variant)]
                        for key in ('rmse','mae','bias','mean_predicted_ghi'):
                            differences.append(abs(got[key]-float(row[key])))
                        assert got['n']==int(row['n'])
        assert len(lookup)==3*7*2*4 and max(differences)<1e-10
    result=dict(state='PASS_INDEPENDENT_NUMPY_RECOMPUTE_ALL_168_ROWS',test_used=False,
        report_sha256=sha(run/'report.json'),predictions_sha256=sha(pred_path),metrics_sha256=sha(metrics_path),
        rows_checked=len(lookup),scalar_metrics_checked=len(differences),max_abs_error=max(differences))
    output.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.run,a.output)
