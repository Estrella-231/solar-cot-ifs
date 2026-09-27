"""Paired validation audit for the GHI-objective Zhujia residual experiment."""
import argparse, json
from pathlib import Path
import numpy as np


def metrics(y,p):
    e=p-y
    return {'n':int(len(y)),'rmse':float(np.mean(e**2)**.5),
            'mae':float(np.mean(np.abs(e))),'bias':float(np.mean(e))}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--source-run',type=Path,required=True)
    ap.add_argument('--bank-contract',type=Path,required=True)
    ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    arms=('agri_residual','cot_residual')
    arrays={k:np.load(a.run/f'{k}_val_predictions.npz') for k in arms}
    ref=arrays[arms[0]]
    for k in arms[1:]:
        for field in ('row_ids','lead','true_ghi'):
            assert np.array_equal(arrays[k][field],ref[field]),f'{field} differs: {k}'
    row=ref['row_ids']; y=ref['true_ghi']; lead=ref['lead']
    pred={k:arrays[k]['pred_ghi'] for k in arms}
    station=np.load(a.base/'station.npy',mmap_mode='r')
    seq=np.load(a.base/'sequence.npy',mmap_mode='r')
    assert np.all(station[row]==1), 'validation rows contain non-Zhujia station'
    day_map=json.loads(a.bank_contract.read_text())['initialization_BJT_days']['val']
    days=np.asarray([day_map[str(int(s))] for s in seq[row]])
    unique=np.unique(days)
    anchor_file=np.load(a.source_run/'zero_val_predictions.npz')
    anchor_lookup={int(r):i for i,r in enumerate(anchor_file['row_ids'])}
    assert all(int(r) in anchor_lookup for r in row), 'row missing in frozen baseline'
    anchor_i=np.asarray([anchor_lookup[int(r)] for r in row])
    assert np.array_equal(anchor_file['lead'][anchor_i],lead)
    assert np.array_equal(anchor_file['true_ghi'][anchor_i],y)
    anchor=anchor_file['pred_ghi'][anchor_i]
    allpred={'frozen_zero_anchor':anchor,**pred}
    overall={k:metrics(y,p) for k,p in allpred.items()}
    lead_metrics={}
    for li in np.unique(lead):
        mask=lead==li
        lead_metrics[str(int(li)*15+15)]={k:metrics(y[mask],p[mask])['rmse'] for k,p in allpred.items()}
    rng=np.random.default_rng(20260928)
    blocks=[np.flatnonzero(days==d) for d in unique]
    boot={key:[] for key in ('cot_minus_agri','agri_minus_anchor','cot_minus_anchor')}
    for _ in range(5000):
        ix=np.concatenate([blocks[i] for i in rng.integers(0,len(blocks),len(blocks))])
        rmse=lambda p: float(np.mean((p[ix]-y[ix])**2)**.5)
        boot['cot_minus_agri'].append(rmse(pred['cot_residual'])-rmse(pred['agri_residual']))
        boot['agri_minus_anchor'].append(rmse(pred['agri_residual'])-rmse(anchor))
        boot['cot_minus_anchor'].append(rmse(pred['cot_residual'])-rmse(anchor))
    paired={k:{'ci95':[float(v) for v in np.quantile(x,[.025,.975])],
               'fraction_delta_below_zero':float(np.mean(np.asarray(x)<0))}
            for k,x in boot.items()}
    delta_from_anchor={k:float(np.max(np.abs(pred[k]-anchor))) for k in arms}
    out={'state':'PAIRED_ZHUJIA_GHI_OBJECTIVE_AUDIT_COMPLETE','test_used':False,
      'rows':int(len(row)),'initialization_days':int(len(unique)),
      'same_rows_truth_and_leads':True,'overall':overall,'lead_rmse':lead_metrics,
      'paired_initialization_day_bootstrap':{'repetitions':5000,'seed':20260928,'deltas':paired},
      'max_abs_prediction_difference_from_zero_anchor_W_m2':delta_from_anchor,
      'selected_exact_baseline_fallback':{k:bool(v<=1e-4) for k,v in delta_from_anchor.items()},
      'run_complete':json.loads((a.run/'complete.json').read_text())}
    a.out.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in out.items() if k!='run_complete'},indent=2))


if __name__=='__main__': main()
