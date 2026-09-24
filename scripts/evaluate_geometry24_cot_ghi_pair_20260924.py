"""Recalculate paired GHI metrics from saved geometry24 head predictions.

O_diag uses real future AGRI through frozen R and is reported as an oracle
diagnostic only.  No model is trained or selected here and test is unopened.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


LABELS = {"A_ST": "A0", "AC_H": "AH", "AC_ST": "AI"}
STATIONS = ("Sili", "Zhujia")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def metrics(error):
    if not len(error) or not np.isfinite(error).all():
        raise RuntimeError('invalid nonempty error cohort required')
    return {'n':int(len(error)), 'rmse_wm2':float(np.sqrt(np.mean(error.astype(np.float64)**2))),
            'mae_wm2':float(np.mean(np.abs(error))), 'bias_wm2':float(np.mean(error))}


def load_arm(directory, group):
    report = json.loads((directory/'complete.json').read_text())
    if report['state'] != 'COMPLETE_EXPLORATORY_CAUSAL_COT_TRAJECTORY' or report['test_used']:
        raise RuntimeError('head run incomplete or test contaminated')
    result = {}
    for station in STATIONS:
        path = directory/f'{station}_{group}_validation_predictions.npz'
        if sha(path) != report['runs'][station][group]['predictions_sha256']:
            raise RuntimeError('saved validation prediction hash drift')
        with np.load(path,allow_pickle=False) as z:
            result[station] = {key:z[key] for key in ('pack_row','pred_kt','observed_ghi','clear_sky_ghi','station','lead')}
    merged = {key:np.concatenate([result[s][key] for s in STATIONS]) for key in result[STATIONS[0]]}
    order = np.argsort(merged['pack_row'])
    merged = {key:value[order] for key,value in merged.items()}
    if len(np.unique(merged['pack_row'])) != len(merged['pack_row']):
        raise RuntimeError('duplicate validation row')
    if not np.isfinite(merged['pred_kt']).all() or (merged['clear_sky_ghi'] <= 0).any():
        raise RuntimeError('invalid GHI prediction values')
    return merged, {'run_complete_sha256':sha(directory/'complete.json'),
                    'files':{station:sha(directory/f'{station}_{group}_validation_predictions.npz')
                             for station in STATIONS}}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--simvp-head-run',type=Path,required=True)
    p.add_argument('--oracle-head-run',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise RuntimeError('refuse to overwrite paired evaluation')
    base=a.root/'data/head_pack_trainval_20260908_v1'
    split=np.load(base/'split.npy',mmap_mode='r')
    expected_rows=np.flatnonzero(split==1)
    if len(expected_rows)!=99849:
        raise RuntimeError('validation row cohort drift')
    arms={};sources={}
    for group,label in LABELS.items():
        arms[label],sources[label]=load_arm(a.simvp_head_run,group)
    if a.oracle_head_run is not None:
        arms['O_diag'],sources['O_diag']=load_arm(a.oracle_head_run,'AC_ST')
        oracle_contract=json.loads((a.oracle_head_run/'execution_contract.json').read_text())
        if oracle_contract['future_cot_kind']!='oracle' or oracle_contract['deployable']:
            raise RuntimeError('O_diag must remain a real-future oracle')
    reference=arms['A0']
    if not np.array_equal(reference['pack_row'],expected_rows):
        raise RuntimeError('not the complete common validation cohort')
    for label,arm in arms.items():
        for key in ('pack_row','observed_ghi','clear_sky_ghi','station','lead'):
            if not np.array_equal(arm[key],reference[key]):
                raise RuntimeError(f'{label} differs on paired {key}')
    rows=reference['pack_row']
    station=reference['station'];lead=reference['lead']
    truth=reference['observed_ghi'].astype(np.float64)
    clear=reference['clear_sky_ghi'].astype(np.float64)
    predictions={label:arm['pred_kt'].astype(np.float64)*clear for label,arm in arms.items()}
    report={'state':'COMPLETE_GEOMETRY24_GHI_PAIRED_VALIDATION','test_used':False,
            'reference':{'n_rows':len(rows),'head_pack_audit_sha256':sha(base/'audit.json'),
                         'split':'validation','oracle_is_deployable':False},
            'sources':sources,'arms':{},'paired_differences':{}}
    for label,pred in predictions.items():
        error=pred-truth
        per_station={name:metrics(error[station==i]) for i,name in enumerate(STATIONS)}
        report['arms'][label]={'all':metrics(error),'stations':per_station,
                               'equal_station_rmse_wm2':float(np.mean([per_station[s]['rmse_wm2'] for s in STATIONS])),
                               'per_lead':{str(15*(k+1)):metrics(error[lead==k]) for k in range(16)}}
    for first,second in (('AH','A0'),('AI','AH'),('AI','A0'),('O_diag','AI'),('O_diag','AH')):
        if first in report['arms'] and second in report['arms']:
            report['paired_differences'][f'{first}_minus_{second}_equal_station_rmse_wm2']=(
                report['arms'][first]['equal_station_rmse_wm2']-
                report['arms'][second]['equal_station_rmse_wm2'])
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'matched_validation_predictions.npz',pack_row=rows,station=station,lead=lead,
                        observed_ghi=truth,clear_sky_ghi=clear,**{f'pred_ghi_{k}':v for k,v in predictions.items()})
    report['matched_predictions_sha256']=sha(a.output/'matched_validation_predictions.npz')
    (a.output/'paired_metrics.json').write_text(json.dumps(report,indent=2,sort_keys=True))
    print(json.dumps({'state':report['state'],'equal_station_rmse_wm2':
        {k:v['equal_station_rmse_wm2'] for k,v in report['arms'].items()}},sort_keys=True),flush=True)


if __name__=='__main__':
    main()
