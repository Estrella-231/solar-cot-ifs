"""Synthetic paired-key/block test for the solar-path COT H-CV scorer."""
from __future__ import annotations
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
STATIONS=('sili','zhujia'); ARMS=('none','raw','transmission'); SEEDS=(42,43,44)

def main():
    with tempfile.TemporaryDirectory(prefix='cot_traj_score_test_') as td:
        root=Path(td)/'run'; root.mkdir()
        (root/'run_manifest.json').write_text(json.dumps({'state':'COMPLETE_COT_SOLAR_PATH_H_CV','test_payload_opened':False,'buffer_payload_opened':False,'future_observed_agri_used_as_input':False,'cpp_used_as_input':False}))
        for st in STATIONS:
            for fold in range(5):
                keys=[]
                for day in (1,3):
                    stamp=f'2026-01-{fold*4+day:02d} 12:00:00'
                    keys.append({'station':st,'init_time_utc':'2026-01-01 00:00:00','target_time_utc':stamp,'lead_minutes':15 if day==1 else 30})
                for arm in ARMS:
                    for seed in SEEDS:
                        p=root/st/f'fold{fold}'/arm/f'seed{seed}'; p.mkdir(parents=True)
                        (p/'oof_keys.jsonl').write_text('\n'.join(json.dumps(k,sort_keys=True) for k in keys)+'\n')
                        truth=np.asarray([500.,500.]); offset={'none':25.,'raw':20.,'transmission':10.}[arm]
                        np.savez_compressed(p/'oof_predictions.npz',prediction_ghi=truth+offset,truth_ghi=truth)
        output=Path(td)/'score.json'
        proc=subprocess.run([sys.executable,str(ROOT/'score_direct_cot_solar_path_hcv.py'),'--run',str(root),'--output',str(output),'--bootstrap','100'],capture_output=True,text=True)
        if proc.returncode:
            raise RuntimeError(f'scorer failed ({proc.returncode})\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}')
        result=json.loads(output.read_text())
        assert result['state']=='COMPLETE_COT_SOLAR_PATH_H_CV_SCORE'
        assert result['results']['predeclared_gate']['pass'] is True
        assert result['results']['transmission_minus_raw']['all_leads']['delta_rmse_wm2']<0
        assert result['test_payload_opened'] is False and result['buffer_payload_opened'] is False
    print('DIRECT_COT_SOLAR_PATH_HCV_SCORER_TESTS_PASS')

if __name__=='__main__': main()
