"""Synthetic arithmetic/key witness, explicitly not experimental evidence."""
import csv, hashlib, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='trajectory_score_fixture_') as temp:
    root = Path(temp); (root/'fits_attempt1').mkdir(); runs=[]; rows=[]; keys=[]; ref=[]
    for day in ['20250101', '20250102', '20250103', '20250104']:
        for lead in range(15,241,15):
            init=day+'0900';target=day+f'{9+lead//60:02d}{lead%60:02d}'
            keys.append((init,target,lead)); value={'20250101':0,'20250102':5,'20250103':15,'20250104':35}[day];ref.append(value)
            for station in ['Niushou','Wangzhai','Mizhuang']:
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        rows.append(dict(init=init,target=target,date=day,station=station,lead=lead,dr=dr,dc=dc,
                                         reference=value,common=1,persistence=value+2,transport=value+3,forecast=value+5))
    baseline=root/'baseline.csv'
    with baseline.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for arm in ['full','repeat_last']:
        for seed in [42,43,44]:
            folder=root/'eval_attempt1'/f'seed{seed}_{arm}';folder.mkdir(parents=True)
            pred=np.broadcast_to(np.log1p(np.array(ref,float)+(1 if arm=='full' else 4))[:,None,None,None],(len(keys),3,3,3)).copy()
            path=folder/'predictions.npz'
            np.savez(path,init=[k[0] for k in keys],target=[k[1] for k in keys],lead=[k[2] for k in keys],pred_log=pred)
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            (folder/'COMPLETE.json').write_text(json.dumps(dict(state='FROZEN_TRAJECTORY_VAL_PREDICTIONS_SAVED',checkpoint_sha256='fixture',prediction_sha256=digest)))
            runs.append(dict(arm=arm,seed=seed,best_sha256='fixture',best_select_mse_normalized=1 if arm=='full' else 2))
    (root/'fits_attempt1/ARM_SELECTION.json').write_text(json.dumps(dict(state='SIX_FITS_COMPLETE_ARM_FROZEN_BEFORE_OUTER_VAL',selected_arm='full',runs=runs)))
    result=subprocess.run([sys.executable,str(P/'score_eval.py'),'--root',str(root),'--baseline',str(baseline),'--out',str(root/'scores')],capture_output=True)
    assert result.returncode==0,result.stderr.decode(errors='replace')
    r=json.loads((root/'scores/SCORES.json').read_text())
    assert r['may_advance_paired_GHI'] and r['common_center_rows']==192
    for name,value in [('persistence',2),('transport',3),('forecast',5),('seed42_full',1),('seed44_repeat_last',4)]:
        assert abs(r['main_rmse'][name]-value)<1e-10,(name,r['main_rmse'][name])
    assert r['comparisons']['full']['persistence']['date_block_ci95'][1]<-.999999
    # Duplicate/missing trajectory keys must fail rather than align by row number.
    path=root/'eval_attempt1/seed42_full/predictions.npz'
    with np.load(path) as z:data={k:z[k] for k in z.files}
    for k in ['init','target','lead']:data[k][0]=data[k][1]
    np.savez(path,**data)
    receipt=json.loads(path.with_name('COMPLETE.json').read_text());receipt['prediction_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name('COMPLETE.json').write_text(json.dumps(receipt))
    failed=subprocess.run([sys.executable,str(P/'score_eval.py'),'--root',str(root),'--baseline',str(baseline),'--out',str(root/'bad')],capture_output=True)
    assert failed.returncode!=0 and not (root/'bad').exists()
print('PASS_SYNTHETIC_FP64_METRICS_DATE_BLOCKS_AND_KEY_REJECTION_NOT_MODEL_RESULT')
