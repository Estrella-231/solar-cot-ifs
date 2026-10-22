"""Replay frozen heads on their saved validation tensors before new inference."""
import json,sys,argparse
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments')
HEAD=ROOT/'hubei_forecast_cot_increment_20261022';HERE=ROOT/'hubei_history_cot_confirmation_20261022'
sys.path.insert(0,str(HEAD/'code_attempt1'))
from history_base import HistoryCOTGHIHead
torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=HERE/'HEAD_REPLAY_PASS.json');args=ap.parse_args()
assert not args.output.exists()
checks=[]
for st in ['Niushou','Wangzhai','Mizhuang']:
    p=HEAD/'prepared_attempt1'/st;roles=np.load(p/'roles.npy');ix=np.where(roles=='val')[0]
    arrays=[np.load(p/(k+'.npy'))[ix] for k in ['future','history','cot']];clear=np.load(p/'clear.npy')[ix]
    for seed in [42,43,44]:
        d=HEAD/'training_attempt1/pairs'/f'{st}_seed{seed}'
        for arm in ['G0','G1']:
            model=HistoryCOTGHIHead().cuda().eval();model.load_state_dict(torch.load(d/(arm+'_best.pt'),map_location='cuda',weights_only=False)['model'])
            pred=[]
            with torch.inference_mode():
                for j in range(0,len(ix),1536):
                    x=[torch.from_numpy(np.ascontiguousarray(a[j:j+1536])).cuda() for a in arrays]
                    if arm=='G0':x[2].zero_()
                    pred.append(model(*x).cpu().numpy())
            actual=np.concatenate(pred).astype(np.float64)*clear.astype(np.float64)
            expected=np.load(d/(arm+'_validation_predictions.npz'))['pred_ghi_wm2']
            error=float(np.max(abs(actual-expected)));assert error<1e-3,(st,seed,arm,error)
            checks.append(dict(station=st,seed=seed,arm=arm,max_abs_difference_wm2=error))
args.output.write_text(json.dumps(dict(state='PASS',checks=checks,old_test_payloads_read=0,new_period_scores_computed=0),indent=2)+'\n')
print('HEAD_REPLAY_PASS',max(r['max_abs_difference_wm2'] for r in checks))
