"""CPU selected-case input intervention; zeroing is distribution shift, diagnostic only."""
import json
from pathlib import Path
import numpy as np
import torch
from train_regional_cot_ghi_pilot_20260927 import Rows,RegionalStationHead
torch.set_num_threads(2)
solar=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
exp=solar/'experiments/regional_station_cot_20260925'
run=exp/'regional_ghi_pilot_seed42_20260927_v1'
bank=exp/'regional_ghi_pilot_bank_20260927_v1'
base=solar/'data/head_pack_trainval_20260908_v1'
zero=dict(np.load(run/'zero_validation_predictions.npz'))
result={}
for arm in ['forecast','oracle']:
    pred=dict(np.load(run/f'{arm}_validation_predictions.npz'))
    ds=(pred['pred_ghi']-pred['true_ghi'])**2-(zero['pred_ghi']-zero['true_ghi'])**2
    take=np.where(zero['station']==1)[0]; take=take[np.argsort(ds[take])[::-1][:12]]
    data=Rows(bank,base,'val',arm)
    lookup={int(v):i for i,v in enumerate(data.ids)}
    batch=[data[lookup[int(zero['row_ids'][i])]] for i in take]
    x=[torch.as_tensor(np.stack([b[k] for b in batch])) for k in range(10)]
    model=RegionalStationHead();model.load_state_dict(torch.load(run/f'{arm}_best.pt',map_location='cpu',weights_only=False)['model']);model.eval()
    def calc(lc,rc):
        with torch.no_grad(): return (model(x[0],x[1],lc,rc,x[4],x[5]).double()*x[7]).numpy()
    original=calc(x[2],x[3]); no=calc(torch.zeros_like(x[2]),torch.zeros_like(x[3]))
    localonly=calc(x[2],torch.zeros_like(x[3])); regiononly=calc(torch.zeros_like(x[2]),x[3])
    lh=x[2].clone();rh=x[3].clone();lh[:,:8]=0;rh[:,:8]=0
    nohistory=calc(lh,rh)
    result[arm+'_cpu_reload_max_abs_difference']=float(np.max(abs(original-pred['pred_ghi'][take])))
    result[arm]=[{'row_id':int(zero['row_ids'][i]),'sequence':int(data.labels['sequence'][zero['row_ids'][i]]),'clear':float(x[7][j]),'true':float(zero['true_ghi'][i]),'zero_head':float(zero['pred_ghi'][i]),'own_original':float(original[j]),'own_cot_zero':float(no[j]),'own_local_only':float(localonly[j]),'own_region_only':float(regiononly[j]),'own_no_history':float(nohistory[j])} for j,i in enumerate(take)]
print(json.dumps(result,indent=2))
