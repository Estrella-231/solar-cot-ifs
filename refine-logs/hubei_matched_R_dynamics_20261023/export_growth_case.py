"""One calendar-selected C_select case, real arrays only, no cherry-picked errors."""
import csv,json
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;B=P.parent/'hubei_solar_geometry_repair_20261021';G=P.parent/'hubei_cot_calibration_coverage_20261023'
rows=list(csv.DictReader((P/'raw_attempt1/pairs.csv').open()));frames=list(csv.DictReader((P/'raw_attempt1/frames.csv').open()));fi={r['bjt']:int(r['frame_index']) for r in frames};raw=np.load(P/'raw_attempt1/raw16.npy',mmap_mode='r')
refs={}
for role in [f'fold_{i}' for i in range(5)]:
    with np.load(P/'reference_attempt1'/role/'reference.npz') as z:refs[role]={k:z[k] for k in z.files}
candidates=sorted({(r['init_time'],r['source_role'],int(r['source_sequence_index'])) for r in rows if r['scope']=='C_select'})
for init,role,seq in candidates:
    t=datetime.strptime(init,'%Y%m%d%H%M');times=[(t+timedelta(minutes=k)).strftime('%Y%m%d%H%M') for k in range(0,241,15)];lookup={v:i for i,v in enumerate(refs[role]['bjt'])}
    if all(v in lookup and v in fi for v in times):
        cot=refs[role]['cot'][[lookup[v] for v in times],0];valid=refs[role]['valid'][[lookup[v] for v in times],0]
        growth=any(valid[0,cy,cx] and np.any(valid[1:,cy,cx] & ((cot[1:,cy,cx]-cot[0,cy,cx])>3)) for cy,cx in [(8,6),(9,8),(7,10)])
        if growth:break
else:raise RuntimeError('No complete calendar case')
observed=np.array(raw[[fi[v] for v in times]])
forecast=np.load(B/'S_banks_attempt12'/role/'val/pred_agri_patch.npy',mmap_mode='r')[seq].copy();forecastcot=np.load(B/'COT_banks_attempt12'/f'{role}.npy',mmap_mode='r')[seq].copy();actualcot=refs[role]['cot'][[lookup[v] for v in times]];actualvalid=refs[role]['valid'][[lookup[v] for v in times]]
truth=np.full((3,16),np.nan);predictions=np.full((3,3,16),np.nan)
for si,st in enumerate(['Niushou','Wangzhai','Mizhuang']):
    values=[]
    for seed in [42,43,44]:
        with np.load(G/'replay'/st/f'seed{seed}_Cselect.npz') as z:
            pp=np.full((3,16),np.nan)
            for lead in range(15,241,15):
                ii=np.flatnonzero((z['init_time']==init)&(z['lead']==lead))
                if len(ii):assert len(ii)==1;truth[si,lead//15-1]=z['truth'][ii[0]];pp[:,lead//15-1]=z['prediction'][[0,3,4],ii[0]]
            values.append(pp)
    v=np.array(values);n=np.isfinite(v).sum(0);predictions[si]=np.divide(np.nansum(v,0),n,out=np.full_like(v[0],np.nan),where=n>0)
np.savez_compressed(P/'GROWTH_CASE.npz',init=init,source_role=role,times=times,observed=observed,forecast=forecast,forecast_cot=forecastcot,observed_R=actualcot,observed_valid=actualvalid,ghi_truth=truth,ghi_prediction=predictions)
(P/'GROWTH_CASE_SCOPE.json').write_text(json.dumps(dict(rule='Exploratory event illustration: earliest C_select init with all 16 reference target frames and observed-R growth >3 at any station; existing analysis event threshold; not selected by forecast/GHI error or gain',init=init,role=role,AGRI_display='fixed C03/C02/C01 false colour, square-root stretch 0..0.7, same transform for every tile',GHI='right-ending 15-minute means; 3-seed means for plotting only; missing kept as gaps',reference='same-fold R(actual future AGRI), not independent truth or deployable input'),indent=2));print(init,role)
