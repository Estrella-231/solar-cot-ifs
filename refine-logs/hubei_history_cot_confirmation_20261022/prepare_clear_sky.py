"""Frozen label/clear-sky convention for new-source confirmation, no scores."""
import argparse,csv,hashlib,json,sys
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import pvlib

STATIONS={'Niushou':(32.0800,112.0400),'Wangzhai':(32.0400,112.1200),'Mizhuang':(32.1200,112.2000)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clear_values(station,times):
    t=pd.DatetimeIndex(pd.to_datetime(times)).tz_localize('Asia/Shanghai')
    loc=pvlib.location.Location(*STATIONS[station],tz='Asia/Shanghai',altitude=0.0,name=station)
    return np.mean([loc.get_clearsky(t-pd.Timedelta(minutes=k),model='ineichen')['ghi'].to_numpy(np.float64)
                    for k in [12.5,7.5,2.5]],axis=0)
def main():
    p=argparse.ArgumentParser();p.add_argument('--labels',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False)
    old=json.loads(a.reference.read_text());errors=[]
    for st in STATIONS:
        rr=[r for r in old if r['station']==st]
        x=clear_values(st,[r['target_time_BJT'] for r in rr]);y=np.array([r['clear_sky_wm2'] for r in rr])
        errors.extend(abs(x-y).tolist())
    error=max(errors);assert error<1e-3,('Existing clear-sky reconstruction mismatch; do not score',error)
    rows=list(csv.DictReader(a.labels.open()));summary={}
    # Reuse the same source function for the label daylight mask.
    sys.path.insert(0,str(Path(__file__).parent))
    from attach_hubei_daylight_mask import solar_cosine_noaa
    for st in STATIONS:
        ids=[i for i,r in enumerate(rows) if r['station']==st]
        clear=clear_values(st,[rows[i]['target_time_BJT'] for i in ids]);assert np.isfinite(clear).all() and np.all(clear>=0)
        for ix,c in zip(ids,clear):
            r=rows[ix];t=datetime.fromisoformat(r['target_time_BJT']);day=solar_cosine_noaa(t,*STATIONS[st])>0
            r.update(clear_sky_ghi_15min_wm2=float(c),day_mask_sza_lt_90=str(day),
                valid_daylight_clear_sky_label=int(day and c>0 and r['mask']=='1'))
        summary[st]=sum(rows[i]['valid_daylight_clear_sky_label'] for i in ids)
    dest=a.out/'labels_with_clear.csv'
    with dest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report=dict(state='NEW_LABELS_CLEAR_SKY_PREPARED_NO_SCORES',valid_daylight_labels=summary,
        pvlib_version=pvlib.__version__,numpy_version=np.__version__,pandas_version=pd.__version__,
        old_clear_reference_max_abs_error_wm2=error,old_clear_reference_sha256=sha(a.reference),
        source_labels_sha256=sha(a.labels),output_sha256=sha(dest),script_sha256=sha(__file__),
        interval='three five-minute midpoints T-12.5/-7.5/-2.5 minutes, Ineichen, altitude0, BJT',
        old_test_payloads_read=0,old_buffer_payloads_read=0,new_period_scores_computed=0)
    (a.out/'CLEAR_SKY_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
