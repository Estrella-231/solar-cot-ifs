#!/usr/bin/env python3
"""Read raw HDF timing metadata only at selected validation station rows."""
import argparse,csv,json
from datetime import datetime,timedelta
from pathlib import Path
import h5py
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--data-root',type=Path,required=True)
p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
cases=json.loads(a.manifest.read_text(encoding='utf-8'))['selected_cases']
selected={(c['station'],t) for c in cases for t in c['future_target_times_utc']}
source={}
for r in csv.DictReader((a.data_root/'source_manifest.csv').open(encoding='utf-8')):
    if r['time_utc'] in {t for _,t in selected}:source[r['time_utc']]=r['source_hdf']
with np.load(a.data_root/'grid_static.npz',allow_pickle=False) as z:
    row_index=np.asarray(z['row_index']);col_index=np.asarray(z['col_index'])
centers={'sili':(163,148),'zhujia':(73,150)}
records=[]
for st,t in sorted(selected):
    gr,gc=centers[st];hr=int(row_index[gr,gc]);hc=int(col_index[gr,gc]);path=source.get(t)
    record={'station':st,'target_time_utc':t,'true_HDF_rc':[hr,hc],'source_HDF':path}
    try:
        with h5py.File(path,'r') as f:
            ds=f['NOMObsTime'];record['NOMObsTime_shape']=list(ds.shape);record['NOMObsTime_dtype']=str(ds.dtype)
            record['NOMObsTime_attributes']={k:str(v) for k,v in ds.attrs.items()}
            value=ds[hr,hc] if ds.ndim==2 and ds.shape[1]>hc else ds[hr]
            record['station_NOMObsTime_raw']=np.asarray(value).tolist()
            record['state']='READ_TIMING_METADATA'
    except (OSError,KeyError,IndexError,TypeError) as e:
        record['state']='UNAVAILABLE_TIMING_METADATA';record['error']=type(e).__name__+': '+str(e)
    bjt=datetime.fromisoformat(t)+timedelta(hours=8)
    aux=list((a.data_root/'aux_256'/bjt.strftime('%Y%m')).glob(bjt.strftime('%Y%m%d%H%M')+'_*_aux.npz'))
    if len(aux)==1:
        with np.load(aux[0],allow_pickle=False) as z:
            if 'obs_time_offset_s' in z:
                record['cached_scan_midpoint_offset_s']=float(z['obs_time_offset_s'][gr,gc])
                record['cached_scan_midpoint_UTC']=(datetime.fromisoformat(t)+timedelta(seconds=record['cached_scan_midpoint_offset_s'])).isoformat(' ')
    records.append(record)
result={'scope':'51 unique frozen validation station/target keys; HDF NOMObsTime only','records':records,
        'test_payload_opened':False,'AGRI_science_payload_read':False}
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'n':len(records),'n_success':sum(r['state']=='READ_TIMING_METADATA' for r in records),'first':records[:1]}),flush=True)
