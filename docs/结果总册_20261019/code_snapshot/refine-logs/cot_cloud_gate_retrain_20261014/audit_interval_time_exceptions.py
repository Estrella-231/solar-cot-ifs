"""Read original HDF timing for quarantined train/val metadata exceptions."""
import argparse,csv,json
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['collect','raw'],required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.mode=='collect':
        exc=json.loads(a.input.read_text());times=[r['utc'] for r in exc if 'error' in r];data=Path('/public/home/slfu/ttzhou/Auxiliary_data/HuNan_AGRI_Preprocessed');records=[]
        for utc in times:
            t=datetime.fromisoformat(utc);b=t+timedelta(hours=8);paths=list((data/'aux_256'/b.strftime('%Y%m')).glob(b.strftime('%Y%m%d%H%M')+'_*_aux.npz'));assert len(paths)==1
            with np.load(paths[0]) as z:
                rec=dict(utc=utc,aux_path=str(paths[0]),obs_offsets=[float(z['obs_time_offset_s'][r,c]) for r,c in [(163,148),(73,150)]])
                for k in ['source_hdf','source_type','obs_time_valid_fraction','obs_time_offset_min_s','obs_time_offset_max_s','incomplete_scans','data_quality','aux_schema_version']:
                    if k in z:
                        v=z[k];rec[k]=v.item() if v.ndim==0 else v.tolist()
                        if isinstance(rec[k],bytes):rec[k]=rec[k].decode()
                records.append(rec)
        a.output.write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
    else:
        import h5py
        records=json.loads(a.input.read_text())
        for rec in records:
            try:
                path=Path(rec['source_hdf']);rec['original_rows']={}
                with h5py.File(path) as h:
                    rec['HDF_time_attributes']={k:str(v) for k,v in h.attrs.items() if 'time' in k.lower() or 'date' in k.lower()}
                    names=[]
                    h.visititems(lambda name,obj:names.append(name) if isinstance(obj,h5py.Dataset) and name.split('/')[-1]=='NOMObsTime' else None)
                    rec['timing_dataset_paths']=names
                    if len(names)!=1:raise ValueError(f'expected exactly one NOMObsTime, found {names}')
                    v=h[names[0]]
                    for st,r in [('sili',694),('zhujia',612)]:
                        vv=v[r].tolist();parsed=[]
                        for t in vv:
                            try:parsed.append(datetime.strptime(str(int(t)),'%Y%m%d%H%M%S%f'))
                            except ValueError:pass
                        row=dict(raw=vv)
                        if len(parsed)==2:
                            mid=parsed[0]+(parsed[1]-parsed[0])/2;row['mid_utc']=mid.isoformat();row['offset_s']=(mid-datetime.fromisoformat(rec['utc'])).total_seconds()
                        else:row['state']='UNPARSEABLE_SOURCE_TIMING'
                        rec['original_rows'][st]=row
            except Exception as e:rec['raw_read_error']=str(e)
        a.output.write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
if __name__=='__main__':main()
