"""Inspect only the six failed new-source headers and same-time alternatives."""
import csv,json,os
from pathlib import Path
import netCDF4 as nc
BASE=Path('/tmp/hubei_history_cot_new_period_full_20261022_attempt1')
INV=Path('/home/Data_Pool/zjnu/hubei_history_cot_confirmation_20261022_inventory_attempt1/hdf_files.csv')
failed=json.loads((BASE/'PREPROCESS_REPORT.json').read_text())['failures']
rows=list(csv.DictReader(INV.open()));answer=[]
for failure in failed:
    row=next(r for r in rows if r['path']==failure['source'])
    alternatives=[r for r in rows if r['time_BJT']==row['time_BJT']]
    record=dict(time_BJT=row['time_BJT'],original_failure=failure,alternatives=[])
    for r in alternatives:
        x=dict(path=r['path'],size_bytes=os.path.getsize(r['path']))
        try:
            with nc.Dataset(r['path']) as f:
                x['root_variables']=list(f.variables);x['groups']={k:list(v.variables) for k,v in f.groups.items()}
                ds=f.groups.get('Data',f);cal=f.groups.get('Calibration',f)
                x['all_15_NOM_channels_present']=all(f'NOMChannel{k:02d}' in ds.variables for k in range(1,16))
                x['all_15_CAL_channels_present']=all((('CAL' if 'Data' in f.groups else 'Cal')+f'Channel{k:02d}') in cal.variables for k in range(1,16))
        except Exception as e:x['error']=repr(e)
        record['alternatives'].append(x)
    answer.append(record)
dest=Path('/home/Data_Pool/zjnu/hubei_history_cot_confirmation_20261022/FAILED_HDF_HEADERS.json')
dest.write_text(json.dumps(answer,indent=2)+'\n');print(json.dumps(answer,indent=2))
