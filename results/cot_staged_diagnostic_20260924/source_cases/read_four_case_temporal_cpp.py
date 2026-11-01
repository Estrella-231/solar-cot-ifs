import csv,json
from pathlib import Path
import numpy as np
from netCDF4 import Dataset
BASE=Path('/tmp/cpp_audit_20260924')
rows=list(csv.DictReader((BASE/'four_case_temporal_cpp_manifest.csv').open()))
starts={'sili':(1368,2201),'zhujia':(1278,2203)}
cot=[];clp=[];qa=[];rowsout=[]
for r in rows:
 stamp=r['timestamp_utc'].replace('-','').replace(':','').replace('T','').replace('Z','');day=stamp[:8]
 path=Path('/home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP/2025')/day/f'FY4B_AGRI_{stamp}.nc'
 rr,cc=starts[r['station_id']]
 with Dataset(path,'r') as ds:
  cot.append(np.ma.filled(ds['COT'][rr:rr+16,cc:cc+16],np.nan).astype(np.float32))
  clp.append(np.ma.filled(ds['CLP'][rr:rr+16,cc:cc+16],np.nan).astype(np.float32))
  qa.append(np.ma.filled(ds['QA'][rr:rr+16,cc:cc+16],-1).astype(np.int8) if 'QA' in ds.variables else np.full((16,16),-1,np.int8))
  rowsout.append({**r,'source_path':str(path),'QA_variable_present':'QA' in ds.variables,'COT_min':float(np.nanmin(cot[-1])),'COT_max':float(np.nanmax(cot[-1])),'CLP_values':np.unique(clp[-1][np.isfinite(clp[-1])]).astype(int).tolist()})
np.savez_compressed(BASE/'four_case_temporal_cpp_windows.npz',cot=np.stack(cot),clp=np.stack(clp),qa=np.stack(qa))
(BASE/'four_case_temporal_cpp_receipt.json').write_text(json.dumps(rowsout,indent=2),encoding='utf-8')
print('read',len(rows),'QA files',sum(x['QA_variable_present'] for x in rowsout))
for x in rowsout: print(x['case_id'],x['station_id'],x['timestamp_utc'],'COT',x['COT_min'],x['COT_max'],'CLP',x['CLP_values'],'QA',x['QA_variable_present'])
