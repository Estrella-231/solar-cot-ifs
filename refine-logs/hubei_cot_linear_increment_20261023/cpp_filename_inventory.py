"""Bounded original-CPP filename inventory on frozen train/val AGRI keys only."""
import csv,json,os
from pathlib import Path
from datetime import datetime,timedelta
from collections import Counter
M=Path(__file__).with_name('agri_paths_trainval_staged.csv');R=Path('/home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP');O=Path('/home/Data_Pool/zjnu/hubei_cpp_regional_inventory_20261023');O.mkdir(exist_ok=False)
rows=[r for r in csv.DictReader(M.open()) if r['time_bjt']<='202603022359'];assert set(r['split'] for r in rows)=={'train','val'};cache={};records=[]
for r in rows:
 b=datetime.strptime(r['time_bjt'],'%Y%m%d%H%M');u=b-timedelta(hours=8);parent=R/u.strftime('%Y')/u.strftime('%Y%m%d')
 if str(parent) not in cache:
  try:cache[str(parent)]={x.name for x in os.scandir(parent) if x.name.endswith('.nc')}
  except FileNotFoundError:cache[str(parent)]=set()
  if len(cache)%25==0:print('directories',len(cache),flush=True)
 name='FY4B_AGRI_'+u.strftime('%Y%m%d%H%M%S')+'.nc';records.append(dict(time_bjt=r['time_bjt'],split=r['split'],path=str(parent/name),listed=int(name in cache[str(parent)])))
with (O/'source_keys.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
summary={}
for split in ['train','val']:
 rr=[r for r in records if r['split']==split];times=sorted(datetime.strptime(r['time_bjt'],'%Y%m%d%H%M') for r in rr if r['listed']);present=set(times);inits=[t for t in times if all(t+timedelta(minutes=15*k) in present for k in range(-7,17))]
 summary[split]=dict(agri_keys=len(rr),cpp_filenames_listed=sum(r['listed'] for r in rr),days_with_listed_files=len(set(t.strftime('%Y%m%d') for t in times)),candidate_24frame_windows=len(inits),candidate_init_dates=len(set(t.strftime('%Y%m%d') for t in inits)),by_month=dict(Counter(t.strftime('%Y%m') for t in times)))
report=dict(state='CPP_FILENAME_INVENTORY_COMPLETE_NOT_PAYLOAD_QC',summary=summary,utc_directories=len(cache),CPP_payloads_read=0,GHI_labels_read=0,test_payloads_read=0,buffer_payloads_read=0,note='filename coverage only; candidate sequence support not complete QA or R-input consistency')
(O/'REPORT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
