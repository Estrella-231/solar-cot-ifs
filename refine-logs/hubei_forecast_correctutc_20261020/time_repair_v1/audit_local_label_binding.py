import csv,io,json,tarfile,hashlib
from pathlib import Path
from datetime import datetime,timedelta
from collections import Counter
import numpy as np
r=Path(__file__).resolve().parent;p=r/'runtime_attempt3/pack'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
a=r/'time_lineage_v2.tar.gz';assert sha(a)==(r/'metadata_archive_sha256.txt').read_text(encoding='utf-8-sig').split()[0]
with tarfile.open(a) as t:
 def records(n):return list(csv.DictReader(io.TextIOWrapper(t.extractfile(n),encoding='utf-8-sig')))
 labs=records('ghi15_train_val_daylight_clear.csv');dates={x['date_BJT'].replace('-',''):x['split'] for x in records('day_assignment.csv')}
 label={}
 for x in labs:
  assert x['split'] in ('train','val')
  tm=x['target_time_BJT'].replace('-','').replace(':','').replace(' ','')[:12]
  if x['valid_daylight_clear_sky_label'] not in ('1','1.0','True','true'):continue
  k=x['station'],tm;assert k not in label;label[k]=x
rows=list(csv.DictReader((p/'rows.csv').open()));y=np.load(p/'ghi_wm2.npy');cl=np.load(p/'clear_sky_wm2.npy');seen=set();support=Counter();date_support=Counter();yerr=[];cerr=[]
for i,x in enumerate(rows):
 k=x['station'],x['init_time_bjt'],x['target_time_bjt'];assert k not in seen;seen.add(k)
 assert int(x['index'])==i and int(x['lead_minutes']) in range(15,241,15)
 assert datetime.strptime(k[2],'%Y%m%d%H%M')-datetime.strptime(k[1],'%Y%m%d%H%M')==timedelta(minutes=int(x['lead_minutes']))
 l=label[k[0],k[2]];assert dates[k[1][:8]]==dates[k[2][:8]]==l['split']==x['split']
 assert l['reason']=='complete' and int(l['source_records'])>=15 and int(l['mask'])==1
 assert y[i]==np.float32(float(l['ghi15_raw'])) and cl[i]==np.float32(float(l['clear_sky_ghi_15min_wm2']))
 assert float(x['ghi_wm2'])==float(l['ghi15_raw']) and float(x['clear_sky_wm2'])==float(l['clear_sky_ghi_15min_wm2'])
 yerr.append(abs(float(y[i])-float(l['ghi15_raw'])));cerr.append(abs(float(cl[i])-float(l['clear_sky_ghi_15min_wm2'])));support[x['split']+'/'+x['station']]+=1;date_support[x['split']+'/'+k[2][:8]]+=1
receipt={'state':'LOCAL_INDEPENDENT_ALL_TRAIN_VAL_GHI_BINDING_PASS','samples':len(rows),'split_station':dict(support),'target_dates':{s:len([d for d in date_support if d.startswith(s+'/')]) for s in ('train','val')},'checks':['unique station/init/target','target minus init equals canonical 15..240min lead','actual BJT init/target in same declared split','original complete15 source records and mask','row GHI and clear sky exact against source labels','packed GHI and clear sky exact float32 conversion against source labels'],'max_float32_ghi_rounding_wm2':max(yerr),'max_float32_clear_rounding_wm2':max(cerr),'sha256':{n:sha(p/n) for n in ('rows.csv','ghi_wm2.npy','clear_sky_wm2.npy','pack_report.json')},'metadata_archive_sha256':sha(a),'test_payloads_read':0,'buffer_payloads_read':0}
(r/'LOCAL_INDEPENDENT_LABEL_BINDING_AUDIT.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt,indent=2))
