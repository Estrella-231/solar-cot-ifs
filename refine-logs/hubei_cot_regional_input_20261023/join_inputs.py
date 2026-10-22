import csv,json,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent;src=P.parent/'hubei_cot_linear_increment_20261023/CPP_SOURCE_KEYS.csv'
cpp=list(csv.DictReader(src.open()));agri={r['bjt']:r for r in csv.DictReader((P/'corrected_agri_manifest.csv').open())};rr=[]
for r in cpp:
 if r['listed']=='1' and r['time_bjt'] in agri:
  a=agri[r['time_bjt']];assert r['split']==a['split'];rr.append(dict(time_bjt=r['time_bjt'],split=r['split'],CPP_path=r['path'],AGRI_UTC=a['utc'],AGRI_path=a['data_path'],AGRI_sha256=a['data_sha256'],aux_path=a['aux_path'],aux_sha256=a['aux_sha256']))
with (P/'selected_cpp_agri.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
rep=dict(state='METADATA_JOIN_COMPLETE_NO_PIXEL_READ',rows=len(rr),by_split={s:dict(frames=sum(r['split']==s for r in rr),dates=len(set(r['time_bjt'][:8] for r in rr if r['split']==s))) for s in ['train','val']},missing_corrected_agri_rows=sum(r['listed']=='1' and r['time_bjt'] not in agri for r in cpp),cpp_source_index_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),corrected_agri_manifest_sha256=hashlib.sha256((P/'corrected_agri_manifest.csv').read_bytes()).hexdigest(),test_payloads_read=0,buffer_payloads_read=0)
(P/'JOIN_REPORT.json').write_text(json.dumps(rep,indent=2));print(rep)
