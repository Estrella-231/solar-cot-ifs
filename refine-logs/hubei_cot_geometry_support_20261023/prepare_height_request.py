"""Bind only eight documented TRAIN source files for one calendar-fixed history."""
import csv,hashlib,json
from pathlib import Path
from datetime import datetime,timedelta,timezone
P=Path(__file__).resolve().parent;manifest=P.parent/'hubei_cot_regional_input_20261023/selected_cpp_agri.csv'
rows=list(csv.DictReader(manifest.open(encoding='utf-8')));lookup={r['time_bjt']:r for r in rows};init=datetime(2025,8,6,12,0);selected=[]
for off in range(105,-1,-15):
    t=init-timedelta(minutes=off);r=lookup[t.strftime('%Y%m%d%H%M')]
    assert r['split']=='train' and r['AGRI_UTC']==(t-timedelta(hours=8)).strftime('%Y%m%d%H%M')
    assert (t-timedelta(hours=8)).strftime('%Y%m%d%H%M%S') in Path(r['CPP_path']).name
    selected.append(dict(time_bjt=r['time_bjt'],split='train',UTC=r['AGRI_UTC'],source=r['CPP_path']))
(P/'HEIGHT_SOURCE_REQUEST.json').write_text(json.dumps(dict(state='EIGHT_TRAIN_HISTORY_SOURCES_BOUND_NOT_EXTRACTED',init_time_BJT=init.isoformat(),rows=selected,
    input_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),release_time_verified=False,CTH_units_verified=False,
    reason='Current input-source audit only; acquisition time is not a release-time receipt',test_payloads_read=0),indent=2))
(P/'ACCESS_RECEIPT.json').write_text(json.dumps(dict(recorded_UTC=datetime.now(timezone.utc).isoformat(),
    FD107=dict(host='10.158.205.107',port=22,user='zjnu',command='ssh -o BatchMode=yes -o ConnectTimeout=12 FD-107 hostname',exit_code=1,error='Connection closed by 10.158.205.107 port 22'),
    FD201=dict(host='10.158.205.201',port=22,user='zjnu',command='ssh -o BatchMode=yes -o ConnectTimeout=12 FD-201 hostname',exit_code=1,error='Connection closed by 10.158.205.201 port 22'),
    college_check='CPP_NOT_VISIBLE_AT_DOCUMENTED_PATH',claim='Source access unavailable, not evidence that files are absent',new_GPU_jobs=0),indent=2))
print('EIGHT_TRAIN_SOURCES_BOUND')
