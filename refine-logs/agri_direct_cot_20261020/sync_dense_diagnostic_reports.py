"""Mirror reports/receipts only; arrays and model weights remain local."""
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

BASE=Path(__file__).resolve().parent
REPO=BASE.parents[1]
REMOTE='/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907'
REL=BASE.relative_to(REPO).as_posix()
OUT=BASE/'results/dense_r_field_event_diagnostic_v1_20261021'
names=['DENSE_OBSERVED_R_REFERENCE_PLAN_20261021.md','DENSE_REFERENCE_INFERENCE_PROTOCOL_20261021.md',
       'INITIAL_CLOUD_VISIBILITY_DIAGNOSTIC_PLAN_20261021.md','DENSE_R_FIELD_EVENT_RESULTS_20261021.md',
       'retrieve_dense_observed_reference.py','verify_dense_archive_inputs.py','audit_dense_reference_events.py',
       'audit_initial_cloud_visibility.py','plot_dense_reference_diagnostics.py','sync_dense_diagnostic_reports.py']
files=[BASE/n for n in names]
for sub in ('dense_observed_v_r_reference_v1_20261021','dense_r_field_event_diagnostic_v1_20261021',
            'initial_cloud_visibility_v1_20261021'):
    files += [p for p in (BASE/'results'/sub).rglob('*') if p.is_file()
              and p.suffix.lower() in ('.csv','.json','.md','.png','.pdf')
              and p.name not in ('server_sync_verified.json','sync_report_bundle.tar')]
files=sorted(set(files))
expected={p.relative_to(REPO).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
marker='## DENSE_R_FIELD_EVENT_COMPLETE_20261021'
tracker=(BASE/'EXPERIMENT_TRACKER.md').read_text(encoding='utf-8')
entry=marker+tracker.split(marker,1)[1]
verifier=OUT/'verify_dense_report_sync.py'
verifier.write_text('import hashlib,json\nfrom pathlib import Path\n'
    'root=Path('+repr(REMOTE)+')\nexpected='+repr(expected)+'\n'
    'for n,h in expected.items():\n assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n\n'
    'p=root/'+repr(REL+'/EXPERIMENT_TRACKER.md')+'\n'
    'marker='+ascii(marker)+'\nentry='+ascii(entry)+'\n'
    'if marker not in p.read_text(encoding="utf-8"):\n'
    ' with p.open("a",encoding="utf-8") as f:f.write("\\n"+entry)\n'
    'assert marker in p.read_text(encoding="utf-8")\n'
    'result={"state":"PASS_DENSE_DIAGNOSTIC_REPORT_SYNC","verified_files":len(expected),'
    '"files":expected,"r_arrays_uploaded":False,"case_npz_uploaded":False,'
    '"new_observation_workbook_uploaded":False,"weights_uploaded":False}\n'
    '(root/'+repr(REL+'/results/dense_r_field_event_diagnostic_v1_20261021/server_sync_verified.json')+
    ').write_text(json.dumps(result,indent=2)+"\\n",encoding="utf-8")\n'
    'print(json.dumps({k:v for k,v in result.items() if k!="files"}))\n',encoding='ascii')
bundle=OUT/'sync_report_bundle.tar'
with tarfile.open(bundle,'w') as tf:
    for p in files:tf.add(p,arcname=p.relative_to(REPO).as_posix(),recursive=False)
    tf.add(verifier,arcname=verifier.relative_to(REPO).as_posix(),recursive=False)
subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15',str(bundle),
                'zjnu-hpc:'+REMOTE+'/refine-logs/agri_direct_cot_20261020/results/dense_diagnostic_reports_20261021.tar'],check=True)
command=('cd '+REMOTE+' && tar -xf '+REL+'/results/dense_diagnostic_reports_20261021.tar && '
         '/public/home/slfu/miniconda3/envs/swc/bin/python '+REL+'/results/dense_r_field_event_diagnostic_v1_20261021/verify_dense_report_sync.py')
subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','zjnu-hpc',command],check=True)
subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15',
                'zjnu-hpc:'+REMOTE+'/'+REL+'/results/dense_r_field_event_diagnostic_v1_20261021/server_sync_verified.json',
                str(OUT/'server_sync_verified.json')],check=True)
receipt=json.loads((OUT/'server_sync_verified.json').read_text())
assert receipt['files']==expected
print(json.dumps({k:v for k,v in receipt.items() if k!='files'},indent=2))
