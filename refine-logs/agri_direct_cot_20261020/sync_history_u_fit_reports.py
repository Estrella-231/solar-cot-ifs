"""Sync this diagnostic's code, reports and receipts, excluding arrays/weights."""
import hashlib,json,subprocess,tarfile
from pathlib import Path

B=Path(__file__).resolve().parent;REPO=B.parents[1]
REMOTE='/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907'
REL=B.relative_to(REPO).as_posix()
OUT=B/'results/u_train_fit_diagnostic_v1_20261021'
names=['VISIBLE_NEIGHBOR_HISTORY_CONTEXT_PLAN_20261021.md','VISIBLE_NEIGHBOR_HISTORY_CONTEXT_RESULTS_20261021.md',
       'extract_visible_neighbor_context.py','run_visible_neighbor_context_cpu_20261021.pbs','submit_visible_context_guard.py',
       'plot_visible_neighbor_context.py','U_TRAIN_FIT_DIAGNOSTIC_PLAN_20261021.md','U_TRAIN_FIT_DIAGNOSTIC_RESULTS_20261021.md',
       'extract_u_train_fit_sample.py','run_u_train_fit_sample_cpu_20261021.pbs','submit_u_fit_guard.py',
       'audit_u_train_fit.py','plot_u_train_fit.py','NEXT_EVIDENCE_ROUTE_20261021.md','sync_history_u_fit_reports.py']
files=[B/n for n in names]
for n in ('visible_context_submission_v2_20261021.json','visible_context_sources_20261021.json','u_fit_submission_20261021.json','u_fit_sources_20261021.json'):
    files.append(B/'results'/n)
for sub in ('visible_neighbor_history_context_diagnostic_v1_20261021','u_train_fit_diagnostic_v1_20261021'):
    files += [p for p in (B/'results'/sub).iterdir() if p.is_file() and p.suffix in ('.md','.csv','.json','.jsonl','.png','.pdf')
              and p.name!='server_sync_verified.json']
files=sorted(set(files));expected={p.relative_to(REPO).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
marker='## VISIBLE_HISTORY_AND_U_FIT_COMPLETE_20261021'
entry=marker+(B/'EXPERIMENT_TRACKER.md').read_text(encoding='utf-8').split(marker,1)[1]
verifier=OUT/'verify_history_u_fit_sync.py'
destinations=[REL+'/results/'+s+'/server_sync_verified.json' for s in
              ('visible_neighbor_history_context_diagnostic_v1_20261021','u_train_fit_diagnostic_v1_20261021')]
verifier.write_text('import hashlib,json\nfrom pathlib import Path\nroot=Path('+repr(REMOTE)+')\nexpected='+repr(expected)+'\n'
    'for n,h in expected.items():\n assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n\n'
    'p=root/'+repr(REL+'/EXPERIMENT_TRACKER.md')+'\nmarker='+ascii(marker)+'\nentry='+ascii(entry)+'\n'
    'text=p.read_text(encoding="utf-8")\nif marker not in text:\n'
    ' with p.open("a",encoding="utf-8") as f:f.write("\\n"+entry)\n'
    'assert entry in p.read_text(encoding="utf-8")\n'
    'r={"state":"PASS_HISTORY_U_FIT_REPORT_SYNC","verified_files":len(expected),"files":expected,'
    '"weights_uploaded":False,"new_prediction_npy_uploaded":False,"raw_sample_arrays_uploaded":False}\n'
    'for d in '+repr(destinations)+':\n (root/d).write_text(json.dumps(r,indent=2)+"\\n",encoding="utf-8")\n'
    'print(json.dumps({k:v for k,v in r.items() if k!="files"}))\n',encoding='ascii')
bundle=OUT/'history_u_fit_reports.tar'
with tarfile.open(bundle,'w') as tf:
    for p in files+[verifier]:
        info=tf.gettarinfo(str(p),arcname=p.relative_to(REPO).as_posix());info.mtime=0
        with p.open('rb') as f:tf.addfile(info,f)
remote_tar=REMOTE+'/'+REL+'/results/history_u_fit_reports_20261021.tar'
subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15',str(bundle),'zjnu-hpc:'+remote_tar],check=True)
cmd='cd '+REMOTE+' && tar -xf '+remote_tar+' && /public/home/slfu/miniconda3/envs/swc/bin/python '+verifier.relative_to(REPO).as_posix()
subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','zjnu-hpc',cmd],check=True)
for rel in destinations:
    local=REPO/rel
    subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15','zjnu-hpc:'+REMOTE+'/'+rel,str(local)],check=True)
    r=json.loads(local.read_text());assert r['files']==expected
print(json.dumps({k:v for k,v in r.items() if k!='files'},indent=2))
