"""CPU-only date-role selection and one local-to-local history transfer."""
import csv,hashlib,json,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent;BASE=P.parent
assert not (P/'TRANSFER.json').exists() and not (P/'SELECTION.json').exists()
roles=json.loads((BASE/'R_INTERNAL_ROLES.json').read_text())
dates={'fit':set(roles['fit_dates']),'select':set(roles['select_dates'])}
assert not dates['fit'].intersection(dates['select'])
pairs=[]
for r in csv.DictReader((BASE/'G4_CANDIDATE_KEYS.csv').open()):
    if r['split']!='train':continue
    hist=r['history_BJT'].split('|')
    for role,allowed in dates.items():
        if r['target_time_BJT'][:8] in allowed and all(k[:8] in allowed for k in hist):
            pairs.append(dict(role=role,**r));break
assert sum(r['role']=='fit' for r in pairs)==19420
assert sum(r['role']=='select' for r in pairs)==2840
lookup={r['bjt']:r for r in csv.DictReader((BASE/'corrected_agri_manifest.csv').open())}
keys=sorted({k for r in pairs for k in r['history_BJT'].split('|')});frames=[lookup[k] for k in keys]
assert all(r['split']=='train' for r in frames)
selection=dict(state='FROZEN_TRAIN_ONLY_TRAJECTORY_SELECTION',frames=frames,pairs=pairs,
    date_roles_sha256=hashlib.sha256((BASE/'R_INTERNAL_ROLES.json').read_bytes()).hexdigest(),
    counts={role:dict(pairs=sum(r['role']==role for r in pairs),
         init_dates=len({r['init_time_BJT'][:8] for r in pairs if r['role']==role}),
         target_dates=len({r['target_time_BJT'][:8] for r in pairs if r['role']==role})) for role in dates},
    test_buffer_payloads_read=0,outer_validation_payloads_read=0)
(P/'SELECTION.json').write_text(json.dumps(selection,indent=2))
source=Path('/dev/shm/hubei_solar_corrected_20261021_attempt10')
files=[str(Path(r[k]).relative_to(source)) for r in frames for k in ('data_path','aux_path')]
(P/'source_files.txt').write_text('\n'.join(files)+'\n')
dest='/dev/shm/hubei_region_trajectory_train_20261023_attempt1'
free=subprocess.check_output(['ssh','-o','BatchMode=yes','node23','df -B1 --output=avail /dev/shm'],text=True)
assert int(free.splitlines()[-1])>len(frames)*7_000_000+20_000_000_000
start=time.monotonic();h=hashlib.sha256();size=0
with (P/'transfer_source.log').open('wb') as se,(P/'transfer_destination.log').open('wb') as de:
    src=subprocess.Popen(['ssh','-o','BatchMode=yes','node22',f'tar -C {source} -cf - -T {P}/source_files.txt'],stdout=subprocess.PIPE,stderr=se)
    dst=subprocess.Popen(['ssh','-o','BatchMode=yes','node23',f'test ! -e {dest} && mkdir {dest} && tar -C {dest} -xf -'],stdin=subprocess.PIPE,stdout=de,stderr=de)
    for b in iter(lambda:src.stdout.read(8<<20),b''):
        dst.stdin.write(b);h.update(b);size+=len(b)
    dst.stdin.close();a=src.wait();b=dst.wait()
result=dict(state='TRAIN_HISTORY_TRANSFER_COMPLETE' if a==b==0 else 'FAILED_PRESERVE_PARTIAL',
    source_exit=a,destination_exit=b,bytes=size,stream_sha256=h.hexdigest(),seconds=time.monotonic()-start,
    frames=len(frames),destination=dest,counts=selection['counts'],GPU_used=False,test_payloads_read=0)
(P/'TRANSFER.json').write_text(json.dumps(result,indent=2));assert a==b==0,result
subprocess.run(['ssh','-o','BatchMode=yes','node23',
    f'/public/home/slfu/miniconda3/envs/swc/bin/python {P}/verify_history.py --root {dest} --selection {P}/SELECTION.json --out {P}/STAGE_AUDIT.json'],check=True)
print(json.dumps(result),flush=True)
