"""Freeze and copy only required validation history AGRI from node22 to node23."""
import csv,hashlib,json,subprocess,time
from pathlib import Path

P=Path(__file__).resolve().parent
manifest=list(csv.DictReader((P/'corrected_agri_manifest.csv').open()))
lookup={r['bjt']:r for r in manifest}
pairs=[r for r in csv.DictReader((P/'G4_CANDIDATE_KEYS.csv').open()) if r['split']=='val']
keys=sorted({k for r in pairs for k in r['history_BJT'].split('|')})
rows=[lookup[k] for k in keys];assert all(r['split']=='val' for r in rows)
selection=dict(state='FROZEN_G4_VALIDATION_HISTORY_KEYS',frames=rows,
               initializations=sorted({r['init_time_BJT'] for r in pairs}),
               pairs=pairs,split='val',test_buffer_payloads_read=0)
assert not (P/'G4_SELECTION.json').exists()
(P/'G4_SELECTION.json').write_text(json.dumps(selection,indent=2))
source=Path('/dev/shm/hubei_solar_corrected_20261021_attempt10')
files=[]
for r in rows:
    files.extend([str(Path(r['data_path']).relative_to(source)),str(Path(r['aux_path']).relative_to(source))])
(P/'g4_source_files.txt').write_text('\n'.join(files)+'\n')
dest='/dev/shm/hubei_g4_region_20261023_attempt1'
free=subprocess.check_output(['ssh','-o','BatchMode=yes','node23','df -B1 --output=avail /dev/shm'],text=True)
assert int(free.splitlines()[-1])>10_000_000_000
start=time.monotonic();h=hashlib.sha256();size=0
with (P/'g4_source_transfer.log').open('wb') as se,(P/'g4_destination_transfer.log').open('wb') as de:
    src=subprocess.Popen(['ssh','-o','BatchMode=yes','node22',f'tar -C {source} -cf - -T {P}/g4_source_files.txt'],stdout=subprocess.PIPE,stderr=se)
    dst=subprocess.Popen(['ssh','-o','BatchMode=yes','node23',f'test ! -e {dest} && mkdir {dest} && tar -C {dest} -xf -'],stdin=subprocess.PIPE,stdout=de,stderr=de)
    for chunk in iter(lambda:src.stdout.read(8<<20),b''):
        dst.stdin.write(chunk);size+=len(chunk);h.update(chunk)
    dst.stdin.close();a=src.wait();b=dst.wait()
report=dict(state='G4_HISTORY_TRANSFER_COMPLETE' if a==b==0 else 'FAILED_PRESERVE_PARTIAL',
            source_exit=a,destination_exit=b,bytes=size,stream_sha256=h.hexdigest(),seconds=time.monotonic()-start,
            frames=len(rows),initializations=len(selection['initializations']),destination=dest)
(P/'G4_TRANSFER.json').write_text(json.dumps(report,indent=2));assert a==b==0,report
subprocess.run(['ssh','-o','BatchMode=yes','node23',
    f'/public/home/slfu/miniconda3/envs/swc/bin/python {P}/verify_g4_stage.py --root {dest} --selection {P}/G4_SELECTION.json --out {P}/G4_STAGE_AUDIT.json'],check=True)
print(json.dumps(report),flush=True)
