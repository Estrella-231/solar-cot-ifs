"""One local-disk to local-disk pack transfer; no raw/shared-source reread."""
import hashlib,json,subprocess,time
from pathlib import Path

P=Path(__file__).resolve().parent;PACK='/dev/shm/hubei_regional_r_pack_20261023_attempt2'
PY='/public/home/slfu/miniconda3/envs/swc/bin/python'
assert not (P/'STAGE_NODE23_TRANSFER.json').exists()
free=subprocess.check_output(['ssh','-o','BatchMode=yes','node23','df -B1 --output=avail /dev/shm'],text=True)
assert int(free.splitlines()[-1])>20_000_000_000
start=time.monotonic();count=0;h=hashlib.sha256()
with (P/'stage_node23_source.log').open('wb') as se,(P/'stage_node23_destination.log').open('wb') as de:
    source=subprocess.Popen(['ssh','-o','BatchMode=yes','node22',f'tar -C {PACK} -cf - .'],stdout=subprocess.PIPE,stderr=se)
    dest=subprocess.Popen(['ssh','-o','BatchMode=yes','node23',f'test ! -e {PACK} && mkdir {PACK} && tar -xf - -C {PACK}'],stdin=subprocess.PIPE,stdout=de,stderr=de)
    try:
        for block in iter(lambda:source.stdout.read(8<<20),b''):
            dest.stdin.write(block);h.update(block);count+=len(block)
        dest.stdin.close();a=source.wait();b=dest.wait()
    except Exception:
        source.terminate();dest.terminate();raise
r=dict(state='LOCAL_PACK_TRANSFER_COMPLETE' if a==b==0 else 'FAILED_PRESERVE_PARTIAL',source_node='node22',destination_node='node23',
       source_exit=a,destination_exit=b,tar_stream_bytes=count,tar_stream_sha256=h.hexdigest(),seconds=time.monotonic()-start)
(P/'STAGE_NODE23_TRANSFER.json').write_text(json.dumps(r,indent=2));assert a==b==0,r
subprocess.run(['ssh','-o','BatchMode=yes','node23',f'{PY} {P}/verify_staged_pack.py --pack {PACK} --out {P}/LOCAL_STAGE_REPORT_node23.json'],check=True)
