"""Verify node-local copied arrays against the already frozen pack hashes."""
import argparse,hashlib,json,socket
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('--pack',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args();assert not a.out.exists()
expected=json.loads((a.pack/'SHA256.json').read_text())
for name,digest in expected.items():
    h=hashlib.sha256()
    with (a.pack/name).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    assert h.hexdigest()==digest,name
assert json.loads((a.pack/'NORMALIZATION_AUDIT.json').read_text())['state']=='PASS_NORMALIZED_CONTEXT_FP16_RANGE'
r=dict(state='PASS_NODE_LOCAL_PACK_COPY',hostname=socket.gethostname(),pack=str(a.pack),files_verified=len(expected),
       bytes=sum(p.stat().st_size for p in a.pack.iterdir() if p.is_file()),GPU_used=False)
a.out.write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
