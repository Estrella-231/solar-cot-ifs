import argparse,hashlib,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--selection',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
d=json.loads(a.selection.read_text());n=0
for r in d['frames']:
    assert r['split']=='val'
    for name,digest in [('data_path','data_sha256'),('aux_path','aux_sha256')]:
        f=a.root/('data' if name=='data_path' else 'aux')/Path(r[name]).name
        assert hashlib.sha256(f.read_bytes()).hexdigest()==r[digest],str(f);n+=1
a.out.write_text(json.dumps(dict(state='PASS_G4_HISTORY_LOCAL_HASHES',files=n,frames=len(d['frames']),
    selection_sha256=hashlib.sha256(a.selection.read_bytes()).hexdigest(),root=str(a.root),test_payloads_read=0),indent=2))
print('PASS_G4_HISTORY_LOCAL_HASHES',n,flush=True)
