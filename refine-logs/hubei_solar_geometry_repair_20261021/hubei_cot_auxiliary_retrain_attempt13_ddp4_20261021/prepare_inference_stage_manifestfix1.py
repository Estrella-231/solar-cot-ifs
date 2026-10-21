"""Adapt the corrected cache ledger to the existing inference remapping schema.

No satellite data are copied or changed. The compatibility view reads the exact
node-local train/validation payloads used for the completed paired training.
"""
import csv
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
base=root.parent
source=base/'prepared_attempt12/stage_manifest.csv'
cache=Path('/dev/shm/hubei_solar_corrected_20261021_attempt10')
out=root/'inference_stage_view_manifestfix1'
assert cache.is_dir() and (root/'PAIR_COMPLETE_attempt13.json').is_file()
rows=list(csv.DictReader(source.open()))
assert len(rows)==10473 and {v['split'] for v in rows}<={'train','val'}
mapping={}
for row in rows:
    for kind in ('data','aux'):
        path=Path(row[kind+'_path'])
        assert path.is_file(),path
        path.resolve().relative_to((cache/kind).resolve())
        key=(kind,str(path))
        payload=dict(kind=kind,source=str(path),destination=str(path),
                     expected_payload_sha256=row[kind+'_sha256'],split=row['split'])
        if key in mapping: assert mapping[key]==payload
        mapping[key]=payload
# The frozen full index already names these exact node-local paths. Every
# train/val input and target reference must resolve through this unchanged view.
references=0
for row in csv.DictReader((base/'prepared_attempt12/indexes/full.csv').open()):
    if row['split'] not in ('train','val'): raise ValueError('Unexpected split')
    for field,kind in [('data_paths','data'),('aux_paths','aux')]:
        for path in row[field].split('|'):
            assert (kind,path) in mapping,(kind,path)
            references+=1
out.mkdir(exist_ok=False)
for kind in ('data','aux'): (out/kind).symlink_to(cache/kind,target_is_directory=True)
manifest=out/'stage_manifest.csv'
with manifest.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(next(iter(mapping.values())).keys()))
    writer.writeheader();writer.writerows(mapping.values())
receipt=dict(state='PASS_INFERENCE_CACHE_MANIFEST_COMPATIBILITY',source_manifest=str(source),
             source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
             stage_root=str(out),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
             unique_payloads=len(mapping),all_index_references_checked=references,
             payloads_copied=0,payloads_changed=0,test_payloads_read=0,buffer_payloads_read=0)
(root/'MANIFEST_FIX_PASS.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
