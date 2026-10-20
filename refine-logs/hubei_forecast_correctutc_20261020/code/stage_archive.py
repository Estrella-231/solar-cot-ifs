"""Verify allowlisted train/val export before extracting on node-local storage."""
import argparse,hashlib,json,tarfile
from pathlib import Path
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True,type=Path);p.add_argument('--sha-file',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if sha(a.archive)!=a.sha_file.read_text().split()[0]:raise ValueError('archive hash')
    with tarfile.open(a.archive) as t:
        members=t.getmembers()
        for m in members:
            n=Path(m.name)
            if n.is_absolute() or '..' in n.parts or not m.isfile() or 'true_agri' in m.name:raise ValueError('unexpected archive path')
        a.output.mkdir(parents=True);t.extractall(a.output)
    receipt=json.loads((a.output/'export_receipt.json').read_text())
    if receipt['test_payloads_read'] or receipt['buffer_payloads_read'] or receipt['true_future_agri_exported']:raise ValueError('export scope')
    for n,v in receipt['files'].items():
        if sha(a.output/n)!=v['sha256']:raise ValueError(('file hash',n))
    print('LOCAL_STAGE_FROZEN_FORECAST_INPUTS_PASS',sum(g['sequences'] for g in receipt['groups']),flush=True)
if __name__=='__main__':main()
