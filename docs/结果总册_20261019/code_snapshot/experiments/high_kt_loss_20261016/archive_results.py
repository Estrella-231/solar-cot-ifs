"""Keep full weights off Git and export compact evidence plus source hashes."""
import hashlib,json,shutil,tarfile
from pathlib import Path
from PIL import Image

here=Path(__file__).resolve().parent;run=here/'runs/formal_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
e=here/'evidence';e.mkdir(exist_ok=True)
for name in ('completion.json','setup.json','profile.json','validation_rows.csv'):
    shutil.copy2(run/name,e/name)
for f in run.glob('seed*/*/*/*.json'):
    p=e/f.relative_to(run);p.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(f,p)
completion=json.loads((run/'completion.json').read_text())
assert len(completion['runs'])==24 and completion['test_payloads_read']==0
for r in completion['runs']:
    folder=run/f'seed{r["seed"]}'/r['station']/r['arm']
    assert sha(folder/'best.pt')==r['checkpoint_sha256'] and sha(folder/'validation_predictions.npz')==r['prediction_sha256']
archives=here/'archives';archives.mkdir(exist_ok=True)
archive=archives/'formal_v1_weights_predictions_and_logs.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
    tar.add(run,arcname='formal_v1')
receipt=dict(state='COMPLETE_24_LOCAL_GPU_RUNS_AND_INDEPENDENT_ANALYSIS',full_training_archive_sha256=sha(archive),full_training_archive_bytes=archive.stat().st_size,
             all_checkpoint_and_prediction_hashes_match=True,training_seconds=completion['elapsed_seconds'],
             source_transfer_sha256=sha(here/'source_transfer_receipt.json'),test_payloads_read=0,qsub_used=False,
             routine_execution_and_monitor_model='gpt-6-luna',documented_invocation_matches_actual=True,
             source_caches_and_weights='data/ and runs/ local, remote archive under experiments/high_kt_loss_20261016; not Git',
             candidate_status='REJECTED_PREDECLARED_OVERALL_AND_LOW_REGIME_PROTECTION',model_replaced=False)
(here/'completion_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
files=[]
for p in sorted(here.rglob('*')):
    if not p.is_file() or any(part in ('data','runs','archives','__pycache__') for part in p.relative_to(here).parts) or p.name=='artifact_manifest.json':continue
    if p.suffix=='.png':
        with Image.open(p) as im:im.verify()
    files.append(dict(path=p.relative_to(here).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
(here/'artifact_manifest.json').write_text(json.dumps(dict(state='PASS_24_RUN_HASHES_COMPACT_ARTIFACTS_AND_FIGURES',files=files,test_payloads_read=0),indent=2)+'\n')
print(json.dumps(receipt,indent=2))
