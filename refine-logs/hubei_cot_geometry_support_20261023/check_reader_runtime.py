"""Isolated CPU reader dependencies and actual NetCDF fixture check, no GPU."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;spec=json.loads((P/'READER_RUNTIME_SPEC.json').read_text());wheels=json.loads((P/'READER_WHEELS.json').read_text())
target=P/spec['dependency_target'];assert not target.exists()
for r in wheels:assert hashlib.sha256((P/'reader_wheels'/r['file']).read_bytes()).hexdigest()==r['sha256']
subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--find-links',str(P/'reader_wheels'),'--target',str(target),*spec['pins']],check=True)
env=os.environ.copy();env.update(PYTHONPATH=str(target),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
subprocess.run([sys.executable,str(P/'test_reader.py')],env=env,check=True)
receipt=json.loads((P/'SYNTHETIC_READER_TEST.json').read_text());assert receipt['state']=='SYNTHETIC_NETCDF_READER_PASS'
(P/'READER_RUNTIME_RECEIPT.json').write_text(json.dumps(dict(state='CPU_READER_SYNTHETIC_RUNTIME_READY',python=sys.executable,target=str(target),wheels=wheels,
    global_environment_modified=False,synthetic_test_sha256=hashlib.sha256((P/'SYNTHETIC_READER_TEST.json').read_bytes()).hexdigest(),
    actual_CPP_source_access_verified=False,GPU_environment_ready_claim=False),indent=2))
print('ISOLATED_CPU_READER_READY',flush=True)
