"""Reuse verified GPU histories, rebuild only CPU output in a fresh directory."""
import json,subprocess,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent;out=Path('/dev/shm/hubei_trajectory_pair_pack_20261023_attempt2')
assert not out.exists() and not(P/'CPU_PACK_REPAIR_attempt2.json').exists()
frames=0
for i in range(4):
    r=json.loads((P/f'cache_attempt1/shard{i}/COMPLETE.json').read_text())
    assert r['state']=='FROZEN_R_TRAIN_HISTORY_CACHED' and r['max_geometry_difference']<1e-6;frames+=r['frames']
assert frames==4598
start=time.monotonic()
subprocess.run([sys.executable,str(P/'build_pair_pack.py'),'--cache',str(P/'cache_attempt1'),'--out',str(out)],check=True)
report=json.loads((out/'REPORT.json').read_text());assert report['state']=='TRAIN_ONLY_TRAJECTORY_PAIR_PACK_COMPLETE_NOT_TRAINED'
(P/'PAIR_PACK_COMPLETE.json').write_text(json.dumps(report,indent=2))
receipt=dict(state='PASS_CPU_ONLY_PAIR_PACK_REPAIR',pack=str(out),frames=frames,seconds=time.monotonic()-start,
    prior_failure='215754 exit1; glob matched shard0.log; partial attempt1 retained',
    reused_GPU_history_shards=4,new_GPU_inference_runs=0,test_buffer_payloads_read=0)
(P/'CPU_PACK_REPAIR_attempt2.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
