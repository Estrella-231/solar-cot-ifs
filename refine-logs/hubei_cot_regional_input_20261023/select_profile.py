"""Choose 2 or 4 GPUs from measured throughput, never validation scores."""
import argparse,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
a=ap.parse_args();assert not a.out.exists()
gate=json.loads((a.root/'GPU_EQUIVALENCE_attempt1.json').read_text());assert gate['state']=='PASS_2_4_GPU_GLOBAL_BATCH_EQUIVALENCE'
pp={n:json.loads((a.root/f'profile{n}_attempt1/PROFILE.json').read_text()) for n in (2,4)}
assert all(p['amp_checkpoint_replay_max_abs_difference']<1e-7 and p['global_batch']==128 and not p['val_scored'] for p in pp.values())
speedup=pp[4]['frames_per_second']/pp[2]['frames_per_second']
result=dict(state='PROFILE_COMPLETE_NO_TRAINING',selected_world=4 if speedup>=1.15 else 2,speedup_4_vs_2=speedup,
            profiles=pp,selection_rule='use4 if at least15 percent faster; otherwise2',
            auto_start_training=False,global_batch=128)
a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
