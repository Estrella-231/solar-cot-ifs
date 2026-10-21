"""Verify no-padding four-rank full-epoch sample and optimizer-step coverage."""
import json
from pathlib import Path
from ddp_runtime import GlobalBatchShard

records=[]
for n,shuffle in ((4086,True),(407,False)):
    rank_batches=[list(GlobalBatchShard(n,r,4,shuffle)) for r in range(4)]
    joined=[]
    for batches in zip(*rank_batches):
        ids=[i for part in batches for i in part]
        assert len(ids)==len(set(ids)) and len(ids)<=16
        joined.extend(ids)
    assert sorted(joined)==list(range(n))
    assert len(rank_batches[0])==(n+15)//16
    records.append(dict(rows=n,steps=len(rank_batches[0]),tail_sizes=[r[-1].__len__() for r in rank_batches],
                        duplicates=0,dropped=0))
out=dict(state='PASS_EXACT_DDP4_GLOBAL_BATCH_COVERAGE',global_batch=16,records=records)
(Path(__file__).parent/'SAMPLER_PASS.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
