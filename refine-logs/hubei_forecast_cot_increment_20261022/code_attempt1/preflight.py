import argparse,json,subprocess,hashlib,shutil
from pathlib import Path
import torch,numpy as np
from train import ConditionalHead
from history_base import sha256
a=argparse.ArgumentParser();a.add_argument('--prepared',type=Path,required=True);a.add_argument('--gpus',required=True);a.add_argument('--output',type=Path,required=True);x=a.parse_args()
prep=json.loads((x.prepared/'PREPARATION.json').read_text());assert prep['state']=='COMPLETE_F_C_SPLIT_AND_FROZEN_INPUTS'
assert prep['test_payloads_read']==prep['buffer_payloads_read']==0
for st,meta in prep['stations'].items():
    for name,digest in meta['arrays'].items():assert sha256(x.prepared/st/(name+'.npy'))==digest,(st,name)
uuids=set(subprocess.check_output(['nvidia-smi','-i',x.gpus,'--query-gpu=uuid','--format=csv,noheader'],text=True).split())
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
busy=[line for line in apps.splitlines() if line.split(',')[0].strip() in uuids];assert not busy, busy
torch.manual_seed(42);m=ConditionalHead(308);z=torch.randn(9,308);base=torch.rand(9);available=torch.ones(9)
assert torch.equal(m(z,base,available),base)
with torch.no_grad():m.delta[-1].bias.fill_(.25)
assert torch.equal(m(z,base,torch.zeros(9)),base)
loss=m(z,base,available).square().mean();loss.backward();assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in m.parameters())
assert shutil.disk_usage('/dev/shm').free>32*1024**3
x.output.write_text(json.dumps({'state':'PASS_INPUT_HASHES_CONDITIONAL_ZERO_INIT_MASK_AND_GRADIENT','torch':torch.__version__,'numpy':np.__version__,'gpu_ids':x.gpus,'allocated_uuid':sorted(uuids),'preexisting_compute_processes':busy,'test_payloads_read':0,'buffer_payloads_read':0},indent=2))
print('PREFLIGHT_PASS',flush=True)
