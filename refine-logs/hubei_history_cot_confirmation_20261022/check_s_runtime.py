import json,sys
from pathlib import Path
import numpy as np
root=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_history_cot_confirmation_20261022')
sys.path.insert(0,str(root/'code_attempt3'))
from infer_new_period import s_model
f=json.loads((root/'MODEL_FREEZE.json').read_text())
with np.load(f['S_norm_path'],allow_pickle=False) as z:
    n={k:np.asarray(z[k],dtype=np.float32) for k in ['agri_mean','agri_std','input_mean','input_std']}
m=s_model(root/'inference_attempt3',f,n,device='cpu')
report=dict(state='FULL_SIMVP_STRICT_CPU_LOAD_PASS',python=sys.executable,parameters=sum(p.numel() for p in m.parameters()),checkpoint=f['S']['best_checkpoint_sha256'])
(root/'CPU_MODEL_WITNESS_attempt3.json').write_text(json.dumps(report,indent=2))
print(report)
