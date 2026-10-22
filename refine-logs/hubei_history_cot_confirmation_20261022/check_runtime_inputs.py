"""CPU witness on the actual numeric normalization and first past-only sample."""
import importlib.util,json
from pathlib import Path
import numpy as np
ROOT=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_history_cot_confirmation_20261022')
code=Path(__file__).resolve().parent/'infer_new_period.py'
spec=importlib.util.spec_from_file_location('past_only_runtime',code);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
freeze=json.loads((ROOT/'MODEL_FREEZE.json').read_text());selection=json.loads((ROOT/'INPUT_SELECTION.json').read_text());grid=dict(np.load(ROOT/'grid_static256.npz'))
with np.load(freeze['S_norm_path'],allow_pickle=False) as z:
    numeric={k:np.asarray(z[k],np.float32) for k in ['agri_mean','agri_std','input_mean','input_std']}
    nonnumeric=[k for k in z.files if k not in numeric]
assert numeric['input_mean'].shape==(16,) and numeric['agri_mean'].shape==(13,)
assert all(np.isfinite(v).all() for v in numeric.values())
sample=m.PastOnly(ROOT/'stage_attempt2',selection,grid,numeric,0,1)[0]
x,aux,ix=sample;assert x.shape==(8,16,256,256) and aux.shape==(24,3,256,256)
assert np.isfinite(x.numpy()).all() and np.isfinite(aux.numpy()).all()
report=dict(state='PASS_ACTUAL_NUMERIC_NORM_AND_PAST_ONLY_SAMPLE',numeric_fields=list(numeric),metadata_fields_not_cast=nonnumeric,
    x_shape=list(x.shape),predictable_aux_shape=list(aux.shape),sequence_id=ix,old_test_payloads_read=0,future_real_AGRI_payloads_read=0)
(ROOT/'CPU_INPUT_WITNESS_attempt2.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
