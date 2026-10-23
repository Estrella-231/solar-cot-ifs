"""Paired date sums: direction-constrained versus frozen unconstrained control."""
import json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;PRE=P.parent/'hubei_forecast_domain_cot_20261023'
new=np.load(P/'F_SELECT_DATE_SUMS.npz');old=np.load(PRE/'F_SELECT_DATE_SUMS.npz')
for k in ['den','dates','methods']:assert np.array_equal(new[k],old[k]),k
assert np.array_equal(new['sse'][:2],old['sse'][:2])
for seed in [42,43,44]:
    a=json.loads((P/f'training_attempt1/seed{seed}/run_metadata.json').read_text());b=json.loads((PRE/f'training_attempt1/seed{seed}/run_metadata.json').read_text())
    for k in ['seed','parameters','global_batch','updates','lr','weight_decay','initialization_sha256','schedule_sha256','fit_fields','select_fields','pack_manifest_sha256']:assert a[k]==b[k],(seed,k)
den=new['den'];ds=new['dates'];rng=np.random.default_rng(20261023);w=rng.multinomial(len(ds),np.ones(len(ds))/len(ds),5000);bd=w@den.T;w=w[(bd>0).all(1)];bd=w@den.T
def boot(z):return np.sqrt(np.einsum('mgd,bd->bmg',z['sse'][2:],w)/bd[:,None]).mean((1,2))
def point(z):return float(np.sqrt(z['sse'][2:].sum(-1)/den.sum(-1)[None]).mean())
diff=boot(new)-boot(old);pnew=point(new);pold=point(old)
out=dict(state='PAIRED_DIRECTION_CONSTRAINT_CONTROL_RESCORING_COMPLETE',directional=pnew,unconstrained=pold,difference=pnew-pold,ci95=np.quantile(diff,[.025,.975]).tolist(),same_rows_masks_labels_initializations_orders_budgets=True,scope='Reused F_select development; not independent test or GHI gain',outer_payloads_read=0)
(P/'PAIRED_CONTROL.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
