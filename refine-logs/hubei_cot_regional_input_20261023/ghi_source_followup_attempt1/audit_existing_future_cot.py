"""Compare old GHI-input COT with regional-R COT on exact existing val keys."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;BASE=P.parent
ROOT=BASE.parent
pack=ROOT/'hubei_solar_geometry_repair_20261021/GHI_pack_attempt12'
out=P/'existing_source_audit_attempt1';assert not out.exists()
meta=json.loads((pack/'pack_report.json').read_text());assert meta['state']=='HUBEI_GHI_FORECAST_HEADS_TRAINVAL_PACK_TEST_CLOSED'
assert meta['canonical_time_binding_repaired'] and meta['correctutc_cot_rebuilt']
model=json.loads((BASE/'G4_MODEL_FREEZE.json').read_text())
oldsource=next(s for s in meta['source_correctutc_Rs'] if s['role']=='validation')
assert oldsource['corrected_S_checkpoint_sha256']==model['S']['best_checkpoint_sha256']
rows=list(csv.DictReader((pack/'rows.csv').open()));assert all(r['split'] in ['train','val'] for r in rows)
old=np.load(pack/'cot.npy',mmap_mode='r');assert old.shape==(len(rows),1,3,3)
lookup={(r['station'],r['init_time_bjt'],r['target_time_bjt'],int(r['lead_minutes'])):i for i,r in enumerate(rows) if r['split']=='val'}
assert len(lookup)==sum(r['split']=='val' for r in rows)
base=list(csv.DictReader((BASE/'g4_attempt1/scores/paired_pixels.csv').open()))
records=[];missing_key=0;old_invalid=0;eligible=0
for r in base:
    if r['common']!='1' or r['dr']!='0' or r['dc']!='0':continue
    eligible+=1;key=(r['station'],r['init'],r['target'],int(r['lead']))
    if key not in lookup:missing_key+=1;continue
    value=float(old[lookup[key],0,1,1])
    if not np.isfinite(value) or not 0<=value<=100:old_invalid+=1;continue
    records.append(dict(station=r['station'],init=r['init'],target=r['target'],lead=int(r['lead']),date=r['date'],
        reference=float(r['reference']),old_ghi_input_cot=value,regional_forecast_cot=float(r['forecast'])))
assert records
sites=['Niushou','Wangzhai','Mizhuang'];y=np.array([r['reference'] for r in records]);v=np.array([[r[k] for r in records] for k in ['old_ghi_input_cot','regional_forecast_cot']]);e=v-y
cell=np.array([sites.index(r['station'])*16+r['lead']//15-1 for r in records]);days,di=np.unique([r['date'] for r in records],return_inverse=True)
n=np.zeros((48,len(days)));sse=np.zeros((2,48,len(days)))
for c in range(48):
    m=cell==c;n[c]=np.bincount(di[m],minlength=len(days))
    for j in range(2):sse[j,c]=np.bincount(di[m],weights=e[j,m]**2,minlength=len(days))
assert np.all(n.sum(1)>0)
scores=np.sqrt(sse.sum(2)/n.sum(1)).mean(1)
w=np.random.default_rng(20261023).multinomial(len(days),np.ones(len(days))/len(days),5000);den=n@w.T
ci=None
if np.all(den>0):
    boot=np.sqrt(np.einsum('mcd,bd->mcb',sse,w)/den[None]).mean(1);ci=np.quantile(boot[1]-boot[0],[.025,.975]).tolist()
report=dict(state='EXISTING_GHI_COT_SOURCE_MATCHED_AUDIT_COMPLETE',methods=['old_ghi_input_cot','regional_forecast_cot'],main_rmse=scores.tolist(),
    delta_regional_minus_old=float(scores[1]-scores[0]),date_ci95=ci,rows=len(records),eligible_g4_common_center_rows=eligible,
    missing_old_ghi_key_rows=missing_key,old_invalid_cot_rows=old_invalid,dates=days.tolist(),same_S_checkpoint=True,
    S_sha256=oldsource['corrected_S_checkpoint_sha256'],old_R_sha256=oldsource['R_checkpoint_sha256'],
    new_R_sha256='ed7be2ea2c987557d3753b3eb7d8d0f9ffb4513b63e4afedfcbb54be132600e2',
    by_site={st:np.sqrt(sse[:,i*16:(i+1)*16].sum(2)/n[i*16:(i+1)*16].sum(1)).mean(1).tolist() for i,st in enumerate(sites)},
    thick_n=int((y>=30).sum()),thick_rmse=np.sqrt((e[:,y>=30]**2).mean(1)).tolist(),
    scope='Existing validation intersection only; R packages differ in region, training/config/norm; not a context-size-only ablation',
    GHI_predictions_read=0,GHI_training_started=False,test_buffer_payloads_read=0,
    sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [pack/'pack_report.json',pack/'rows.csv',pack/'cot.npy',BASE/'G4_MODEL_FREEZE.json']})
out.mkdir();(out/'REPORT.json').write_text(json.dumps(report,indent=2))
with (out/'pixels.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
print(json.dumps(report,indent=2))
