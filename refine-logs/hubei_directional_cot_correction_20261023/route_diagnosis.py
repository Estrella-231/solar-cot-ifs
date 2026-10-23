"""Read-only F_select diagnosis of permitted routes and actual corrections."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;manifest=json.loads((P/'PACK_COMPLETE.json').read_text());root=Path(manifest['path']);selected=np.flatnonzero(np.load(root/'select.npy'));idx={int(v):i for i,v in enumerate(selected)}
fields=list(csv.DictReader((root/'fields.csv').open()));members=list(csv.DictReader((root/'members.csv').open()));raw=np.array(np.load(root/'base.npy',mmap_mode='r')[selected]);truth=np.array(np.load(root/'target.npy',mmap_mode='r')[selected]);mask=np.array(np.load(root/'score_mask.npy',mmap_mode='r')[selected]);gm=json.loads((P/'GATE_COMPLETE.json').read_text());directions=np.load(gm['path'],mmap_mode='r');assert hashlib.sha256(Path(gm['path']).read_bytes()).hexdigest()==gm['sha256'];gate=np.array(directions[selected]);corrected=[]
for seed in [42,43,44]:
    with np.load(P/f'training_attempt1/seed{seed}/best_select_predictions.npz') as z:
        assert np.array_equal(z['field_index'],selected);corrected.append(z['prediction'])
corrected=np.array(corrected,dtype='float64');result={};centers={'Niushou':(8,6),'Wangzhai':(9,8),'Mizhuang':(7,10)}
for st,(cy,cx) in centers.items():
    ids=np.array([idx[int(r['field_index'])] for r in members if r['station']==st and int(r['field_index']) in idx]);ids=ids[mask[ids,0,cy,cx]];b=raw[ids,0,cy,cx].astype('float64');y=truth[ids,0,cy,cx].astype('float64');d=corrected[:,ids,0,cy,cx]-b;dire=gate[ids,0,cy,cx];label=np.where(y-b>3,1,np.where(y-b< -3,-1,0));cm=np.bincount((label+1)*3+dire+1,minlength=9).reshape(3,3)
    def metric(m):
        e=b[m]-y[m];delta=d[:,m];change=abs(delta)>1e-4;net=(e[None]+delta)**2-e[None]**2
        return dict(n=int(m.sum()),actual_changed_fraction=float(change.mean()),mean_abs_change_when_changed=float(abs(delta)[change].mean()) if change.any() else 0.,mean_change=float(delta.mean()),delta_MSE=float(net.mean()),interaction_2ed=float((2*e[None]*delta).mean()),change_energy_d2=float((delta**2).mean()),numerical_identity_max_abs=float(abs(net-2*e[None]*delta-delta**2).max()))
    result[st]=dict(direction_confusion_true_rows_pred_cols=cm.tolist(),all=metric(np.ones(len(y),bool)),reference_bins={})
    for key,lo,hi in [('0_to_1',0,1),('1_to_10',1,10),('10_to_30',10,30),('ge30',30,np.inf)]:
        m=(y>=lo)&(y<hi)
        result[st]['reference_bins'][key]=dict(all_routes=metric(m),routes={str(route):metric(m&(dire==route)) for route in [-1,0,1] if np.any(m&(dire==route))})
out=dict(state='READ_ONLY_F_SELECT_DIRECTION_ROUTE_ATTRIBUTION_COMPLETE',numerical_change_threshold_COT=1e-4,threshold_purpose='Numerical replay tolerance only; not a decision/training threshold',station=result,scope='All saved F_select center records; reference R bins, not independent weather truth',outer_payloads_read=0)
(P/'ROUTE_DIAGNOSIS.json').write_text(json.dumps(out,indent=2));print(json.dumps({s:{'confusion':v['direction_confusion_true_rows_pred_cols'],'nearzero':v['reference_bins']['0_to_1']['all_routes']} for s,v in result.items()},indent=2))
archive=P/'direction_maps.npz';assert not archive.exists();np.savez_compressed(archive,direction=np.asarray(directions));(P/'GATE_ARCHIVE.json').write_text(json.dumps(dict(path=archive.name,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),raw_sha256=gm['sha256'],shape=gm['shape']),indent=2))
