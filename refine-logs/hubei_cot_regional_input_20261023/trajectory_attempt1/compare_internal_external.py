"""Describe existing predictions across dates; never select a new model here."""
import csv,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
internal=P/'internal_replay_attempt1'
assert json.loads((internal/'COMPLETE.json').read_text())['state']=='SIX_TRAIN_INTERNAL_FROZEN_REPLAYS_COMPLETE'
rows=list(csv.DictReader((P.parent/'G4_PAIRED_PIXELS_attempt1.csv').open()))
rows=[r for r in rows if r['common']=='1' and r['dr']=='0' and r['dc']=='0']
sites=['Niushou','Wangzhai','Mizhuang'];allout={}
for scope in ['internal_select','outer_validation']:
    learned=[]
    for seed in [42,43,44]:
        name=f'seed{seed}_repeat_last'
        if scope=='internal_select':
            with np.load(internal/name/'predictions.npz') as z:
                m=z['common'][:,:,1,1];target=z['reference'][:,:,1,1][m];p=z['persistence'][:,:,1,1][m];t=z['transport'][:,:,1,1][m];learned.append(z['learned'][:,:,1,1][m])
        else:
            target=np.array([float(r['reference']) for r in rows]);p=np.array([float(r['persistence']) for r in rows]);t=np.array([float(r['transport']) for r in rows])
            with np.load(P/'eval_attempt1'/name/'predictions.npz') as z:
                lookup={(str(i),str(ti),int(l)):pr for i,ti,l,pr in zip(z['init'],z['target'],z['lead'],z['pred_log'])}
            learned.append(np.array([np.expm1(float(lookup[r['init'],r['target'],int(r['lead'])][sites.index(r['station']),1,1])) for r in rows]))
    learned=np.array(learned);classes={}
    for label,m in [('all',np.ones(len(target),bool)),('near_zero',target<1),('thin',(target>=1)&(target<10)),('moderate',(target>=10)&(target<30)),('thick',target>=30),
                    ('source_below_target_by10',target-p>=10),('source_above_target_by10',p-target>=10),('source_target_within10',abs(target-p)<10)]:
        if not m.any():continue
        e=t[m]-target[m];d=learned[:,m]-t[m]
        interaction=float(np.mean(2*e*d));energy=float(np.mean(d*d));change=float(np.mean((learned[:,m]-target[m])**2)-np.mean(e*e))
        assert abs(change-interaction-energy)<1e-8
        classes[label]=dict(n=int(m.sum()),fraction=float(m.mean()),reference_mean=float(target[m].mean()),
            persistence_mean=float(p[m].mean()),transport_mean=float(t[m].mean()),learned_mean=float(learned[:,m].mean()),
            persistence_rmse=float(np.sqrt(np.mean((p[m]-target[m])**2))),transport_rmse=float(np.sqrt(np.mean(e*e))),
            learned_rmse_seed_mean=float(np.sqrt(np.mean((learned[:,m]-target[m])**2,axis=1)).mean()),
            learned_minus_transport_mean=float(d.mean()),correction_interaction=interaction,correction_energy=energy,
            learned_minus_transport_mse=change,learned_better_fraction=float(np.mean((learned[:,m]-target[m])**2<e*e)))
    allout[scope]=classes
report=dict(state='EXISTING_PREDICTIONS_CONDITIONAL_SHIFT_DESCRIBED',scope='descriptive pooled station COT, not independent test or GHI gain',
    selected_arm_unchanged='repeat_last',results=allout,model_selection_or_training_performed=False,
    caveat='Future reference used only for diagnostic grouping; not operational features or sample selection')
(internal/'CONDITIONAL_SHIFT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
