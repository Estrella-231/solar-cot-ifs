"""Second arithmetic path from saved pixel predictions; no Torch or scorer import."""
import csv,hashlib,json,math
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
rows=list(csv.DictReader((P.parent/'G4_PAIRED_PIXELS_attempt1.csv').open()))
rows=[r for r in rows if r['common']=='1' and r['dr']=='0' and r['dc']=='0']
groups={}
for row in rows:groups.setdefault((row['station'],int(row['lead'])),[]).append(row)
assert len(groups)==48 and len(rows)==7669
report=json.loads((P/'eval_attempt1/scores_cpu_repair_attempt2/SCORES.json').read_text())
scores={}
for method in ['persistence','transport','forecast']:
    scores[method]=math.fsum(math.sqrt(math.fsum((float(r[method])-float(r['reference']))**2 for r in rr)/len(rr)) for rr in groups.values())/48
for arm in ['full','repeat_last']:
    for seed in [42,43,44]:
        name=f'seed{seed}_{arm}';path=P/'eval_attempt1'/name/'predictions.npz'
        with np.load(path) as z:
            lookup={(str(i),str(t),int(l)):p for i,t,l,p in zip(z['init'],z['target'],z['lead'],z['pred_log'])}
        terms=[]
        for (station,lead), rr in groups.items():
            ix=['Niushou','Wangzhai','Mizhuang'].index(station)
            errors=[(math.expm1(float(lookup[r['init'],r['target'],lead][ix,1,1]))-float(r['reference']))**2 for r in rr]
            terms.append(math.sqrt(math.fsum(errors)/len(errors)))
        scores[name]=math.fsum(terms)/48
maximum=max(abs(v-report['main_rmse'][k]) for k,v in scores.items())
assert maximum<1e-10,maximum
out=dict(state='PASS_INDEPENDENT_RAW_PIXEL_FP64_RECALC',main_rmse=scores,max_difference=maximum,
    common_rows=len(rows),cells=len(groups),implementation='Python math.expm1/fsum grouped raw saved predictions; no primary scorer import',
    primary_score_sha256=hashlib.sha256((P/'eval_attempt1/scores_cpu_repair_attempt2/SCORES.json').read_bytes()).hexdigest())
(P/'INDEPENDENT_RECALC_attempt2.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
