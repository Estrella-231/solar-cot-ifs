"""Second implementation of reported metrics from saved FP64 per-frame sums."""
import csv,json,math
from pathlib import Path

root=Path(__file__).resolve().parent
report=json.loads((root/'R_RETRIEVAL_SCORES_attempt1.json').read_text())
rows=list(csv.DictReader((root/'R_VAL_ERROR_SUMS_attempt1.csv').open()))
maximum=0.;counts=0
for scope,bins in report['scores'].items():
    for label,old in bins.items():
        rr=[r for r in rows if r['scope']==scope and r['stratum']==label]
        n=sum(int(r['n']) for r in rr);assert n==old['n']
        if n:
            new=dict(rmse=math.sqrt(math.fsum(float(r['sse']) for r in rr)/n),
                     mae=math.fsum(float(r['sae']) for r in rr)/n,
                     bias=math.fsum(float(r['se']) for r in rr)/n)
            for name,value in new.items():maximum=max(maximum,abs(value-old[name]));counts+=1
assert maximum<1e-10
result=dict(state='PASS_SECOND_IMPLEMENTATION_ERROR_SUM_AGGREGATION',metrics=counts,
            max_absolute_difference=maximum,frames=report['frames'],
            scope='same-time R retrieval only; not independent CPP truth or forecast/GHI improvement',
            original_payloads_recomputed_in_this_check=False,test_payloads_read=0)
(root/'R_SCORE_AGGREGATION_AUDIT_attempt1.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
