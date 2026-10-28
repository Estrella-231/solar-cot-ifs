"""Compare date-assignment metadata only; do not open cloud or GHI payloads."""
import collections,csv,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
ours_file=ROOT/'refine-logs/hubei_cot_independent_20261014/frozen_split_20261015_v1/day_assignment.csv'
theirs_file=P/'metadata/dydiff_day_assignment_common.csv'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return {r['date_BJT']:r['split'] for r in csv.DictReader(f)}
ours=read(ours_file);theirs=read(theirs_file);common=sorted(set(ours)&set(theirs));assert len(theirs)==len(common)==306
counts=collections.Counter((theirs[d],ours[d]) for d in common)
rows=[dict(date_BJT=d,dydiff=theirs[d],our=ours[d]) for d in common if theirs[d]!=ours[d]]
with (P/'SPLIT_DATE_MISMATCHES.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=['date_BJT','dydiff','our']);w.writeheader();w.writerows(rows)
result={'state':'DATE_METADATA_COMPARISON_COMPLETE',
    'remote_full_day_table_sha256':'97c8f76b7f1de348074524f781fffed3db0ce3d78d1e9486fe87b0b8243011a6',
    'local_filtered_remote_table_sha256':hashlib.sha256(theirs_file.read_bytes()).hexdigest(),
    'our_day_table_sha256':hashlib.sha256(ours_file.read_bytes()).hexdigest(),
    'shared_dates':len(common),'matrix':[dict(dydiff=a,ours=b,count=n) for (a,b),n in sorted(counts.items())],
    'date_mismatch_rows':len(rows),'date_roles_are_not_proof_of_exact_frame_exposure':True,
    'cloud_payloads_read':0,'test_labels_read':0}
(P/'SPLIT_METADATA_AUDIT.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
