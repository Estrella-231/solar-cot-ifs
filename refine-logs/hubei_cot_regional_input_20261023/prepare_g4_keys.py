"""Train/val-only causal history and target-reference keys, without payload reads.

Do not require 24 consecutive CPP files: R can retrieve history from AGRI;
CPP is only needed at each evaluated target. Pixel QC remains a later step.
"""
import csv, hashlib, json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


def main():
    root=Path(__file__).resolve().parent
    parse=lambda s:datetime.strptime(s,'%Y%m%d%H%M')
    fmt=lambda t:t.strftime('%Y%m%d%H%M')
    agri=list(csv.DictReader((root/'corrected_agri_manifest.csv').open(encoding='utf-8-sig')))
    cpp=list(csv.DictReader((root/'selected_cpp_agri.csv').open(encoding='utf-8-sig')))
    assert all(r['split'] in ('train','val') for r in agri+cpp)
    amap={r['bjt']:r for r in agri};cmap={r['time_bjt']:r for r in cpp}
    assert len(amap)==len(agri) and len(cmap)==len(cpp)
    pairs=[]
    for key,r in sorted(amap.items()):
        init=parse(key);history=[fmt(init+timedelta(minutes=o)) for o in range(-105,1,15)]
        if not all(k in amap and amap[k]['split']==r['split'] for k in history):continue
        for lead in range(15,241,15):
            target=fmt(init+timedelta(minutes=lead))
            if target not in cmap or cmap[target]['split']!=r['split']:continue
            assert target in amap and amap[target]['split']==r['split']
            pairs.append(dict(split=r['split'],init_time_BJT=key,target_time_BJT=target,lead_minutes=lead,
                              history_BJT='|'.join(history)))
    out=root/'G4_CANDIDATE_KEYS.csv';assert not out.exists()
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['split','init_time_BJT','target_time_BJT','lead_minutes','history_BJT'])
        w.writeheader();w.writerows(pairs)
    result=dict(state='CAUSAL_TRAIN_VAL_REFERENCE_KEY_CANDIDATES_NOT_PIXEL_QC',
                source_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in
                               ('corrected_agri_manifest.csv','selected_cpp_agri.csv')},
                key_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                rule='8 corrected AGRI history keys in same split; CPP only at target; no future input',
                test_buffer_payloads_read=0,all_payloads_read=0,counts={})
    for split in ('train','val'):
        rr=[r for r in pairs if r['split']==split]
        result['counts'][split]=dict(pairs=len(rr),initializations=len({r['init_time_BJT'] for r in rr}),
             target_dates=len({r['target_time_BJT'][:8] for r in rr}),
             per_lead=dict(sorted(Counter(r['lead_minutes'] for r in rr).items())))
    (root/'G4_KEY_REPORT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
