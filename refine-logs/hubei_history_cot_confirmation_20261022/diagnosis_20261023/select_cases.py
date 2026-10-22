import csv,json
from pathlib import Path
p=Path('solar-cot-ifs/refine-logs/hubei_history_cot_confirmation_20261022/diagnosis_20261023'); rows=list(csv.DictReader((p/'all_common_initializations.csv').open()));choices=json.loads((p/'cases.json').read_text(encoding='utf-8-sig'))
for r in rows:
    if '09:00:00'<=r['init'][11:]<='14:00:00' and float(r['Niushou'])<0 and float(r['Wangzhai'])>0 and float(r['Mizhuang'])>0:
        if r['init'][:10] not in [x['init'][:10] for x in choices[3:]]: choices.append(dict(init=r['init'],reason='daytime sign-contrast case; selected by error for mechanism illustration'))
    if len(choices)==5:break
(p/'cases.json').write_text(json.dumps(choices,indent=2));print(choices)
