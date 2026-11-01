import csv, datetime as dt, json
from pathlib import Path
base = Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_solar_geometry_repair_20261021')
rpack = Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907/experiments/hubei_cloud_probability_20261020/R_pack_correctutc_attempt2')
flat = Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/hubei_independent_trainval_stage_20261017/data')
rows = list(csv.DictReader((rpack/'rows.csv').open(encoding='utf-8-sig')))
out = []
for row in rows:
    bjt = dt.datetime.strptime(row['time_bjt'], '%Y%m%d%H%M')
    utc = (bjt-dt.timedelta(hours=8)).strftime('%Y%m%d%H%M')
    if not (flat/f'{utc}.npy').is_file():
        out.append({'bjt':row['time_bjt'],'utc':utc,'split':row['split'],
                    'relative':f'{bjt.year:04d}/{utc[:6]}/{utc[:8]}/{utc}.npy'})
assert len(out) == 202
assert all(x['split'] in ('train','val') for x in out)
dest=base/'missing_r_sources_attempt6.json'
dest.write_text(json.dumps(out,indent=2))
print(json.dumps({'count':len(out),'split_counts':{s:sum(x['split']==s for x in out) for s in ('train','val')},'first':out[:5],'last':out[-5:],'test_payloads_read':0,'buffer_payloads_read':0},indent=2))
