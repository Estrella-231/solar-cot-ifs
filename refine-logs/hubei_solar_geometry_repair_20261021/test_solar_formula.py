from pathlib import Path
from datetime import datetime,timedelta,timezone
import sys,json
r=Path(__file__).resolve().parent;sys.path.insert(0,str(r/'code'));from solar_geometry_noaa import compute_solar_angles,_julian_day
assert _julian_day(datetime(2000,1,1,12))==2451545.0
u=datetime(2025,11,1,9,30);b=u.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=8)));assert compute_solar_angles(32.04,112.12,u)[0]==compute_solar_angles(32.04,112.12,b)[0]
rows=[]
for yy in (2024,2025,2026):
 for mm in range(1,13):
  for hh in (0,3,6,9,12,15,18,21):
   t=datetime(yy,mm,15,hh)
   for lat,lon in ((32.08,112.04),(32.04,112.12),(32.12,112.20)):
    z,a,d=compute_solar_angles(lat,lon,t);assert d in (0,1) and 0<=a<360 and 0<=z<=180
    rows.append(dict(utc=t.strftime('%Y%m%d%H%M'),lat=lat,lon=lon,zenith=float(z),azimuth=float(a),split='synthetic_formula_check'))
(r/'formula_unit_cases.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in rows),encoding='utf-8');print('FORMULA_UNIT_CASES',len(rows))
