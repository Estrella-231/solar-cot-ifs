"""Train/val-only full-region CPP supervision mask audit; never reads test."""
import argparse,csv,json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import netCDF4,numpy as np
CHANNELS=(0,1,2,3,4,5,8,9,10,11,12,13,14)

def main():
 p=argparse.ArgumentParser(); p.add_argument('--inventory',type=Path,required=True); p.add_argument('--grid',type=Path,required=True); p.add_argument('--aux-root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--offset',type=int,default=0); p.add_argument('--limit',type=int,default=0); p.add_argument('--skip-utc',action='append',default=[]); a=p.parse_args()
 with np.load(a.grid,allow_pickle=False) as g: glat=np.asarray(g['lat']); glon=np.asarray(g['lon'])
 rows=[]
 with a.inventory.open(newline='',encoding='utf-8-sig') as f: records=[r for r in csv.DictReader(f) if r['cpp_present']=='1' and r['split'] in ('train','val')]
 records.sort(key=lambda r:(r['split'],r['utc']))
 records=records[a.offset:a.offset+a.limit] if a.limit else records[a.offset:]
 with netCDF4.Dataset(records[0]['cpp_path']) as nc:
  lat=np.asarray(np.ma.filled(nc.variables['LAT'][:],np.nan)); lon=np.asarray(np.ma.filled(nc.variables['LON'][:],np.nan))
  rr=np.abs(lat[:,None]-glat[:,0][None,:]).argmin(0); cc=np.abs(lon[:,None]-glon[0,:][None,:]).argmin(0)
  err=max(float(np.max(np.abs(glat-lat[rr,None]))),float(np.max(np.abs(glon-lon[None,cc]))))
  if err>1e-4: raise RuntimeError(f'CPP grid mismatch {err}')
 skip=set(a.skip_utc); fail=[]; cpp_payloads_read=0; no_day_skipped=0
 for i,r in enumerate(records,1):
  utc=r['utc']; bjt=r['bjt']; item={'split':r['split'],'utc':utc,'bjt':bjt,'month_bjt':r['month_bjt'],'season_bjt':r['season_bjt'],'target_occurrences':int(r['target_occurrences'])}
  if utc in skip:
   item.update({'cpp_valid':0,'phase_valid':0,'agri_day_valid':0,'supervised':0,'nearzero':0,'cloud_ge_0_1':0,'thick_ge30':0,'error':'known problematic read excluded'}); rows.append(item); fail.append(utc); continue
  try:
   auxfiles=list((a.aux_root/bjt[:6]).glob(bjt+'_*_aux.npz'))
   if len(auxfiles)!=1: raise RuntimeError(f'aux match count={len(auxfiles)}')
   with np.load(auxfiles[0],allow_pickle=False) as z:
    bits=z['agri_valid_bits']; agri=np.stack([(bits&(1<<c))!=0 for c in CHANNELS],axis=0).all(0)
    day=(z['day_mask']>.5)&(z['cosSOZ']>np.cos(np.deg2rad(80)))
   agri_day=agri&day
   if not agri_day.any():
    no_day_skipped+=1
    item.update({'cpp_valid':0,'phase_valid':0,'agri_day_valid':0,'supervised':0,'nearzero':0,'cloud_ge_0_1':0,'thick_ge30':0,'error':'no target pixel passes daytime/AGRI support; CPP payload not read'})
    rows.append(item)
    if i%100==0: print(f'progress={i}/{len(records)} failures={len(fail)} no_day_skipped={no_day_skipped}',flush=True)
    continue
   cpp_payloads_read+=1
   with netCDF4.Dataset(r['cpp_path']) as nc:
    r0,r1=int(rr.min()),int(rr.max()+1); c0,c1=int(cc.min()),int(cc.max()+1)
    cv=nc.variables['COT']; pv=nc.variables.get('CLP')
    if pv is None: raise RuntimeError('CPP missing CLP')
    cot=np.asarray(np.ma.filled(cv[r0:r1,c0:c1],np.nan),np.float32)[np.ix_(rr-r0,cc-c0)]
    phase=np.asarray(np.ma.filled(pv[r0:r1,c0:c1],np.nan),np.float32)[np.ix_(rr-r0,cc-c0)]
   if rr[0]>rr[-1]: cot,phase=cot[::-1],phase[::-1]
   if cc[0]>cc[-1]: cot,phase=cot[:,::-1],phase[:,::-1]
   numeric=np.isfinite(cot)&(cot>=0)&(cot<=100); phaseok=np.isfinite(phase)&np.isin(phase,(0,1,2)); mask=numeric&phaseok&agri_day
   item.update({'cpp_valid':int(numeric.sum()),'phase_valid':int((numeric&phaseok).sum()),'agri_day_valid':int(agri_day.sum()),'supervised':int(mask.sum()),'nearzero':int(((cot<.1)&mask).sum()),'cloud_ge_0_1':int(((cot>=.1)&mask).sum()),'thick_ge30':int(((cot>=30)&mask).sum()),'error':''})
  except Exception as e:
   item.update({'cpp_valid':0,'phase_valid':0,'agri_day_valid':0,'supervised':0,'nearzero':0,'cloud_ge_0_1':0,'thick_ge30':0,'error':f'{type(e).__name__}: {e}'}); fail.append(utc)
  rows.append(item)
  if i%100==0: print(f'progress={i}/{len(records)} failures={len(fail)} no_day_skipped={no_day_skipped}',flush=True)
 a.output.parent.mkdir(parents=True,exist_ok=True)
 fields=list(rows[0]);
 with a.output.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 sums=defaultdict(lambda:defaultdict(int))
 for x in rows:
  for key in [('all','all'),(x['split'],'all'),(x['split'],x['season_bjt']),(x['split'],x['month_bjt'])]:
   d=sums[key];d['products']+=1;d['supported_products']+=int(x['supervised']>0);d['target_occurrences']+=x['target_occurrences'];d['supervised_pixels']+=x['supervised']*x['target_occurrences'];d['nearzero_pixels']+=x['nearzero']*x['target_occurrences'];d['cloud_ge_0_1_pixels']+=x['cloud_ge_0_1']*x['target_occurrences'];d['thick_ge30_pixels']+=x['thick_ge30']*x['target_occurrences']
 summary={'state':'TRAIN_VAL_FULL_REGION_CPP_MASK_AUDIT','test_used':False,'grid_max_coordinate_error_degrees':err,'mask':'finite COT in [0,100] AND CLP in {0,1,2} AND all13 AGRI valid bits AND day_mask>0.5 AND cosSOZ>cos80','products_processed':len(records),'cpp_payloads_read':cpp_payloads_read,'no_day_or_agri_support_skipped_before_cpp_read':no_day_skipped,'read_failure_count':len(fail),'read_failures':fail,'aggregates':{'/'.join(k):dict(v) for k,v in sums.items()},'scope':'full 256x256 region; CPP supported timestamps only; CPP is not read when no pixel can satisfy the predefined daylight/AGRI mask'}
 a.output.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
