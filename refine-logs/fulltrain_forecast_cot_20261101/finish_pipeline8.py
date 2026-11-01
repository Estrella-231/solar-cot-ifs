import csv,json,os,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
B=Path(__file__).resolve().parent
P=Path('/public/home/slfu/ttzhou/swc/irradiance_forecast/SolarCOTIFS_20260907')
OLD=P/'refine-logs/cot_aware_simvp_20261017'
def main():
 snap=json.loads((B/'resume_snapshot.json').read_text())
 ds=[B/f'tail_rank{i}' for i in range(8)]
 ms=[json.loads((d/'bank_manifest.json').read_text()) for d in ds]
 assert all(m['state']=='COMPLETE_FORECAST_AGRI_R_TRAIN_VAL_BANK' for m in ms)
 sources=[(B/f'bank_rank{i}',snap['prefix_counts'][i]) for i in range(4)]+[(d,m['rows']) for d,m in zip(ds,ms)]
 assert len({m['simvp_checkpoint_sha256'] for m in ms})==1
 assert len({m['r_checkpoint_sha256'] for m in ms})==1
 out=B/'bank_merged';out.mkdir()
 rows=[]
 for d,count in sources:
  rows.extend(list(csv.DictReader((d/'index.csv').open()))[:count])
 assert len({(r['split'],r['sequence_index_within_split']) for r in rows})==len(rows)
 total=sum(n for _,n in sources)
 assert total==33253,(total,snap) 
 for name in ms[0]['layouts']:
  xs=[np.load(d/name,mmap_mode='r')[:n] for d,n in sources]
  a=np.lib.format.open_memmap(out/name,mode='w+',dtype=np.float32,shape=(total,*xs[0].shape[1:]))
  k=0
  for x in xs:a[k:k+len(x)]=x;k+=len(x)
  a.flush()
 with (out/'index.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
  for i,r in enumerate(rows):r['bank_index']=i;w.writerow(r)
 m=ms[0];m['rows']=total;m['layouts']={k:[total,*v[1:]] for k,v in m['layouts'].items()};m['screening_pilot_only']=False
 (out/'bank_manifest.json').write_text(json.dumps(m,indent=2))
 subprocess.run([sys.executable,str(B/'build_future_ghi_head_pack_vectorized.py'),'--forecast-bank',str(out),'--crosswalk',str(OLD/'results/future_ghi_crosswalk.json'),'--keys',str(OLD/'results/future_ghi_crosswalk_keys.npz'),'--ghi-pack',str(P/'experiments/monthly_R_station16_20261013_v1/data/ghi_head_pack_new_r_v1'),'--output',str(B/'pack')],check=True)
 gpus=os.environ['ASSIGNED_GPUS'].split(',')
 tasks=[(s,seed) for s in ('sili','zhujia') for seed in (42,43,44)]
 def lane(i):
  for s,seed in tasks[i::len(gpus)]:
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus[i]);run=B/'heads'/f'{s}_seed{seed}'
   with (B/f'train_{s}_{seed}.log').open('w') as log:
    subprocess.run([sys.executable,'-u',str(B/'train_future_ghi_heads.py'),'--pack',str(B/'pack'),'--output',str(run),'--station',s,'--seeds',str(seed),'--epochs','30','--batch-size','1792'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 with ThreadPoolExecutor(len(gpus)) as ex:list(ex.map(lane,range(len(gpus))))
 (B/'TRAINING_COMPLETE.json').write_text(json.dumps({'state':'COMPLETE','stations':['sili','zhujia'],'seeds':[42,43,44],'arms':['A','AC'],'test_payloads_read':0}))
 subprocess.run([sys.executable,str(B/'score_pair.py')],check=True)
if __name__=='__main__':main()
