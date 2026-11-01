"""Preserve committed prefixes, partition only remaining work, submit 8 GPUs."""
import ast,csv,hashlib,json,re,subprocess,time
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parent
def command(*args):return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT)
def main():
 for name in ('build_bank_fast_v2.py','finish_pipeline8.py'):ast.parse((B/name).read_text())
 subprocess.run(['bash','-n',str(B/'run_fast8.pbs')],check=True)
 assert hashlib.sha256((B/'build_bank.py').read_bytes()).hexdigest()=='9a2aa1158fa94a3237a09799cff09e1c455a3d9e0cb639d46cab885188723a19'
 diffs={}
 for n in ('forecast_agri_raw.npy','forecast_geometry.npy','forecast_cot_log1p.npy'):
  a=np.load(B/'profile_batch2'/n,mmap_mode='r');c=np.load(B/'profile_batch8'/n,mmap_mode='r')
  diffs[n]=float(np.max(np.abs(a-c)))
  if 'cot' in n:
   diffs['physical_COT_max']=float(np.max(np.abs(np.expm1(a)-np.expm1(c))))
   assert diffs['physical_COT_max']<0.1
  else:assert np.array_equal(a,c)
 info=command('qstat','-f','217013.tc6000')
 assert 'Job_Owner = slfu@' in info and 'Job_Name = dy_full_ghi' in info
 archive=B/'switch_checkpoint';archive.mkdir()
 (archive/'original_qstat.txt').write_text(info)
 (archive/'source_build_bank.py').write_bytes((B/'build_bank.py').read_bytes())
 subprocess.run(['qdel','217013.tc6000'],check=True)
 for i in range(25):
  try:
   status=command('qstat','-f','217013.tc6000')
   if re.search(r'job_state\s*=\s*C',status):break
  except subprocess.CalledProcessError:break
  time.sleep(2)
 else:raise RuntimeError('Old job not quiescent; no replacement submitted')
 snap={'old_job':'217013.tc6000','prefix_counts':[],'profile_differences':diffs,'batch':8,'gpus':8,'preserve_original_arrays':True}
 for rank in range(4):
  d=B/f'bank_rank{rank}';p=json.loads((d/'progress.json').read_text());n=p['sequences_complete']
  assert p['state']=='BUILDING' and n>0
  rows=list(csv.DictReader((d/'index.csv').open()));assert len(rows)>=n
  for name in ('history_agri_raw.npy','history_geometry.npy','history_cot_log1p.npy','forecast_agri_raw.npy','forecast_geometry.npy','forecast_cot_log1p.npy'):
   x=np.load(d/name,mmap_mode='r+');assert np.isfinite(x[:n]).all();x.flush()
  (archive/f'progress_rank{rank}.json').write_text(json.dumps(p,indent=2))
  snap['prefix_counts'].append(n)
  snap.setdefault('prefix_keys',[]).append([(r['split'],int(r['sequence_index_within_split'])) for r in rows[:n]])
 keypath=B.parent/'cot_aware_simvp_20261017/results/future_ghi_crosswalk_keys.npz'
 with np.load(keypath) as z:
  sc=z['split_code'];ix=z['sequence_index_within_split']
  allkeys=[(sn,int(i)) for code,sn in ((0,'train'),(1,'val')) for i in sorted(set(ix[sc==code]))]
 assert len(allkeys)==33253
 combined=[]
 for r in range(4):
  seq=allkeys[r::4];n=snap['prefix_counts'][r]
  assert list(map(tuple,snap['prefix_keys'][r]))==seq[:n]
  combined+=seq[:n]+seq[n::2]+seq[n+1::2]
 assert len(combined)==len(set(combined))==len(allkeys) and set(combined)==set(allkeys)
 snap['preserved_sequences']=sum(snap['prefix_counts']);snap['remaining_sequences']=len(allkeys)-snap['preserved_sequences']
 (B/'resume_snapshot.json').write_text(json.dumps(snap,indent=2))
 queue=command('qstat','-u','slfu');active=[l for l in queue.splitlines() if re.match(r'^\d+\.',l.strip()) and l.split()[9] not in ('C','E')]
 assert len(active)<3,queue
 job=command('qsub',str(B/'run_fast8.pbs')).strip();snap['new_job']=job
 (B/'SWITCH_COMPLETE.json').write_text(json.dumps(snap,indent=2))
 print(json.dumps({k:v for k,v in snap.items() if k!='prefix_keys'}))
 print(command('qstat','-u','slfu'))
if __name__=='__main__':main()
