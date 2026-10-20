from pathlib import Path
import numpy as np,csv,json,sys
b=Path(sys.argv[1]);p=b.parent/'hubei_cloud_probability_20261020/R_pack_correctutc_attempt2';s=b/'diagnosis_v1/forecast/validation'
k=list(csv.DictReader((s/'keys.csv').open()));obs=np.load(s/'true_agri_patch.npy',mmap_mode='r');rr=list(csv.DictReader((p/'rows.csv').open()));rx=np.load(p/'x_norm.npy',mmap_mode='r');norm=json.loads((p/'norm.json').read_text());mu=np.array(norm['mean'],np.float32)[:,None,None];sd=np.array(norm['std'],np.float32)[:,None,None];ri={r['time_bjt']:j for j,r in enumerate(rr) if r['split']=='val'}
entries=[];per=[]
for i,key in enumerate(k):
 for l,t in enumerate(key['target_times_bjt'].split('|')):
  if t not in ri:continue
  raw=rx[ri[t]]*sd+mu;err=abs(raw[:13]-obs[i,l]);per.append(err.max((1,2)));entries.append((float(err.max()),i,l,t))
entries.sort(reverse=True);print('nmatch',len(entries),'per_channel_max',np.max(per,0).tolist(),'median',np.median(per,0).tolist());print('worst',entries[:6]);print('best',entries[-3:])
for e,i,l,t in entries[:2]+entries[-1:]:
 raw=rx[ri[t]]*sd+mu;v=obs[i,l];print('detail',t,'R_source',rr[ri[t]]['path_agri'],'obs_center',v[:,8,8].tolist(),'pack_center',raw[:13,8,8].tolist())
 print('ranges',[(float(v[c].min()),float(v[c].max()),float(raw[c].min()),float(raw[c].max())) for c in range(13)])
