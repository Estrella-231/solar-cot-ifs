"""Unique forecast fields and F_fit-only normalization; no GHI values."""
import csv,hashlib,json
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;B=P.parent/'hubei_solar_geometry_repair_20261021'
OUT=Path('/dev/shm/hubei_forecast_domain_cot_20261023_pack_attempt1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(p.open()))
assert json.loads((P/'REFERENCE_AUDIT.json').read_text())['state'].startswith('PASS_ALL_F_REFERENCE')
pairs=rows(P/'raw_attempt1/pairs.csv');unique={};members=[]
for r in pairs:
    assert r['scope'] in ['F_fit','F_select'] and r['all_frame_metadata_available']=='True'
    key=(r['source_role'],int(r['source_sequence_index']),int(r['lead']))
    if key not in unique:unique[key]=dict(r,field_index=len(unique))
    q=unique[key];assert (q['scope'],q['init_time'],q['target_time'])==(r['scope'],r['init_time'],r['target_time'])
    members.append(dict(field_index=q['field_index'],station=r['station'],prepared_index=r['prepared_index']))
records=list(unique.values());N=len(records);OUT.mkdir(exist_ok=False)
def mem(name,shape,dtype='float32'):return np.lib.format.open_memmap(OUT/(name+'.npy'),mode='w+',dtype=dtype,shape=shape)
x=mem('x',(N,34,16,16));target=mem('target',(N,1,16,16));base=mem('base',(N,1,16,16));persist=mem('persistence',(N,1,16,16));mask=mem('mask',(N,1,16,16),'bool');scoremask=mem('score_mask',(N,1,16,16),'bool')
forecastmask=mem('forecast_mask',(N,1,16,16),'bool')
folds={};refs={};sources={}
for role in sorted({r['source_role'] for r in records}):
    folder=B/'S_banks_attempt12'/role/'val';keys=rows(folder/'keys.csv')
    folds[role]=(keys,np.load(folder/'pred_agri_patch.npy',mmap_mode='r'),np.load(folder/'future_geometry_patch.npy',mmap_mode='r'),np.load(B/'COT_banks_attempt12'/f'{role}.npy',mmap_mode='r'))
    with np.load(P/'reference_attempt1'/role/'reference.npz') as z:refs[role]={k:z[k] for k in z.files}
    refs[role]['lookup']={str(t):i for i,t in enumerate(refs[role]['bjt'])}
    sources[role]={'keys_sha256':sha(folder/'keys.csv'),'R_reference_sha256':sha(P/'reference_attempt1'/role/'reference.npz')}
for i,r in enumerate(records):
    role=r['source_role'];seq=int(r['source_sequence_index']);lead=int(r['lead']);kk=lead//15-1;keys,agri,geom,cot=folds[role]
    assert keys[seq]['init_time_bjt']==r['init_time'] and keys[seq]['target_times_bjt'].split('|')[kk]==r['target_time']
    ref=refs[role];li=ref['lookup'];init=datetime.strptime(r['init_time'],'%Y%m%d%H%M')
    histids=[li[(init-timedelta(minutes=k)).strftime('%Y%m%d%H%M')] for k in [105,90,75,60,45,30,15,0]]
    tid=li[r['target_time']];forecast=np.asarray(cot[seq,kk]).reshape(1,16,16);geo=np.asarray(geom[seq,kk]);past=ref['cot'][histids,0];hv=ref['valid'][histids,0]
    pv=(geo[2:3]>.5)&(geo[0:1]>np.cos(np.deg2rad(80.)))&np.isfinite(forecast)&(forecast>=0)&(forecast<=100)
    x[i]=np.concatenate([agri[seq,kk],geo,np.log1p(np.maximum(forecast,0)),np.where(hv,np.log1p(past),0),hv.astype('float32'),np.full((1,16,16),lead/240,dtype='float32')])
    target[i]=ref['cot'][tid];base[i]=forecast;persist[i]=past[-1:];forecastmask[i]=pv;mask[i]=pv&ref['valid'][tid];scoremask[i]=mask[i]&hv[-1:]
    assert np.isfinite(x[i]).all() and np.isfinite(target[i]).all()
    if i%4000==0:print('PACK',i,N,flush=True)
fit=np.array([r['scope']=='F_fit' for r in records]);select=~fit
mu=np.zeros(25);sd=np.ones(25);counts=np.zeros(25,dtype='int64')
for ch in range(25):
    su=ss=0.;n=0
    for j in range(0,N,256):
        chosen=fit[j:j+256];v=np.asarray(x[j:j+256,ch],dtype='float64')[chosen]
        if ch==16:vm=np.asarray(forecastmask[j:j+256,0])[chosen]
        elif ch>=17:vm=np.asarray(x[j:j+256,25+ch-17])[chosen]>.5
        else:vm=np.ones(v.shape,dtype=bool)
        vals=v[vm];su+=float(vals.sum());ss+=float(np.square(vals).sum());n+=len(vals)
    assert n>0;mu[ch]=su/n;sd[ch]=max(np.sqrt(max(ss/n-mu[ch]**2,0)),1e-4);counts[ch]=n
for j in range(0,N,256):
    xx=np.array(x[j:j+256],copy=True);xx[:,:25]=(xx[:,:25]-mu[None,:,None,None])/sd[None,:,None,None]
    xx[:,16]=np.where(forecastmask[j:j+256,0],xx[:,16],0)
    xx[:,17:25]=np.where(xx[:,25:33]>.5,xx[:,17:25],0);x[j:j+256]=xx
for a in [x,target,base,persist,mask,scoremask,forecastmask]:a.flush()
np.save(OUT/'fit.npy',fit);np.save(OUT/'select.npy',select)
for name,data in [('fields',records),('members',members)]:
    with (OUT/(name+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
norm=dict(mean=mu.tolist(),std=sd.tolist(),counts=counts.tolist(),fitted_on='F_fit unique forecast fields only; invalid COT values excluded',channels=34)
(OUT/'norm.json').write_text(json.dumps(norm,indent=2))
report=dict(state='F_ONLY_CAUSAL_COT_CORRECTION_PACK_READY',path=str(OUT),fields=N,fit_fields=int(fit.sum()),select_fields=int(select.sum()),station_members=len(members),sampling='Uniform unique forecast fields; station duplicates removed before fitting, retained for evaluation',sources=sources,hashes={p.name:sha(p) for p in OUT.iterdir() if p.is_file()},GHI_values_read=0,outer_payloads_read=0)
(OUT/'COMPLETE.json').write_text(json.dumps(report,indent=2));(P/'PACK_COMPLETE.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
