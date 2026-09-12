"""Materialize verified real-AGRI -> repaired-COT train/val arrays; no test labels."""
import argparse,csv,hashlib,io,json,os,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from cpp_quality_mask import cot_reference_mask
from repartition_cot_manifest import paper_split

INDICES=(0,1,2,3,4,5,8,9,10,11,12,13,14)
NAMES=['C01','C02','C03','C04','C05','C06','C09','C10','C11','C12','C13','C14','C15','cosSOZ','cosRAA','day_mask']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def load_row(row):
    raw=Path(row['original_path']).read_bytes();fixed=Path(row['sidecar_path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=row['original_sha256']:raise ValueError('original SHA mismatch')
    if hashlib.sha256(fixed).hexdigest()!=row['sidecar_sha256']:raise ValueError('sidecar SHA mismatch')
    with np.load(io.BytesIO(raw),allow_pickle=False) as z, np.load(io.BytesIO(fixed),allow_pickle=False) as q:
        for key in ('station_id','timestamp_utc'):
            if str(z[key].item())!=row[key] or str(q[key].item())!=row[key]:raise ValueError('identity mismatch')
        if str(q['original_sha256'].item())!=row['original_sha256']:raise ValueError('sidecar binding mismatch')
        if str(q['schema_version'].item())!='cpp_aligned_sidecar_v1':raise ValueError('wrong sidecar schema')
        if paper_split(row['timestamp_utc'])[0]!=row['split']:raise ValueError('time split mismatch')
        agri=z['agri'];cot=q['cpp'][0];mask=q['cot_reference_mask']
        if agri.shape!=(20,16,16) or cot.shape!=(16,16):raise ValueError('shape mismatch')
        expected,_=cot_reference_mask(cot,q['cpp_valid_mask'][0],agri[19],agri[18])
        if not bool(q['cpp_present']) or not np.array_equal(mask,expected):raise ValueError('reference mask mismatch')
        if int(mask.sum())!=int(row['valid_pixels']) or not mask.any():raise ValueError('invalid candidate mask')
        if not np.array_equal(z['grid_lat'],q['grid_lat']) or not np.array_equal(z['grid_lon'],q['grid_lon']):
            raise ValueError('grid mismatch')
        x=np.concatenate([agri[list(INDICES)],np.cos(np.deg2rad(agri[18]))[None],
             np.cos(np.deg2rad(agri[17]-agri[15]))[None],(agri[19]>0.5)[None]],axis=0).astype('float32')
        y=np.where(mask,cot/100.0,0).astype('float32')[None]
        return x,y,mask[None]


def atomic(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--acceptance',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    acceptance=json.loads(a.acceptance.read_text());assert acceptance['status']=='PASS'
    assert acceptance['counts'].get('error',0)==0
    a.output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(a.output/'pack.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config={'manifest_sha256':sha(a.manifest),'acceptance_sha256':sha(a.acceptance),
            'code_sha256':{n:sha(Path(__file__).with_name(n)) for n in
                           ('pack_cot_training.py','cpp_quality_mask.py','repartition_cot_manifest.py')}}
    cp=a.output/'pack_config.json'
    if cp.exists():assert json.loads(cp.read_text())==config,'resume code/config changed'
    else:atomic(cp,config)
    with a.manifest.open(newline='') as f:allrows=list(csv.DictReader(f))
    assert len(allrows)==acceptance['cot_candidates']
    assert len({(r['station_id'],r['timestamp_utc']) for r in allrows})==len(allrows)
    rows=[r for r in allrows if r['split'] in ('train','validation')]
    assert sum(r['split']=='train' for r in rows)==11978
    assert sum(r['split']=='validation' for r in rows)==6536
    start=0;state=a.output/'pack_status.json'
    if state.exists():
        old=json.loads(state.read_text())
        if old['state']=='COMPLETE':print('ALREADY_COMPLETE',flush=True);return
        start=old['processed']
    arrays=[]
    for name,shape,dtype in [('x_raw',(len(rows),16,16,16),'float32'),
                              ('target',(len(rows),1,16,16),'float32'),('mask',(len(rows),1,16,16),'bool')]:
        path=a.output/(name+'.npy')
        arrays.append(np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=dtype,shape=shape))
    t=time.monotonic()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i,(x,y,v) in enumerate(pool.map(load_row,rows[start:]),start+1):
            for arr,item in zip(arrays,(x,y,v)):arr[i-1]=item
            if i%100==0 or i==len(rows):
                for arr in arrays:arr.flush()
                report={'state':'PACKING','processed':i,'total':len(rows),'elapsed_this_run':time.monotonic()-t}
                atomic(state,report);print(json.dumps(report),flush=True)
    train_idx=np.array([i for i,r in enumerate(rows) if r['split']=='train'])
    sums=np.zeros(16);squares=sums.copy();counts=np.zeros(16,np.int64)
    for i in train_idx:
        x=arrays[0][i].astype('float64');v=np.isfinite(x);x=np.where(v,x,0)
        sums+=x.sum((1,2));squares+=(x*x).sum((1,2));counts+=v.sum((1,2))
    if (counts==0).any():raise ValueError('empty feature')
    mean=sums/counts;std=np.sqrt(np.maximum(squares/counts-mean*mean,1e-8))
    mean[15]=0;std[15]=1
    atomic(a.output/'norm.json',{'mean':mean.tolist(),'std':std.tolist(),'count':counts.tolist(),
           'feature_names':NAMES,'train_samples':len(train_idx),'cot_scale':100.0,
           'test_used':False,'manifest_sha256':config['manifest_sha256']})
    with (a.output/'rows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['index']+list(rows[0]));w.writeheader()
        for i,r in enumerate(rows):w.writerow({'index':i,**r})
    # Test references only; no test input/target payloads opened.
    with (a.output/'test_references.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader()
        w.writerows(r for r in allrows if r['split']=='test')
    paths=['x_raw.npy','target.npy','mask.npy','rows.csv','norm.json','test_references.csv','pack_config.json']
    atomic(a.output/'SHA256.json',{n:sha(a.output/n) for n in paths})
    atomic(state,{'state':'COMPLETE','processed':len(rows),'total':len(rows),'train':11978,'validation':6536,
                  'test_payloads_read':0,'provenance':'all original/sidecar hashes verified',
                  'elapsed_this_run':time.monotonic()-t})
    print('PACK_COMPLETE',flush=True)


if __name__=='__main__':main()
