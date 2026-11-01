"""Read-only deterministic U sample using the original training dataset."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from train_future_cot_u import UFrameSequenceDataset,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();b=a.base
    pr=b/'results/predictor_u_v1';cache=pr/'u_teacher_cache'
    meta=json.loads((pr/'P-XC/run_metadata.json').read_text())
    assert sha(b/'train_future_cot_u.py')==meta['p_code_sha256']=='152451e366da51d45ba93a682b1ef6fd5d9d1a0522d678b619fa38e520a0e93d'
    assert sha(cache/'cache_manifest.json')==meta['u_cache_manifest_sha256']
    for arm,h in [('P-C','7258ea13ea210347f94a49174ec2e6ac88a241463ec74220aa9d30a2c7ba1b37'),
                  ('P-XC','4b926f3a94d2228505d9acce0a7d36adbd90cbd8ef221d9db02045b35fea50df')]:
        assert sha(pr/arm/'epoch_012.pt')==h,arm
    assert all(meta[n]==0 for n in ('h_payloads_read','outer_validation_payloads_read','test_payloads_read','buffer_payloads_read'))
    ds=UFrameSequenceDataset(cache,'XC');assert len(ds)==52344
    assert abs(ds.cot_mean-meta['history_cot_log1p_normalization']['mean'])<1e-12
    assert abs(ds.cot_std-meta['history_cot_log1p_normalization']['std'])<1e-12
    selected=np.floor(np.linspace(0,len(ds.indices)-1,512)).astype(int);assert len(set(selected))==512
    seq_ids=(cache/'sequence_ids.txt').read_text().splitlines()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    shapes={'x':(1024,8,31,16,16),'g':(1024,16,3,16,16),'y':(1024,16,1,16,16),'m':(1024,16,1,16,16)}
    arrays={n:np.lib.format.open_memmap(a.output/(n+'.npy'),mode='w+',dtype='bool' if n=='m' else 'float32',shape=s) for n,s in shapes.items()}
    rows=[]
    for k,sq in enumerate(selected):
        for st in range(2):
            j=k*2+st;item=int(sq)*2+st
            x,g,y,m=ds[item]
            for n,value in zip(('x','g','y','m'),(x,g,y,m)):arrays[n][j]=value
            rows.append({'sample_index':j,'dataset_item':item,'sequence_index':int(sq),'seq_id':seq_ids[int(sq)],'station':('sili','zhujia')[st]})
        if (k+1)%64==0:print('U_SAMPLE',k+1,'/512',flush=True)
    for ar in arrays.values():ar.flush()
    with (a.output/'rows.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    receipt={'state':'COMPLETE_FIXED_U_TRAIN_FIT_SAMPLE','u_sequences':512,'station_windows':1024,
        'selection':'floor(linspace(0,26171,512)), both stations, no value-based replacement',
        'teacher_value_space':'physical COT returned by original Dataset; source cache log1p(COT) expm1 once',
        'source':{'script_sha256':sha(Path(__file__)),'plan_sha256':sha(b/'U_TRAIN_FIT_DIAGNOSTIC_PLAN_20261021.md'),
            'trainer_sha256':meta['p_code_sha256'],'u_cache_manifest_sha256':meta['u_cache_manifest_sha256'],
            'r_checkpoint_sha256':meta['r_u_checkpoint_sha256']},
        'artifact_sha256':{n:sha(a.output/n) for n in ('x.npy','g.npy','y.npy','m.npy','rows.csv')},
        'training_started':False,'ghi_labels_read':False,'test_payload_opened':False,'buffer_payload_opened':False}
    (a.output/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':main()
