"""Bounded 5-case V-only observed 64x64 C03/C13 background extraction."""
import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

STATIONS=('sili','zhujia')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--days',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--inference-index',type=Path,required=True);a=ap.parse_args();base=a.base
    gate=json.loads((base/'data/dense_observed_reference_gate_20261021.json').read_text())
    for k,p in [('days',a.days),('v_support',base/'data/sequence_support_v1/v_sequences.csv'),
                ('grid',base/'data/grid_static.npz'),('r_rows',base/'data/r_u_pack_v2/rows_u_supervised.csv'),
                ('reader',base/'data/observed_reference_reader_20261021/hunan_data.py')]:assert sha(p)==gate['sha256'][k],k
    source=base/'results/initial_cloud_visibility_v1_20261021/event_input_visibility.csv'
    assert sha(source)=='0999d66761d5ba4fed781228c7846c5b8a2ab80bb540bb5a915a17eb15fdc58b'
    cases=[r for r in rows(source) if r['arm']=='cot_p_xc' and r['detected']=='False'
           and r['initial_visibility']=='NEIGHBOR_THICK_CENTER_NOT'];assert len(cases)==5
    assert sha(a.inference_index)=='618f001ef89f46ddaaf6eb1bc7e6837e4a439eae5669a4444fdedf7c1847cc13'
    vindex=rows(a.inference_index)
    seq={r['seq_id']:r for r in rows(base/'data/sequence_support_v1/v_sequences.csv')}
    day={r['date_BJT']:r['split'] for r in rows(a.days)}
    centers={}
    for r in rows(base/'data/r_u_pack_v2/rows_u_supervised.csv'):
        rc=tuple(map(int,r['station_grid_rc'].split(',')))
        if r['station'] in centers:assert centers[r['station']]==rc
        centers[r['station']]=rc
    sys.path.insert(0,str(base/'data/observed_reference_reader_20261021'))
    from hunan_data import resolve_aux,CHANNELS
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    values=np.lib.format.open_memmap(a.output/'channels.npy',mode='w+',dtype='float32',shape=(5,24,2,64,64))
    valid=np.lib.format.open_memmap(a.output/'channel_valid.npy',mode='w+',dtype='bool',shape=(5,24,2,64,64))
    joint=np.lib.format.open_memmap(a.output/'all13_qa_valid.npy',mode='w+',dtype='bool',shape=(5,24,64,64))
    metadata=[];began=time.monotonic()
    for ci,c in enumerate(cases):
        sq=int(c['sequence_index']);index=vindex[sq];assert int(index['sequence_index_within_split'])==sq
        row=seq[index['seq_id']];assert row['split']=='val'
        paths=row['data_relpaths'].split('|');assert len(paths)==24
        times=[datetime.strptime(Path(p).stem,'%Y%m%d%H%M') for p in paths]
        assert all(times[i+1]-times[i]==timedelta(minutes=15) for i in range(23))
        assert times[7].isoformat(' ')==c['initial_time_utc']
        assert all(day[(t+timedelta(hours=8)).date().isoformat()]=='val' for t in times)
        rr,cc=centers[c['station']];r0,c0=rr-32,cc-32
        assert 0<=r0 and 0<=c0 and r0+64<=256 and c0+64<=256
        for ti,(rel,t) in enumerate(zip(paths,times)):
            rawpath=a.data_root/'data'/rel;aux=resolve_aux(a.data_root,t+timedelta(hours=8))
            before=[(p.stat().st_size,p.stat().st_mtime_ns) for p in (rawpath,aux)]
            raw=np.load(rawpath,mmap_mode='r',allow_pickle=False)
            assert raw.shape==(20,256,256) and raw.dtype==np.float32
            cropped=np.asarray(raw[[2,12],r0:r0+64,c0:c0+64],np.float32)
            with np.load(aux,allow_pickle=False) as z:bits=np.asarray(z['agri_valid_bits'][r0:r0+64,c0:c0+64],np.uint16)
            vv=np.stack([(bits&(1<<k))!=0 for k in (2,12)])&np.isfinite(cropped)
            jj=np.stack([(bits&(1<<k))!=0 for k in CHANNELS]).all(axis=0)
            assert before==[(p.stat().st_size,p.stat().st_mtime_ns) for p in (rawpath,aux)]
            values[ci,ti]=cropped;valid[ci,ti]=vv;joint[ci,ti]=jj
            metadata.append({'case_index':ci,'event_id':c['event_id'],'station':c['station'],
                'frame_index':ti,'lead_minutes':15*(ti-7),'data_relpath':rel,
                'time_utc':t.isoformat(' '),'time_bjt':(t+timedelta(hours=8)).isoformat(' '),
                'phase':'history' if ti<8 else 'offline_future_reference',
                'station_grid_row':rr,'station_grid_col':cc,'crop_row':r0,'crop_col':c0,
                'raw_size':before[0][0],'raw_mtime_ns':before[0][1],'aux_path':str(aux),
                'aux_size':before[1][0],'aux_mtime_ns':before[1][1],
                'crop_sha256':hashlib.sha256(cropped.tobytes()).hexdigest(),
                'channel_qa_sha256':hashlib.sha256(vv.tobytes()).hexdigest()})
        print('CONTEXT_CASE',ci+1,c['event_id'],'elapsed',time.monotonic()-began,flush=True)
    for x in (values,valid,joint):x.flush()
    for name,rr in [('frames.csv',metadata),('cases.csv',cases)]:
        with (a.output/name).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    receipt={'state':'COMPLETE_FIVE_CASE_OBSERVED_CONTEXT','cases':5,'frames':120,
        'channel_raw_indices':[2,12],'channel_names':['C03','C13'],'shape':[5,24,2,64,64],
        'source':{'script_sha256':sha(Path(__file__)),'case_table_sha256':sha(source),
                  'plan_sha256':sha(base/'VISIBLE_NEIGHBOR_HISTORY_CONTEXT_PLAN_20261021.md'),
                  'v_support_sha256':gate['sha256']['v_support'],'grid_sha256':gate['sha256']['grid'],
                  'inference_index_path':str(a.inference_index),'inference_index_sha256':sha(a.inference_index)},
        'artifact_sha256':{n:sha(a.output/n) for n in ('channels.npy','channel_valid.npy','all13_qa_valid.npy','frames.csv','cases.csv')},
        'elapsed_seconds':time.monotonic()-began,'future_observed_agri_read_only_for_offline_diagnostic':True,
        'new_training':False,'ghi_labels_read':False,'test_payload_opened':False,'buffer_payload_opened':False,
        'r_applied_outside_16x16':False,'deployable_input_changed':False}
    (a.output/'complete.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':main()
