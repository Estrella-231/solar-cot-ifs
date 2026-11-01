"""Read-only comparison of auxiliary CPP support and corrected R training support."""
import argparse,csv,hashlib,json,math,re
from collections import Counter
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):h.update(x)
    return h.hexdigest()

def read(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def key(s):
    q=re.sub(r'\D','',str(s));assert len(q)>=12,s
    return q[:12]

def compare_masks(old,new,cos,day):
    assert old.dtype==new.dtype==np.bool_
    assert old.shape==new.shape==cos.shape==day.shape
    outside=(cos<=math.cos(math.radians(80)))|(day<=.5)
    unambiguous=(cos<math.cos(math.radians(80))-1e-6)|(day<.5-1e-6)
    return {'old_valid':int(old.sum()),'corrected_valid':int(new.sum()),
            'same_valid':int((old&new).sum()),'old_only':int((old&~new).sum()),
            'corrected_only':int((new&~old).sum()),
            'old_valid_outside_corrected_geometry':int((old&outside).sum()),
            'old_valid_unambiguously_outside_corrected_geometry':int((old&unambiguous).sum()),
            'corrected_valid_outside_corrected_geometry':int((new&outside).sum())}

def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();meta=json.loads((a.attempt/'prepared_inputs.json').read_text())
    assert meta['state']=='PASS_ATTEMPT13_TRAIN_VAL_INPUTS_AND_CPP_PHYSICAL_TARGETS'
    assert meta['test_payloads_read']==meta['buffer_payloads_read']==0
    old=Path(meta['r_pack']);new=Path(meta['prepared_dir'])/'R_packs/full';side=Path(meta['sidecar'])
    paths={'prepared_inputs':a.attempt/'prepared_inputs.json','index':Path(meta['index']),
           'side_manifest':side/'manifest.json','old_rows':old/'rows.csv','new_rows':new/'rows.csv',
           'old_target':old/'target.npy','old_mask':old/'mask.npy','new_target':new/'target.npy',
           'new_mask':new/'mask.npy','new_x_norm':new/'x_norm.npy','new_norm':new/'norm.json',
           'old_status':old/'pack_status.json','new_status':new/'pack_status.json'}
    for d in (old,new):
        h=json.loads((d/'SHA256.json').read_text())
        for n in ('rows.csv','target.npy','mask.npy','norm.json'):assert sha(d/n)==h[n],str(d/n)
    assert sha(new/'x_norm.npy')==json.loads((new/'SHA256.json').read_text())['x_norm.npy']
    assert sha(paths['index'])==meta['prepared_data']['index_sha256']
    assert sha(paths['side_manifest'])==meta['sidecar_audit']['sidecar_manifest_sha256']
    ro,rn=read(paths['old_rows']),read(paths['new_rows'])
    assert len(ro)==len(rn)==3479
    mo,mn=(np.load(paths[n],mmap_mode='r',allow_pickle=False) for n in ('old_mask','new_mask'))
    yo,yn=(np.load(paths[n],mmap_mode='r',allow_pickle=False) for n in ('old_target','new_target'))
    assert mo.dtype==mn.dtype==np.bool_ and mo.shape==mn.shape==(3479,1,16,16)
    assert yo.shape==yn.shape==mo.shape
    no=json.loads(paths['new_norm'].read_text());x=np.load(paths['new_x_norm'],mmap_mode='r')
    assert x.shape==(3479,16,16,16)
    geom=np.asarray(x[:,13:16],np.float32)*np.array(no['std'][13:16],np.float32)[None,:,None,None]+np.array(no['mean'][13:16],np.float32)[None,:,None,None]
    assert np.isfinite(geom).all()
    bynew={key(r['time_bjt']):i for i,r in enumerate(rn)};assert len(bynew)==3479
    order=np.array([bynew[key(r['time_bjt'])] for r in ro]);assert len(set(order))==3479
    assert all(ro[i]['split']==rn[j]['split'] for i,j in enumerate(order))
    newmask=np.asarray(mn[order,0]);oldmask=np.asarray(mo[:,0]);cos=geom[order,0];day=geom[order,2]
    newy=np.asarray(yn[order,0]);oldy=np.asarray(yo[:,0])
    assert np.isfinite(oldy).all() and np.isfinite(newy).all()
    source_stats={}
    examples=[]
    for split in ('train','val'):
        sel=np.array([r['split']==split for r in ro])
        stat=compare_masks(oldmask[sel],newmask[sel],cos[sel],day[sel])
        common=oldmask[sel]&newmask[sel]
        stat['max_abs_target_difference_on_common_valid_physical_cot']=float(np.max(np.abs((oldy[sel]-newy[sel])[common]))*100) if common.any() else None
        source_stats[split]=stat
    for i,r in enumerate(ro):
        dif=oldmask[i]^newmask[i]
        if dif.any():
            yy,xx=np.argwhere(dif)[0]
            examples.append({'time_bjt':key(r['time_bjt']),'split':r['split'],'row':int(yy),'col':int(xx),
                'old_mask':bool(oldmask[i,yy,xx]),'corrected_mask':bool(newmask[i,yy,xx]),
                'corrected_cosSOZ':float(cos[i,yy,xx]),'corrected_day_mask':float(day[i,yy,xx]),
                'old_physical_cot':float(oldy[i,yy,xx]*100),'corrected_physical_cot':float(newy[i,yy,xx]*100)})
    index=read(paths['index']);assert Counter(r['split'] for r in index)=={'train':4086,'val':407}
    byold={key(r['time_bjt']):i for i,r in enumerate(ro)}
    sm=json.loads(paths['side_manifest'].read_text());sidestats={}
    for split,n in (('train',4086),('val',407)):
        sr=read(side/(split+'_keys.csv'));st=np.load(side/(split+'_target_physical.npy'),mmap_mode='r');sk=np.load(side/(split+'_mask.npy'),mmap_mode='r')
        assert sk.dtype==np.bool_ and st.shape==sk.shape==(n,16,1,16,16)
        for filename,mkey in [(split+'_keys.csv','keys_sha256'),(split+'_target_physical.npy','target_sha256'),(split+'_mask.npy','mask_sha256')]:
            assert sha(side/filename)==sm['splits'][split][mkey]
            paths['side_'+filename]=side/filename
        ix=[r for r in index if r['split']==split];assert len(sr)==len(ix)==n
        counter=Counter();compare_count=0
        for i,(r,s) in enumerate(zip(ix,sr)):
            dp=r['data_paths'].split('|');ap=r['aux_paths'].split('|');assert len(dp)==len(ap)==24
            dt=[datetime.strptime(Path(z).stem,'%Y%m%d%H%M')+timedelta(hours=8) for z in dp]
            at=[datetime.strptime(Path(z).name[:12],'%Y%m%d%H%M') for z in ap]
            assert dt==at and all(dt[j+1]-dt[j]==timedelta(minutes=15) for j in range(23))
            future=[t.strftime('%Y%m%d%H%M') for t in dt[8:]]
            assert int(s['sample_index'])==i and s['split']==split
            assert key(s['init_aux_bjt'])==dt[7].strftime('%Y%m%d%H%M')
            assert s['target_times_bjt'].split('|')==future
            for li,t in enumerate(future):
                oi=byold.get(t)
                if oi is None or ro[oi]['split']!=split:
                    assert not sk[i,li].any() and not st[i,li].any();continue
                assert np.array_equal(sk[i,li,0],oldmask[oi])
                assert np.array_equal(st[i,li,0],oldy[oi]*np.float32(100))
                one=compare_masks(np.asarray(sk[i,li,0]),newmask[oi],cos[oi],day[oi])
                counter.update(one);compare_count+=1
        sidestats[split]={'matched_forecast_targets':compare_count,**dict(counter)}
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    if examples:
        with (a.output/'first_differing_pixel_per_source_row.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(examples[0]));w.writeheader();w.writerows(examples)
    result={'state':'COMPLETE_READONLY_CPP_GEOMETRY_SUPPORT_COMPARISON',
        'unique_source_row_stats':source_stats,'repeated_sidecar_target_stats':sidestats,'source_rows_with_mask_difference':len(examples),
        'source_pack_paths':{'old_correct_utc':str(old),'corrected_solar':str(new),'actual_sidecar':str(side)},
        'old_status':json.loads(paths['old_status'].read_text()),'new_status':json.loads(paths['new_status'].read_text()),
        'sources_sha256':{n:sha(p) for n,p in paths.items()},'script_sha256':sha(Path(__file__)),
        'plan_sha256':sha(Path(__file__).with_name('INPUT_SUPPORT_CHECK_PLAN.md')),
        'activity':'read-only train/val support diagnostic; not a model/GHI result',
        'test_payload_opened':False,'buffer_payload_opened':False,'ghi_labels_read':False,'active_files_changed':False}
    (a.output/'support_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources_sha256','old_status','new_status')},indent=2),flush=True)

if __name__=='__main__':main()
