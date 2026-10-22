"""No-training Hubei source-domain diagnostic; CPP history is not frozen-R input.

One fixed 10:00 BJT initialization per eligible validation date, independent of
CPP availability or weather. Motion: inherited seven-pair C13 phase median.
Only source domain changes; missing source never becomes clear-sky zero.
"""
import argparse,csv,json,hashlib,time
from pathlib import Path
from datetime import datetime,timedelta,timezone
from functools import lru_cache
import numpy as np
import netCDF4
from scipy.ndimage import map_coordinates

STATIONS={'Niushou':(111,124),'Wangzhai':(112,126),'Mizhuang':(110,128)}
ROOT=Path('/home/Data_Pool_3/chenyi/Auxiliary_data/HuBei_AGRI_Preprocessed')
CPP=Path('/home/Data_Pool_3/data/Global_CPP/FY/FY4B/Result/CPP')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def phase(a,b):
    a=np.nan_to_num(a.astype(np.float64)); b=np.nan_to_num(b.astype(np.float64))
    win=np.outer(np.hanning(a.shape[0]),np.hanning(a.shape[1]))
    a=(a-a.mean())*win; b=(b-b.mean())*win
    f=np.fft.fft2(b)*np.conj(np.fft.fft2(a)); f/=np.maximum(abs(f),1e-12)
    c=np.fft.ifft2(f).real; y,x=np.unravel_index(c.argmax(),c.shape)
    def sub(v,m,p):
        d=m-2*v+p
        return .5*(m-p)/d if abs(d)>1e-12 else 0.
    dy=y+sub(c[y,x],c[(y-1)%256,x],c[(y+1)%256,x])
    dx=x+sub(c[y,x],c[y,(x-1)%256],c[y,(x+1)%256])
    return np.array([dy-256 if dy>128 else dy,dx-256 if dx>128 else dx])
def cosz(t,lat,lon):
    # NOAA fractional-year declination and equation of time; UTC input.
    u=t-timedelta(hours=8); hour=u.hour+u.minute/60
    g=2*np.pi/365*(u.timetuple().tm_yday-1+(hour-12)/24)
    dec=.006918-.399912*np.cos(g)+.070257*np.sin(g)-.006758*np.cos(2*g)+.000907*np.sin(2*g)-.002697*np.cos(3*g)+.00148*np.sin(3*g)
    eq=229.18*(.000075+.001868*np.cos(g)-.032077*np.sin(g)-.014615*np.cos(2*g)-.040849*np.sin(2*g))
    ha=np.deg2rad((hour*60+eq+4*lon)/4-180); la=np.deg2rad(lat)
    return np.sin(la)*np.sin(dec)+np.cos(la)*np.cos(dec)*np.cos(ha)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(open(a.manifest,encoding='utf-8-sig')))
    allowed={r['time_bjt']:r['split'] for r in rows}
    assert len(allowed)==len(rows) and set(allowed.values())=={'train','val'}
    grid=np.load(ROOT/'grid_static.npz');lat=grid['lat'];lon=grid['lon']
    candidates=[]
    for day in sorted({k[:8] for k,s in allowed.items() if s=='val'}):
        t=datetime.strptime(day+'1000','%Y%m%d%H%M')
        keys=[(t+timedelta(minutes=15*k)).strftime('%Y%m%d%H%M') for k in range(-7,17)]
        if all(allowed.get(k)=='val' for k in keys):candidates.append(t)
    contract={'scope':'exploratory validation CPP-history source-domain diagnostic, NOT R-chain or GHI skill',
        'manifest_sha256':sha(a.manifest),'script_sha256':sha(__file__),'grid_sha256':sha(ROOT/'grid_static.npz'),
        'init_bjt':[t.isoformat() for t in candidates],'selection':'10:00 BJT each validation date with 24 authorized AGRI timestamps, before any CPP read',
        'domains':{'16':[103,118,16,16],'64':[79,94,64,64],'256':[0,0,256,256]},
        'motion':'median of seven full-region consecutive historical raw C13 phase-correlation shifts (20-channel raw index 12)',
        'interpolation':'bilinear physical COT; all interpolation weight must have finite [0,100] source',
        'stations':STATIONS,'daylight':'NOAA UTC solar cosine > cos(80 degrees)',
        'test_payloads_read':0,'buffer_payloads_read':0,'training':False}
    (out/'contract.json').write_text(json.dumps(contract,indent=2))
    def agri(t):
        k=t.strftime('%Y%m%d%H%M');assert allowed.get(k)=='val'
        p=ROOT/'data'/k[:4]/k[:6]/k[:8]/(k+'.npy')
        x=np.load(p,mmap_mode='r');assert x.shape==(20,256,256)
        assert np.array_equal(x[15],grid['saa']) and np.array_equal(x[16],grid['saz'])
        return np.asarray(x[12],dtype=np.float32).copy()
    sources=[]
    @lru_cache(maxsize=20)
    def cpp(t):
        k=t.strftime('%Y%m%d%H%M');assert allowed.get(k)=='val'
        u=t-timedelta(hours=8);p=CPP/u.strftime('%Y')/u.strftime('%Y%m%d')/('FY4B_AGRI_'+u.strftime('%Y%m%d%H%M%S')+'.nc')
        if not p.is_file():
            sources.append({'key':k,'path':str(p),'state':'missing'});return None
        with netCDF4.Dataset(p) as ds:
            la=np.asarray(ds['LAT'][:]);lo=np.asarray(ds['LON'][:])
            ri=np.abs(la[:,None]-lat[:,0]).argmin(0);ci=np.abs(lo[:,None]-lon[0,:]).argmin(0)
            err=max(np.max(abs(la[ri,None]-lat)),np.max(abs(lo[ci][None,:]-lon)))
            assert err<1e-4 and len(set(ri))==256 and len(set(ci))==256
            r0,r1=int(ri.min()),int(ri.max())+1;c0,c1=int(ci.min()),int(ci.max())+1
            v=np.ma.filled(ds['COT'][r0:r1,c0:c1],np.nan).astype(np.float32)[np.ix_(ri-r0,ci-c0)]
            v[~np.isfinite(v)|(v<0)|(v>100)]=np.nan
            sources.append({'key':k,'path':str(p),'state':'read','source_bytes':p.stat().st_size,'max_coordinate_error_deg':float(err),'region_valid_fraction':float(np.isfinite(v).mean())})
            return v
    records=[];failures=[];witness_done=False
    for ii,t in enumerate(candidates):
        start=time.time();init=t.strftime('%Y%m%d%H%M')
        try:
            hist=[agri(t+timedelta(minutes=15*k)) for k in range(-7,1)]
            pairs=np.stack([phase(hist[k-1],hist[k]) for k in range(1,8)]);motion=np.median(pairs,axis=0)
            base=cpp(t)
            if base is None:failures.append({'init':init,'reason':'missing historical CPP'});continue
            # Sign and interpolation witness independent of the weather data.
            if not witness_done:
                z=np.zeros((256,256));z[111,124]=1
                assert phase(z,np.roll(z,(2,-3),(0,1)))[0]>1.9
                assert np.allclose(map_coordinates(z,[[113-2],[121+3]],order=1),1)
                witness_done=True
            truths=[];preds=[];masks=[]
            for lead in range(1,17):
                target=t+timedelta(minutes=15*lead);ref=cpp(target)
                tr=np.full((3,3,3),np.nan);pr=np.full((4,3,3,3),np.nan);day=np.zeros((3,3,3),bool)
                for si,(name,(r,c)) in enumerate(STATIONS.items()):
                    yy,xx=np.mgrid[r-1:r+2,c-1:c+2];sy=yy-lead*motion[0];sx=xx-lead*motion[1]
                    coords=np.array([sy,sx]); val=map_coordinates(np.nan_to_num(base,nan=0),coords,order=1,mode='constant',cval=0)
                    good=map_coordinates(np.isfinite(base).astype(float),coords,order=1,mode='constant',cval=0)>1-1e-6
                    day[si]=cosz(target,lat[yy,xx],lon[yy,xx])>np.cos(np.deg2rad(80))
                    if ref is not None:tr[si]=ref[yy,xx]
                    pr[0,si]=base[yy,xx]
                    for mi,(sz,(r0,c0,h,w)) in enumerate(contract['domains'].items(),1):
                        inside=(sy>=r0)&(sy<=r0+h-1)&(sx>=c0)&(sx<=c0+w-1)
                        pr[mi,si]=np.where(good&inside,val,np.nan)
                    for py in range(3):
                        for px in range(3):
                            rec={'init_bjt':init,'target_bjt':target.strftime('%Y%m%d%H%M'),'date':target.strftime('%Y%m%d'),'station':name,'lead_min':15*lead,'dy':float(motion[0]),'dx':float(motion[1]),'dr':py-1,'dc':px-1,'day':int(day[si,py,px]),'reference':float(tr[si,py,px])}
                            rec.update({n:float(pr[j,si,py,px]) for j,n in enumerate(['persistence','domain16','domain64','domain256'])});records.append(rec)
                truths.append(tr);preds.append(pr);masks.append(day)
            np.savez_compressed(out/(init+'.npz'),base_cpp=base,motion=motion,pair_motion=pairs,reference=np.stack(truths),prediction=np.stack(preds),day_mask=np.stack(masks),history_c13=np.stack(hist))
            print(json.dumps({'completed':ii+1,'total':len(candidates),'init':init,'seconds':round(time.time()-start,2),'motion':motion.tolist()}),flush=True)
        except (OSError,ValueError) as e:
            failures.append({'init':init,'reason':repr(e)});print('CASE_FAILURE',failures[-1],flush=True)
    if records:
        with (out/'paired_pixels.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (out/'source_inventory.json').write_text(json.dumps(sources,indent=2))
    (out/'completion.json').write_text(json.dumps({'status':'COMPLETE','cases':len(candidates),'saved_cases':len(list(out.glob('*.npz'))),'records':len(records),'failures':failures,'motion_sign_witness':witness_done,'test_payloads_read':0,'buffer_payloads_read':0},indent=2))
    print('DOMAIN_COMPARISON_COMPLETE',flush=True)
if __name__=='__main__':main()
