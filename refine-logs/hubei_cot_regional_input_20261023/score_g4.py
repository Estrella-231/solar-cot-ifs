"""Paired physical-COT errors on identical pixel support; no GHI labels."""
import argparse,csv,hashlib,json
from datetime import datetime,timedelta
from pathlib import Path
import numpy as np
from g4_motion import phase,advect,witness

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    root=Path(__file__).resolve().parent;selection=json.loads((root/'G4_SELECTION.json').read_text())
    hist={};pred={};motions={};records=[]
    for p in sorted(a.run.glob('history*/history.npz')):
        with np.load(p) as z:
            for i,k in enumerate(z['keys']):
                assert str(k) not in hist;hist[str(k)]=(z['log_cot'][i].copy(),z['valid'][i].copy(),z['c13'][i].copy())
    assert set(hist)=={r['bjt'] for r in selection['frames']}
    for p in sorted(a.run.glob('forecast*/forecast.npz')):
        with np.load(p) as z:
            assert z['log_cot'].shape==(len(z['ids']),16,3,3,3)
            for i,ix in enumerate(z['ids']):
                assert int(ix) not in pred;pred[int(ix)]=z['log_cot'][i].copy()
    assert set(pred)==set(range(len(selection['initializations'])))
    init_ids={k:i for i,k in enumerate(selection['initializations'])}
    pack=Path('/dev/shm/hubei_regional_r_pack_20261023_attempt2')
    rr={r['time_bjt']:r for r in csv.DictReader((pack/'rows.csv').open())};mask=np.load(pack/'mask.npy',mmap_mode='r')
    sources={r['time_bjt']:r for r in json.loads((root/'source_ready_attempt1/frames.json').read_text())}
    truth_cache={}
    for pair in selection['pairs']:
        init=pair['init_time_BJT'];target=pair['target_time_BJT'];lead=int(pair['lead_minutes']);li=lead//15-1
        if init not in motions:
            keys=pair['history_BJT'].split('|');assert keys[-1]==init
            motions[init]=np.median(np.stack([phase(hist[keys[j-1]][2],hist[keys[j]][2]) for j in range(1,8)]),axis=0)
        motion=motions[init];log,valid,_=hist[init]
        if target not in truth_cache:
            source=root/'source_ready_attempt1/payload/val'/(target+'.npz')
            assert hashlib.sha256(source.read_bytes()).hexdigest()==sources[target]['output_sha256']
            with np.load(source) as z:truth=z['cot'].astype(np.float64)
            truth_cache[target]=(truth,np.asarray(mask[int(rr[target]['index']),0]))
        truth,reference_valid=truth_cache[target]
        for si,(name,(r,c)) in enumerate(zip(['Niushou','Wangzhai','Mizhuang'],[(111,124),(112,126),(110,128)])):
            yy,xx=np.mgrid[r-1:r+2,c-1:c+2];sl=np.s_[r-1:r+2,c-1:c+2]
            transported,good=advect(log,valid,np.array([yy-(lead//15)*motion[0],xx-(lead//15)*motion[1]]))
            persistence=np.expm1(log[sl].astype(np.float64));transport=np.expm1(transported.astype(np.float64))
            forecast=np.expm1(pred[init_ids[init]][li,si].astype(np.float64));ref=truth[sl]
            vm=reference_valid[sl];pm=valid[sl];common=vm&pm&good&np.isfinite(forecast)
            for j in range(3):
                for k in range(3):
                    records.append(dict(init=init,target=target,date=target[:8],station=name,lead=lead,dr=j-1,dc=k-1,
                        reference=float(ref[j,k]) if vm[j,k] else '',reference_valid=int(vm[j,k]),
                        persistence_valid=int(pm[j,k]),transport_valid=int(good[j,k]),common=int(common[j,k]),
                        persistence=float(persistence[j,k]),transport=float(transport[j,k]),forecast=float(forecast[j,k]),
                        motion_dy=float(motion[0]),motion_dx=float(motion[1])))
    assert not a.out.exists();a.out.mkdir(parents=True)
    with (a.out/'paired_pixels.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    # Summaries retain each lead and support; later independent date-block scoring
    # compares equal-station/equal-lead scores, not pooled overlapping records.
    sums=[]
    for scope in ['1x1','3x3']:
      for name in ['Niushou','Wangzhai','Mizhuang']:
       for lead in range(15,241,15):
        rows=[r for r in records if r['station']==name and r['lead']==lead and (scope=='3x3' or r['dr']==r['dc']==0)]
        shared=[r for r in rows if r['common']]
        metrics={}
        for method in ['persistence','transport','forecast']:
            e=np.array([r[method]-r['reference'] for r in shared],dtype=np.float64)
            metrics[method]=dict(n=len(e),sse=float(e@e),sae=float(abs(e).sum()),se=float(e.sum()),
                 rmse=float(np.sqrt(np.mean(e*e))) if len(e) else None)
        sums.append(dict(scope=scope,station=name,lead=lead,reference_n=sum(r['reference_valid'] for r in rows),
              persistence_available_n=sum(r['reference_valid'] and r['persistence_valid'] for r in rows),
              transport_available_n=sum(r['reference_valid'] and r['transport_valid'] for r in rows),metrics=metrics))
    result=dict(state='FROZEN_REGIONAL_COT_ROUTES_COMMON_PIXEL_ERRORS_SAVED',scope='validation future COT, not GHI',
       reference='CPP satellite retrieval, not independent truth',witness=witness(),initializations=len(motions),
       target_dates=sorted({r['date'] for r in records}),pixel_rows=len(records),summaries=sums,
       transport='seven historical full-region C13 phase shifts median; bilinear log1p COT; no fallback',
       common_mask='CPP/spectral target validity & persistence source validity & transported-source validity & finite forecast',
       test_buffer_payloads_read=0,GHI_labels_read=0,training=False)
    (a.out/'REPORT.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='summaries'}),flush=True)
if __name__=='__main__':main()
