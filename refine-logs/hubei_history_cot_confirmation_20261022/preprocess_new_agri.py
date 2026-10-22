"""Reuse the audited full256 pixel routine on explicit new-source HDFs only."""
import argparse,csv,hashlib,importlib.util,json,os,time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timedelta
from pathlib import Path
from types import SimpleNamespace
import netCDF4 as nc
import numpy as np
from solar_geometry_noaa import compute_solar_angles

GRID=None;CODE=None;OUT=None
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def detect(f):
    if 'Data' in f.groups:
        assert 'Calibration' in f.groups
        return 'group',f.groups['Data'],f.groups['Calibration']
    assert 'NOMChannel01' in f.variables and 'CalChannel01' in f.variables
    return 'flat',f,f
def init(grid,existing,out):
    global GRID,CODE,OUT
    z=np.load(grid,allow_pickle=False)
    GRID=SimpleNamespace(sub_lat=z['lat'],sub_lon=z['lon'],sub_saa=z['saa'],sub_saz=z['saz'],
        sub_lat_idx=z['row_index'],sub_lon_idx=z['col_index'],compute_solar_angles=compute_solar_angles,detect_hdf_format=detect)
    spec=importlib.util.spec_from_file_location('existing_full256',existing);CODE=importlib.util.module_from_spec(spec);spec.loader.exec_module(CODE)
    OUT=Path(out)
def work(row):
    t=datetime.fromisoformat(row['time_BJT']);assert datetime(2026,7,15)<=t<datetime(2026,8,19)
    u=t-timedelta(hours=8);assert u==datetime.fromisoformat(row['start_UTC'])
    assert row['orbital_longitude']=='1050E' and u.strftime('%Y%m%d%H%M%S') in row['path']
    start=time.monotonic();path=Path(row['path']);arr=CODE._make_full_scene(path,GRID)
    # Independent pointwise source DN/LUT check at the three station centers.
    maximum=0.
    with nc.Dataset(path) as f:
        fmt,data,cal=detect(f)
        for rr,cc in [(111,124),(112,126),(110,128)]:
            i,j=int(GRID.sub_lat_idx[rr,cc]),int(GRID.sub_lon_idx[rr,cc])
            for ch in [1,9,13,15]:
                raw=int(np.asarray(data[f'NOMChannel{ch:02d}'][i,j]));raw=4096 if raw in [65534,65535] else min(4096,max(0,raw))
                lut=np.append(np.asarray(cal[('CAL' if fmt=='group' else 'Cal')+f'Channel{ch:02d}'][:],dtype=np.float32),np.float32(0))
                val=lut[raw]
                if ch<=6 and arr[19,rr,cc]<.5:val=np.float32(0)
                maximum=max(maximum,abs(float(arr[ch-1,rr,cc])-float(val)))
    assert maximum==0
    name=u.strftime('%Y%m%d%H%M');dest=OUT/'data'/f'{name}.npy';aux=OUT/'aux'/f'{name}.npz'
    assert not dest.exists() and not aux.exists()
    CODE.atomic_npy(dest,arr)
    g=np.stack((np.cos(np.deg2rad(arr[18])),np.cos(np.deg2rad(arr[17]-arr[15])),arr[19])).astype(np.float32)
    np.savez(aux,cosSOZ=g[0],cosRAA=g[1],day_mask=g[2])
    return dict(time_BJT=row['time_BJT'],utc=name,data_path=str(dest),aux_path=str(aux),
        data_sha256=sha(dest),aux_sha256=sha(aux),source_hdf=str(path),source_size_bytes=path.stat().st_size,
        independent_station_LUT_max_abs_difference=maximum,seconds=time.monotonic()-start)
def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);p.add_argument('--grid',type=Path,required=True)
    p.add_argument('--existing-processor',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--required-times',type=Path);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'data').mkdir();(a.out/'aux').mkdir()
    bytime={}
    for r in csv.DictReader(a.inventory.open()):
        if r['orbital_longitude']=='1050E':bytime.setdefault(r['time_BJT'],[]).append(r)
    jobs=[min(v,key=lambda r:('real_time_data' not in r['path'],r['path'])) for _,v in sorted(bytime.items())]
    if a.required_times:
        needed=set(json.loads(a.required_times.read_text())['history_times_BJT'])
        jobs=[j for j in jobs if j['time_BJT'] in needed]
        assert {j['time_BJT'] for j in jobs}==needed
    if a.max_frames:jobs=jobs[:a.max_frames]
    (a.out/'SOURCE_SELECTION.json').write_text(json.dumps(dict(priority='realtime first, historical fallback: match original audited preprocessing',
        inventory_note='inventory_attempt1 source_selection was a proposed order only; no payload selected then',
        frames=len(jobs),grid_sha256=sha(a.grid),existing_routine_sha256=sha(a.existing_processor),script_sha256=sha(__file__),
        solar_geometry_sha256=sha(Path(__file__).parent/'solar_geometry_noaa.py'),source_inventory_sha256=sha(a.inventory)),indent=2))
    started=time.monotonic();done=[];failures=[]
    with ProcessPoolExecutor(max_workers=a.workers,initializer=init,initargs=(str(a.grid),str(a.existing_processor),str(a.out))) as pool:
        pending={pool.submit(work,j):j for j in jobs}
        for future in as_completed(pending):
            try:done.append(future.result())
            except Exception as e:
                failures.append(dict(source=pending[future]['path'],error=repr(e)))
            if (len(done)+len(failures))%25==0:
                print(json.dumps(dict(done=len(done),failed=len(failures),total=len(jobs),seconds=time.monotonic()-started)),flush=True)
    done.sort(key=lambda r:r['time_BJT'])
    if done:
        with (a.out/'frame_manifest.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(done[0]));w.writeheader();w.writerows(done)
    report=dict(state='COMPLETE' if not failures else 'FAILED_RETAIN_PARTIAL',written=len(done),expected=len(jobs),failures=failures,
        workers=a.workers,seconds=time.monotonic()-started,frame_manifest_sha256=sha(a.out/'frame_manifest.csv') if done else None,
        old_test_payloads_read=0,old_buffer_payloads_read=0,new_HDF_payloads_read=len(done)+len(failures),
        output_shape=[20,256,256],dtype='float32')
    (a.out/'PREPROCESS_REPORT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if failures:raise RuntimeError('Preserved failed attempt; inspect PREPROCESS_REPORT before retry')
if __name__=='__main__':main()
