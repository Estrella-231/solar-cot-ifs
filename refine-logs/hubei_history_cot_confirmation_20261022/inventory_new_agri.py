"""Bounded filename-only HDF inventory; no science or test/buffer arrays opened."""
import argparse,csv,hashlib,json,os,re,signal,socket
from collections import Counter
from datetime import datetime,timedelta
from pathlib import Path

ROOTS=['/home/Data_Pool/data/FY/FY4B/AGRI/4KM',
       '/home/Data_Pool/data/FY/FY4B/real_time_data/AGRI/4KM']
PAT=re.compile(r'_(\d{14})_(\d{14})_4000M(?:_V\d+)?\.HDF$')
def alarm(*args):raise TimeoutError('directory operation exceeded 8 seconds')
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);signal.signal(signal.SIGALRM,alarm)
    dirs=[];files=[];bytime={};end=datetime(2026,8,19);start=datetime(2026,7,15)
    # BJT starts at UTC previous day 16h. List explicit UTC dates only.
    for n in range(36):
        utcday=datetime(2026,7,14)+timedelta(days=n)
        for root in ROOTS:
            path=Path(root)/utcday.strftime('%Y')/utcday.strftime('%Y%m%d')
            try:
                signal.alarm(8)
                with os.scandir(path) as it:names=sorted(e.name for e in it)
                signal.alarm(0);state='LISTED'
            except (OSError,TimeoutError) as e:
                signal.alarm(0);dirs.append(dict(path=str(path),state=type(e).__name__,detail=str(e)));continue
            count=0
            for name in names:
                m=PAT.search(name)
                if not m:continue
                utc=datetime.strptime(m[1],'%Y%m%d%H%M%S');bjt=utc+timedelta(hours=8)
                if not start<=bjt<end or utc.second or utc.minute%15:continue
                row=dict(time_BJT=bjt.isoformat(' '),start_UTC=utc.isoformat(' '),
                    end_UTC=datetime.strptime(m[2],'%Y%m%d%H%M%S').isoformat(' '),
                    orbital_longitude='1050E' if '_1050E_' in name else '1330E' if '_1330E_' in name else 'OTHER',
                    path=str(path/name),source_root=root)
                files.append(row);bytime.setdefault(bjt,[]).append(row);count+=1
            dirs.append(dict(path=str(path),state=state,matching_files=count))
    with (a.out/'hdf_files.csv').open('w',newline='') as f:
        fields=['time_BJT','start_UTC','end_UTC','orbital_longitude','path','source_root'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(files)
    compatible={t for t,rr in bytime.items() if any(r['orbital_longitude']=='1050E' for r in rr)}
    windows=[];t=start+timedelta(minutes=105)
    while t<end:
        missing=[(t-timedelta(minutes=k)).isoformat(' ') for k in range(105,-1,-15) if t-timedelta(minutes=k) not in compatible]
        windows.append(dict(init_time_BJT=t.isoformat(' '),eight_history_frames_available=int(not missing),
            missing_history_frames='|'.join(missing)))
        t+=timedelta(minutes=15)
    with (a.out/'history_input_support.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(windows[0]));w.writeheader();w.writerows(windows)
    report=dict(state='FILENAME_METADATA_ONLY',host=socket.gethostname(),roots=ROOTS,directories=dirs,
        filenames=len(files),unique_times=len(bytime),compatible_1050E_times=len(compatible),
        orbital_versions=dict(Counter(r['orbital_longitude'] for r in files)),
        daily_compatible=dict(sorted(Counter(t.date().isoformat() for t in compatible).items())),
        complete_eight_frame_initializations=sum(r['eight_history_frames_available'] for r in windows),
        candidate_initializations=len(windows),first_BJT=min(bytime).isoformat(' ') if bytime else None,
        last_BJT=max(bytime).isoformat(' ') if bytime else None,HDF_pixel_payloads_read=0,
        old_test_payloads_read=0,old_buffer_payloads_read=0,
        source_selection='1050E only; historical then realtime after payload equivalence/grid check, not yet performed',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for file in ['hdf_files.csv','history_input_support.csv']:
        report[file+'_sha256']=hashlib.sha256((a.out/file).read_bytes()).hexdigest()
    (a.out/'SATELLITE_INVENTORY.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['directories','daily_compatible']},indent=2))
if __name__=='__main__':main()
